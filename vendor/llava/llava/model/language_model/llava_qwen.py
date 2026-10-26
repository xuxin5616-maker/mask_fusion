#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from .fastv_attention import last_prompt_attention
from .pruning_timing import measure_pruning
from .vico import ViCoMixin

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(ViCoMixin, LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

        # FastV is opt-in.  These values are intentionally runtime-only so an
        # ordinary LLaVA checkpoint can be loaded without editing config.json.
        self.fastv_enabled = False
        self.fastv_layer = 2
        self.fastv_keep_ratio = 0.5
        self.fastv_min_tokens = 1
        self.fastv_preserve_image_newline = True
        self.fastv_capture_attention = False
        self._fastv_image_spans = []
        self._fastv_keep_mask = None
        self._fastv_image_attentions = []
        self._fastv_stats = {}
        self.init_vico()

    def configure_fastv(
        self,
        enabled=False,
        layer=2,
        keep_ratio=0.5,
        min_tokens=1,
        preserve_image_newline=True,
        capture_attention=False,
    ):
        """Configure FastV mask evaluation for subsequent ``generate`` calls.

        ``layer`` is the first decoder layer that receives the reduced visual
        key set. Ranking is independently recomputed from the input and Q/K
        projections of ``layer - 1``; decoder attention outputs are never read.
        This follows the layer convention in the original FastV implementation.
        """
        layer = int(layer)
        keep_ratio = float(keep_ratio)
        min_tokens = int(min_tokens)

        if enabled and not 1 <= layer < len(self.layers):
            raise ValueError(
                f"fastv_layer must be in [1, {len(self.layers) - 1}], got {layer}"
            )
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"fastv_keep_ratio must be in (0, 1], got {keep_ratio}")
        if min_tokens < 1:
            raise ValueError(f"fastv_min_tokens must be >= 1, got {min_tokens}")

        self.vico_enabled = False
        self.vico_configured = False
        self.fastv_enabled = bool(enabled)
        self.fastv_layer = layer
        self.fastv_keep_ratio = keep_ratio
        self.fastv_min_tokens = min_tokens
        self.fastv_preserve_image_newline = bool(preserve_image_newline)
        self.fastv_capture_attention = bool(capture_attention)
        self.reset_fastv_state()

    def reset_fastv_state(self):
        """Clear sample-specific spans, masks and statistics before generation."""
        self._fastv_image_spans = []
        self._fastv_keep_mask = None
        self._fastv_image_attentions = []
        self._fastv_stats = {}
        self.reset_vico_state()

    def set_fastv_image_spans(self, image_spans):
        """Receive dynamic image-token spans produced by multimodal packing.

        The structure is ``batch -> images -> (start, end)`` and uses the final
        padded multimodal sequence coordinates.  End positions are exclusive.
        """
        self._fastv_image_spans = image_spans

    def get_fastv_stats(self):
        return dict(self._fastv_stats)

    def get_fastv_image_masks(self):
        """Copy the actual per-image decisions to CPU, only when requested.

        Coordinates include prompt padding; each keep list follows the packed
        image token order, including structural tokens such as image_newline.
        No attention hooks or second ranking pass are needed for visualization.
        """
        if self._fastv_keep_mask is None:
            raise RuntimeError("No FastV mask is available; generate an image prompt first.")
        keep_mask = self._fastv_keep_mask.detach().cpu()
        return [
            [
                {"span": [int(start), int(end)],
                 "keep": keep_mask[batch_index, start:end].tolist()}
                for start, end in spans
            ]
            for batch_index, spans in enumerate(self._fastv_image_spans)
        ]

    def get_fastv_image_attentions(self):
        """Return the ranking layer's last-prompt-token attention per image."""
        if not self._fastv_image_attentions:
            raise RuntimeError("No FastV attention is available; enable visualization and generate an image prompt first.")
        return [
            [
                {"span": list(image["span"]), "scores": list(image["scores"])}
                for image in batch
            ]
            for batch in self._fastv_image_attentions
        ]

    def _build_fastv_keep_mask(self, scores):
        # Scores already contain the independently computed last-query row,
        # averaged over heads, rather than a decoder's full attention matrix.
        batch_size, key_length = scores.shape
        if len(self._fastv_image_spans) != batch_size:
            raise RuntimeError(
                "FastV image spans do not match the model batch: "
                f"{len(self._fastv_image_spans)} spans for batch size {batch_size}."
            )

        keep_mask = torch.ones(
            (batch_size, key_length), dtype=torch.bool, device=scores.device
        )
        batch_stats = []
        captured_attention = []

        for batch_index, spans in enumerate(self._fastv_image_spans):
            token_scores = scores[batch_index]
            captured_images = []
            image_tokens = 0
            kept_image_tokens = 0
            normalized_spans = []

            for raw_start, raw_end in spans:
                start = max(0, min(int(raw_start), key_length))
                end = max(start, min(int(raw_end), key_length))
                if end <= start:
                    continue

                # LLaVA randomroi with add_newl appends one global image-newline
                # token.  It is structural rather than a patch, so keep it.
                protected_tokens = 1 if self.fastv_preserve_image_newline else 0
                protected_tokens = min(protected_tokens, end - start)
                candidate_end = end - protected_tokens
                candidate_count = candidate_end - start

                image_tokens += end - start
                if candidate_count > 0:
                    keep_count = max(
                        self.fastv_min_tokens,
                        int(math.ceil(candidate_count * self.fastv_keep_ratio)),
                    )
                    keep_count = min(keep_count, candidate_count)
                    keep_mask[batch_index, start:candidate_end] = False
                    selected = torch.topk(
                        token_scores[start:candidate_end], keep_count, sorted=False
                    ).indices + start
                    keep_mask[batch_index, selected] = True
                else:
                    keep_count = 0

                kept_image_tokens += keep_count + protected_tokens
                normalized_spans.append([start, end])
                if getattr(self, "fastv_capture_attention", False):
                    captured_images.append({
                        "span": [start, end],
                        "scores": token_scores[start:end].detach().cpu().tolist(),
                    })

            batch_stats.append(
                {
                    "image_spans": normalized_spans,
                    "image_tokens": image_tokens,
                    "kept_image_tokens": kept_image_tokens,
                    "masked_image_tokens": image_tokens - kept_image_tokens,
                }
            )
            captured_attention.append(captured_images)

        self._fastv_keep_mask = keep_mask
        self._fastv_image_attentions = captured_attention if getattr(self, "fastv_capture_attention", False) else []
        self._fastv_stats = {
            "mode": "mask",
            "attention_source": "independent_qk",
            "score_dtype": "float32",
            "fastv_layer": self.fastv_layer,
            "keep_ratio": self.fastv_keep_ratio,
            "prompt_tokens": key_length,
            "batch": batch_stats,
        }

    def _apply_fastv_mask(self, causal_mask, key_length):
        if self._fastv_keep_mask is None:
            return causal_mask
        keep_mask = self._fastv_keep_mask
        if keep_mask.shape[-1] < key_length:
            # Tokens generated after the prompt are never removed by FastV.
            generated_keep = torch.ones(
                (keep_mask.shape[0], key_length - keep_mask.shape[-1]),
                dtype=torch.bool,
                device=keep_mask.device,
            )
            keep_mask = torch.cat((keep_mask, generated_keep), dim=-1)
        else:
            keep_mask = keep_mask[:, :key_length]

        if self.config._attn_implementation == "flash_attention_2":
            # FlashAttention accepts a 2D key/padding mask, not a 4D bias.
            # Its prefill kernel also unpads the discarded query positions;
            # those positions remain masked in all later layers/cache steps.
            if causal_mask is None:
                return keep_mask
            if causal_mask.ndim != 2:
                raise RuntimeError("FlashAttention2 requires a two-dimensional mask")
            return causal_mask[:, :key_length].to(keep_mask.device).bool() & keep_mask
        if causal_mask is None or causal_mask.ndim != 4:
            raise RuntimeError("Non-flash attention requires a four-dimensional mask")
        # SDPA's explicit mask can have a spare column during cache decoding.
        extra = causal_mask.shape[-1] - keep_mask.shape[-1]
        if extra > 0:
            keep_mask = torch.nn.functional.pad(keep_mask, (0, extra), value=True)
        min_dtype = torch.finfo(causal_mask.dtype).min
        return causal_mask.masked_fill(~keep_mask[:, None, None, :].to(causal_mask.device), min_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Original Qwen2 at rate zero; independent scoring at nonzero rates.

        The surrounding flow intentionally mirrors Transformers 4.46.1.  The
        sequence itself is not shortened, which keeps DynamicCache generation
        correct and makes this implementation suitable for accuracy evaluation.
        """
        if self.vico_enabled:
            return self.vico_forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position,
            )
        if not self.fastv_enabled or not any(self._fastv_image_spans):
            # Crucial baseline invariant: do not run our decoder loop, scoring
            # helper, or mask builder when FastV is disabled.
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
                cache_position=cache_position,
            )
        if self.training:
            raise RuntimeError("This FastV implementation is inference-only")
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        if output_attentions:
            raise ValueError("FastV uses independent Q/K scores; output_attentions must be False")
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = (
                DynamicCache()
                if past_key_values is None
                else DynamicCache.from_legacy_cache(past_key_values)
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        fastv_active = self.fastv_enabled and bool(self._fastv_image_spans)
        fastv_prefill = (
            fastv_active
            and past_seen_tokens == 0
            and inputs_embeds.shape[1] > 1
            and self._fastv_keep_mask is None
        )
        # FlashAttention returns a 2D mask (or None) here and never falls back
        # to eager attention. Other implementations need an explicit 4D mask.
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            self.config._attn_implementation != "flash_attention_2",
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = None

        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if fastv_prefill and layer_index == self.fastv_layer - 1:
                with measure_pruning(self, hidden_states.device):
                    scores = last_prompt_attention(
                        decoder_layer, hidden_states, position_embeddings, attention_mask,
                    )
                    self._build_fastv_keep_mask(scores)
            layer_causal_mask = causal_mask
            if fastv_active and layer_index >= self.fastv_layer:
                with measure_pruning(self, hidden_states.device):
                    layer_causal_mask = self._apply_fastv_mask(
                        causal_mask, past_seen_tokens + inputs_embeds.shape[1],
                    )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]  # KV cache, not attention weights.

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, next_cache, all_hidden_states)
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        masks=None,
        mask_valid=None,
        mask_images=None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, masks=masks, mask_valid=mask_valid, mask_images=mask_images)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **loss_kwargs,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        masks=None,
        mask_valid=None,
        mask_images=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        self.get_model().reset_fastv_state()

        if images is not None:
            # Preserve original reference LLaVA's generation inputs, including its None
            # mask/position_ids behavior. HF prepares the mask after packing.
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, masks=masks, mask_valid=mask_valid, mask_images=mask_images)
        else:
            if masks is not None or mask_valid is not None or mask_images is not None:
                raise ValueError("Masks require images")
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if hasattr(self.get_model(), "mask_fusion"):
            limit = min(getattr(self.config, "tokenizer_model_max_length", None)
                        or self.config.max_position_embeddings, self.config.max_position_embeddings)
            if inputs_embeds.shape[1] + int(kwargs.get("max_new_tokens", 20)) > limit:
                raise ValueError("Packed prompt plus max_new_tokens exceeds the model context limit")
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
