"""Inference-only ViCo/PyramidDrop adaptation for Qwen2 / Transformers 4.46.1.

Algorithm reference: https://github.com/Cooperx521/PyramidDrop
This implementation uses independent FP32 ranking rows with the existing model's
Q/K weights; it never changes the decoder dtype or attention implementation.
Single unpadded image prompt, batch size one, DynamicCache, ordinary decoding.
"""

import math

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .fastv_attention import last_prompt_attention
from .pruning_timing import measure_pruning


class ViCoMixin:
    def init_vico(self):
        self.vico_enabled = False
        self.vico_configured = False
        self.vico_layers = ()
        self.vico_keep_ratios = ()
        self.vico_min_tokens = 1
        self.vico_capture_visualization = False
        self.vico_capture_attention = False
        self.reset_vico_state()

    def reset_vico_state(self):
        self._vico_image_spans = []
        self._vico_prompt_length = None
        self._vico_stats = {}
        self._vico_stages = []

    def configure_vico(self, *, enabled, layers, keep_ratios, min_tokens=1,
                       capture_visualization=False, capture_attention=False):
        layers, ratios = tuple(layers), tuple(keep_ratios)
        if (not layers or any(type(n) is not int or not 1 <= n < len(self.layers) for n in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError("ViCo layers must be increasing 1-based boundaries below the final layer")
        if (len(ratios) != len(layers) or any(not math.isfinite(r) or not 0 < r <= 1 for r in ratios)
                or any(a < b for a, b in zip(ratios, ratios[1:]))):
            raise ValueError("ViCo keep_ratios must be positive, non-increasing cumulative fractions")
        if type(min_tokens) is not int or min_tokens < 1:
            raise ValueError("ViCo min_tokens must be a positive integer")
        self.fastv_enabled = False
        self.reset_fastv_state()
        self.vico_configured = True
        self.vico_enabled = bool(enabled) and ratios[-1] < 1.0
        self.vico_layers = layers
        self.vico_keep_ratios = ratios
        self.vico_min_tokens = min_tokens
        self.vico_capture_visualization = bool(capture_visualization or capture_attention)
        self.vico_capture_attention = bool(capture_attention)
        if self.vico_enabled and len({layer.self_attn.q_proj.weight.device for layer in self.layers}) != 1:
            raise ValueError("ViCo currently requires all decoder layers on one device; set CUDA_VISIBLE_DEVICES to one GPU")

    def set_vico_image_spans(self, spans, prompt_length=None):
        self._vico_image_spans = spans
        self._vico_prompt_length = prompt_length

    def get_vico_stats(self):
        if self.vico_enabled:
            return dict(self._vico_stats)
        spans = self._vico_image_spans[0] if self._vico_image_spans else []
        count = sum(end - start for start, end in spans)
        length = self._vico_prompt_length
        return {
            "mode": "disabled_baseline", "keep_ratio": 1.0,
            "rate_semantics": "final_cumulative", "stages": [],
            "original_image_tokens": count,
            "layers": [self._vico_layer_record(i + 1, count, count, length, length, count)
                       for i in range(len(self.layers))],
        }

    @staticmethod
    def _vico_layer_record(layer, before, after, seq_before, seq_after, original):
        return {"layer": layer, "image_tokens_in": before, "image_tokens_out": after,
                "sequence_tokens_in": seq_before, "sequence_tokens_out": seq_after,
                "removed_after_layer": before - after,
                "cumulative_prune_rate": 100.0 * (1 - after / original) if original else 0.0}

    def get_vico_stages(self):
        if not self._vico_stages:
            raise RuntimeError("No ViCo stage visualization captured")
        start, end = self._vico_image_spans[0][0]
        count = end - start
        result = []
        for item in self._vico_stages:
            kept = item["kept_original_indices"].detach().cpu()
            keep = torch.zeros(count, dtype=torch.bool)
            keep[kept] = True
            stage = dict(item["stats"])
            stage["mask"] = {"span": [start, end], "keep": keep.tolist()}
            if "scores" in item:
                active = item["active_original_indices"].detach().cpu()
                scores = torch.zeros(count, dtype=torch.float32)
                valid = torch.zeros(count, dtype=torch.bool)
                scores[active] = item["scores"].detach().float().cpu()
                valid[active] = True
                stage["attention"] = {"span": [start, end], "scores": scores.tolist(),
                                      "valid": valid.tolist()}
            result.append(stage)
        return result

    @staticmethod
    def _vico_causal_mask(hidden, past_length, implementation):
        # No padding is supported on this inference path. FlashAttention uses
        # its native causal kernel; the small-model CPU tests can use SDPA/eager.
        if implementation == "flash_attention_2":
            return None
        q_len = hidden.shape[1]
        keys = torch.arange(past_length + q_len, device=hidden.device)
        queries = torch.arange(past_length, past_length + q_len, device=hidden.device)
        visible = keys[None, :] <= queries[:, None]
        return torch.zeros((1, 1, q_len, past_length + q_len),
                           dtype=hidden.dtype, device=hidden.device).masked_fill(
                               ~visible[None, None], torch.finfo(hidden.dtype).min)

    def vico_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, use_cache=None,
                     output_attentions=None, output_hidden_states=None,
                     return_dict=None, cache_position=None):
        if self.training:
            raise ValueError("This ViCo adapter is inference-only; call eval()")
        if output_attentions or (output_attentions is None and self.config.output_attentions):
            raise ValueError("ViCo computes independent Q/K scores; output_attentions must be False")
        if getattr(self.config, "use_sliding_window", False):
            raise ValueError("ViCo adapter does not support sliding-window attention")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (self.config.output_hidden_states if output_hidden_states is None
                                else output_hidden_states)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("ViCo currently supports batch size 1 (no beam expansion)")
        if attention_mask is not None and (attention_mask.ndim != 2 or not attention_mask.bool().all()):
            raise ValueError("ViCo currently requires an unpadded single-image prompt")
        legacy = use_cache and not isinstance(past_key_values, Cache)
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("ViCo requires DynamicCache, not static/sliding/quantized caches")
        prefill = past_key_values is None or past_key_values.get_seq_length(0) == 0
        if len(self._vico_image_spans) != 1 or len(self._vico_image_spans[0]) != 1:
            raise ValueError("ViCo requires one image-token span for one prompt")
        start, end = self._vico_image_spans[0][0]
        original = end - start
        hidden = inputs_embeds
        if prefill:
            if not 0 <= start < end < hidden.shape[1]:
                raise ValueError("ViCo requires visual tokens followed by at least one prompt text token")
            if position_ids is not None and not torch.equal(
                    position_ids, torch.arange(hidden.shape[1], device=hidden.device)[None]):
                raise ValueError("ViCo requires contiguous zero-based prefill position IDs")
            self._vico_stats = {"mode": "physical_drop", "attention_source": "independent_qk",
                                "score_dtype": "float32", "rate_semantics": "final_cumulative",
                                "original_image_tokens": original,
                                "keep_ratio": self.vico_keep_ratios[-1],
                                "position_policy": "reindex_after_each_drop", "stages": [], "layers": []}
            self._vico_stages = []
            active = torch.arange(original, device=hidden.device)
        elif not self._vico_stats:
            raise ValueError("ViCo cache continuation requires its matching prefill state")
        all_hidden = () if output_hidden_states else None
        rotary_key = None
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (hidden,)
            past_length = past_key_values.get_seq_length(index) if past_key_values is not None else 0
            # Each stage has its own compacted cache length. Never use layer 0's
            # full prompt length for deeper stages during autoregressive decode.
            key = (past_length, hidden.shape[1])
            if key != rotary_key:
                positions = torch.arange(past_length, past_length + hidden.shape[1], device=hidden.device)
                ids = positions[None]
                rotary = self.rotary_emb(hidden, ids)
                causal = self._vico_causal_mask(hidden, past_length, self.config._attn_implementation)
                rotary_key = key
            layer_output = layer(hidden, attention_mask=causal, position_ids=ids,
                                 past_key_value=past_key_values, output_attentions=False,
                                 use_cache=use_cache, cache_position=positions,
                                 position_embeddings=rotary)
            hidden = layer_output[0]
            if not prefill:
                continue
            before, seq_before = active.numel(), hidden.shape[1]
            if index + 1 in self.vico_layers:
                with measure_pruning(self, hidden.device):
                    stage_index = self.vico_layers.index(index + 1)
                    ratio = self.vico_keep_ratios[stage_index]
                    # Cumulative targets are based on the ORIGINAL packed image span
                    # (including structural newlines), never the already reduced span.
                    keep_count = min(before, max(self.vico_min_tokens, math.ceil(original * ratio - 1e-10)))
                    # Official pdrop_rank_drop uses the NEXT layer's norm and Q/K
                    # projections on the output of this completed boundary layer.
                    scores = last_prompt_attention(self.layers[index + 1], hidden, rotary)
                    visual_scores = scores[0, start:start + before]
                    selected = visual_scores.topk(keep_count, sorted=False).indices.sort().values
                    kept_original = active[selected]
                    stage_stats = {"after_layer": index + 1, "scoring_layer": index + 2,
                                   "target_keep_ratio": ratio, "image_tokens_before": before,
                                   "image_tokens_after": keep_count, "removed_this_stage": before - keep_count,
                                   "cumulative_prune_rate": 100.0 * (1 - keep_count / original)}
                    self._vico_stats["stages"].append(stage_stats)
                    if self.vico_capture_visualization:
                        item = {"stats": stage_stats, "kept_original_indices": kept_original}
                        if self.vico_capture_attention:
                            item.update(active_original_indices=active, scores=visual_scores)
                        self._vico_stages.append(item)
                    selection = torch.cat((torch.arange(start, device=hidden.device), start + selected,
                                           torch.arange(start + before, hidden.shape[1], device=hidden.device)))
                    hidden = hidden.index_select(1, selection)
                    active = kept_original
                # Reindexing on the next iteration follows the author's policy.
                rotary_key = None
            self._vico_stats["layers"].append(self._vico_layer_record(
                index + 1, before, active.numel(), seq_before, hidden.shape[1], original))
        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden += (hidden,)
        cache = past_key_values if use_cache else None
        if legacy:
            cache = cache.to_legacy_cache()
        if not return_dict:
            return tuple(value for value in (hidden, cache, all_hidden) if value is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache,
                                       hidden_states=all_hidden, attentions=None)
