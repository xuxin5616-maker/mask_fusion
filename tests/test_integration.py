"""Offline integration tests through the real vendored Qwen forward/generate.

The tiny randomly initialized language model and fake vision tower require no
checkpoint, tokenizer, network access, or CUDA device.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from mask_fusion.config import FusionConfig
from mask_fusion.loader import activate_vendor
from mask_fusion.model import attach_fusion, configure_image_mode, load_adapter, save_adapter

activate_vendor()
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM


class FakeVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 12
        self.config = SimpleNamespace(patch_size=2, image_size=4, hidden_size=12)
        self.image_size = 4
        self.num_patches = 4
        self.num_patches_per_side = 2
        self.is_loaded = True
        self.patch_projection = nn.Conv2d(3, 12, kernel_size=2, stride=2)
        self.forward_calls = 0
        self.encoded_batch_sizes = []

    def forward(self, pixels):
        self.forward_calls += 1
        self.encoded_batch_sizes.append(pixels.shape[0])
        return self.patch_projection(pixels).flatten(2).transpose(1, 2)


def make_tiny_model(seed=101, fusion_config=None):
    torch.manual_seed(seed)
    config = LlavaQwenConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, attention_dropout=0.0, use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config.tokenizer_model_max_length = 64
    config.tokenizer_padding_side = "right"
    config.mm_hidden_size = 12
    configure_image_mode(config, "square")
    # No mm_vision_tower attribute: construction never loads pretrained weights.
    model = LlavaQwenForCausalLM(config)
    core = model.get_model()
    core.vision_tower = FakeVisionTower()
    core.mm_projector = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 32))
    if fusion_config is None:
        fusion_config = FusionConfig(
            mask_hidden_size=8, expert_dim=16, position_dim=8,
            position_hidden_size=16, attention_dim=16, num_heads=4,
        )
    attach_fusion(model, fusion_config)
    return model.eval()


def make_batch(crop_counts=(1,), seed=37):
    generator = torch.Generator().manual_seed(seed)
    batch = len(crop_counts)
    images = [torch.randn(count, 3, 4, 4, generator=generator) for count in crop_counts]
    masks = [torch.rand(count, 1, 4, 4, generator=generator) * 255 for count in crop_counts]
    valid = [torch.ones(count, 1, 4, 4, dtype=torch.bool) for count in crop_counts]
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 7, 8, 9]] * batch)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 8, 9]] * batch)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "images": images,
        "masks": masks,
        "mask_valid": valid,
        "image_sizes": [(4, 4)] * batch,
        "modalities": ["image"] * batch,
    }


class TinyQwenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_forward_packs_images_and_preserves_answer_supervision(self):
        model = make_tiny_model()
        batch = make_batch()
        result = model(**batch, use_cache=False)
        self.assertEqual(tuple(result.logits.shape), (1, 8, 64))
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(model.get_vision_tower().forward_calls, 1)
        packed = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, batch["labels"],
            batch["images"], batch["modalities"], batch["image_sizes"],
            masks=batch["masks"], mask_valid=batch["mask_valid"],
        )
        packed_labels = packed[-1]
        torch.testing.assert_close(packed_labels[0, -2:], torch.tensor([8, 9]))
        self.assertTrue((packed_labels[0, :-2] == IGNORE_INDEX).all())
        self.assertEqual(model.config.mm_hidden_size, 12)
        self.assertEqual(model.get_model().mm_projector[0].in_features, 22)

    def test_answer_loss_reaches_cnn_attention_and_projector_only(self):
        model = make_tiny_model()
        model.train()
        batch = make_batch()
        backbone_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if not parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
        )
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        parameters = dict(model.named_parameters())
        for name in (
            "model.mask_fusion.mask_encoder.patch_conv.weight",
            "model.mask_fusion.mask_encoder.context_conv.weight",
            "model.mask_fusion.layers.0.q_proj.weight",
            "model.mask_fusion.layers.0.k_proj.weight",
            "model.mm_projector.0.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        trainable_before = parameters["model.mm_projector.0.weight"].detach().clone()
        for name in backbone_before:
            self.assertIsNone(parameters[name].grad, name)
        optimizer.step()
        for name, previous in backbone_before.items():
            self.assertTrue(torch.equal(parameters[name].detach(), previous), name)
        self.assertFalse(torch.equal(parameters["model.mm_projector.0.weight"].detach(), trainable_before))

    def test_generation_encodes_mask_and_image_once_during_prefill(self):
        model = make_tiny_model()
        batch = make_batch()
        counts = {"fusion": 0}

        def record_call(module, inputs):
            counts["fusion"] += 1

        handle = model.get_model().mask_fusion.register_forward_pre_hook(record_call)
        try:
            output = model.generate(
                inputs=batch["input_ids"], attention_mask=batch["attention_mask"],
                images=batch["images"], masks=batch["masks"], mask_valid=batch["mask_valid"],
                image_sizes=batch["image_sizes"], modalities=batch["modalities"],
                do_sample=False, min_new_tokens=3, max_new_tokens=3, use_cache=True,
            )
        finally:
            handle.remove()
        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertEqual(model.get_vision_tower().forward_calls, 1)
        self.assertEqual(counts["fusion"], 1)

    def test_adapter_round_trip_restores_weights_and_logits(self):
        model = make_tiny_model()
        batch = make_batch()
        # Exercise noninitial adapter weights, including newly added columns.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(torch.randn_like(parameter) * 0.001)
            expected = model(**batch, use_cache=False).logits
        with TemporaryDirectory(prefix="mask-fusion-adapter-") as directory:
            save_adapter(model, directory, base_model="offline-tiny-test-backbone")
            self.assertTrue((Path(directory) / "adapter.safetensors").is_file())
            self.assertTrue((Path(directory) / "adapter_config.json").is_file())
            restored = make_tiny_model()  # identical frozen weights from seed
            load_adapter(restored, directory)
            with torch.no_grad():
                actual = restored(**batch, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for prefix in ("mask_fusion", "mm_projector"):
                expected_state = getattr(model.get_model(), prefix).state_dict()
                actual_state = getattr(restored.get_model(), prefix).state_dict()
                for key in expected_state:
                    self.assertTrue(torch.equal(expected_state[key], actual_state[key]), f"{prefix}.{key}")

    def test_different_crop_counts_keep_batch_and_mask_order(self):
        model = make_tiny_model()
        batch = make_batch((1, 2))
        batch["masks"][0].fill_(20)
        batch["masks"][1][0].fill_(80)
        batch["masks"][1][1].fill_(160)
        seen_masks = []
        handle = model.get_model().mask_fusion.register_forward_pre_hook(
            lambda module, inputs: seen_masks.append(inputs[1].detach().clone()))
        try:
            output = model(**batch, use_cache=False)
        finally:
            handle.remove()
        self.assertEqual(tuple(output.logits.shape), (2, 12, 64))
        self.assertEqual(model.get_vision_tower().encoded_batch_sizes, [3])
        torch.testing.assert_close(seen_masks[0][:, 0, 0, 0], torch.tensor([20., 80., 160.]))
        packed = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, batch["labels"],
            batch["images"], batch["modalities"], batch["image_sizes"],
            masks=batch["masks"], mask_valid=batch["mask_valid"],
        )
        torch.testing.assert_close(packed[2].sum(dim=-1), torch.tensor([8, 12]))
        self.assertTrue((packed[-1][0, 8:] == IGNORE_INDEX).all())

    def test_missing_or_misaligned_masks_are_rejected(self):
        model = make_tiny_model()
        batch = make_batch((1, 2))
        missing = dict(batch)
        missing.pop("masks")
        with self.assertRaisesRegex(ValueError, "requires aligned continuous masks"):
            model(**missing)
        misaligned = dict(batch)
        misaligned["masks"] = [batch["masks"][0], batch["masks"][1][:1]]
        with self.assertRaisesRegex(ValueError, "crop's count"):
            model(**misaligned)

    def test_training_context_overflow_is_rejected_before_answer_truncation(self):
        model = make_tiny_model()
        model.config.tokenizer_model_max_length = 7  # packed sample length is 8
        with self.assertRaisesRegex(ValueError, "refusing to silently truncate"):
            model(**make_batch())

    def test_generation_reserves_new_token_context(self):
        model = make_tiny_model()
        model.config.tokenizer_model_max_length = 9  # 8 packed + 2 generated > 9
        batch = make_batch()
        with self.assertRaisesRegex(ValueError, "context limit"):
            model.generate(
                inputs=batch["input_ids"], images=batch["images"], masks=batch["masks"],
                mask_valid=batch["mask_valid"], image_sizes=batch["image_sizes"],
                modalities=batch["modalities"], max_new_tokens=2, do_sample=False,
            )


if __name__ == "__main__":
    unittest.main()
