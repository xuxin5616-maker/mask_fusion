"""Offline phase-1 tests through the real tiny LLaVA-Qwen decoder."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from torch import nn
from safetensors.torch import load_file

from mask_fusion.config import FusionConfig
from mask_fusion.model import (
    configure_image_mode, configure_mask_pretraining, load_adapter,
    read_adapter_config, save_adapter,
)
from mask_fusion.modules import SplitInputLinear
from test_integration import make_batch, make_tiny_model


class MaskPretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_split_linear_preserves_output_and_parameter_partition(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                original = nn.Linear(7, 5, bias=bias).eval()
                split = SplitInputLinear(original, visual_dim=4)
                features = torch.randn(2, 3, 7)
                torch.testing.assert_close(split(features), original(features), rtol=0, atol=0)
                self.assertFalse(split.visual_weight.requires_grad)
                self.assertTrue(split.expert_weight.requires_grad)
                self.assertFalse(split.training)
                if bias:
                    self.assertFalse(split.bias.requires_grad)
                else:
                    self.assertIsNone(split.bias)

    def test_forward_skips_vision_and_attention_and_ignores_rgb_values(self):
        model = make_tiny_model()
        core = model.get_model()
        config_before = dict(model.config.mask_fusion)
        counts = configure_mask_pretraining(model)
        self.assertEqual(model.config.mask_fusion, config_before)
        self.assertTrue(model.config.mask_fusion_pretrain)
        self.assertEqual(core.mask_fusion.config.mode, "both")
        self.assertEqual(len(core.mask_fusion.layers), 1)
        self.assertIsInstance(core.mm_projector[0], SplitInputLinear)
        self.assertEqual(counts["trainable"], sum(
            p.numel() for p in model.parameters() if p.requires_grad))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad,
                             "mask_fusion.mask_encoder." in name
                             or name == "model.mm_projector.0.expert_weight", name)
        batch = make_batch((1, 2))
        seen = {"fusion": 0, "decoder": 0, "inputs": []}

        def fusion_hook(module, inputs):
            seen["fusion"] += 1

        def decoder_hook(module, inputs):
            seen["decoder"] += 1

        def projector_hook(module, inputs):
            seen["inputs"].append(inputs[0].detach().clone())

        handles = [core.mask_fusion.register_forward_pre_hook(fusion_hook),
                   core.layers[0].register_forward_pre_hook(decoder_hook),
                   core.mm_projector.register_forward_pre_hook(projector_hook)]
        try:
            with mock.patch.object(core.vision_tower, "forward", side_effect=AssertionError("vision used")), \
                 mock.patch.object(core.mask_fusion, "encode_prior", side_effect=AssertionError("prior used")), \
                 mock.patch.object(core.mask_fusion.layers[0], "forward", side_effect=AssertionError("attention used")):
                result = model(**batch, use_cache=False)
                changed = dict(batch)
                # Even NaN RGB values cannot enter the phase-1 feature path.
                changed["images"] = [torch.full_like(value, float("nan")) for value in batch["images"]]
                other = model(**changed, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(result.logits.shape), (2, 12, 64))
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(seen["fusion"], 2)
        self.assertEqual(seen["decoder"], 2)
        self.assertEqual(core.vision_tower.forward_calls, 0)
        self.assertEqual(tuple(seen["inputs"][0].shape), (3, 4, 22))
        self.assertTrue(torch.equal(seen["inputs"][0][..., :12], torch.zeros(3, 4, 12)))
        expected_expert, _, _ = core.mask_fusion.encode_masks(torch.cat(batch["masks"]))
        torch.testing.assert_close(seen["inputs"][0][..., 12:], expected_expert, rtol=0, atol=0)
        torch.testing.assert_close(result.logits, other.logits, rtol=0, atol=0)

    def test_two_steps_warm_up_mask_gradients_and_never_change_frozen_weights(self):
        model = make_tiny_model()
        # Clear an existing baseline graph's gradients when changing stages.
        model(**make_batch(), use_cache=False).loss.backward()
        configure_mask_pretraining(model)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.train()
        core = model.get_model()
        frozen_before = {name: p.detach().clone() for name, p in model.named_parameters()
                         if not p.requires_grad}
        cnn_before = core.mask_fusion.mask_encoder.patch_conv.weight.detach().clone()
        expert_before = core.mm_projector[0].expert_weight.detach().clone()
        self.assertEqual(expert_before.count_nonzero().item(), 0)
        # Including frozen parameters demonstrates physical isolation from
        # AdamW decay; production optimizers filter requires_grad parameters.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.2)
        batch = make_batch()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            expert_grad = core.mm_projector[0].expert_weight.grad
            self.assertGreater(expert_grad.abs().sum().item(), 0)
            cnn_grad = core.mask_fusion.mask_encoder.patch_conv.weight.grad
            self.assertIsNotNone(cnn_grad)
            self.assertTrue(torch.isfinite(cnn_grad).all())
            if step == 0:
                self.assertEqual(cnn_grad.count_nonzero().item(), 0)
            else:
                self.assertGreater(cnn_grad.abs().sum().item(), 0)
            for name, p in model.named_parameters():
                if name in frozen_before:
                    self.assertIsNone(p.grad, name)
            optimizer.step()
            for name, p in model.named_parameters():
                if name in frozen_before:
                    self.assertTrue(torch.equal(p, frozen_before[name]), name)
        self.assertFalse(torch.equal(core.mm_projector[0].expert_weight, expert_before))
        self.assertFalse(torch.equal(core.mask_fusion.mask_encoder.patch_conv.weight, cnn_before))
        self.assertTrue(all(p.grad is None for p in core.layers.parameters()))

    def test_canonical_adapter_export_supports_resume_and_normal_phase2(self):
        model = make_tiny_model()
        configure_mask_pretraining(model)
        core = model.get_model()
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.01)
        batch = make_batch()
        with torch.no_grad():
            pretrain_logits = model(**batch, use_cache=False).logits
        with TemporaryDirectory(prefix="mask-pretrain-adapter-") as directory:
            for gathered in (False, True):
                with self.subTest(gathered=gathered):
                    folder = Path(directory) / str(gathered)
                    if gathered:
                        weights = {f"{prefix}.{key}": value.clone()
                                   for prefix in ("mask_fusion", "mm_projector")
                                   for key, value in getattr(core, prefix).state_dict().items()}
                        with mock.patch.object(core.mask_fusion, "state_dict", side_effect=AssertionError("ungathered access")), \
                             mock.patch.object(core.mm_projector, "state_dict", side_effect=AssertionError("ungathered access")):
                            save_adapter(model, folder, "offline-tiny", weights=weights)
                    else:
                        save_adapter(model, folder, "offline-tiny")
                    metadata = read_adapter_config(folder)
                    self.assertEqual(metadata["training_stage"], "mask_pretrain")
                    self.assertNotIn("mask_fusion_pretrain", metadata["fusion_config"])
                    weights = load_file(str(folder / "adapter.safetensors"))
                    self.assertIn("mm_projector.0.weight", weights)
                    self.assertFalse(any("expert_weight" in key or "visual_weight" in key for key in weights))
                    torch.testing.assert_close(weights["mm_projector.0.weight"], torch.cat((
                        core.mm_projector[0].visual_weight, core.mm_projector[0].expert_weight), dim=1), rtol=0, atol=0)
                    restored = make_tiny_model()
                    load_adapter(restored, folder)
                    self.assertIsInstance(restored.get_model().mm_projector[0], nn.Linear)
                    self.assertFalse(restored.config.mask_fusion_pretrain)
                    # Ordinary phase-2 model: both vision and attention active.
                    model.config.mask_fusion_pretrain = False
                    with torch.no_grad():
                        expected = model(**batch, use_cache=False).logits
                        actual = restored(**batch, use_cache=False).logits
                    model.config.mask_fusion_pretrain = True
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    configure_mask_pretraining(restored)
                    with torch.no_grad():
                        resumed = restored(**batch, use_cache=False).logits
                    torch.testing.assert_close(resumed, pretrain_logits, rtol=0, atol=0)

    def test_linear_projector_exports_without_sequential_prefix(self):
        model = make_tiny_model()
        model.get_model().mm_projector = nn.Linear(22, 32, bias=False)
        configure_mask_pretraining(model)
        split = model.get_model().mm_projector
        self.assertIsInstance(split, SplitInputLinear)
        self.assertTrue(torch.isfinite(model(**make_batch(), use_cache=False).loss))
        with TemporaryDirectory(prefix="mask-pretrain-linear-") as directory:
            save_adapter(model, directory, "offline-tiny")
            weights = load_file(str(Path(directory) / "adapter.safetensors"))
            self.assertIn("mm_projector.weight", weights)
            self.assertNotIn("mm_projector.bias", weights)
            restored = make_tiny_model()
            restored.get_model().mm_projector = nn.Linear(22, 32, bias=False)
            load_adapter(restored, directory)
            torch.testing.assert_close(restored.get_model().mm_projector.weight,
                                       torch.cat((split.visual_weight, split.expert_weight), dim=1),
                                       rtol=0, atol=0)

    def test_image_packing_transfer_requires_opt_in_and_keeps_architecture_checks(self):
        model = make_tiny_model()
        configure_mask_pretraining(model)
        with TemporaryDirectory(prefix="mask-pretrain-transfer-") as directory:
            save_adapter(model, directory, "offline-tiny")
            restored = make_tiny_model()
            configure_image_mode(restored.config, "randomroi")
            with self.assertRaisesRegex(ValueError, "mm_patch_merge_type"):
                load_adapter(restored, directory)
            load_adapter(restored, directory, allow_image_mode_change=True)
            self.assertEqual(restored.config.mask_fusion_image_mode, "randomroi")
            restored.config.mask_fusion_vision_dim += 1
            with self.assertRaisesRegex(ValueError, "vision_dim"):
                load_adapter(restored, directory, allow_image_mode_change=True)
            restored.config.mask_fusion_vision_dim -= 1
            restored.config.mask_fusion["num_layers"] += 1
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                load_adapter(restored, directory, allow_image_mode_change=True)

    def test_pretraining_rejects_ablation_mode(self):
        model = make_tiny_model(fusion_config=FusionConfig(
            mask_hidden_size=8, expert_dim=16, position_dim=8,
            position_hidden_size=16, attention_dim=16, num_heads=4, mode="direct"))
        with self.assertRaisesRegex(ValueError, "mode='both'"):
            configure_mask_pretraining(model)


if __name__ == "__main__":
    unittest.main()
