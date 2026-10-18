"""CPU-sized behavioral checks; run with python -m unittest discover -s tests."""

from dataclasses import replace
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mask_fusion.config import FusionConfig
from mask_fusion.modules import ExpertMaskFusion, expand_projector, position_encoding_2d


def small_config(**overrides):
    values = dict(
        mask_hidden_size=4, expert_dim=8, position_dim=8,
        position_hidden_size=8, attention_dim=8, num_heads=2,
    )
    values.update(overrides)
    return FusionConfig(**values)


class ConfigurationTests(unittest.TestCase):
    def test_json_round_trip_and_default_dimensions(self):
        config = FusionConfig()
        self.assertEqual(FusionConfig.from_json(config.to_json()), config)
        self.assertEqual(config.expert_feature_dim, 130)
        self.assertEqual(config.prior_dim, 512)
        self.assertFalse(config.direct_residual_mlp)
        self.assertEqual(config.direct_mlp_hidden_size, 256)
        fusion = ExpertMaskFusion(vision_dim=1152, patch_size=14, config=config)
        self.assertEqual(fusion.layers[0].q_proj.in_features, 512)
        self.assertEqual(fusion.layers[0].k_proj.in_features, 512)
        self.assertEqual(fusion.layers[0].q_proj.out_features, 256)
        self.assertEqual(fusion.layers[0].v_proj.in_features, 1152)
        self.assertEqual(fusion.layers[0].out_proj.out_features, 1152)

    def test_invalid_configuration_is_rejected(self):
        cases = (
            {"mask_hidden_size": 0}, {"num_layers": -1}, {"num_heads": True},
            {"position_dim": 6}, {"attention_dim": 10, "num_heads": 3},
            {"score_scale": 0}, {"score_scale": float("inf")},
            {"alpha_init": float("nan")}, {"normalize_e": "true"},
            {"direct_residual_mlp": 1}, {"direct_mlp_hidden_size": 0},
            {"direct_mlp_hidden_size": True}, {"direct_mlp_hidden_size": 2.5},
            {"mode": "unsupported"},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                FusionConfig(**values)
        with self.assertRaises(ValueError):
            FusionConfig.from_dict({"mask_hiden_size": 128})

    def test_legacy_config_stays_disabled_and_residual_preset_is_explicit(self):
        folder = Path(__file__).resolve().parents[1] / "configs"
        legacy = FusionConfig.from_json((folder / "default.json").read_text())
        residual = FusionConfig.from_json((folder / "residual_mlp.json").read_text())
        self.assertFalse(legacy.direct_residual_mlp)
        self.assertEqual(residual, replace(legacy, direct_residual_mlp=True,
                                           direct_mlp_hidden_size=256))


class MaskEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.fusion = ExpertMaskFusion(6, 2, small_config())

    def test_dynamic_grid_and_output_shape(self):
        features = torch.randn(2, 6, 6)
        masks = torch.randint(0, 256, (2, 1, 4, 6), dtype=torch.uint8)
        output = self.fusion(features, masks)
        self.assertEqual(tuple(output.shape), (2, 6, 12))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(self.fusion.expert_feature_dim, 6)

    def test_statistics_follow_row_major_nonoverlapping_patches(self):
        masks = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) * 10
        expert, valid, grid = self.fusion.encode_masks(masks)
        expected_mean = torch.tensor([[25, 45, 105, 125]], dtype=torch.float32) / 255
        expected_max = torch.tensor([[50, 70, 130, 150]], dtype=torch.float32) / 255
        torch.testing.assert_close(expert[..., -2], expected_mean)
        torch.testing.assert_close(expert[..., -1], expected_max)
        self.assertEqual(grid, (2, 2))
        self.assertTrue(valid.all())

    def test_partial_patch_statistics_ignore_invalid_pixels(self):
        masks = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) * 10
        valid_pixels = torch.ones_like(masks, dtype=torch.bool)
        valid_pixels[0, 0, :2, 0] = False
        valid_pixels[0, 0, 2:, 2:] = False
        masks[~valid_pixels] = 10000  # deliberately larger than every real score
        expert, valid, _ = self.fusion.encode_masks(masks, valid_pixels)
        torch.testing.assert_close(expert[0, 0, -2:], torch.tensor([30.0, 50.0]) / 255)
        self.assertFalse(valid[0, 3])
        self.assertEqual(torch.count_nonzero(expert[0, 3]).item(), 0)

    def test_trailing_pixels_follow_valid_stride_patch_embedding(self):
        # The fifth row/column are outside the 2x2 stride-two patch grid.
        masks = torch.zeros(1, 1, 5, 5)
        masks[..., 4, :] = 255
        masks[..., :, 4] = 255
        expert, _, grid = self.fusion.encode_masks(masks)
        self.assertEqual(grid, (2, 2))
        self.assertEqual(torch.count_nonzero(expert[..., -2:]).item(), 0)

    def test_positions_preserve_row_column_order(self):
        positions = position_encoding_2d(2, 3, 8)
        self.assertEqual(tuple(positions.shape), (6, 8))
        torch.testing.assert_close(positions[0, :4], positions[1, :4])
        torch.testing.assert_close(positions[0, 4:], positions[3, 4:])
        self.assertFalse(torch.equal(positions[0, 4:], positions[1, 4:]))
        self.assertFalse(torch.equal(positions[0, :4], positions[3, :4]))
        torch.testing.assert_close(positions[0], torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]))

    def test_e_is_concatenation_not_an_additional_compression(self):
        fusion = ExpertMaskFusion(6, 2, small_config(normalize_e=False))
        expert = torch.randn(2, 6, fusion.expert_feature_dim)
        prior = fusion.encode_prior(expert, (2, 3))
        positions = position_encoding_2d(2, 3, fusion.config.position_dim)
        expected_expert = fusion.expert_projection(expert)
        expected_spatial = fusion.position_projection(positions).unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(prior[..., :8], expected_expert)
        torch.testing.assert_close(prior[..., 8:], expected_spatial)
        self.assertEqual(prior.shape[-1], 16)
        # Identical scores/features still yield position-dependent priors.
        constant = torch.ones(1, 6, fusion.expert_feature_dim)
        constant_prior = fusion.encode_prior(constant, (2, 3))
        self.assertFalse(torch.equal(constant_prior[0, 0], constant_prior[0, 1]))

    def test_invalid_shapes_fail_before_projector(self):
        with self.assertRaisesRegex(ValueError, "Mask patch grid"):
            self.fusion(torch.randn(1, 5, 6), torch.zeros(1, 1, 4, 6))
        with self.assertRaisesRegex(ValueError, "mask_valid"):
            self.fusion(torch.randn(1, 6, 6), torch.zeros(1, 1, 4, 6), torch.ones(1, 3, 6))
        with self.assertRaisesRegex(ValueError, "Mask inputs"):
            self.fusion(torch.randn(1, 6, 6), None)

    def test_invalid_valid_pixel_scores_are_rejected(self):
        for score in (-1.0, 256.0, float("inf"), float("nan")):
            masks = torch.zeros(1, 1, 4, 4)
            masks[0, 0, 0, 0] = score
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "Valid mask scores"):
                self.fusion.encode_masks(masks)
            valid = torch.ones_like(masks, dtype=torch.bool)
            valid[0, 0, 0, 0] = False
            expert, _, _ = self.fusion.encode_masks(masks, valid)
            self.assertTrue(torch.isfinite(expert).all())

    def test_validity_map_must_be_binary(self):
        masks = torch.zeros(1, 1, 4, 4)
        for value in (0.5, 255.0, -1.0, float("nan")):
            valid = torch.ones_like(masks)
            valid[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "binary"):
                self.fusion.encode_masks(masks, valid)


class AttentionAndInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = small_config()
        self.features = torch.randn(2, 4, 6)
        self.masks = torch.rand(2, 1, 4, 4) * 255

    def test_entirely_padded_sample_is_finite_and_keeps_visual_residual(self):
        fusion = ExpertMaskFusion(6, 2, self.config)
        valid = torch.ones_like(self.masks, dtype=torch.bool)
        valid[0] = False
        masks = self.masks.clone()
        masks[0] = float("nan")  # invalid pixels must never reach the CNN/statistics
        output = fusion(self.features, masks, valid)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[0, :, :6], self.features[0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(output[0, :, 6:]).item(), 0)
        output.square().mean().backward()
        for name, parameter in fusion.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_padding_keys_cannot_affect_valid_queries(self):
        fusion = ExpertMaskFusion(6, 2, self.config).eval()
        valid = torch.ones_like(self.masks, dtype=torch.bool)
        valid[..., 2:, 2:] = False  # last token is padding
        changed = self.features.clone()
        changed[:, -1] += 10000
        with torch.no_grad():
            original_output = fusion(self.features, self.masks, valid)
            changed_output = fusion(changed, self.masks, valid)
        torch.testing.assert_close(original_output[:, :3], changed_output[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(changed_output[:, -1, :6], changed[:, -1], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(changed_output[:, -1, 6:]).item(), 0)

    def test_each_layer_reads_the_same_prior_but_updated_visual_values(self):
        fusion = ExpertMaskFusion(6, 2, replace(self.config, num_layers=2, alpha_init=0.5))
        queries, values = [], []
        handles = []
        for layer in fusion.layers:
            handles.append(layer.q_proj.register_forward_pre_hook(
                lambda module, inputs: queries.append(inputs[0].detach().clone())))
            handles.append(layer.v_proj.register_forward_pre_hook(
                lambda module, inputs: values.append(inputs[0].detach().clone())))
        try:
            fusion(self.features, self.masks)
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(queries[0], queries[1], rtol=0, atol=0)
        torch.testing.assert_close(values[0], self.features, rtol=0, atol=0)
        self.assertFalse(torch.equal(values[0], values[1]))

    def test_mask_location_prior_is_independent_of_visual_content(self):
        fusion = ExpertMaskFusion(6, 2, self.config)
        priors = []
        handle = fusion.layers[0].q_proj.register_forward_pre_hook(
            lambda module, inputs: priors.append(inputs[0].detach().clone()))
        try:
            first = fusion(self.features, self.masks)
            second = fusion(-self.features, self.masks)
        finally:
            handle.remove()
        torch.testing.assert_close(priors[0], priors[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[..., :6], second[..., :6]))

    def test_zero_residual_and_expanded_projector_preserve_baseline(self):
        fusion = ExpertMaskFusion(6, 2, replace(self.config, alpha_init=0)).double()
        projector = nn.Sequential(nn.Linear(6, 9), nn.GELU(), nn.Linear(9, 5)).double().eval()
        expanded = expand_projector(projector, fusion.expert_feature_dim)
        features = self.features.double()
        actual = expanded(fusion(features, self.masks.double()))
        expected = projector(features)
        # GEMM kernel selection can change with input width; verify numerical
        # equivalence rather than promising bitwise-identical GPU reductions.
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(expanded[0].weight[:, :6], projector[0].weight))
        self.assertTrue(torch.equal(expanded[0].bias, projector[0].bias))
        self.assertEqual(torch.count_nonzero(expanded[0].weight[:, 6:]).item(), 0)
        self.assertTrue(torch.equal(expanded[2].weight, projector[2].weight))
        self.assertIsNot(expanded[2], projector[2])
        self.assertFalse(expanded.training)
        self.assertEqual(projector[0].in_features, 6)

    def test_nonzero_alpha_propagates_gradients_through_new_attention_and_cnn(self):
        fusion = ExpertMaskFusion(6, 2, self.config)
        projector = expand_projector(nn.Linear(6, 5), fusion.expert_feature_dim)
        prediction = projector(fusion(self.features, self.masks))
        target = torch.randn_like(prediction)
        (prediction - target).square().mean().backward()
        parameters = dict(fusion.named_parameters())
        for name in (
            "mask_encoder.patch_conv.weight", "mask_encoder.context_conv.weight",
            "expert_projection.0.weight", "position_projection.weight",
            "layers.0.q_proj.weight", "layers.0.k_proj.weight",
            "layers.0.v_proj.weight", "layers.0.out_proj.weight", "layers.0.alpha",
        ):
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertGreater(projector.weight.grad[:, 6:].abs().sum().item(), 0)

    def test_fsdp_alpha_storage_preserves_bf16_outputs_and_gradients(self):
        scalar = ExpertMaskFusion(6, 2, self.config)
        vector = copy.deepcopy(scalar)
        for layer in vector.layers:
            layer.alpha.data = layer.alpha.data.reshape(1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scalar_output = scalar(self.features, self.masks)
            vector_output = vector(self.features, self.masks)
        torch.testing.assert_close(scalar_output, vector_output, rtol=0, atol=0)
        scalar_output.square().mean().backward()
        vector_output.square().mean().backward()
        for (name, parameter), (other_name, other) in zip(scalar.named_parameters(), vector.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad.reshape(-1), other.grad.reshape(-1), rtol=0, atol=0)

    def test_mode_switches_keep_checkpoint_shapes_and_expected_paths(self):
        both = ExpertMaskFusion(6, 2, self.config)
        outputs = {"both": both(self.features, self.masks)}
        for mode in ("direct", "attention", "none"):
            module = ExpertMaskFusion(6, 2, replace(self.config, mode=mode))
            module.load_state_dict(both.state_dict())
            if mode == "none":
                with patch.object(module.mask_encoder, "forward", side_effect=AssertionError("disabled")):
                    outputs[mode] = module(self.features, None)
            elif mode == "direct":
                with patch.object(module.layers[0], "forward", side_effect=AssertionError("disabled")):
                    outputs[mode] = module(self.features, self.masks)
            else:
                outputs[mode] = module(self.features, self.masks)
        for value in outputs.values():
            self.assertEqual(tuple(value.shape), (2, 4, 12))
        torch.testing.assert_close(outputs["direct"][..., :6], self.features, rtol=0, atol=0)
        torch.testing.assert_close(outputs["none"][..., :6], self.features, rtol=0, atol=0)
        torch.testing.assert_close(outputs["both"][..., 6:], outputs["direct"][..., 6:])
        torch.testing.assert_close(outputs["both"][..., :6], outputs["attention"][..., :6])
        self.assertEqual(torch.count_nonzero(outputs["attention"][..., 6:]).item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["none"][..., 6:]).item(), 0)


class DirectResidualMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.config = small_config(direct_residual_mlp=True, direct_mlp_hidden_size=7)
        self.fusion = ExpertMaskFusion(6, 2, self.config)
        self.features = torch.randn(2, 4, 6)
        self.masks = torch.rand(2, 1, 4, 4) * 255

    def test_zero_initialization_preserves_shared_weights_and_outputs(self):
        torch.manual_seed(73)
        baseline = ExpertMaskFusion(6, 2, replace(self.config, direct_residual_mlp=False))
        self.assertIsNone(baseline.direct_mlp)
        self.assertFalse(any(key.startswith("direct_mlp.") for key in baseline.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, self.fusion.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(self.fusion.direct_mlp[2].weight.count_nonzero().item(), 0)
        self.assertEqual(self.fusion.direct_mlp[2].bias.count_nonzero().item(), 0)
        self.assertGreater(self.fusion.direct_mlp[0].weight.count_nonzero().item(), 0)
        for mask_only in (False, True):
            with self.subTest(mask_only=mask_only):
                expected = baseline(self.features, self.masks, mask_only=mask_only)
                actual = self.fusion(self.features, self.masks, mask_only=mask_only)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_learning_changes_only_direct_hidden_features_and_preserves_prior(self):
        fusion = self.fusion
        before = fusion(self.features, self.masks).detach()
        expert_before, valid, grid = fusion.encode_masks(self.masks)
        prior_before = fusion.encode_prior(expert_before, grid, valid).detach()
        expert_before = expert_before.detach()
        target = torch.randn(2, 4, self.config.mask_hidden_size)
        optimizer = torch.optim.SGD(fusion.direct_mlp.parameters(), lr=0.1)
        for step in range(2):
            fusion.zero_grad(set_to_none=True)
            prediction = fusion(self.features, self.masks)[..., 6:10]
            (prediction - target).square().mean().backward()
            self.assertGreater(fusion.direct_mlp[2].weight.grad.abs().sum().item(), 0)
            first_gradient = fusion.direct_mlp[0].weight.grad
            self.assertTrue(torch.isfinite(first_gradient).all())
            if step == 0:
                self.assertEqual(first_gradient.count_nonzero().item(), 0)
            else:
                self.assertGreater(first_gradient.abs().sum().item(), 0)
            # H's identity path keeps the mask CNN differentiable even while
            # the new MLP's final layer starts at zero.
            cnn_gradient = fusion.mask_encoder.patch_conv.weight.grad
            self.assertGreater(cnn_gradient.abs().sum().item(), 0)
            optimizer.step()
        after = fusion(self.features, self.masks).detach()
        expert_after, _, _ = fusion.encode_masks(self.masks)
        prior_after = fusion.encode_prior(expert_after, grid, valid).detach()
        torch.testing.assert_close(expert_after, expert_before, rtol=0, atol=0)
        torch.testing.assert_close(prior_after, prior_before, rtol=0, atol=0)
        torch.testing.assert_close(after[..., :6], before[..., :6], rtol=0, atol=0)
        torch.testing.assert_close(after[..., -2:], before[..., -2:], rtol=0, atol=0)
        self.assertFalse(torch.equal(after[..., 6:10], before[..., 6:10]))

    def test_statistics_bypass_mlp_and_padding_stays_zero_after_learning_bias(self):
        fusion = self.fusion
        with torch.no_grad():
            fusion.direct_mlp[2].weight.fill_(0.2)
            fusion.direct_mlp[2].bias.fill_(3.0)
        valid_pixels = torch.ones_like(self.masks, dtype=torch.bool)
        valid_pixels[0] = False
        valid_pixels[1, :, 2:, 2:] = False
        valid_pixels[1, :, :2, 0] = False  # also exercise a partial patch
        masks = self.masks.clone()
        masks[~valid_pixels] = float("nan")
        expert, valid_tokens, _ = fusion.encode_masks(masks, valid_pixels)
        direct = fusion.encode_direct(expert, valid_tokens)
        self.assertEqual(tuple(direct.shape), (2, 4, 6))
        torch.testing.assert_close(direct[..., -2:], expert[..., -2:], rtol=0, atol=0)
        self.assertEqual(direct[~valid_tokens].count_nonzero().item(), 0)
        for mask_only in (False, True):
            output = fusion(self.features, masks, valid_pixels, mask_only=mask_only)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output[..., 6:], direct, rtol=0, atol=0)
            if mask_only:
                self.assertEqual(output[..., :6].count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, "valid_tokens"):
            fusion.encode_direct(expert, torch.ones(1, 4, dtype=torch.bool))

    def test_mlp_runs_only_for_enabled_direct_paths(self):
        for mode in ("both", "direct", "attention", "none"):
            with self.subTest(mode=mode):
                fusion = ExpertMaskFusion(6, 2, replace(self.config, mode=mode))
                with patch.object(fusion.direct_mlp, "forward", wraps=fusion.direct_mlp.forward) as call:
                    output = fusion(self.features, None if mode == "none" else self.masks)
                    self.assertEqual(call.call_count, int(mode in ("both", "direct")))
                if mode in ("attention", "none"):
                    self.assertEqual(output[..., 6:].count_nonzero().item(), 0)
        with patch.object(self.fusion.direct_mlp, "forward", wraps=self.fusion.direct_mlp.forward) as call, \
             patch.object(self.fusion, "encode_prior", side_effect=AssertionError("prior used in phase 1")):
            self.fusion(self.features, self.masks, mask_only=True)
            self.assertEqual(call.call_count, 1)


class ProjectorExpansionTests(unittest.TestCase):
    def test_linear_without_bias_preserves_dtype_flags_and_values(self):
        original = nn.Linear(3, 4, bias=False).double().eval().requires_grad_(False)
        expanded = expand_projector(original, 2)
        self.assertIsNone(expanded.bias)
        self.assertEqual(expanded.weight.dtype, torch.float64)
        self.assertFalse(expanded.weight.requires_grad)
        self.assertFalse(expanded.training)
        x = torch.randn(2, 3, dtype=torch.float64)
        extra = torch.randn(2, 2, dtype=torch.float64)
        torch.testing.assert_close(expanded(torch.cat((x, extra), -1)), original(x))
        self.assertEqual(original.in_features, 3)

    def test_unsupported_projectors_fail_explicitly(self):
        for module in (nn.Identity(), nn.Sequential(), nn.Sequential(nn.GELU(), nn.Linear(2, 2))):
            with self.subTest(module=module), self.assertRaises(TypeError):
                expand_projector(module, 4)
        with self.assertRaises(ValueError):
            expand_projector(nn.Linear(2, 2), -1)


if __name__ == "__main__":
    unittest.main()
