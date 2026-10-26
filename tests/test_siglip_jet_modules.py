"""CPU checks of externally encoded Jet-mask features; no tower is loaded."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mask_fusion.config import FusionConfig
from mask_fusion.modules import ExpertMaskFusion, PriorAttentionLayer


def jet_config(**changes):
    values = dict(mask_encoder_type="siglip_jet", direct_residual_mlp=True,
                  direct_mlp_hidden_size=7, attention_dim=8, num_heads=2)
    values.update(changes)
    return FusionConfig(**values)


class SiglipJetModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(91)
        self.config = jet_config()
        self.fusion = ExpertMaskFusion(6, 2, self.config)
        self.visual = torch.randn(2, 4, 6)
        self.expert = torch.randn(2, 4, 6)
        self.valid = torch.ones(2, 4, dtype=torch.bool)

    def call_fusion(self, fusion=None, **overrides):
        arguments = dict(expert_features=self.expert, valid_tokens=self.valid)
        arguments.update(overrides)
        return (fusion or self.fusion)(self.visual, **arguments)

    def test_legacy_default_and_explicit_preset(self):
        self.assertEqual(FusionConfig.from_dict({}).mask_encoder_type, "cnn")
        path = Path(__file__).resolve().parents[1] / "configs" / "siglip_jet.json"
        config = FusionConfig.from_json(path.read_text())
        self.assertEqual(config.mask_encoder_type, "siglip_jet")
        self.assertTrue(config.direct_residual_mlp)
        self.assertEqual(config.direct_mlp_hidden_size, 512)
        self.assertEqual(FusionConfig.from_json(config.to_json()), config)
        for value in ("jet", "", True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "mask_encoder_type"):
                FusionConfig(mask_encoder_type=value)

    def test_width_follows_tower_without_cnn_statistics_or_external_positions(self):
        for width in (6, 11, 1152):
            with self.subTest(width=width):
                fusion = ExpertMaskFusion(width, 2, self.config)
                self.assertEqual(fusion.expert_feature_dim, width)
                self.assertEqual(fusion.prior_dim, width)
                self.assertIsNone(fusion.mask_encoder)
                self.assertIsNone(fusion.expert_projection)
                self.assertIsNone(fusion.position_projection)
                self.assertFalse(any(isinstance(module, nn.Conv2d) for module in fusion.modules()))
                keys = fusion.state_dict()
                self.assertFalse(any(key.startswith(("mask_encoder.", "expert_projection.",
                                                     "position_projection.")) for key in keys))
                self.assertEqual(fusion.layers[0].q_proj.in_features, width)
                self.assertEqual(fusion.layers[0].k_proj.in_features, width)
                self.assertEqual(fusion.layers[0].q_proj.out_features, 8)
                self.assertEqual(fusion.direct_mlp[0].in_features, width)
                self.assertEqual(fusion.direct_mlp[2].out_features, width)
                self.assertIsNot(fusion.direct_norm, fusion.prior_norm)
                expert = torch.randn(1, 3, width)
                with patch("mask_fusion.modules.position_encoding_2d", side_effect=AssertionError("external PE used")):
                    prior = fusion.encode_prior(expert)
                    output = fusion(torch.zeros_like(expert), expert_features=expert,
                                    valid_tokens=torch.ones(1, 3, dtype=torch.bool))
                torch.testing.assert_close(prior, fusion.prior_norm(expert), rtol=0, atol=0)
                self.assertEqual(tuple(output.shape), (1, 3, 2 * width))
        with self.assertRaisesRegex(ValueError, "externally extracted"):
            self.fusion.encode_masks(torch.zeros(1, 1, 4, 4))
        # Existing direct callers of PriorAttentionLayer keep their CNN width.
        self.assertEqual(PriorAttentionLayer(6, FusionConfig()).q_proj.in_features, 512)

    def test_direct_residual_initially_identity_and_does_not_change_qk_input(self):
        fusion = self.fusion
        torch.testing.assert_close(fusion.encode_direct(self.expert), self.expert, rtol=0, atol=0)
        self.assertEqual(fusion.direct_mlp[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(fusion.direct_mlp[-1].bias.count_nonzero().item(), 0)
        priors = []
        handle = fusion.layers[0].q_proj.register_forward_pre_hook(
            lambda module, inputs: priors.append(inputs[0].detach().clone()))
        try:
            before = self.call_fusion().detach()
            with torch.no_grad():
                fusion.direct_mlp[-1].weight.normal_(std=0.1)
                fusion.direct_mlp[-1].bias.fill_(0.5)
                fusion.direct_norm.weight.mul_(2)
                fusion.direct_norm.bias.fill_(0.3)
            after = self.call_fusion().detach()
        finally:
            handle.remove()
        torch.testing.assert_close(priors[0], priors[1], rtol=0, atol=0)
        torch.testing.assert_close(after[..., :6], before[..., :6], rtol=0, atol=0)
        self.assertFalse(torch.equal(after[..., 6:], before[..., 6:]))
        expected_direct = self.expert + fusion.direct_mlp(fusion.direct_norm(self.expert))
        torch.testing.assert_close(after[..., 6:], expected_direct, rtol=0, atol=0)

    def test_modes_and_mask_only_use_only_the_requested_branches(self):
        both_output = self.call_fusion()
        for mode in ("direct", "attention", "none"):
            with self.subTest(mode=mode):
                fusion = ExpertMaskFusion(6, 2, replace(self.config, mode=mode))
                fusion.load_state_dict(self.fusion.state_dict())
                if mode == "none":
                    with patch.object(fusion, "encode_prior", side_effect=AssertionError("prior used")), \
                         patch.object(fusion, "encode_direct", side_effect=AssertionError("direct used")):
                        output = fusion(self.visual)
                    torch.testing.assert_close(output[..., :6], self.visual, rtol=0, atol=0)
                    self.assertEqual(output[..., 6:].count_nonzero().item(), 0)
                elif mode == "direct":
                    with patch.object(fusion, "encode_prior", side_effect=AssertionError("prior used")):
                        output = self.call_fusion(fusion)
                    torch.testing.assert_close(output[..., :6], self.visual, rtol=0, atol=0)
                    torch.testing.assert_close(output[..., 6:], both_output[..., 6:], rtol=0, atol=0)
                else:
                    with patch.object(fusion, "encode_direct", side_effect=AssertionError("direct used")):
                        output = self.call_fusion(fusion)
                    torch.testing.assert_close(output[..., :6], both_output[..., :6], rtol=0, atol=0)
                    self.assertEqual(output[..., 6:].count_nonzero().item(), 0)
        with patch.object(self.fusion, "encode_prior", side_effect=AssertionError("prior used")):
            output = self.call_fusion(mask_only=True)
        self.assertEqual(output[..., :6].count_nonzero().item(), 0)
        torch.testing.assert_close(output[..., 6:], self.expert, rtol=0, atol=0)
        disabled = ExpertMaskFusion(6, 2, replace(self.config, direct_residual_mlp=False))
        self.assertIsNone(disabled.direct_mlp)
        self.assertIsNone(disabled.direct_norm)
        torch.testing.assert_close(disabled.encode_direct(self.expert), self.expert, rtol=0, atol=0)

    def test_padding_features_are_ignored_and_all_invalid_is_finite(self):
        valid = self.valid.clone()
        valid[0] = False
        valid[1, -1] = False
        expert = self.expert.clone()
        expert[~valid] = float("nan")
        with torch.no_grad():
            self.fusion.direct_mlp[-1].bias.fill_(4)
        output = self.call_fusion(expert_features=expert, valid_tokens=valid)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[..., 6:][~valid].count_nonzero().item(), 0)
        torch.testing.assert_close(output[..., :6][~valid], self.visual[~valid], rtol=0, atol=0)
        expert[~valid] = 100000
        other = self.call_fusion(expert_features=expert, valid_tokens=valid)
        torch.testing.assert_close(other, output, rtol=0, atol=0)
        output.square().mean().backward()
        for name, parameter in self.fusion.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_shape_validity_and_nonfinite_inputs_fail_clearly(self):
        invalid_cases = (
            ({"expert_features": None}, "expert_features"),
            ({"expert_features": self.expert[..., :5]}, "expert_features"),
            ({"expert_features": self.expert.to(torch.int64)}, "expert_features"),
            ({"valid_tokens": None}, "valid_tokens"),
            ({"valid_tokens": self.valid.float()}, "valid_tokens"),
            ({"valid_tokens": self.valid[:, :3]}, "valid_tokens"),
            ({"expert_features": torch.full_like(self.expert, float("inf"))}, "finite"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(message=message, keys=list(overrides)), self.assertRaisesRegex(ValueError, message):
                self.call_fusion(**overrides)

    def test_sft_gradients_and_phase1_direct_adapter_warmup(self):
        prediction = self.call_fusion()
        target = torch.randn_like(prediction)
        (prediction - target).square().mean().backward()
        for name in ("layers.0.q_proj.weight", "layers.0.k_proj.weight",
                     "layers.0.v_proj.weight", "layers.0.out_proj.weight", "prior_norm.weight"):
            gradient = dict(self.fusion.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

        fusion = ExpertMaskFusion(6, 2, self.config)
        optimizer = torch.optim.SGD(list(fusion.direct_mlp.parameters())
                                    + list(fusion.direct_norm.parameters()), lr=0.1)
        direct_target = torch.randn_like(self.expert)
        for step in range(2):
            fusion.zero_grad(set_to_none=True)
            output = self.call_fusion(fusion, mask_only=True)
            (output[..., 6:] - direct_target).square().mean().backward()
            self.assertGreater(fusion.direct_mlp[-1].weight.grad.abs().sum().item(), 0)
            for name, parameter in list(fusion.direct_mlp[0].named_parameters()) + list(fusion.direct_norm.named_parameters()):
                self.assertIsNotNone(parameter.grad, name)
                if step == 0:
                    self.assertEqual(parameter.grad.count_nonzero().item(), 0, name)
                else:
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            self.assertTrue(all(p.grad is None for p in fusion.layers.parameters()))
            self.assertTrue(all(p.grad is None for p in fusion.prior_norm.parameters()))
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
