"""CPU contracts for mixed-precision adapters; real sharding is in fsdp_smoke.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from mask_fusion.model import configure_trainable, load_adapter, save_adapter
from test_integration import make_batch, make_tiny_model


def adapter_weights(model):
    core = model.get_model()
    return {
        f"{prefix}.{name}": value.detach().cpu().clone()
        for prefix in ("mask_fusion", "mm_projector")
        for name, value in getattr(core, prefix).state_dict().items()
    }


class FrozenBackboneContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_bf16_frozen_decoder_preserves_fp32_adapter_gradients_with_checkpointing(self):
        batch = make_batch((1, 2))
        gradients = []
        for checkpointing in (False, True):
            model = make_tiny_model().to(dtype=torch.bfloat16)
            configure_trainable(model)
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            model.get_vision_tower().eval()
            frozen_before = {name: value.detach().clone() for name, value in model.named_parameters()
                             if not value.requires_grad}
            optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=0.001)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(**batch, use_cache=False).loss
            loss.backward()
            current = {}
            for name, value in model.named_parameters():
                if value.requires_grad:
                    self.assertEqual(value.dtype, torch.float32, name)
                    self.assertIsNotNone(value.grad, name)
                    self.assertEqual(value.grad.dtype, torch.float32, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
                    current[name] = value.grad.detach().clone()
                else:
                    self.assertEqual(value.dtype, torch.bfloat16, name)
                    self.assertIsNone(value.grad, name)
            for name in ("model.mask_fusion.mask_encoder.patch_conv.weight",
                         "model.mask_fusion.layers.0.q_proj.weight", "model.mm_projector.0.weight"):
                self.assertGreater(current[name].abs().sum().item(), 0, name)
            before = model.get_model().mm_projector[0].weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.get_model().mm_projector[0].weight))
            for name, value in model.named_parameters():
                if name in frozen_before:
                    self.assertTrue(torch.equal(value, frozen_before[name]), name)
            gradients.append(current)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)

    def test_rank_zero_writer_never_calls_state_dict_after_collective_collection(self):
        model = make_tiny_model()
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=0.001)
        model(**make_batch()).loss.backward()
        optimizer.step()
        model.eval()
        weights = adapter_weights(model)
        core = model.get_model()
        with torch.no_grad():
            expected_logits = model(**make_batch()).logits
        with TemporaryDirectory(prefix="gathered-adapter-contract-") as directory:
            # A rank-only state_dict here would deadlock an actual FSDP export.
            with patch.object(core.mask_fusion, "state_dict", side_effect=AssertionError("unexpected collective")), \
                    patch.object(core.mm_projector, "state_dict", side_effect=AssertionError("unexpected collective")):
                save_adapter(model, directory, "offline-tiny-backbone", weights=weights)
            saved = load_file(str(Path(directory) / "adapter.safetensors"))
            self.assertEqual(set(saved), set(weights))
            self.assertFalse(any("embed_tokens" in key or "vision_tower" in key for key in saved))
            restored = make_tiny_model()
            load_adapter(restored, directory)
            with torch.no_grad():
                actual_logits = restored(**make_batch()).logits
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            for name, expected in weights.items():
                torch.testing.assert_close(saved[name], expected, rtol=0, atol=0)

    def test_gathered_writer_rejects_frozen_backbone_weights(self):
        model = make_tiny_model()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fusion/projector"):
                save_adapter(model, directory, "offline", weights={"model.embed_tokens.weight": torch.zeros(2, 2)})


if __name__ == "__main__":
    unittest.main()
