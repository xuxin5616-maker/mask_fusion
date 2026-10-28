"""Real tiny-Qwen checks for mask-guided vision with the original projector."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from mask_fusion import train
from mask_fusion.config import FusionConfig
from mask_fusion.model import (
    configure_mask_pretraining, configure_trainable, load_adapter,
    read_adapter_config, save_adapter,
)
from test_integration import make_batch, make_tiny_model
from test_pretrain_training import SavedTokenizer


def tiny_config(**overrides):
    values = dict(mask_hidden_size=8, expert_dim=16, position_dim=8,
                  position_hidden_size=16, attention_dim=16, num_heads=4,
                  mode="attention", mask_encoder_type="cnn", direct_to_projector=False)
    values.update(overrides)
    return FusionConfig(**values)


class AttentionOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_preset_and_invalid_combinations(self):
        folder = Path(__file__).resolve().parents[1] / "configs"
        preset = FusionConfig.from_json((folder / "cnn_attention_only.json").read_text())
        self.assertEqual(preset.mask_encoder_type, "cnn")
        self.assertEqual(preset.mode, "attention")
        self.assertFalse(preset.direct_to_projector)
        self.assertFalse(preset.direct_residual_mlp)
        self.assertEqual((preset.num_layers, preset.num_heads), (1, 8))
        self.assertTrue(FusionConfig.from_json((folder / "attention.json").read_text()).direct_to_projector)
        for change in ({"mode": "both"}, {"mode": "direct"}, {"direct_residual_mlp": True},
                       {"direct_to_projector": "false"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                tiny_config(**change)

    def test_original_projector_is_not_expanded_or_reinitialized(self):
        reference = make_tiny_model(fusion_config=tiny_config(direct_to_projector=True))
        with patch("mask_fusion.model.expand_projector", side_effect=AssertionError("projector expanded")):
            model = make_tiny_model(fusion_config=tiny_config())
        projector = model.get_model().mm_projector
        expected = reference.get_model().mm_projector
        self.assertEqual(projector[0].in_features, 12)
        torch.testing.assert_close(projector[0].weight, expected[0].weight[:, :12], rtol=0, atol=0)
        torch.testing.assert_close(projector[0].bias, expected[0].bias, rtol=0, atol=0)
        torch.testing.assert_close(projector[2].weight, expected[2].weight, rtol=0, atol=0)
        torch.testing.assert_close(projector[2].bias, expected[2].bias, rtol=0, atol=0)

    def test_only_updated_visual_features_enter_projector_and_mask_still_affects_them(self):
        model = make_tiny_model(fusion_config=tiny_config())
        core = model.get_model()
        seen, outputs = [], []
        hooks = [core.mm_projector.register_forward_pre_hook(
            lambda module, values: seen.append(values[0].detach().clone())),
            core.mask_fusion.register_forward_hook(
                lambda module, values, result: outputs.append(result.detach().clone()))]
        try:
            with patch.object(core.mask_fusion, "encode_direct", side_effect=AssertionError("direct path used")):
                result = model(**make_batch((1, 2)))
                changed = make_batch((1, 2))
                changed["masks"] = [torch.zeros_like(mask) for mask in changed["masks"]]
                model(**changed)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(tuple(result.logits.shape), (2, 12, 64))
        self.assertEqual(tuple(seen[0].shape), (3, 4, 12))
        torch.testing.assert_close(seen[0], outputs[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(seen[0], seen[1]))
        self.assertEqual(core.vision_tower.forward_calls, 2)
        self.assertIsNone(core.mask_fusion.direct_mlp)
        self.assertIsNone(core.mask_fusion.direct_norm)

    def test_first_step_gradients_reach_cnn_attention_and_projector_only(self):
        model = make_tiny_model(fusion_config=tiny_config())
        model.train()
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        loss = model(**make_batch((1, 2)), use_cache=False).loss
        loss.backward()
        fusion = model.get_model().mask_fusion
        checked = (fusion.mask_encoder.patch_conv.weight, fusion.mask_encoder.context_conv.weight,
                   fusion.mask_encoder.hidden_norm.weight, fusion.expert_projection[0].weight,
                   fusion.position_projection.weight, fusion.prior_norm.weight,
                   fusion.layers[0].q_proj.weight, fusion.layers[0].k_proj.weight,
                   fusion.layers[0].v_proj.weight, fusion.layers[0].out_proj.weight,
                   fusion.layers[0].alpha, model.get_model().mm_projector[0].weight)
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter, frozen[name]), name)

    def test_bf16_checkpointed_training_keeps_adapter_masters_fp32(self):
        model = make_tiny_model(fusion_config=tiny_config()).to(dtype=torch.bfloat16)
        configure_trainable(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(**make_batch(), use_cache=False).loss
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertEqual(parameter.dtype, torch.float32, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_save_reload_generation_and_cross_architecture_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            model = make_tiny_model(fusion_config=tiny_config())
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
            model(**make_batch()).loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                expected = model(**make_batch()).logits
            save_adapter(model, directory, "offline-base")
            self.assertFalse(read_adapter_config(directory)["fusion_config"]["direct_to_projector"])
            self.assertEqual(load_file(str(Path(directory) / "adapter.safetensors"))["mm_projector.0.weight"].shape,
                             (32, 12))
            reloaded = make_tiny_model(fusion_config=tiny_config())
            load_adapter(reloaded, directory)
            with torch.no_grad():
                torch.testing.assert_close(reloaded(**make_batch()).logits, expected, rtol=0, atol=0)
            prompt = make_batch()
            ids = prompt.pop("input_ids")
            prompt.pop("labels")
            self.assertEqual(reloaded.generate(inputs=ids, **prompt, max_new_tokens=2, do_sample=False,
                                              use_cache=True, pad_token_id=0, eos_token_id=2).shape[0], 1)
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                load_adapter(make_tiny_model(fusion_config=tiny_config(direct_to_projector=True)), directory)
            legacy = Path(directory) / "legacy"
            save_adapter(make_tiny_model(), legacy, "offline-base")
            metadata_path = legacy / "adapter_config.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["fusion_config"].pop("direct_to_projector")
            metadata_path.write_text(json.dumps(metadata))
            load_adapter(make_tiny_model(), legacy)
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                load_adapter(model, legacy, allow_image_mode_change=True, allow_direct_mlp_upgrade=True)

    def test_mask_only_pretraining_is_rejected_and_none_mode_preserves_visual_width(self):
        model = make_tiny_model(fusion_config=tiny_config())
        with self.assertRaisesRegex(ValueError, "real-image SFT"):
            configure_mask_pretraining(model)
        fusion = model.get_model().mask_fusion
        with self.assertRaisesRegex(ValueError, "real-image SFT"):
            fusion(torch.randn(1, 4, 12), torch.zeros(1, 1, 4, 4), mask_only=True)
        disabled = make_tiny_model(fusion_config=tiny_config(mode="none")).get_model().mask_fusion
        pixels = torch.randn(1, 4, 12)
        self.assertIs(disabled(pixels), pixels)

    def test_real_fresh_training_and_resume_restore_same_next_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data.jsonl"
            manifest.write_text("{}\n")
            config_file = root / "fusion.json"
            config_file.write_text(tiny_config().to_json())
            samples = [make_batch(seed=37 + i) for i in range(3)]
            loaded = []

            class Dataset(torch.utils.data.Dataset):
                def __init__(self, *args, **kwargs):
                    self.records = [{"image": f"image-{i}", "mask": f"mask-{i}"} for i in range(3)]

                def __len__(self):
                    return len(samples)

                def __getitem__(self, index):
                    batch = samples[index]
                    return {"input_ids": batch["input_ids"][0], "labels": batch["labels"][0],
                            "images": batch["images"][0], "masks": batch["masks"][0],
                            "mask_valid": batch["mask_valid"][0], "image_size": (4, 4)}

                def set_epoch(self, epoch):
                    pass

            def loader(base, *, fusion_config=None, adapter_path=None, **kwargs):
                config = FusionConfig.from_dict(fusion_config) if fusion_config else tiny_config()
                model = make_tiny_model(fusion_config=config)
                if adapter_path:
                    load_adapter(model, adapter_path)
                loaded.append(model)
                return model, SavedTokenizer(), object()

            common = ["--base-model", str(root / "base"), "--data", str(manifest),
                      "--fusion-config", str(config_file), "--image-mode", "square",
                      "--epochs", "2", "--batch-size", "1", "--grad-accum", "1",
                      "--device", "cpu", "--dtype", "float32", "--seed", "17",
                      "--no-gradient-checkpointing", "--save-every-steps", "1"]
            with patch("mask_fusion.train.QADataset", Dataset), \
                    patch("mask_fusion.model.load_model", side_effect=loader), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main(common + ["--output", str(root / "fresh")])
                fresh = load_file(str(root / "fresh" / "adapter.safetensors"))
                signature = train._load_training_state(root / "fresh" / "training_state.pt")["signature"]
                self.assertIsNone(signature.get("initial_adapter"))
                self.assertEqual(signature["training_stage"], "sft")
                train.main(common + ["--output", str(root / "resumed"), "--resume",
                                     str(root / "fresh" / "checkpoint-step-2")])
                resumed = load_file(str(root / "resumed" / "adapter.safetensors"))
                for name in fresh:
                    torch.testing.assert_close(resumed[name], fresh[name], rtol=0, atol=0, msg=name)
                self.assertEqual(loaded[-1].get_model().mm_projector[0].in_features, 12)


if __name__ == "__main__":
    unittest.main()
