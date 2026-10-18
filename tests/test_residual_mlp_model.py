"""Real tiny-Qwen coverage of residual-mask learning and adapter transitions.

No checkpoint downloads or GPUs are needed. Only the vision tower/tokenizer
and dataset loading are replaced; forward/backward, exports, and training state
restoration use the production implementations.
"""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from mask_fusion import train
from mask_fusion.config import FusionConfig
from mask_fusion.model import (
    configure_image_mode, configure_mask_pretraining, configure_trainable,
    load_adapter, read_adapter_config, save_adapter,
)
from test_integration import make_batch, make_tiny_model
from test_pretrain_training import SavedTokenizer


def tiny_config(enabled=True, **changes):
    config = FusionConfig(
        mask_hidden_size=8, expert_dim=16, position_dim=8,
        position_hidden_size=16, attention_dim=16, num_heads=4,
        direct_residual_mlp=enabled,
    )
    return replace(config, **changes)


def optimize(model, steps=4):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.003, weight_decay=0.0,
    )
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(**make_batch(), use_cache=False).loss.backward()
        optimizer.step()
    model.eval()


class ResidualMLPModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_four_answer_loss_steps_warm_up_projector_then_both_mlp_layers(self):
        model = make_tiny_model(fusion_config=tiny_config())
        configure_mask_pretraining(model)
        model.train()
        core = model.get_model()
        parameters = dict(model.named_parameters())
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items()
                  if not parameter.requires_grad}
        learned = {
            "cnn": core.mask_fusion.mask_encoder.patch_conv.weight,
            "mlp_first": core.mask_fusion.direct_mlp[0].weight,
            "mlp_last": core.mask_fusion.direct_mlp[2].weight,
            "projector": core.mm_projector[0].expert_weight,
        }
        initial = {name: parameter.detach().clone() for name, parameter in learned.items()}
        self.assertEqual(initial["projector"].count_nonzero().item(), 0)
        self.assertEqual(initial["mlp_last"].count_nonzero().item(), 0)
        # Zero decay makes delayed updates evidence of loss gradients, not decay.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        gradients = []
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**make_batch(), use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            current = {}
            for name, parameter in learned.items():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                current[name] = parameter.grad.abs().sum().item()
            gradients.append(current)
            for name in frozen:
                self.assertIsNone(parameters[name].grad, name)
            optimizer.step()
            for name, previous in frozen.items():
                self.assertTrue(torch.equal(parameters[name], previous), name)
            if step < 2:
                self.assertTrue(torch.equal(learned["mlp_first"], initial["mlp_first"]))
        self.assertGreater(gradients[0]["projector"], 0)
        for name in ("cnn", "mlp_first", "mlp_last"):
            self.assertEqual(gradients[0][name], 0, name)
        self.assertGreater(gradients[1]["cnn"], 0)
        self.assertGreater(gradients[1]["mlp_last"], 0)
        self.assertEqual(gradients[1]["mlp_first"], 0)
        self.assertGreater(gradients[2]["mlp_first"], 0)
        for name, parameter in learned.items():
            self.assertFalse(torch.equal(parameter, initial[name]), name)
        self.assertEqual(core.vision_tower.forward_calls, 0)

    def test_pretrain_export_restores_phase1_and_phase2_logits_exactly(self):
        model = make_tiny_model(fusion_config=tiny_config())
        configure_mask_pretraining(model)
        optimize(model)
        with torch.no_grad():
            expected_phase1 = model(**make_batch(), use_cache=False).logits
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(model, directory, "offline-tiny")
            weights = load_file(str(Path(directory) / "adapter.safetensors"))
            self.assertEqual(read_adapter_config(directory)["training_stage"], "mask_pretrain")
            self.assertIn("mask_fusion.direct_mlp.0.weight", weights)
            self.assertIn("mask_fusion.direct_mlp.2.weight", weights)
            self.assertIn("mm_projector.0.weight", weights)
            self.assertFalse(any("expert_weight" in name or "visual_weight" in name for name in weights))
            restored = make_tiny_model(fusion_config=tiny_config())
            load_adapter(restored, directory)
            configure_trainable(model)
            model.eval()
            with torch.no_grad():
                expected_phase2 = model(**make_batch(), use_cache=False).logits
                actual_phase2 = restored(**make_batch(), use_cache=False).logits
            torch.testing.assert_close(actual_phase2, expected_phase2, rtol=0, atol=0)
            self.assertGreater(restored.get_vision_tower().forward_calls, 0)
            configure_mask_pretraining(restored)
            with torch.no_grad():
                actual_phase1 = restored(**make_batch(), use_cache=False).logits
            torch.testing.assert_close(actual_phase1, expected_phase1, rtol=0, atol=0)
            for name, value in model.get_model().mask_fusion.direct_mlp.state_dict().items():
                torch.testing.assert_close(restored.get_model().mask_fusion.direct_mlp.state_dict()[name],
                                           value, rtol=0, atol=0)

    def test_attention_only_never_updates_the_unused_direct_mlp(self):
        model = make_tiny_model(fusion_config=tiny_config(mode="attention"))
        direct = model.get_model().mask_fusion.direct_mlp
        before = {name: parameter.detach().clone() for name, parameter in direct.named_parameters()}
        self.assertTrue(all(not parameter.requires_grad for parameter in direct.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.2)
        model.train()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(**make_batch(), use_cache=False).loss.backward()
            self.assertTrue(all(parameter.grad is None for parameter in direct.parameters()))
            optimizer.step()
        for name, parameter in direct.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_old_metadata_without_new_fields_loads_strictly_in_legacy_model(self):
        model = make_tiny_model(fusion_config=tiny_config(False))
        optimize(model, steps=2)
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(model, directory, "offline-tiny")
            config_path = Path(directory) / "adapter_config.json"
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            for name in ("direct_residual_mlp", "direct_mlp_hidden_size"):
                del metadata["fusion_config"][name]
            config_path.write_text(json.dumps(metadata), encoding="utf-8")
            normalized = read_adapter_config(directory)["fusion_config"]
            self.assertFalse(normalized["direct_residual_mlp"])
            self.assertEqual(normalized["direct_mlp_hidden_size"], 256)
            restored = make_tiny_model(fusion_config=tiny_config(False))
            load_adapter(restored, directory)
            self.assertIsNone(restored.get_model().mask_fusion.direct_mlp)
            with torch.no_grad():
                torch.testing.assert_close(restored(**make_batch(), use_cache=False).logits,
                                           model(**make_batch(), use_cache=False).logits, rtol=0, atol=0)

    def test_explicit_old_to_new_upgrade_restores_identity_even_on_reused_model(self):
        legacy = make_tiny_model(fusion_config=tiny_config(False))
        optimize(legacy, steps=2)  # Make expert projector columns nonzero.
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(legacy, directory, "offline-tiny")
            upgraded = make_tiny_model(fusion_config=tiny_config(direct_mlp_hidden_size=13))
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                load_adapter(upgraded, directory)
            direct = upgraded.get_model().mask_fusion.direct_mlp
            first_before = direct[0].weight.detach().clone()
            with torch.no_grad():
                direct[2].weight.fill_(0.5)
                direct[2].bias.fill_(2)
            load_adapter(upgraded, directory, allow_direct_mlp_upgrade=True)
            self.assertEqual(direct[2].weight.count_nonzero().item(), 0)
            self.assertEqual(direct[2].bias.count_nonzero().item(), 0)
            torch.testing.assert_close(direct[0].weight, first_before, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(upgraded(**make_batch(), use_cache=False).logits,
                                           legacy(**make_batch(), use_cache=False).logits, rtol=0, atol=0)

    def test_upgrade_never_tolerates_unrelated_missing_or_corrupt_new_mlp_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path, new_path = root / "old", root / "new"
            save_adapter(make_tiny_model(fusion_config=tiny_config(False)), legacy_path, "offline-tiny")
            old_weights = load_file(str(legacy_path / "adapter.safetensors"))
            del old_weights["mask_fusion.mask_encoder.patch_conv.weight"]
            save_file(old_weights, str(legacy_path / "adapter.safetensors"))
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_adapter(make_tiny_model(fusion_config=tiny_config()), legacy_path,
                             allow_direct_mlp_upgrade=True)
            save_adapter(make_tiny_model(fusion_config=tiny_config()), new_path, "offline-tiny")
            new_weights = load_file(str(new_path / "adapter.safetensors"))
            del new_weights["mask_fusion.direct_mlp.0.weight"]
            save_file(new_weights, str(new_path / "adapter.safetensors"))
            for allow in (False, True):
                with self.subTest(allow_upgrade=allow), self.assertRaisesRegex(ValueError, "exactly match"):
                    load_adapter(make_tiny_model(fusion_config=tiny_config()), new_path,
                                 allow_direct_mlp_upgrade=allow)

    def test_upgrade_opt_in_does_not_permit_other_configuration_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_adapter(make_tiny_model(fusion_config=tiny_config(False)), root / "old", "offline-tiny")
            save_adapter(make_tiny_model(fusion_config=tiny_config()), root / "new", "offline-tiny")
            for source, target in (
                ("old", tiny_config(score_scale=1.0)),
                ("old", tiny_config(num_layers=2)),
                ("new", tiny_config(False)),
                ("new", tiny_config(direct_mlp_hidden_size=17)),
            ):
                with self.subTest(source=source, target=target), self.assertRaisesRegex(ValueError, "fusion_config"):
                    load_adapter(make_tiny_model(fusion_config=target), root / source,
                                 allow_direct_mlp_upgrade=True)

    @contextlib.contextmanager
    def training_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            records, samples, loaded = [], [], []
            for index in range(3):
                batch = make_batch(seed=37 + index)
                image_path, mask_path = root / f"{index}.png", root / f"{index}.npy"
                Image.new("RGB", (4, 4)).save(image_path)
                np.save(mask_path, batch["masks"][0][0, 0].numpy())
                records.append({"image": str(image_path), "mask": str(mask_path)})
                samples.append({
                    "input_ids": batch["input_ids"][0], "labels": batch["labels"][0],
                    "images": batch["images"][0], "masks": batch["masks"][0],
                    "mask_valid": batch["mask_valid"][0], "image_size": (4, 4),
                })

            class Dataset(torch.utils.data.Dataset):
                def __init__(self, *args, **kwargs):
                    self.records = records

                def __len__(self):
                    return len(samples)

                def __getitem__(self, index):
                    return samples[index]

                def set_epoch(self, epoch):
                    pass

            def loader(base, *, adapter_path=None, fusion_config=None, image_mode="square", **kwargs):
                if fusion_config is None and adapter_path:
                    fusion_config = read_adapter_config(adapter_path)["fusion_config"]
                config = FusionConfig.from_dict(fusion_config) if fusion_config is not None else tiny_config(False)
                model = make_tiny_model(fusion_config=config)
                model.get_model().image_newline = torch.nn.Parameter(torch.zeros(32), requires_grad=False)
                configure_image_mode(model.config, image_mode)
                if adapter_path:
                    load_adapter(model, adapter_path)
                loaded.append(model)
                return model, SavedTokenizer(), object()

            common = ["--base-model", str(root / "base"), "--data", str(manifest),
                      "--batch-size", "1", "--grad-accum", "1", "--device", "cpu",
                      "--dtype", "float32", "--seed", "17", "--no-gradient-checkpointing",
                      "--lr", "0.003", "--projector-lr", "0.003", "--weight-decay", "0"]
            for name, config in (("residual", tiny_config()), ("legacy", tiny_config(False))):
                (root / f"{name}.json").write_text(config.to_json(), encoding="utf-8")
            with patch("mask_fusion.train.QADataset", Dataset), \
                    patch("mask_fusion.model.load_model", side_effect=loader), \
                    contextlib.redirect_stdout(io.StringIO()):
                yield root, common, loaded

    def test_real_train_pretrain_resume_and_phase2_weight_initialization(self):
        with self.training_fixture() as (root, common, loaded):
            config_args = ["--fusion-config", str(root / "residual.json")]
            stage1 = common + config_args + ["--training-stage", "mask_pretrain", "--image-mode", "square",
                                            "--epochs", "2", "--save-every-steps", "1"]
            train.main(stage1 + ["--output", str(root / "pretrained")])
            self.assertEqual(loaded[-1].get_vision_tower().forward_calls, 0)
            original = load_file(str(root / "pretrained" / "adapter.safetensors"))
            initial = make_tiny_model(fusion_config=tiny_config()).get_model().mask_fusion.direct_mlp
            self.assertFalse(torch.equal(original["mask_fusion.direct_mlp.0.weight"], initial[0].weight))
            self.assertGreater(original["mask_fusion.direct_mlp.2.weight"].count_nonzero().item(), 0)
            train.main(stage1 + ["--output", str(root / "resumed"), "--resume",
                                 str(root / "pretrained" / "checkpoint-step-2")])
            resumed = load_file(str(root / "resumed" / "adapter.safetensors"))
            self.assertEqual(set(original), set(resumed))
            for name in original:
                torch.testing.assert_close(resumed[name], original[name], rtol=0, atol=0, msg=name)
            self.assertEqual(train._load_training_state(root / "resumed" / "training_state.pt")["global_step"], 6)

            initial_states = []

            def capture_phase2(model):
                core = model.get_model()
                initial_states.append({
                    "mask_fusion.direct_mlp.0.weight": core.mask_fusion.direct_mlp[0].weight.detach().clone(),
                    "mask_fusion.direct_mlp.2.weight": core.mask_fusion.direct_mlp[2].weight.detach().clone(),
                    "mm_projector.0.weight": core.mm_projector[0].weight.detach().clone(),
                })
                return configure_trainable(model)

            with patch("mask_fusion.model.configure_trainable", side_effect=capture_phase2):
                train.main(common + ["--epochs", "1", "--image-mode", "randomroi",
                                     "--init-adapter", str(root / "pretrained"), "--output", str(root / "sft")])
            for name, value in initial_states[-1].items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0, msg=name)
            self.assertTrue(loaded[-1].config.mask_fusion["direct_residual_mlp"])
            self.assertFalse(loaded[-1].config.mask_fusion_pretrain)
            self.assertGreater(loaded[-1].get_vision_tower().forward_calls, 0)
            state = train._load_training_state(root / "sft" / "training_state.pt")
            self.assertEqual(state["global_step"], 3)
            self.assertEqual(state["scheduler"]["last_epoch"], 3)
            self.assertEqual(state["signature"]["training_stage"], "sft")
            self.assertIsNotNone(state["signature"]["initial_adapter"])

    def test_train_init_can_upgrade_legacy_but_resume_cannot(self):
        with self.training_fixture() as (root, common, loaded):
            stage = common + ["--training-stage", "mask_pretrain", "--image-mode", "square", "--epochs", "1"]
            train.main(stage + ["--fusion-config", str(root / "legacy.json"), "--output", str(root / "old")])
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                train.main(stage + ["--fusion-config", str(root / "residual.json"),
                                    "--resume", str(root / "old"), "--output", str(root / "bad_resume")])
            train.main(common + ["--fusion-config", str(root / "residual.json"),
                                 "--image-mode", "square", "--epochs", "1",
                                 "--init-adapter", str(root / "old"), "--output", str(root / "new_sft")])
            self.assertTrue(loaded[-1].config.mask_fusion["direct_residual_mlp"])
            upgraded = load_file(str(root / "new_sft" / "adapter.safetensors"))
            self.assertGreater(upgraded["mask_fusion.direct_mlp.2.weight"].count_nonzero().item(), 0)
            self.assertEqual(train._load_training_state(root / "new_sft" / "training_state.pt")["global_step"], 3)


if __name__ == "__main__":
    unittest.main()
