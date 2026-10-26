"""Jet masks through the shared frozen tower and real tiny-Qwen training path."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from mask_fusion import train
from mask_fusion.config import FusionConfig
from mask_fusion.data import collate_samples
from mask_fusion.model import (
    configure_image_mode, configure_mask_pretraining, configure_trainable,
    load_adapter, read_adapter_config, save_adapter,
)
from test_integration import make_batch, make_tiny_model
from test_pretrain_training import SavedTokenizer


def jet_config(mode="both"):
    return FusionConfig(
        mask_encoder_type="siglip_jet", attention_dim=16, num_heads=4,
        direct_residual_mlp=True, direct_mlp_hidden_size=8, mode=mode,
    )


def jet_batch(crop_counts=(1,), seed=37):
    batch = make_batch(crop_counts, seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    batch["mask_images"] = [
        torch.rand(count, 3, 4, 4, generator=generator) * 2 - 1
        for count in crop_counts
    ]
    return batch


class JetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_sft_uses_one_shared_tower_for_both_streams_and_keeps_token_count(self):
        model = make_tiny_model(fusion_config=jet_config())
        core = model.get_model()
        batch = jet_batch((1, 2))
        inputs = []
        handle = core.mm_projector.register_forward_pre_hook(
            lambda module, values: inputs.append(values[0].detach().clone()))
        try:
            result = model(**batch, use_cache=False)
        finally:
            handle.remove()
        self.assertEqual(core.vision_tower.encoded_batch_sizes, [3, 3])
        self.assertEqual(tuple(result.logits.shape), (2, 12, 64))
        self.assertEqual(tuple(inputs[0].shape), (3, 4, 24))
        with torch.no_grad():
            expected = core.vision_tower(torch.cat(batch["mask_images"]))
        torch.testing.assert_close(inputs[0][..., 12:], expected, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in core.vision_tower.parameters()))
        self.assertFalse(any("mask_encoder." in name for name, _ in core.mask_fusion.named_parameters()))

    def test_pretrain_encodes_only_mask_and_skips_attention_and_rgb(self):
        model = make_tiny_model(fusion_config=jet_config())
        configure_mask_pretraining(model)
        batch = jet_batch((1, 2))
        core = model.get_model()
        captured = []
        handle = core.mm_projector.register_forward_pre_hook(
            lambda module, values: captured.append(values[0].detach().clone()))
        try:
            with patch.object(core.mask_fusion.layers[0], "forward", side_effect=AssertionError("attention used")):
                first = model(**batch, use_cache=False)
                changed = dict(batch, images=[torch.full_like(x, float("nan")) for x in batch["images"]])
                second = model(**changed, use_cache=False)
        finally:
            handle.remove()
        self.assertEqual(core.vision_tower.encoded_batch_sizes, [3, 3])
        self.assertTrue(torch.equal(captured[0][..., :12], torch.zeros(3, 4, 12)))
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)

    def test_pretrain_learns_adapter_and_norm_without_changing_backbone_or_old_projector(self):
        model = make_tiny_model(fusion_config=jet_config())
        configure_mask_pretraining(model)
        core = model.get_model()
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        for name, parameter in model.named_parameters():
            expected = ("mask_fusion.direct_mlp." in name or "mask_fusion.direct_norm." in name
                        or name == "model.mm_projector.0.expert_weight")
            self.assertEqual(parameter.requires_grad, expected, name)
        old_mlp = core.mask_fusion.direct_mlp[0].weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model.train()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**jet_batch(), use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(core.mm_projector[0].expert_weight.grad.abs().sum().item(), 0)
            if step == 2:
                self.assertGreater(core.mask_fusion.direct_mlp[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(core.mask_fusion.direct_norm.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(core.mask_fusion.direct_mlp[0].weight, old_mlp))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter, frozen[name]), name)

    def test_sft_trains_both_routes_through_frozen_qwen(self):
        model = make_tiny_model(fusion_config=jet_config())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(**jet_batch(), use_cache=False).loss.backward()
            optimizer.step()
        fusion = model.get_model().mask_fusion
        for parameter in (fusion.layers[0].q_proj.weight, fusion.layers[0].k_proj.weight,
                          fusion.direct_norm.weight, fusion.direct_mlp[0].weight,
                          model.get_model().mm_projector[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.get_vision_tower().parameters()))
        self.assertTrue(all(p.grad is None for p in model.get_model().layers.parameters()))

    def test_export_reload_and_generation_preserve_both_training_stages(self):
        for pretrain in (False, True):
            with self.subTest(pretrain=pretrain), tempfile.TemporaryDirectory() as directory:
                model = make_tiny_model(fusion_config=jet_config())
                if pretrain:
                    configure_mask_pretraining(model)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    model(**jet_batch(), use_cache=False).loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model(**jet_batch()).logits
                save_adapter(model, directory, "offline-base")
                reloaded = make_tiny_model(fusion_config=jet_config())
                load_adapter(reloaded, directory)
                if pretrain:
                    configure_mask_pretraining(reloaded)
                reloaded.eval()
                with torch.no_grad():
                    actual = reloaded(**jet_batch()).logits
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                prompt = jet_batch()
                ids = prompt.pop("input_ids")
                prompt.pop("labels")
                before = reloaded.get_vision_tower().forward_calls
                generated = reloaded.generate(inputs=ids, **prompt, max_new_tokens=2, do_sample=False,
                                              use_cache=True, pad_token_id=0, eos_token_id=2)
                self.assertEqual(generated.shape[0], 1)
                self.assertEqual(reloaded.get_vision_tower().forward_calls - before, 1 if pretrain else 2)

    def test_invalid_inputs_fail_and_entirely_padded_mask_stays_finite(self):
        model = make_tiny_model(fusion_config=jet_config())
        batch = jet_batch()
        without = dict(batch)
        without.pop("mask_images")
        with self.assertRaisesRegex(ValueError, "mask_images"):
            model(**without)
        wrong = dict(batch, mask_images=[torch.randn(2, 3, 4, 4)])
        with self.assertRaisesRegex(ValueError, "crop"):
            model(**wrong)
        batch["mask_valid"] = [torch.zeros_like(x) for x in batch["mask_valid"]]
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(**batch).loss))

    def test_cnn_adapters_cannot_initialize_jet_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(make_tiny_model(), directory, "offline-base")
            with self.assertRaisesRegex(ValueError, "fusion_config"):
                load_adapter(make_tiny_model(fusion_config=jet_config()), directory,
                             allow_image_mode_change=True, allow_direct_mlp_upgrade=True)

    def test_attention_ablation_freezes_direct_norm_and_mlp(self):
        model = make_tiny_model(fusion_config=jet_config("attention"))
        fusion = model.get_model().mask_fusion
        self.assertTrue(all(not p.requires_grad for p in fusion.direct_norm.parameters()))
        self.assertTrue(all(not p.requires_grad for p in fusion.direct_mlp.parameters()))
        self.assertTrue(all(p.requires_grad for p in fusion.prior_norm.parameters()))
        configure_trainable(make_tiny_model(fusion_config=jet_config("direct")))

    def test_collate_forwards_jet_views_and_rejects_mixed_architectures(self):
        batch = jet_batch((1, 2))
        samples = [{"input_ids": batch["input_ids"][i], "labels": batch["labels"][i],
                    "images": batch["images"][i], "masks": batch["masks"][i],
                    "mask_valid": batch["mask_valid"][i], "mask_images": batch["mask_images"][i],
                    "image_size": (4, 4)} for i in range(2)]
        result = collate_samples(samples, 0)
        self.assertEqual([x.shape[0] for x in result["mask_images"]], [1, 2])
        samples[1].pop("mask_images")
        with self.assertRaisesRegex(ValueError, "mix"):
            collate_samples(samples, 0)

    def test_bf16_frozen_backbone_and_checkpointing_keep_adapter_gradients(self):
        model = make_tiny_model(fusion_config=jet_config()).to(dtype=torch.bfloat16)
        configure_trainable(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(**jet_batch(), use_cache=False).loss
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertEqual(parameter.dtype, torch.float32, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_real_training_resume_and_phase2_weight_only_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            config_file = root / "fusion.json"
            config_file.write_text(jet_config().to_json(), encoding="utf-8")
            records, samples, loaded = [], [], []
            for index in range(3):
                batch = jet_batch(seed=37 + index)
                image_path, mask_path = root / f"{index}.png", root / f"{index}.npy"
                Image.new("RGB", (4, 4)).save(image_path)
                np.save(mask_path, batch["masks"][0][0, 0].numpy())
                records.append({"image": str(image_path), "mask": str(mask_path)})
                samples.append({
                    "input_ids": batch["input_ids"][0], "labels": batch["labels"][0],
                    "images": batch["images"][0], "masks": batch["masks"][0],
                    "mask_images": batch["mask_images"][0], "mask_valid": batch["mask_valid"][0],
                    "image_size": (4, 4),
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
                config = FusionConfig.from_dict(fusion_config) if fusion_config is not None else jet_config()
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
            phase1 = common + ["--fusion-config", str(config_file), "--training-stage", "mask_pretrain",
                               "--image-mode", "square", "--epochs", "2", "--save-every-steps", "1"]
            with patch("mask_fusion.train.QADataset", Dataset), \
                    patch("mask_fusion.model.load_model", side_effect=loader), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main(phase1 + ["--output", str(root / "pretrained")])
                self.assertEqual(loaded[-1].get_vision_tower().forward_calls, 6)
                original = load_file(str(root / "pretrained" / "adapter.safetensors"))
                self.assertGreater(original["mask_fusion.direct_mlp.2.weight"].count_nonzero().item(), 0)
                train.main(phase1 + ["--resume", str(root / "pretrained" / "checkpoint-step-2"),
                                     "--output", str(root / "resumed")])
                resumed = load_file(str(root / "resumed" / "adapter.safetensors"))
                for name in original:
                    torch.testing.assert_close(resumed[name], original[name], rtol=0, atol=0, msg=name)
                initial_states = []

                def capture_phase2(model):
                    initial_states.append(model.get_model().mask_fusion.direct_mlp[2].weight.detach().clone())
                    return configure_trainable(model)

                with patch("mask_fusion.model.configure_trainable", side_effect=capture_phase2):
                    train.main(common + ["--epochs", "1", "--image-mode", "randomroi",
                                         "--init-adapter", str(root / "pretrained"),
                                         "--output", str(root / "sft")])
                torch.testing.assert_close(initial_states[-1], original["mask_fusion.direct_mlp.2.weight"],
                                           rtol=0, atol=0)
                self.assertEqual(loaded[-1].get_vision_tower().forward_calls, 6)
                self.assertFalse(loaded[-1].config.mask_fusion_pretrain)
                state = train._load_training_state(root / "sft" / "training_state.pt")
                self.assertEqual(state["global_step"], 3)
                self.assertEqual(state["scheduler"]["last_epoch"], 3)
                self.assertEqual(state["signature"]["training_stage"], "sft")


if __name__ == "__main__":
    unittest.main()
