"""Exercise both stages and resumable training through a real tiny Qwen."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mask_fusion import train
from mask_fusion.config import FusionConfig
from mask_fusion.model import configure_image_mode, load_adapter, configure_trainable
from test_data import FakeTokenizer
from test_integration import make_batch, make_tiny_model


class SavedTokenizer(FakeTokenizer):
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class StageArgumentTests(unittest.TestCase):
    def test_stages_cannot_silently_change_labels_or_restore_old_optimizer(self):
        common = ["--base-model", "base", "--data", "data.jsonl", "--output", "run"]
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (
                ["--training-stage", "mask_pretrain"],
                ["--training-stage", "mask_pretrain", "--image-mode", "randomroi"],
                ["--init-adapter", "pretrain", "--resume", "old"],
            ):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    train.parse_args(common + extra)
        parsed = train.parse_args(common + ["--training-stage", "mask_pretrain", "--image-mode", "square"])
        self.assertEqual(parsed.training_stage, "mask_pretrain")

    def test_initialization_fingerprint_detects_changed_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "adapter_config.json").write_text("{}")
            (folder / "adapter.safetensors").write_bytes(b"first")
            before = train._initial_adapter_signature(folder)
            (folder / "adapter.safetensors").write_bytes(b"second")
            after = train._initial_adapter_signature(folder)
            self.assertNotEqual(before["adapter.safetensors"], after["adapter.safetensors"])


class StageTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_pretrain_resume_and_real_stage_weight_only_initialization(self):
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n")
            records = []
            samples = []
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

            loaded = []

            def loader(base, *, adapter_path=None, fusion_config=None, image_mode="square", **kwargs):
                fusion = FusionConfig.from_dict(fusion_config) if fusion_config is not None else None
                model = make_tiny_model(fusion_config=fusion)
                # Real OneVision has this frozen packing token even when stage
                # one uses square; the minimal original fixture does not.
                model.get_model().image_newline = torch.nn.Parameter(torch.zeros(32), requires_grad=False)
                configure_image_mode(model.config, image_mode)
                if adapter_path:
                    load_adapter(model, adapter_path)
                loaded.append(model)
                return model, SavedTokenizer(), object()

            common = ["--base-model", str(root / "base"), "--data", str(manifest),
                      "--batch-size", "1", "--grad-accum", "1", "--device", "cpu",
                      "--dtype", "float32", "--seed", "17", "--no-gradient-checkpointing",
                      "--lr", "0.002", "--projector-lr", "0.002", "--weight-decay", "0.1"]
            pretrain = common + ["--training-stage", "mask_pretrain", "--image-mode", "square",
                                 "--epochs", "2", "--save-every-steps", "1"]
            with patch("mask_fusion.train.QADataset", Dataset), \
                 patch("mask_fusion.model.load_model", side_effect=loader), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main(pretrain + ["--output", str(root / "pretrained")])
                self.assertEqual(loaded[-1].get_vision_tower().forward_calls, 0)
                original = load_file(str(root / "pretrained" / "adapter.safetensors"))
                train.main(pretrain + ["--output", str(root / "resumed"), "--resume",
                                       str(root / "pretrained" / "checkpoint-step-1")])
                resumed = load_file(str(root / "resumed" / "adapter.safetensors"))
                self.assertEqual(original.keys(), resumed.keys())
                for name in original:
                    torch.testing.assert_close(original[name], resumed[name], rtol=0, atol=0)

                # The new stage receives learned weights before any optimizer step,
                # can change packing, and does not restore the pretraining optimizer.
                initial_states = []

                def capture_configuration(model):
                    initial_states.append(model.get_model().mm_projector[0].weight.detach().clone())
                    return configure_trainable(model)

                with patch("mask_fusion.model.configure_trainable", side_effect=capture_configuration):
                    train.main(common + ["--epochs", "1", "--image-mode", "randomroi",
                                         "--init-adapter", str(root / "pretrained"),
                                         "--output", str(root / "sft")])
                torch.testing.assert_close(initial_states[-1], original["mm_projector.0.weight"], rtol=0, atol=0)
                self.assertGreater(loaded[-1].get_vision_tower().forward_calls, 0)
                self.assertFalse(getattr(loaded[-1].config, "mask_fusion_pretrain", False))
                state = train._load_training_state(root / "sft" / "training_state.pt")
                self.assertEqual(state["global_step"], 3)
                self.assertEqual(state["signature"]["training_stage"], "sft")
                self.assertIsNotNone(state["signature"]["initial_adapter"])
                self.assertEqual(state["scheduler"]["last_epoch"], 3)
                # Resuming stage two does not require the initialization CLI option.
                train.main(common + ["--epochs", "1", "--image-mode", "randomroi",
                                     "--resume", str(root / "sft"), "--output", str(root / "sft_resumed")])
                sft = load_file(str(root / "sft" / "adapter.safetensors"))
                sft_resumed = load_file(str(root / "sft_resumed" / "adapter.safetensors"))
                for name in sft:
                    torch.testing.assert_close(sft[name], sft_resumed[name], rtol=0, atol=0)

                with self.assertRaisesRegex(ValueError, "use --init-adapter"):
                    train.main(common + ["--epochs", "2", "--image-mode", "square",
                                         "--resume", str(root / "pretrained"),
                                         "--output", str(root / "wrong_stage")])

            # Precomputed QA must not silently get paired with a resized mask.
            Image.new("RGB", (6, 6)).save(records[0]["image"])
            with self.assertRaisesRegex(ValueError, "without resizing"):
                train._validate_pretrain_sizes(Dataset(), make_tiny_model())


if __name__ == "__main__":
    unittest.main()
