"""Multiple source files stay ordered and are all checked on training resume."""

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
from test_data import FakeTokenizer, TinyTrainingModel, fake_image_tokenize


class ManifestArgumentTests(unittest.TestCase):
    def test_single_file_arguments_keep_the_existing_representation(self):
        args = train.parse_args(["--base-model", "base", "--data", "one.json", "--output", "run"])
        self.assertEqual(args.data, "one.json")
        self.assertIsNone(args.val_data)

    def test_multiple_and_repeated_arguments_preserve_order_and_spaces(self):
        args = train.parse_args([
            "--base-model", "base", "--data", "first set.jsonl", "second.json",
            "--data", "third.json", "--val-data", "val one.json",
            "--validation", "val two.jsonl", "--output", "run",
        ])
        self.assertEqual(args.data, ["first set.jsonl", "second.json", "third.json"])
        self.assertEqual(args.val_data, ["val one.json", "val two.jsonl"])

    def test_fingerprint_covers_every_file_and_its_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = [Path(temporary) / name for name in ("a.json", "b.jsonl")]
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            original = train._fingerprint([first, second])
            self.assertEqual(train._fingerprint(first), train._fingerprint([first]))
            self.assertNotEqual(original, train._fingerprint([second, first]))
            second.write_text("changed", encoding="utf-8")
            self.assertNotEqual(original, train._fingerprint([first, second]))
            self.assertEqual(original[0], train._fingerprint(first))


class MultipleManifestTrainingTests(unittest.TestCase):
    def test_real_combined_dataset_resumes_and_rejects_a_changed_second_file(self):
        def load_model(base_model, *, adapter_path=None, **kwargs):
            model = TinyTrainingModel()
            if adapter_path:
                model.load_state_dict(torch.load(Path(adapter_path) / "toy_adapter.pt", weights_only=True))
            return model, FakeTokenizer(), object()

        def save_adapter(model, output, base_model, tokenizer=None):
            Path(output).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(output) / "toy_adapter.pt")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for source_index, count in enumerate((2, 3)):
                folder = root / f"source {source_index}"
                folder.mkdir()
                Image.new("RGB", (4, 4), (source_index * 100, 0, 0)).save(folder / "image.png")
                np.save(folder / "mask.npy", np.full((4, 4), source_index + 20, dtype=np.float32))
                records = [dict(id=str(i), image="image.png", mask="mask.npy", question=f"Question {i}?", answer="normal") for i in range(count)]
                path = folder / ("train.json" if source_index == 0 else "train.jsonl")
                path.write_text(json.dumps(records) if source_index == 0 else "\n".join(map(json.dumps, records)), encoding="utf-8")
                manifests.append(path)
            prepared = {
                "images": torch.zeros(1, 3, 4, 4), "masks": torch.ones(1, 1, 4, 4),
                "mask_valid": torch.ones(1, 1, 4, 4, dtype=torch.bool), "image_size": (4, 4),
            }
            common = [
                "--base-model", str(root / "base"), "--data", *map(str, manifests),
                "--epochs", "2", "--batch-size", "2", "--grad-accum", "2",
                "--device", "cpu", "--dtype", "float32", "--seed", "19",
                "--save-every-steps", "1", "--lr", "0.01", "--projector-lr", "0.02",
            ]
            with patch("mask_fusion.model.load_model", side_effect=load_model), \
                 patch("mask_fusion.model.configure_trainable"), \
                 patch("mask_fusion.model.save_adapter", side_effect=save_adapter), \
                 patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize), \
                 patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared) as prepare, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                train.main(common + ["--output", str(root / "full")])
                consumed_masks = {float(call.args[1][0, 0]) for call in prepare.call_args_list}
                self.assertEqual(consumed_masks, {20.0, 21.0})
                checkpoint = root / "full" / "checkpoint-step-1"
                train.main(common + ["--output", str(root / "resumed"), "--resume", str(checkpoint)])
                full = torch.load(root / "full" / "toy_adapter.pt", weights_only=True)
                resumed = torch.load(root / "resumed" / "toy_adapter.pt", weights_only=True)
                for key in full:
                    torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([event["qa_records"] for event in events if event["event"] == "train_start"], [5, 5])
                content = manifests[1].read_text(encoding="utf-8")
                manifests[1].write_text(content.replace("Question 0?", "A different question?"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Resume settings/data differ.*train_data"):
                    train.main(common + ["--output", str(root / "changed"), "--resume", str(checkpoint)])


if __name__ == "__main__":
    unittest.main()
