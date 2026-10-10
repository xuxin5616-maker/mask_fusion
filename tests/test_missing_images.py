"""Missing-image records are filtered consistently before multimodal training."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mask_fusion.data import IGNORE_INDEX, QADataset
from mask_fusion.validate_data import validate_splits
from test_data import FakeTokenizer, fake_image_tokenize


class MissingImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        Image.new("RGB", (6, 4)).save(self.root / "image.png")
        np.save(self.root / "mask.npy", np.ones((4, 6), dtype=np.float32))
        self.record = {
            "id": "same_id", "category": "bottle", "image": "image.png", "mask": "mask.npy",
            "question": "Is it defective?", "answer": "No.",
        }

    def write(self, name, records):
        path = self.root / name
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def dataset(self, path, **kwargs):
        return QADataset(path, FakeTokenizer(), object(), SimpleNamespace(), **kwargs)

    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_filters_missing_images_and_keeps_original_positions_and_supervision(self, _):
        # Text-only examples may lack even QA/category/mask fields; skip them first.
        missing = [{}, *({"image": value} for value in (
            None, "", "  ", [], "missing.png", str(self.root / "absent.png"),
        ))]
        rows = [self.record, *missing, {**self.record, "answer": "A scratch is visible."}]
        path = self.write("mixed.json", rows)
        with self.assertWarnsRegex(UserWarning, "skipped 7 records"):
            dataset = self.dataset(path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual([r["id"] for r in dataset.records], ["same_id", "same_id"])
        self.assertEqual([s["source_index"] for s in dataset.sources], [0, 8])
        self.assertEqual([s["source_index"] for s in dataset.skipped_records], list(range(1, 8)))
        self.assertTrue(all(s["source_file"] == str(path.resolve()) for s in dataset.skipped_records))
        self.assertIn("does not exist", dataset.skipped_records[-1]["reason"])
        prepared = {
            "images": torch.zeros(1, 3, 8, 8), "masks": torch.ones(1, 1, 8, 8),
            "mask_valid": torch.ones(1, 1, 8, 8, dtype=torch.bool), "image_size": (6, 4),
        }
        with patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared):
            sample = dataset[1]
        self.assertEqual(sample["labels"][sample["labels"] != IGNORE_INDEX].tolist(),
                         FakeTokenizer()("A scratch is visible.<|im_end|>").input_ids)
        report = validate_splits(path, check_pixels=True)
        self.assertEqual(report["total"]["qa_count"], 2)
        self.assertEqual(report["total"]["unique_images"], 1)
        self.assertEqual(report["skipped_missing_images"], 7)
        self.assertEqual(report["splits"]["train"]["skipped_missing_images"], 7)
        self.assertEqual(report["splits"]["train"]["sources"][0]["skipped_missing_images"], 7)
        self.assertEqual([{k: v for k, v in item.items() if k != "split"}
                          for item in report["skipped_records"]], dataset.skipped_records)

    def test_all_missing_images_fail_for_dataset_and_each_requested_split(self):
        missing = self.write("missing.json", [{}, {"image": "missing.png"}])
        valid = self.write("valid.json", [self.record])
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "No usable samples"):
            self.dataset(missing)
        with self.assertRaisesRegex(ValueError, "train: No usable samples"):
            validate_splits(missing)
        with self.assertRaisesRegex(ValueError, "val: No usable samples"):
            validate_splits(valid, missing)

    def test_all_skipped_source_can_be_combined_with_a_usable_source(self):
        missing = self.write("missing.json", [{"image": None}])
        valid = self.write("valid.json", [self.record])
        with self.assertWarns(UserWarning):
            dataset = self.dataset([missing, valid])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.sources, [{"source_file": str(valid.resolve()), "source_index": 0}])
        report = validate_splits([missing, valid])
        self.assertEqual([s["qa_count"] for s in report["splits"]["train"]["sources"]], [0, 1])
        self.assertEqual(report["skipped_missing_images"], 1)

    def test_unsupported_image_types_and_other_record_errors_are_not_silently_skipped(self):
        cases = [
            ({"image": ["image.png"]}, "image must be"),
            ({"image": {"path": "image.png"}}, "image must be"),
            ({"image": 123}, "image must be"),
            ({"question": None}, "question must be"),
            ({"mask": "missing.npy"}, "Missing mask"),
        ]
        for change, error in cases:
            path = self.write("invalid.json", [{**self.record, **change}])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                self.dataset(path)
            with self.subTest(check="preflight", change=change), self.assertRaisesRegex(ValueError, error):
                validate_splits(path)

    def test_inference_uses_the_same_image_filter_and_preserves_source_index(self):
        raw = {key: value for key, value in self.record.items() if key not in ("answer", "id")}
        path = self.write("infer.json", [{"image": None}, raw])
        with self.assertWarns(UserWarning):
            dataset = self.dataset(path, training=False)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.records[0]["id"], "1")
        self.assertNotIn("answer", dataset.records[0])
        self.assertEqual(dataset.sources[0]["source_index"], 1)


if __name__ == "__main__":
    unittest.main()
