"""Manifest validation tests; no model, tokenizer, or GPU is loaded."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from mask_fusion.preprocessing import load_mask
from mask_fusion.validate_data import main, validate_splits


class ValidateDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for number in (1, 2, 3):
            Image.new("RGB", (8, 8), "white").save(self.root / f"image{number}.png")
            np.save(self.root / f"mask{number}.npy", np.full((8, 8), number + 0.125, np.float32))

    def record(self, number=1, category="bottle", identifier="q1", **overrides):
        record = {
            "id": identifier, "category": category, "image": f"image{number}.png",
            "mask": f"mask{number}.npy", "question": "Is this object defective?",
            "options": ["Yes", "No"], "answer": "B",
        }
        record.update(overrides)
        return record

    def write(self, name, records):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def test_multiple_questions_count_once_per_visual_and_mask_checked_once(self):
        train = self.write("train.jsonl", [
            self.record(identifier="question1"),
            self.record(identifier="question2", image="./image1.png", answer="A"),
            self.record(2, "bottle", "question3", options=None, answer="A scratch is visible."),
        ])
        with patch("mask_fusion.preprocessing.load_mask", wraps=load_mask) as load:
            report = validate_splits(train, check_pixels=True)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["total"], {"qa_count": 3, "unique_images": 2, "unique_masks": 2, "category_count": 1})
        split = report["splits"]["train"]
        self.assertEqual(split["per_category"]["bottle"], {"qa_count": 3, "unique_images": 2})
        self.assertEqual(split["mcq_count"], 2)
        self.assertEqual(split["free_text_count"], 1)
        self.assertIsNone(split["normality_counts"])
        self.assertEqual(report["observed_score_range"], {"min": 1.125, "max": 2.125})
        json.dumps(report)

    def test_disjoint_category_splits_with_local_id_reuse_are_valid(self):
        train = self.write("train.jsonl", [self.record(1, "bottle", "q1")])
        val = self.write("val.jsonl", [self.record(2, "capsule", "q1")])
        test = self.write("test.jsonl", [self.record(3, "cable", "q1")])
        report = validate_splits(train, val, test)
        self.assertEqual(report["protocol"], "category_held_out")
        self.assertEqual(report["total"]["unique_images"], 3)
        self.assertEqual(report["total"]["category_count"], 3)
        self.assertEqual(report["shared_categories"], {})
        self.assertEqual(report["checks"]["mask_pixels"], "not_checked")

    def test_category_overlap_requires_explicit_override(self):
        train = self.write("train.jsonl", [self.record(1)])
        val = self.write("val.jsonl", [self.record(2)])
        with self.assertRaisesRegex(ValueError, "Category overlap.*bottle"):
            validate_splits(train, val)
        report = validate_splits(train, val, allow_shared_categories=True)
        self.assertEqual(report["shared_categories"], {"bottle": ["train", "val"]})
        self.assertEqual(report["protocol"], "shared_categories_allowed")

    def test_image_leakage_is_rejected_even_when_category_sharing_allowed(self):
        train = self.write("train.jsonl", [self.record()])
        val = self.write("val.jsonl", [self.record(image="./image1.png", identifier="q2")])
        with self.assertRaisesRegex(ValueError, "Image leakage.*share resolved image"):
            validate_splits(train, val, allow_shared_categories=True)

    def test_image_cannot_have_two_categories_within_or_across_splits(self):
        train = self.write("train.jsonl", [self.record(), self.record(category="cable", identifier="q2")])
        with self.assertRaisesRegex(ValueError, "conflicting category"):
            validate_splits(train, allow_shared_categories=True)
        train = self.write("train.jsonl", [self.record()])
        val = self.write("val.jsonl", [self.record(category="cable")])
        with self.assertRaisesRegex(ValueError, "conflicting category"):
            validate_splits(train, val)

    def test_requires_category_and_id_and_unique_ids(self):
        for key in ("id", "category"):
            record = self.record()
            del record[key]
            path = self.write("train.jsonl", [record])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_splits(path)
        for invalid_id in (None, "", "   ", True, 1.5):
            path = self.write("train.jsonl", [self.record(identifier=invalid_id)])
            with self.subTest(id=invalid_id), self.assertRaisesRegex(ValueError, "id"):
                validate_splits(path)
        path = self.write("train.jsonl", [self.record(identifier=0)])
        self.assertEqual(validate_splits(path)["total"]["qa_count"], 1)
        path = self.write("train.jsonl", [self.record(identifier=1), self.record(2, identifier="1")])
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            validate_splits(path)

    def test_missing_files_invalid_qa_and_empty_manifest_are_rejected(self):
        for key in ("image", "mask"):
            path = self.write("train.jsonl", [self.record(**{key: "missing.file"})])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Missing {key}"):
                validate_splits(path)
        path = self.write("train.jsonl", [self.record(answer="C")])
        with self.assertRaisesRegex(ValueError, "MCQ answer"):
            validate_splits(path)
        path = self.write("train.jsonl", [])
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_splits(path)

    def test_data_root_matches_dataset_path_resolution(self):
        path = self.write("manifests/train.jsonl", [self.record()])
        with self.assertRaisesRegex(ValueError, "Missing image"):
            validate_splits(path)
        report = validate_splits(path, data_root=self.root)
        self.assertEqual(report["splits"]["train"]["data_root"], str(self.root.resolve()))

    def test_pixel_checks_are_opt_in_and_reject_invalid_raw_scores(self):
        path = self.write("train.jsonl", [self.record()])
        for values, expected in [
            (np.array([[np.nan]], np.float32), "finite"),
            (np.array([[-0.125]], np.float32), "nonnegative"),
            (np.array([[255.125]], np.float32), "exceeds score_scale"),
        ]:
            np.save(self.root / "mask1.npy", values)
            with self.subTest(values=values.tolist()):
                with patch("mask_fusion.preprocessing.load_mask") as reader:
                    report = validate_splits(path)
                    reader.assert_not_called()
                    self.assertEqual(report["checked_mask_count"], 0)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_splits(path, check_pixels=True)
        np.save(self.root / "mask1.npy", np.array([[255.125]], np.float32))
        self.assertEqual(validate_splits(path, score_scale=512, check_pixels=True)["status"], "ok")

    def test_rejects_invalid_score_scale(self):
        path = self.write("train.jsonl", [self.record()])
        for scale in (0, -1, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "score_scale"):
                validate_splits(path, score_scale=scale)

    def test_cli_emits_json_and_nonzero_error_status(self):
        path = self.write("train.jsonl", [self.record()])
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--train", str(path), "--check-pixels"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "ok")
        bad = self.write("bad.jsonl", [self.record(category=None)])
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--train", str(bad)])
        self.assertNotEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
