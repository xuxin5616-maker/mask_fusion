import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from mask_fusion.infer import extract_choice, summarize_predictions, main


class PredictionTests(unittest.TestCase):
    def test_default_cuda_does_not_pass_an_unindexed_device_to_set_device(self):
        metadata = {"base_model": "base", "image_mode": "square"}
        with tempfile.TemporaryDirectory() as folder, \
                patch("mask_fusion.infer.read_adapter_config", return_value=metadata), \
                patch("mask_fusion.infer._resolve_dtype", return_value=("float32", torch.float32)), \
                patch("mask_fusion.infer.load_model", side_effect=RuntimeError("load sentinel")), \
                patch("torch.cuda.set_device") as set_device:
            with self.assertRaisesRegex(RuntimeError, "load sentinel"):
                main(["--adapter", "adapter", "--data", "data.jsonl", "--output", str(Path(folder) / "out.jsonl")])
            set_device.assert_not_called()

    def test_choice_parser_does_not_guess_letters_from_prose(self):
        choices = {"A": "normal", "B": "defective"}
        for text in ("B", "b", "(B)", "B.", "Defective"):
            self.assertEqual(extract_choice(text, choices), "B")
        for text in ("Based on the mask, ...", "A or B", "C", ""):
            self.assertIsNone(extract_choice(text, choices))

    def test_macro_category_accuracy_and_unparsed_answers(self):
        predictions = [
            {"category": "small", "correct": False, "options": {"A": "yes"}, "choice": None},
            {"category": "large", "correct": True, "options": {"A": "yes"}, "choice": "A"},
            {"category": "large", "correct": True, "options": {"A": "yes"}, "choice": "A"},
            {"category": "large", "correct": None, "options": None, "choice": None},
        ]
        result = summarize_predictions(predictions)
        self.assertEqual(result["scored_mcq"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["macro_category_accuracy"], 0.5)
        self.assertEqual(result["unparseable_mcq"], 1)


if __name__ == "__main__":
    unittest.main()
