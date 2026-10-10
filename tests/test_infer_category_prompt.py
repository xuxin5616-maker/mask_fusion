"""Verify category MCQ prompts at the inference/model boundary, without downloads."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from mask_fusion.data import IMAGE_TOKEN_INDEX
from mask_fusion.infer import main
from test_data import fake_image_tokenize
from test_infer_anomaly import DecodeTokenizer, RecordingModel


class CategoryPromptInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        Image.new("RGB", (6, 4), color=(20, 30, 40)).save(self.root / "image.png")
        Image.new("L", (6, 4), color=91).save(self.root / "mask.png")
        self.manifest = self.root / "questions.jsonl"
        self.output = self.root / "results" / "predictions.jsonl"
        self.tokenizer = DecodeTokenizer()
        self.record = {
            "question_id": "sample-0", "image": "image.png", "mask": "mask.png",
            "text": "LEGACY QUESTION: describe whether the product is normal.",
            "answer": "OLD_REFERENCE_ANSWER_DO_NOT_SEND", "class": "metal_nut", "gt": 1,
        }

    def arguments(self):
        return [
            "--adapter", "offline-adapter", "--data", str(self.manifest),
            "--data-root", str(self.root), "--output", str(self.output),
            "--device", "cpu", "--dtype", "float32",
        ]

    def run_infer(self, records, replies):
        self.manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        self.model = RecordingModel(replies, self.tokenizer)
        prepared = {
            "images": torch.zeros(1, 3, 4, 6), "masks": torch.full((1, 1, 4, 6), 91.0),
            "mask_valid": torch.ones(1, 1, 4, 6, dtype=torch.bool), "image_size": (6, 4),
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("mask_fusion.infer.read_adapter_config", return_value={
                "base_model": "offline-base", "image_mode": "square",
            }))
            stack.enter_context(patch("mask_fusion.infer.load_model", return_value=(
                self.model, self.tokenizer, object(),
            )))
            stack.enter_context(patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize))
            stack.enter_context(patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main(self.arguments() + ["--eval-task", "anomaly", "--anomaly-prompt", "category-mcq"])
        predictions = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        metrics = json.loads(self.output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        return predictions, metrics

    def actual_prompt(self, index):
        # Reverse the fake character tokenizer, preserving ChatML/image markers.
        pieces = []
        for token in self.model.calls[index]["inputs"][0].tolist():
            if token == IMAGE_TOKEN_INDEX:
                pieces.append("<image>")
            elif token == self.tokenizer.eos_token_id:
                pieces.append("<|im_end|>")
            elif token > 9:
                pieces.append(chr(token - 10))
            else:
                self.fail(f"Unexpected input token: {token}")
        return "".join(pieces)

    @staticmethod
    def expected_template(category):
        # Literal user-requested wording; intentionally independent of constants
        # imported from the implementation so accidental text changes fail.
        return (
            f"Are there any defects on the {category} in this image?\n"
            "A. Yes\n"
            "B. No\n"
            "Answer with the option's letter from the given choices directly."
        )

    def test_exact_model_input_label_isolation_and_ab_scoring(self):
        records = [self.record, {**self.record, "question_id": "sample-1", "gt": 0},
                   {**self.record, "question_id": "sample-2", "gt": 1}]
        predictions, metrics = self.run_infer(records, ["A", "B", "I cannot decide."])
        template = self.expected_template("metal nut")
        expected_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{template}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for index in range(3):
            actual = self.actual_prompt(index)
            self.assertEqual(actual, expected_prompt)
            self.assertEqual(actual.count(template), 1)
            self.assertNotIn(self.record["text"], actual)
            self.assertNotIn(self.record["answer"], actual)
            self.assertNotIn("Answer with the option letter only.", actual)
            self.assertNotIn("labels", self.model.calls[index])
            self.assertNotIn("gt", self.model.calls[index])
            self.assertFalse(self.model.calls[index]["do_sample"])
        torch.testing.assert_close(self.model.calls[0]["inputs"], self.model.calls[1]["inputs"], rtol=0, atol=0)
        self.assertEqual([row["choice"] for row in predictions], ["A", "B", None])
        self.assertEqual([row["anomaly_prediction"] for row in predictions], [1, 0, None])
        self.assertEqual([row["answer"] for row in predictions], ["A", "B", "A"])
        self.assertEqual([row["correct"] for row in predictions], [True, True, False])
        for row in predictions:
            self.assertEqual(row["original_question"], self.record["text"])
            self.assertEqual(row["options"], {"A": "Yes", "B": "No"})
            self.assertEqual(row["mcq_instruction"], template.splitlines()[-1])
            self.assertEqual(row["category"], "metal_nut")
        self.assertEqual(metrics["eval_task"], "anomaly")
        self.assertEqual(metrics["anomaly_prompt"], "category-mcq")
        self.assertEqual(metrics["scored_anomaly"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["unparseable_anomaly"], 1)
        self.assertEqual(metrics["by_category"]["metal_nut"]["count"], 3)
        self.assertAlmostEqual(metrics["positive_recall"], 0.5)
        self.assertAlmostEqual(metrics["negative_recall"], 1.0)

    def test_category_precedence_and_fallback_only_change_prompt_spelling(self):
        without_class = {key: value for key, value in self.record.items() if key != "class"}
        records = [
            {**self.record, "category": "cable_connector", "class": "ignored_class"},
            {**self.record, "question_id": "class-fallback", "class": "wood_block"},
            {**without_class, "question_id": "subdataset-fallback", "sub_dataset": "capsule_case"},
            {**self.record, "question_id": "null-category", "category": None, "class": "screw_head"},
        ]
        predictions, metrics = self.run_infer(records, ["A"] * len(records))
        categories = ["cable_connector", "wood_block", "capsule_case", "screw_head"]
        self.assertEqual([row["category"] for row in predictions], categories)
        self.assertEqual(set(metrics["by_category"]), set(categories))
        for index, category in enumerate(categories):
            self.assertIn(self.expected_template(category.replace("_", " ")), self.actual_prompt(index))
            self.assertNotIn(category, self.actual_prompt(index))
            self.assertEqual(metrics["by_category"][category]["count"], 1)
        self.assertNotIn("ignored class", self.actual_prompt(0))

    def test_missing_category_rejects_entire_batch_before_generation_or_output(self):
        missing = {key: value for key, value in self.record.items() if key != "class"}
        with self.assertRaisesRegex(ValueError, "record 1:.*requires category"):
            self.run_infer([self.record, missing], [])
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".metrics.json").exists())

    def test_category_without_readable_characters_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty readable category"):
            self.run_infer([{**self.record, "class": "___"}], [])
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.output.exists())

    def test_category_prompt_requires_anomaly_task_before_loading(self):
        with patch("mask_fusion.infer.read_adapter_config") as read_config, \
                patch("mask_fusion.infer.load_model") as load_model, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                main(self.arguments() + ["--eval-task", "qa", "--anomaly-prompt", "category-mcq"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("requires --eval-task anomaly", stderr.getvalue())
        read_config.assert_not_called()
        load_model.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
