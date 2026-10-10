"""Exercise Triad manifests through the inference CLI without model downloads."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from mask_fusion.infer import main
from test_data import FakeTokenizer, fake_image_tokenize


class DecodeTokenizer(FakeTokenizer):
    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(token) - 10) for token in tokens if int(token) > 9)


class RecordingModel(torch.nn.Module):
    def __init__(self, replies, tokenizer):
        super().__init__()
        self.config = SimpleNamespace()
        self.replies = iter(replies)
        self.tokenizer = tokenizer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([self.tokenizer(next(self.replies)).input_ids])


class AnomalyInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        image_path = self.root / "Traid_eval_data/mvtec/imgs/000000000.png"
        mask_path = self.root / "Traid_eval_data/mvtec/masks_muscv2_classnorm_group_score/000000000.png"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        Image.new("RGB", (6, 4)).save(image_path)
        Image.fromarray(np.full((4, 6), 91, dtype=np.uint8)).save(mask_path)
        self.record = {
            "question_id": "000000000", "image": image_path.relative_to(self.root).as_posix(),
            "text": "Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it.",
            "gt": 1, "origin_path": "toothbrush/test/defective/006.png",
            "mask": mask_path.relative_to(self.root).as_posix(), "class": "toothbrush",
            "musc_scores": 0.5304691195487976,
            "musc_mask": "unused_musc.png", "muscv2_mask": "unused_muscv2.png",
        }
        self.manifest = self.root / "questions.jsonl"
        self.output = self.root / "results/predictions.jsonl"
        self.tokenizer = DecodeTokenizer()

    def run_infer(self, records, replies, *, task="anomaly"):
        self.manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        self.model = RecordingModel(replies, self.tokenizer)
        prepared = {
            "images": torch.zeros(1, 3, 4, 6), "masks": torch.ones(1, 1, 4, 6),
            "mask_valid": torch.ones(1, 1, 4, 6, dtype=torch.bool), "image_size": (6, 4),
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("mask_fusion.infer.read_adapter_config", return_value={"base_model": "base", "image_mode": "square"}))
            stack.enter_context(patch("mask_fusion.infer.load_model", return_value=(self.model, self.tokenizer, object())))
            stack.enter_context(patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize))
            stack.enter_context(patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = [
                "--adapter", "adapter", "--data", str(self.manifest), "--data-root", str(self.root),
                "--output", str(self.output), "--device", "cpu", "--dtype", "float32",
            ]
            if task is not None:
                args.extend(["--eval-task", task])
            main(args)
        predictions = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        metrics = json.loads(self.output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        return predictions, metrics

    def test_triad_yes_no_scoring_preserves_prompt_and_counts_unparsed_and_skipped(self):
        records = [
            self.record,
            {**self.record, "question_id": "000000001", "gt": 0},
            {**self.record, "question_id": "000000002"},
            {**self.record, "question_id": "missing", "image": "missing.png"},
        ]
        with self.assertWarnsRegex(UserWarning, "skipped 1 records"):
            predictions, metrics = self.run_infer(records, ["Yes, the bristles are damaged.", "No. The object looks intact.", "I cannot decide."])
        self.assertEqual([row["id"] for row in predictions], ["000000000", "000000001", "000000002"])
        self.assertEqual([row["anomaly_prediction"] for row in predictions], [1, 0, None])
        self.assertEqual([row["correct"] for row in predictions], [True, True, False])
        for index, row in enumerate(predictions):
            self.assertEqual(row["question"], self.record["text"])
            self.assertEqual(row["category"], "toothbrush")
            self.assertEqual(row["mask"], str((self.root / self.record["mask"]).resolve()))
            self.assertEqual(row["source_index"], index)
            self.assertIsNone(row["options"])
            self.assertIsNone(row["answer"])
        # The two opposite ground-truth labels must not change model inputs.
        self.assertTrue(torch.equal(self.model.calls[0]["inputs"], self.model.calls[1]["inputs"]))
        self.assertNotIn("labels", self.model.calls[0])
        self.assertNotIn("gt", self.model.calls[0])
        self.assertFalse(self.model.calls[0]["do_sample"])
        self.assertEqual(metrics["eval_task"], "anomaly")
        self.assertEqual(metrics["scored_anomaly"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["unparseable_anomaly"], 1)
        self.assertEqual(metrics["input_records"], 4)
        self.assertEqual(metrics["skipped_missing_images"], 1)
        self.assertEqual(len(metrics["skipped_records"]), 1)
        self.assertEqual(metrics["skipped_records"][0]["id"], "missing")

    def test_missing_gt_fails_before_generating_or_writing_predictions(self):
        record = {key: value for key, value in self.record.items() if key != "gt"}
        with self.assertRaisesRegex(ValueError, "record 0:.*gt"):
            self.run_infer([record], [])
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.output.exists())

    def test_anomaly_mode_does_not_guess_letter_to_label_mapping(self):
        record = {**self.record, "options": ["Normal", "Abnormal"], "answer": "B"}
        with self.assertRaisesRegex(ValueError, "use --eval-task qa"):
            self.run_infer([record], [])
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.output.exists())

    def test_default_mcq_mode_still_scores_option_letters(self):
        record = {**self.record, "question": "Is this image defective?", "options": ["Normal", "Abnormal"], "answer": "B"}
        predictions, metrics = self.run_infer([record], ["B"], task=None)
        self.assertEqual(predictions[0]["choice"], "B")
        self.assertTrue(predictions[0]["correct"])
        self.assertNotIn("anomaly_prediction", predictions[0])
        self.assertEqual(metrics["eval_task"], "qa")
        self.assertEqual(metrics["scored_mcq"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
