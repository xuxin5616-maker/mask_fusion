"""Compatibility tests for the original Triad yes/no evaluation JSON schema."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from mask_fusion.data import IMAGE_TOKEN_INDEX, QADataset, build_prompt, normalize_record


QUESTION = "Is there any defect in this image? If yes, say 'yes', otherwise say 'no'. Then describe it."


def triad_record(**changes):
    record = {
        "question_id": "000000000",
        "image": "Traid_eval_data/mvtec/imgs/000000000.png",
        "text": QUESTION,
        "gt": 1,
        "origin_path": "toothbrush/test/defective/006.png",
        "mask": "Traid_eval_data/mvtec/masks_muscv2_classnorm_group_score/000000000.png",
        "class": "toothbrush",
        "musc_mask": "Traid_eval_data/mvtec/masks_musc/000000000.png",
        "muscv2_mask": "Traid_eval_data/mvtec/masks_muscv2/000000000.png",
    }
    record.update(changes)
    return record


class ResizeProcessor:
    size = (14, 14)
    crop_size = {"height": 14, "width": 14}

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        pixels = [np.asarray(image.resize(self.size), dtype=np.float32).copy() for image in images]
        return {"pixel_values": torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2)}


class TriadRecordTests(unittest.TestCase):
    def test_aliases_preserve_ground_truth_mask_prompt_and_input_record(self):
        record = triad_record()
        original = copy.deepcopy(record)
        normalized = normalize_record(record, require_answer=False)
        self.assertEqual(normalized["id"], "000000000")
        self.assertEqual(normalized["question"], QUESTION)
        self.assertEqual(normalized["category"], "toothbrush")
        self.assertEqual(normalized["gt"], 1)
        self.assertEqual(normalized["text"], QUESTION)
        self.assertEqual(normalized["mask"], original["mask"])
        self.assertEqual(normalized["musc_mask"], original["musc_mask"])
        self.assertEqual(normalized["muscv2_mask"], original["muscv2_mask"])
        self.assertNotIn("answer", normalized)
        self.assertIsNone(normalized["options"])
        prompt = build_prompt(record)
        self.assertIn(QUESTION, prompt)
        self.assertEqual(prompt.count("<image>"), 1)
        self.assertNotIn("Answer with the option letter only.", prompt)
        self.assertEqual(record, original)

    def test_binary_gt_does_not_supply_a_training_answer(self):
        for gt in (0, 1):
            with self.subTest(gt=gt), self.assertRaisesRegex(ValueError, "answer must be a nonempty string"):
                normalize_record(triad_record(gt=gt), require_answer=True)
        normalized = normalize_record(triad_record(answer="Yes, the bristles are damaged."))
        self.assertEqual(normalized["answer"], "Yes, the bristles are damaged.")

    def test_explicit_flat_fields_take_priority_over_legacy_aliases(self):
        normalized = normalize_record(
            triad_record(id="explicit-id", question="What is visible?", category="explicit-category"),
            require_answer=False,
        )
        self.assertEqual(normalized["id"], "explicit-id")
        self.assertEqual(normalized["question"], "What is visible?")
        self.assertEqual(normalized["category"], "explicit-category")
        self.assertEqual(normalized["text"], QUESTION)

    def test_conversations_take_priority_over_legacy_text(self):
        normalized = normalize_record(triad_record(conversations=[
            {"from": "human", "value": "<image>\nDescribe the bristles."},
            {"from": "gpt", "value": "The bristles are damaged."},
        ]))
        self.assertEqual(normalized["question"], "Describe the bristles.")
        self.assertEqual(normalized["answer"], "The bristles are damaged.")
        self.assertEqual(normalized["id"], "000000000")

    def test_alias_does_not_hide_empty_questions_or_malformed_conversations(self):
        for question in ("", "   ", None, 5):
            with self.subTest(question=question), self.assertRaises(ValueError):
                normalize_record(triad_record(question=question), require_answer=False)
        for conversations in ([], None, "bad", [{"from": "gpt", "value": "wrong first role"}]):
            with self.subTest(conversations=conversations), self.assertRaises(ValueError):
                normalize_record(triad_record(conversations=conversations), require_answer=False)

    def test_invalid_legacy_text_is_rejected(self):
        for text in ("", "  ", None, 1, [], "an extra <image> placeholder"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_record(triad_record(text=text), require_answer=False)

    def test_dataset_loads_the_actual_declared_png_mask_and_original_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = triad_record()
            original = copy.deepcopy(record)
            for key, value in (("image", 40), ("mask", 71), ("musc_mask", 13), ("muscv2_mask", 199)):
                path = root / record[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (14, 14), (value, value, value)) if key == "image" else Image.new("L", (14, 14), value)
                image.save(path)
            manifest = root / "triad_eval.json"
            manifest.write_text(json.dumps([record]), encoding="utf-8")
            dataset = QADataset(
                manifest, object(), ResizeProcessor(), SimpleNamespace(), training=False, mode="square",
            )
            with patch("mask_fusion.data._image_tokenize", return_value=torch.tensor([7, IMAGE_TOKEN_INDEX, 8])) as tokenize:
                sample = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.records[0]["id"], "000000000")
            self.assertEqual(dataset.records[0]["category"], "toothbrush")
            self.assertEqual(dataset.records[0]["gt"], 1)
            self.assertEqual(dataset.records[0]["question"], QUESTION)
            self.assertEqual(dataset.records[0]["mask"], str((root / record["mask"]).resolve()))
            self.assertIn(QUESTION, tokenize.call_args.args[0])
            self.assertNotIn("labels", sample)
            self.assertEqual(sample["image_size"], (14, 14))
            self.assertTrue(torch.all(sample["images"] == 40))
            self.assertTrue(torch.all(sample["masks"] == 71))
            self.assertTrue(sample["mask_valid"].all())
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), [original])
            self.assertEqual(record, original)
            with self.assertRaisesRegex(ValueError, "answer must be a nonempty string"):
                QADataset(manifest, object(), ResizeProcessor(), SimpleNamespace(), training=True)


if __name__ == "__main__":
    unittest.main()
