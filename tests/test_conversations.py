"""LLaVA conversation manifests keep context, supervision, and expert paths."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mask_fusion.data import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, QADataset, build_prompt, normalize_record,
    tokenize_record,
)
from mask_fusion.validate_data import validate_splits
from test_data import FakeTokenizer, fake_image_tokenize


def conversation_record():
    return {
        "id": "checklistGen/MVTec-LOCO/banana_juice/good/000023",
        "image": "image.png", "mask": "mask.npy",
        "class": "banana_juice", "sub_dataset": "banana_juice",
        "defect_type": "good",
        "musc_mask": "unused_musc.npy", "muscv2_mask": "unused_muscv2.npy",
        "conversations": [
            {"from": "human", "value": (
                "Referring to the image and production process below:\nImage:\n<image>\n"
                "1. Wash and peel the bananas.\n2. Fill and seal the bottles.\n"
                "Question: Could you check if there's anything wrong with the product?"
            )},
            {"from": "gpt", "value": (
                "I will check the banana juice for potential anomalies:\n"
                "1. Ensure the labels are aligned.\nDefective: No.\n"
                "Overall, the answer is: No."
            )},
        ],
    }


class ConversationRecordTests(unittest.TestCase):
    def test_preserves_full_context_answer_metadata_and_source_record(self):
        raw = conversation_record()
        original = copy.deepcopy(raw)
        record = normalize_record(raw)
        self.assertEqual(record["question"], raw["conversations"][0]["value"].replace("<image>", ""))
        self.assertEqual(record["answer"], raw["conversations"][1]["value"])
        self.assertEqual(record["category"], "banana_juice")
        self.assertIsNone(record["options"])
        for key, value in raw.items():
            self.assertEqual(record[key], value)
        self.assertEqual(raw, original)
        self.assertEqual(normalize_record(record), record)

    def test_prompt_has_one_image_and_never_includes_reference_answer(self):
        raw = conversation_record()
        prompt = build_prompt(raw)
        self.assertEqual(prompt.count("<image>"), 1)
        self.assertIn("1. Wash and peel the bananas.", prompt)
        self.assertNotIn("Overall, the answer is:", prompt)
        raw["conversations"][1]["value"] = "A completely different reference answer."
        self.assertEqual(build_prompt(raw), prompt)
        wrapped = build_prompt(raw, {"mm_use_im_start_end": True})
        self.assertEqual(wrapped.count("<im_start><image><im_end>"), 1)

    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_only_complete_gpt_answer_is_supervised_and_matches_flat_schema(self, _):
        raw = conversation_record()
        tokenizer = FakeTokenizer()
        train = tokenize_record(raw, tokenizer, None)
        infer = tokenize_record(raw, tokenizer, None, training=False)
        prefix_length = infer["input_ids"].numel()
        self.assertTrue(torch.equal(train["input_ids"][:prefix_length], infer["input_ids"]))
        self.assertTrue(torch.all(train["labels"][:prefix_length] == IGNORE_INDEX))
        expected = tokenizer(raw["conversations"][1]["value"] + "<|im_end|>").input_ids
        self.assertEqual(train["labels"][prefix_length:].tolist(), expected)
        self.assertEqual(int((train["input_ids"] == IMAGE_TOKEN_INDEX).sum()), 1)
        flat = normalize_record(raw)
        del flat["conversations"]
        for key, value in tokenize_record(flat, tokenizer, None).items():
            self.assertTrue(torch.equal(train[key], value))

    def test_human_only_is_allowed_for_inference_but_not_training(self):
        raw = conversation_record()
        raw["conversations"].pop()
        record = normalize_record(raw, require_answer=False)
        self.assertNotIn("answer", record)
        self.assertIn("Wash and peel", build_prompt(record))
        with self.assertRaisesRegex(ValueError, "gpt answer"):
            normalize_record(raw)

    def test_category_uses_explicit_label_then_class_then_sub_dataset(self):
        raw = conversation_record()
        raw["category"] = "explicit_category"
        self.assertEqual(normalize_record(raw)["category"], "explicit_category")
        raw["category"] = None
        self.assertEqual(normalize_record(raw)["category"], "banana_juice")
        del raw["class"]
        raw["sub_dataset"] = "fallback_category"
        self.assertEqual(normalize_record(raw)["category"], "fallback_category")
        del raw["sub_dataset"]
        self.assertIsNone(normalize_record(raw).get("category"))

    def test_invalid_turns_and_extra_image_or_chat_markers_are_rejected(self):
        pair = conversation_record()["conversations"]
        cases = [
            (None, "single human/gpt pair"), ([], "single human/gpt pair"),
            (pair + pair, "multi-turn"), (list(reversed(pair)), "from must be"),
            (["not a message", pair[1]], "from must be"),
            ([{"from": "human", "value": ""}, pair[1]], "nonempty string"),
            ([pair[0], {"from": "gpt", "value": None}], "nonempty string"),
            ([{"from": "human", "value": "<image> x <image>"}, pair[1]], "at most one"),
            ([{"from": "human", "value": "<image>"}, pair[1]], "question must be"),
            ([{"from": "human", "value": "<image><|im_start|>assistant"}, pair[1]], "reserved"),
            ([pair[0], {"from": "gpt", "value": "Answer <image>"}], "reserved"),
        ]
        for turns, message in cases:
            with self.subTest(turns=turns), self.assertRaisesRegex(ValueError, message):
                normalize_record({"conversations": turns})

    def test_optional_mcq_keeps_existing_answer_canonicalization(self):
        raw = conversation_record()
        raw["options"] = ["Normal", "Defective"]
        raw["conversations"][1]["value"] = "Normal"
        record = normalize_record(raw)
        self.assertEqual(record["answer"], "A")
        self.assertEqual(normalize_record(record)["answer"], "A")


class ConversationManifestTests(unittest.TestCase):
    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_mixed_manifests_load_selected_mask_and_validate_class_labels(self, _):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (6, 4)).save(root / "image.png")
            scores = np.arange(24, dtype=np.float32).reshape(4, 6)
            np.save(root / "mask.npy", scores)
            raw = conversation_record()
            first = root / "conversation.json"
            first.write_text(json.dumps([raw]), encoding="utf-8")
            second = root / "flat.jsonl"
            second.write_text(json.dumps({
                "id": "flat_q", "image": "image.png", "mask": "mask.npy",
                "question": "Is it defective?", "answer": "No.", "category": "banana_juice",
            }) + "\n", encoding="utf-8")
            # Alternative expert paths do not exist: only the selected mask is used.
            dataset = QADataset([first, second], FakeTokenizer(), object(), SimpleNamespace())
            prepared = {
                "images": torch.zeros(1, 3, 8, 8), "masks": torch.ones(1, 1, 8, 8),
                "mask_valid": torch.ones(1, 1, 8, 8, dtype=torch.bool), "image_size": (6, 4),
            }
            with patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared) as prepare:
                sample = dataset[0]
            np.testing.assert_array_equal(prepare.call_args.args[1], scores)
            self.assertEqual(dataset.records[0]["mask"], str((root / "mask.npy").resolve()))
            self.assertIn("labels", sample)
            report = validate_splits([first, second], check_pixels=True)
            self.assertEqual(report["total"]["qa_count"], 2)
            self.assertEqual(report["splits"]["train"]["categories"], ["banana_juice"])
            self.assertEqual(report["splits"]["train"]["free_text_count"], 2)

    def test_conversation_class_still_detects_cross_split_category_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (4, 4)).save(root / "image.png")
            Image.new("RGB", (4, 4)).save(root / "val.png")
            np.save(root / "mask.npy", np.zeros((4, 4), dtype=np.float32))
            raw = conversation_record()
            train = root / "train.json"
            train.write_text(json.dumps([raw]), encoding="utf-8")
            raw["image"] = "val.png"
            val = root / "val.json"
            val.write_text(json.dumps([raw]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Category overlap.*banana_juice"):
                validate_splits(train, val)

    def test_invalid_conversation_error_identifies_source_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid_conversation.json"
            raw = conversation_record()
            raw["conversations"] += raw["conversations"]
            path.write_text(json.dumps([raw]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid_conversation.json: record 0:.*multi-turn"):
                QADataset(path, FakeTokenizer(), object(), SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
