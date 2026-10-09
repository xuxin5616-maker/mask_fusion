"""CPU tests for prompt supervision, schema checks, and training bookkeeping."""
import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from mask_fusion.data import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, QADataset, build_prompt, collate_samples,
    normalize_record, read_records, tokenize_record,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<|im_end|>"
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        # Preserve the end marker as one token; ordinary characters are separate.
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(self.eos_token_id)
                text = text[len(self.eos_token) :]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return SimpleNamespace(input_ids=ids)


def fake_image_tokenize(prompt, tokenizer):
    pieces = prompt.split("<image>")
    ids = []
    for index, piece in enumerate(pieces):
        if index:
            ids.append(IMAGE_TOKEN_INDEX)
        ids.extend(tokenizer(piece, add_special_tokens=False).input_ids)
    return torch.tensor(ids, dtype=torch.long)


def example(**changes):
    record = {"id": "q1", "question": "Is the product defective?", "options": ["Normal", "Defective"], "answer": "B"}
    record.update(changes)
    return record


class RecordTests(unittest.TestCase):
    def test_mcq_text_and_letter_answers_have_one_canonical_form(self):
        self.assertEqual(normalize_record(example(answer="defective"))["answer"], "B")
        self.assertEqual(normalize_record(example(answer="b"))["answer"], "B")
        mapped = normalize_record(example(options={"B": "Defective", "A": "Normal"}))
        self.assertEqual(list(mapped["options"]), ["A", "B"])

    def test_invalid_and_ambiguous_choices_are_rejected(self):
        cases = [
            {"answer": "C"}, {"answer": "possibly defective"},
            {"options": ["Normal", " normal "]}, {"options": {"A": "Normal", "C": "Defective"}},
            {"options": ["B", "A"], "answer": "A"}, {"options": ["Only one"]},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                normalize_record(example(**case))

    def test_ordinary_qa_and_eight_options_are_supported(self):
        qa = normalize_record({"question": "Describe the product.", "answer": "A clean bottle."})
        self.assertIsNone(qa["options"])
        self.assertEqual(qa["answer"], "A clean bottle.")
        eight = normalize_record(example(options=[f"Choice {i}" for i in range(8)], answer="H"))
        self.assertEqual(eight["answer"], "H")

    def test_reserved_tokens_cannot_create_extra_images_or_turns(self):
        for field in ("question", "answer"):
            with self.assertRaises(ValueError):
                normalize_record(example(**{field: "text <image> text"}))

    def test_inference_prompt_does_not_include_the_answer(self):
        first = build_prompt(example(answer="A"))
        second = build_prompt(example(answer="B"))
        self.assertEqual(first, second)
        self.assertEqual(first.count("<image>"), 1)
        self.assertTrue(first.endswith("<|im_start|>assistant\n"))
        self.assertIn("A. Normal\nB. Defective", first)
        self.assertIn("<im_start><image><im_end>", build_prompt(example(), {"mm_use_im_start_end": True}))

    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_only_answer_and_end_marker_receive_labels(self, _):
        tokenizer = FakeTokenizer()
        inference = tokenize_record(example(), tokenizer, None, training=False)
        training = tokenize_record(example(answer="Defective"), tokenizer, None, training=True)
        prefix_length = inference["input_ids"].numel()
        self.assertNotIn("labels", inference)
        self.assertTrue(torch.equal(training["input_ids"][:prefix_length], inference["input_ids"]))
        self.assertTrue(torch.all(training["labels"][:prefix_length] == IGNORE_INDEX))
        self.assertEqual(training["labels"][prefix_length:].tolist(), [ord("B") + 10, tokenizer.eos_token_id])
        self.assertEqual(training["labels"][training["input_ids"] == IMAGE_TOKEN_INDEX].tolist(), [IGNORE_INDEX])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        Image.new("RGB", (6, 4)).save(self.root / "image.png")
        np.save(self.root / "mask.npy", np.arange(24, dtype=np.float32).reshape(4, 6))
        self.record = example(image="image.png", mask="mask.npy")
        self.data = self.root / "train.jsonl"
        self.data.write_text(json.dumps(self.record) + "\n", encoding="utf-8")

    def dataset(self, **kwargs):
        return QADataset(self.data, FakeTokenizer(), object(), SimpleNamespace(), **kwargs)

    def test_json_array_and_jsonl_readers(self):
        self.assertEqual(read_records(self.data), [self.record])
        array = self.root / "train.json"
        array.write_text(json.dumps([self.record]), encoding="utf-8")
        self.assertEqual(read_records(array), [self.record])

    def test_paths_are_relative_to_json_or_explicit_root(self):
        dataset = self.dataset()
        self.assertEqual(dataset.records[0]["image"], str((self.root / "image.png").resolve()))
        folder = self.root / "annotations"
        folder.mkdir()
        self.data = folder / "train.jsonl"
        self.data.write_text(json.dumps(self.record), encoding="utf-8")
        self.assertEqual(self.dataset(data_root=self.root).records[0]["mask"], str((self.root / "mask.npy").resolve()))

    def test_missing_mask_and_duplicate_record_id_fail_early(self):
        self.data.write_text(json.dumps({**self.record, "mask": "missing.npy"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing mask"):
            self.dataset()
        self.data.write_text("\n".join([json.dumps(self.record)] * 2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate record id"):
            self.dataset()

    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_text_is_not_silently_truncated(self, _):
        with self.assertRaisesRegex(ValueError, "never silently truncated"):
            self.dataset(max_text_length=1)[0]

    @patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize)
    def test_preprocessing_receives_original_mask_and_reproducible_epoch_seed(self, _):
        prepared = {
            "images": torch.zeros(2, 3, 8, 8), "masks": torch.ones(2, 1, 8, 8),
            "mask_valid": torch.ones(2, 1, 8, 8, dtype=torch.bool), "image_size": (6, 4),
        }
        dataset = self.dataset(seed=7)
        dataset.set_epoch(2)
        with patch("mask_fusion.preprocessing.prepare_image_mask", return_value=prepared) as prepare:
            sample = dataset[0]
        self.assertEqual(prepare.call_args.kwargs["seed"], 9)
        self.assertEqual(prepare.call_args.kwargs["mode"], "anyres_max_9")
        np.testing.assert_array_equal(prepare.call_args.args[1], np.arange(24, dtype=np.float32).reshape(4, 6))
        self.assertEqual(sample["images"].shape, (2, 3, 8, 8))
        self.assertIs(sample["mask_valid"], prepared["mask_valid"])


class CollationTests(unittest.TestCase):
    @staticmethod
    def sample(ids, views):
        return {
            "input_ids": torch.tensor(ids), "labels": torch.tensor([-100] + ids[1:]),
            "images": torch.zeros(views, 3, 8, 8), "masks": torch.zeros(views, 1, 8, 8),
            "mask_valid": torch.ones(views, 1, 8, 8, dtype=torch.bool), "image_size": (12, 8),
        }

    def test_right_padding_preserves_valid_eos_tokens_and_variable_views(self):
        batch = collate_samples([self.sample([7, 2, 9], 2), self.sample([8, 2], 5)], pad_token_id=2)
        self.assertEqual(batch["input_ids"].tolist(), [[7, 2, 9], [8, 2, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 9], [-100, 2, -100]])
        self.assertEqual([x.shape[0] for x in batch["images"]], [2, 5])
        self.assertEqual(batch["modalities"], ["image", "image"])

    def test_mixed_inference_training_batch_is_rejected(self):
        sample = self.sample([1, 2], 1)
        inference = {key: value for key, value in sample.items() if key != "labels"}
        with self.assertRaises(ValueError):
            collate_samples([sample, inference], 0)


class TinyTrainingModel(torch.nn.Module):
    """Frozen decoder still propagates gradients into both trainable modules."""
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.mask_fusion = torch.nn.Linear(1, 2)
        self.core.mm_projector = torch.nn.Linear(2, 1)
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        self.decoder.weight.data.fill_(1.5)
        self.decoder.requires_grad_(False)
        self.vision = torch.nn.Identity()
        self.config = SimpleNamespace(use_cache=True)

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def get_vision_tower(self):
        return self.vision

    def forward(self, input_ids, **kwargs):
        value = input_ids[:, :1].float() / 10
        if self.training:
            # Exercise RNG restoration in the mid-epoch resume check.
            value = value + torch.rand_like(value) * 0.1
        prediction = self.decoder(self.core.mm_projector(self.core.mask_fusion(value)))
        return SimpleNamespace(loss=(prediction - 0.5).square().mean())


class TinyTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.records = [{"image": f"image_{i}.png"} for i in range(5)]

    def __len__(self):
        return 5

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return CollationTests.sample([index + 1, 2], 1)


class FakeRecoveringScaler:
    """CPU stand-in for GradScaler: one real inf gradient, then finite updates."""
    def __init__(self, **kwargs):
        self.loss_scale = 8.0
        self.inject_overflow = True
        self.found_inf = False

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.loss_scale

    def scale(self, loss):
        return loss * self.loss_scale

    def unscale_(self, optimizer):
        gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None]
        for gradient in gradients:
            gradient.div_(self.loss_scale)
        if self.inject_overflow:
            gradients[0].flatten()[0] = float("inf")
            self.inject_overflow = False
        self.found_inf = any(not torch.isfinite(gradient).all() for gradient in gradients)

    def step(self, optimizer):
        if not self.found_inf:
            optimizer.step()

    def update(self):
        if self.found_inf:
            self.loss_scale /= 2

    def state_dict(self):
        return {"loss_scale": self.loss_scale, "inject_overflow": self.inject_overflow}

    def load_state_dict(self, state):
        self.loss_scale = state["loss_scale"]
        self.inject_overflow = state["inject_overflow"]


class TrainingLoopTests(unittest.TestCase):
    def test_partial_accumulation_and_mid_epoch_resume_match_uninterrupted_run(self):
        self._assert_resumed_run()

    def test_fp16_overflow_skips_update_and_checkpoint_resumes_with_reduced_scale(self):
        self._assert_resumed_run(FakeRecoveringScaler)

    def _assert_resumed_run(self, scaler_class=None):
        from mask_fusion import train

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
            data = root / "data.jsonl"
            data.write_text("{}\n", encoding="utf-8")
            common = [
                "--base-model", str(root / "base"), "--data", str(data),
                "--epochs", "2", "--batch-size", "2", "--grad-accum", "2",
                "--device", "cpu", "--dtype", "float32", "--seed", "19",
                "--lr", "0.01", "--projector-lr", "0.02", "--save-every-steps", "1",
            ]
            initial = load_model("unused")[0].decoder.weight.detach().clone()
            log = io.StringIO()
            with patch("mask_fusion.train.QADataset", TinyTrainingDataset), \
                 patch("mask_fusion.model.load_model", side_effect=load_model), \
                 patch("mask_fusion.model.configure_trainable"), \
                 patch("mask_fusion.model.save_adapter", side_effect=save_adapter), \
                 contextlib.redirect_stdout(log), contextlib.ExitStack() as stack:
                if scaler_class is not None:
                    stack.enter_context(patch("mask_fusion.train.torch.cuda.amp.GradScaler", scaler_class))
                train.main(common + ["--output", str(root / "full")])
                resume_step = 0 if scaler_class is not None else 1
                train.main(common + ["--output", str(root / "resumed"), "--resume", str(root / "full" / f"checkpoint-step-{resume_step}")])
            full = torch.load(root / "full" / "toy_adapter.pt", weights_only=True)
            resumed = torch.load(root / "resumed" / "toy_adapter.pt", weights_only=True)
            for key in full:
                self.assertTrue(torch.equal(full[key], resumed[key]), key)
            self.assertTrue(torch.equal(full["decoder.weight"], initial))
            state = train._load_training_state(root / "resumed" / "training_state.pt")
            # Five records, batch 2 -> three microbatches, accumulation 2 -> two updates per epoch.
            expected_steps = 3 if scaler_class is not None else 4
            self.assertEqual(state["global_step"], expected_steps)
            self.assertEqual(state["scheduler"]["last_epoch"], expected_steps)
            self.assertEqual((state["epoch"], state["next_batch"]), (2, 0))
            if scaler_class is not None:
                skipped = [json.loads(line) for line in log.getvalue().splitlines() if '"event": "optimizer_step_skipped"' in line]
                self.assertEqual(len(skipped), 1)
                self.assertEqual(skipped[0]["loss_scale_after"], 4.0)
                self.assertEqual(skipped[0]["step"], 0)
                overflow_state = train._load_training_state(root / "full" / "checkpoint-step-0" / "training_state.pt")
                self.assertEqual(overflow_state["next_batch"], 2)
                self.assertEqual(overflow_state["consecutive_overflows"], 1)
                self.assertEqual(state["consecutive_overflows"], 0)


if __name__ == "__main__":
    unittest.main()
