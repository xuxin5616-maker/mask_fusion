"""Exact prompt contracts for an optional per-record MCQ instruction."""
import copy
import unittest

from mask_fusion.data import build_prompt, normalize_record


def record(**changes):
    value = {
        "question": "Is the toothbrush anomalous?",
        "options": {"A": "Anomalous", "B": "Normal"},
        "answer": "A",
        "gt": 1,
    }
    value.update(changes)
    return value


class MCQInstructionTests(unittest.TestCase):
    def test_default_prompt_is_byte_for_byte_unchanged(self):
        expected = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<image>\nIs the toothbrush anomalous?\n"
            "A. Anomalous\nB. Normal\nAnswer with the option letter only.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(build_prompt(record()), expected)

    def test_custom_instruction_appears_once_and_replaces_only_default_instruction(self):
        instruction = "Return exactly one letter: A or B. Do not provide any explanation."
        source = record(mcq_instruction=instruction)
        original = copy.deepcopy(source)
        expected = build_prompt(record()).replace("Answer with the option letter only.", instruction)
        prompt = build_prompt(source)
        self.assertEqual(prompt, expected)
        self.assertEqual(prompt.count(instruction), 1)
        self.assertNotIn("Answer with the option letter only.", prompt)
        self.assertEqual(source, original)

    def test_answer_and_ground_truth_do_not_change_custom_prompt(self):
        instruction = "Reply with the letter A or B only."
        prompts = {
            build_prompt(record(answer=answer, gt=gt, mcq_instruction=instruction))
            for answer in ("A", "B", "Anomalous", "Normal")
            for gt in (0, 1)
        }
        self.assertEqual(len(prompts), 1)

    def test_instruction_uses_nonempty_text_and_reserved_token_validation(self):
        invalid = [None, "", " \n ", 12, [], {}]
        invalid += [f"Return {token} here." for token in ("<image>", "<im_start>", "<im_end>", "<|im_start|>", "<|im_end|>")]
        for instruction in invalid:
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, "mcq_instruction"):
                normalize_record(record(mcq_instruction=instruction))

    def test_instruction_is_ignored_and_not_validated_for_non_mcq_records(self):
        source = {"question": "Describe the toothbrush.", "answer": "It is normal."}
        baseline = build_prompt(source)
        for unused in (None, "", [], "An unused <image> marker"):
            with self.subTest(unused=unused):
                self.assertEqual(build_prompt({**source, "mcq_instruction": unused}), baseline)


if __name__ == "__main__":
    unittest.main()
