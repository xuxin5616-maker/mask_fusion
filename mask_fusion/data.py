"""Shared one-image QA formatting, tokenization, and aligned data loading.

An MCQ answer is serialized as its canonical option letter. Without ``options``
the answer is ordinary text. Training and inference use the same ChatML prefix.
"""
from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
SYSTEM_MESSAGE = "You are a helpful assistant."
_RESERVED = ("<image>", "<im_start>", "<im_end>", "<|im_start|>", "<|im_end|>")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    value = value.strip()
    if any(token in value for token in _RESERVED):
        raise ValueError(f"{name} contains a reserved image/chat token")
    return value


def _comparison_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def normalize_record(record: Mapping[str, Any], *, require_answer: bool = True) -> Dict[str, Any]:
    """Validate QA semantics without opening files; return a normalized copy.

    Options may contain 2--8 entries, either a list or a contiguous A--H map.
    Text answers to MCQs must uniquely match an option; unsupported answers
    are rejected instead of being guessed or silently assigned to an option.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Each QA record must be a JSON object")
    result = dict(record)
    result["question"] = _text(record.get("question"), "question")
    options = record.get("options")
    if options is not None:
        if isinstance(options, list):
            if not 2 <= len(options) <= 8:
                raise ValueError("options must contain between 2 and 8 choices")
            normalized = {string.ascii_uppercase[i]: _text(v, f"option {i}") for i, v in enumerate(options)}
        elif isinstance(options, Mapping):
            if not 2 <= len(options) <= 8:
                raise ValueError("options must contain between 2 and 8 choices")
            expected = list(string.ascii_uppercase[: len(options)])
            if set(options) != set(expected):
                raise ValueError(f"Option keys must be contiguous uppercase letters: {expected}")
            normalized = {key: _text(options[key], f"option {key}") for key in expected}
        else:
            raise ValueError("options must be a list or an A--H mapping")
        values = [_comparison_text(value) for value in normalized.values()]
        if len(set(values)) != len(values):
            raise ValueError("Option texts must be distinct")
        result["options"] = normalized
    else:
        result["options"] = None

    if require_answer or record.get("answer") is not None:
        answer = _text(record.get("answer"), "answer")
        if result["options"] is not None:
            options = result["options"]
            by_letter = answer.upper() if len(answer) == 1 and answer.upper() in options else None
            by_text = [key for key, value in options.items() if _comparison_text(value) == _comparison_text(answer)]
            if by_letter is not None and by_text and by_text[0] != by_letter:
                raise ValueError("Ambiguous answer: letter and option text refer to different choices")
            if by_letter is not None:
                answer = by_letter
            elif len(by_text) == 1:
                answer = by_text[0]
            else:
                raise ValueError("MCQ answer must be an existing option letter or the exact text of one option")
        result["answer"] = answer
    if "category" in result and result["category"] is not None:
        result["category"] = _text(result["category"], "category")
    return result


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_prompt(record: Mapping[str, Any], model_config: Any = None) -> str:
    """Return the exact inference prefix, excluding the assistant's answer."""
    record = normalize_record(record, require_answer=False)
    image_token = "<image>"
    if _config_value(model_config, "mm_use_im_start_end", False):
        image_token = "<im_start><image><im_end>"
    question = image_token + "\n" + record["question"]
    if record["options"] is not None:
        question += "\n" + "\n".join(f"{key}. {value}" for key, value in record["options"].items())
        question += "\nAnswer with the option letter only."
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _image_tokenize(prompt: str, tokenizer: Any) -> torch.Tensor:
    from .loader import activate_vendor

    activate_vendor()
    from llava.mm_utils import tokenizer_image_token

    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")


def tokenize_record(record: Mapping[str, Any], tokenizer: Any, model_config: Any, training: bool = True) -> Dict[str, torch.Tensor]:
    """Only answer and its ChatML end token are supervised; the prefix is -100."""
    record = normalize_record(record, require_answer=training)
    prefix = _image_tokenize(build_prompt(record, model_config), tokenizer).long()
    if prefix.ndim != 1 or int((prefix == IMAGE_TOKEN_INDEX).sum()) != 1:
        raise ValueError("One-image QA requires exactly one image placeholder")
    if not training:
        return {"input_ids": prefix}
    # Tokenize the continuation separately, as generation does, with no BOS.
    answer_ids = tokenizer(record["answer"] + "<|im_end|>", add_special_tokens=False).input_ids
    answer = torch.as_tensor(answer_ids, dtype=torch.long)
    if answer.ndim != 1 or answer.numel() == 0:
        raise ValueError("The tokenizer produced an empty or invalid answer")
    return {
        "input_ids": torch.cat((prefix, answer)),
        "labels": torch.cat((torch.full_like(prefix, IGNORE_INDEX), answer.clone())),
    }


def read_records(data_path: Any) -> List[Dict[str, Any]]:
    """Read a JSONL file or a JSON array, retaining source line diagnostics."""
    path = Path(data_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = handle.read()
    if raw.lstrip().startswith("["):
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array")
    else:
        records = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not a JSON object")
    return records


class QADataset(Dataset):
    def __init__(
        self, data_path: Any, tokenizer: Any, processor: Any, model_config: Any,
        *, data_root: Optional[Any] = None, mode: str = "anyres_max_9", seed: int = 0,
        training: bool = True, max_text_length: int = 2048,
    ) -> None:
        self.data_path = Path(data_path).resolve()
        self.data_root = Path(data_root).resolve() if data_root is not None else self.data_path.parent
        self.tokenizer, self.processor, self.model_config = tokenizer, processor, model_config
        self.mode, self.seed, self.training = mode, int(seed), bool(training)
        self.max_text_length = int(max_text_length)
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be positive")
        self.epoch = 0
        self.records = []
        seen_ids = set()
        for index, raw in enumerate(read_records(self.data_path)):
            try:
                record = normalize_record(raw, require_answer=training)
                record.setdefault("id", str(index))
                record_id = str(record["id"])
                if record_id in seen_ids:
                    raise ValueError(f"duplicate record id {record_id!r}")
                seen_ids.add(record_id)
                for key in ("image", "mask"):
                    value = raw.get(key)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{key} must be a nonempty file path")
                    path = Path(value)
                    record[key] = str((path if path.is_absolute() else self.data_root / path).resolve())
                    if not Path(record[key]).is_file():
                        raise FileNotFoundError(f"Missing {key}: {record[key]}")
                self.records.append(record)
            except (ValueError, OSError) as error:
                raise ValueError(f"{self.data_path}: record {index}: {error}") from error

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from .preprocessing import load_mask, prepare_image_mask

        record = self.records[index]
        sample = tokenize_record(record, self.tokenizer, self.model_config, training=self.training)
        if sample["input_ids"].numel() > self.max_text_length:
            raise ValueError(
                f"Record {record['id']}: text length {sample['input_ids'].numel()} exceeds "
                f"max_text_length={self.max_text_length}; shorten the example or raise the limit. "
                "Supervision is never silently truncated."
            )
        with Image.open(record["image"]) as image_file:
            image = image_file.convert("RGB")
        mask = load_mask(record["mask"])
        prepared = prepare_image_mask(
            image, mask, self.processor, self.model_config, mode=self.mode,
            seed=self.seed + self.epoch * len(self) + index,
        )
        for name in ("images", "masks", "mask_valid", "image_size"):
            sample[name] = prepared[name]
        return sample


def collate_samples(batch: Sequence[Mapping[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Right-pad text; keep variable numbers of aligned image views as lists."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if pad_token_id is None:
        raise ValueError("A valid tokenizer pad_token_id is required")
    has_labels = ["labels" in sample for sample in batch]
    if any(has_labels) and not all(has_labels):
        raise ValueError("Cannot mix training and inference samples in a batch")
    max_length = max(sample["input_ids"].numel() for sample in batch)
    ids = torch.full((len(batch), max_length), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, IGNORE_INDEX) if all(has_labels) else None
    for index, sample in enumerate(batch):
        length = sample["input_ids"].numel()
        ids[index, :length] = sample["input_ids"]
        # Compare lengths, not token IDs: Qwen can use EOS as its pad token.
        attention[index, :length] = True
        if labels is not None:
            if sample["labels"].shape != sample["input_ids"].shape:
                raise ValueError("input_ids and labels must have the same shape")
            labels[index, :length] = sample["labels"]
    result = {
        "input_ids": ids, "attention_mask": attention,
        "images": [sample["images"] for sample in batch],
        "masks": [sample["masks"] for sample in batch],
        "mask_valid": [sample["mask_valid"] for sample in batch],
        "image_sizes": [sample["image_size"] for sample in batch],
        "modalities": ["image"] * len(batch),
    }
    if labels is not None:
        result["labels"] = labels
    return result
