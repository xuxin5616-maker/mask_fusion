"""Greedy single-image QA inference and optional MCQ accuracy reporting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import torch

from .data import QADataset, collate_samples
from .model import load_model, read_adapter_config
from .train import _move, _resolve_dtype, _autocast


def extract_choice(text, options):
    """Accept one unambiguous option, never find a random letter in a sentence."""
    if not options:
        return None
    cleaned = text.strip()
    candidate = re.fullmatch(r"\(?([A-Ha-h])\)?[.。]?", cleaned)
    if candidate and candidate.group(1).upper() in options:
        return candidate.group(1).upper()
    matching = [key for key, value in options.items()
                if " ".join(value.split()).casefold() == " ".join(cleaned.split()).casefold()]
    return matching[0] if len(matching) == 1 else None


def summarize_predictions(predictions):
    buckets = defaultdict(list)
    scored = []
    for result in predictions:
        if result.get("correct") is None:
            continue
        scored.append(bool(result["correct"]))
        buckets[result.get("category") or "__unspecified__"].append(bool(result["correct"]))
    categories = {name: {"count": len(values), "accuracy": sum(values) / len(values)}
                  for name, values in sorted(buckets.items())}
    return {
        "qa_records": len(predictions), "scored_mcq": len(scored),
        "accuracy": sum(scored) / len(scored) if scored else None,
        "macro_category_accuracy": (sum(v["accuracy"] for v in categories.values()) / len(categories)
                                    if categories else None),
        "unparseable_mcq": sum(result.get("options") is not None and result.get("choice") is None
                               for result in predictions),
        "by_category": categories,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", help="Override the original checkpoint's saved local path")
    parser.add_argument("--vision-tower", help="Override an unavailable SigLIP path in the base config")
    parser.add_argument("--data", required=True, help="JSONL/JSON QA records; answer is optional")
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True, help="Predictions JSONL")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    output = Path(args.output)
    metrics_path = output.with_suffix(".metrics.json")
    if not args.overwrite and (output.exists() or metrics_path.exists()):
        raise FileExistsError("Output already exists; choose another path or use --overwrite")
    metadata = read_adapter_config(args.adapter)
    base = args.base_model or metadata["base_model"]
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    dtype_name, dtype = _resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    model, tokenizer, processor = load_model(
        base, adapter_path=args.adapter, image_mode=metadata["image_mode"],
        device=str(device), dtype=dtype_name, attn_implementation=args.attn_implementation,
        vision_tower=args.vision_tower,
    )
    model.eval()
    dataset = QADataset(
        args.data, tokenizer, processor, model.config, data_root=args.data_root,
        mode=metadata["image_mode"], seed=args.seed, training=False,
        max_text_length=args.max_text_length,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        # One prompt per call avoids decoder-only right-padding generation ambiguity.
        for index in range(len(dataset)):
            record = dataset.records[index]
            batch = _move(collate_samples([dataset[index]], tokenizer.pad_token_id), device)
            input_ids = batch.pop("input_ids")
            with _autocast(device, dtype):
                generated = model.generate(
                    inputs=input_ids, **batch, max_new_tokens=args.max_new_tokens,
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # LLaVA generate passes inputs_embeds to HF and returns the continuation.
            answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            choice = extract_choice(answer, record["options"])
            expected = record.get("answer")
            result = {
                "id": record["id"], "image": record["image"], "category": record.get("category"),
                "question": record["question"], "options": record["options"],
                "prediction": answer, "choice": choice, "answer": expected,
                "correct": ((choice == expected) if expected is not None and record["options"] else None),
            }
            predictions.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(dataset)}] {record['id']}: {answer}", flush=True)
    metrics = summarize_predictions(predictions)
    metrics.update({"adapter": str(Path(args.adapter).resolve()), "base_model": str(base),
                    "image_mode": metadata["image_mode"], "dtype": dtype_name,
                    "attention": args.attn_implementation, "seed": args.seed})
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
