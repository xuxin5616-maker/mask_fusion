"""Greedy single-image inference with MCQ and binary anomaly scoring."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch

from .data import QADataset, collate_samples
from .anomaly_metrics import (
    extract_anomaly_prediction, summarize_anomaly_predictions, validate_anomaly_target,
)
from .model import load_model, read_adapter_config
from .progress import ProgressTimer
from .train import _move, _resolve_dtype, _autocast


CATEGORY_MCQ_INSTRUCTION = "Answer with the option's letter from the given choices directly."
CATEGORY_MCQ_MODES = ("category-mcq", "category-knowledge-mcq")
CATEGORY_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "prompts" / "mvtec_category_knowledge.json"


def load_category_knowledge():
    """Read the bundled category criteria once and record their exact version."""
    raw = CATEGORY_KNOWLEDGE_PATH.read_bytes()
    knowledge = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(knowledge, dict) or not isinstance(knowledge.get("categories"), dict):
        raise ValueError(f"Invalid category knowledge: {CATEGORY_KNOWLEDGE_PATH}")
    return knowledge, {
        "file": str(CATEGORY_KNOWLEDGE_PATH),
        "version": knowledge["version"],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def category_anomaly_record(record, *, knowledge=None):
    """Build the requested test-only prompt without using any target label."""
    category = record.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category-mcq requires category (or class/sub_dataset in the source record)")
    readable_category = " ".join(category.replace("_", " ").split())
    if not readable_category:
        raise ValueError("category-mcq requires a nonempty readable category")
    question = f"Are there any defects on the {readable_category} in this image?"
    if knowledge is not None:
        # Only the product category selects knowledge, never gt, origin_path,
        # the reference answer, or the image's defect subdirectory.
        key = readable_category.casefold().replace(" ", "_")
        if key not in knowledge["categories"]:
            raise ValueError(
                f"category-knowledge-mcq has no inspection criteria for category {category!r}; "
                "use category-mcq for products outside MVTec AD"
            )
        entry = knowledge["categories"][key]
        question = "\n\n".join([
            knowledge["common_prefix"],
            f"Product: {entry['object_name']}.",
            f"Normal criteria: {entry['normal_criteria']}",
            f"Normal appearance: {entry['normal_appearance']}",
            "Possible anomalies:\n" + "\n".join(
                f"- {name}: {description}" for name, description in entry["defects"].items()
            ),
            f"Does the {entry['object_name']} in this image show any anomaly under these criteria?",
        ])
    return {
        **record,
        "original_question": record["question"],
        "question": question,
        "options": {"A": "Yes", "B": "No"},
        "mcq_instruction": CATEGORY_MCQ_INSTRUCTION,
        # Keep reference answers out of the runtime prompt record. gt is read
        # separately for scoring after generation, never by this helper.
        "answer": None,
    }


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
    parser.add_argument(
        "--eval-task", choices=("qa", "anomaly"), default="qa",
        help="qa: score labeled MCQs; anomaly: score defect judgments against gt=1/0",
    )
    parser.add_argument(
        "--anomaly-prompt", choices=("original", *CATEGORY_MCQ_MODES), default="original",
        help="With --eval-task anomaly: original yes/no, category A/B, or MVTec inspection knowledge followed by A/B",
    )
    parser.add_argument("--output", required=True, help="Predictions JSONL")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False,
                        help="Save actual image/mask input views alongside predictions")
    parser.add_argument("--visualization-dir", help="Save visualizations here (also enables them); must be empty/new")
    parser.add_argument("--visualization-limit", type=int, default=0,
                        help="Visualize the first N retained QA records; 0 saves all")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.anomaly_prompt != "original" and args.eval_task != "anomaly":
        parser.error(f"--anomaly-prompt {args.anomaly_prompt} requires --eval-task anomaly")
    if args.visualization_limit < 0:
        parser.error("--visualization-limit must be nonnegative")
    output = Path(args.output)
    metrics_path = output.with_suffix(".metrics.json")
    if not args.overwrite and (output.exists() or metrics_path.exists()):
        raise FileExistsError("Output already exists; choose another path or use --overwrite")
    visualization_dir = None
    if args.visualize or args.visualization_dir:
        visualization_dir = (Path(args.visualization_dir) if args.visualization_dir else
                             output.with_name(output.stem + "_visualizations")).resolve()
        if visualization_dir.exists() and (not visualization_dir.is_dir() or any(visualization_dir.iterdir())):
            raise FileExistsError("Visualization directory is not empty; choose a new --output or --visualization-dir")
        from .visualization import save_input_visualization
    knowledge, knowledge_metadata = None, None
    if args.anomaly_prompt == "category-knowledge-mcq":
        knowledge, knowledge_metadata = load_category_knowledge()
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
    if args.eval_task == "anomaly":
        # Validate the complete retained set before writing partial predictions.
        # gt is only an evaluation target: it is never converted into a prompt.
        for index, record in enumerate(dataset.records):
            try:
                validate_anomaly_target(record)
                if args.anomaly_prompt in CATEGORY_MCQ_MODES:
                    dataset.records[index] = category_anomaly_record(record, knowledge=knowledge)
                elif record["options"] is not None:
                    raise ValueError("anomaly scoring requires yes/no prompts without options; use --eval-task qa for MCQs")
            except ValueError as error:
                source = dataset.sources[index]
                raise ValueError(f"{source['source_file']}: record {source['source_index']}: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    visualization_count = 0
    visualization_score_max = float(metadata.get("fusion_config", {}).get("score_scale", 255.0))
    timer = ProgressTimer(len(dataset))
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        # One prompt per call avoids decoder-only right-padding generation ambiguity.
        for index in range(len(dataset)):
            record = dataset.records[index]
            # Prepare exactly once: random ROI selection must be shared by the
            # model and its diagnostic images, not independently rerun.
            sample = dataset[index]
            batch = _move(collate_samples([sample], tokenizer.pad_token_id), device)
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
                "mask": record["mask"], **dataset.sources[index],
                "question": record["question"], "options": record["options"],
                "prediction": answer, "choice": choice, "answer": expected,
                "correct": ((choice == expected) if expected is not None and record["options"] else None),
            }
            if args.eval_task == "anomaly":
                if args.anomaly_prompt in CATEGORY_MCQ_MODES:
                    predicted_label = {"A": 1, "B": 0}.get(choice)
                    result.update({
                        "original_question": record["original_question"],
                        "mcq_instruction": record["mcq_instruction"],
                        "answer": "A" if record["gt"] == 1 else "B",
                    })
                else:
                    predicted_label = extract_anomaly_prediction(answer)
                result.update({
                    "gt": record["gt"], "anomaly_prediction": predicted_label,
                    "correct": predicted_label == record["gt"],
                })
            # Annotate only after generation/scoring. Ground truth is display
            # metadata and is never added to model inputs or crop selection.
            if visualization_dir is not None and (args.visualization_limit == 0 or index < args.visualization_limit):
                result["visualization"] = save_input_visualization(
                    sample, record, dataset.sources[index], processor, visualization_dir, index,
                    score_max=visualization_score_max, prediction_result=result,
                )
                visualization_count += 1
            predictions.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            progress = timer.snapshot(index + 1)
            print(f"[{index + 1}/{len(dataset)}] elapsed={progress['elapsed']} "
                  f"ETA={progress['eta']} {record['id']}: {answer}", flush=True)
    metrics = (summarize_anomaly_predictions(predictions) if args.eval_task == "anomaly"
               else summarize_predictions(predictions))
    metrics.update({"adapter": str(Path(args.adapter).resolve()), "base_model": str(base),
                    "image_mode": metadata["image_mode"], "dtype": dtype_name,
                    "attention": args.attn_implementation, "seed": args.seed,
                    "eval_task": args.eval_task,
                    "anomaly_prompt": args.anomaly_prompt,
                    "input_records": len(dataset) + len(dataset.skipped_records),
                    "skipped_missing_images": len(dataset.skipped_records),
                    "skipped_records": dataset.skipped_records})
    metrics["inference_progress"] = {"progress_unit": "record", **timer.snapshot()}
    if knowledge_metadata is not None:
        metrics["anomaly_knowledge"] = knowledge_metadata
    if visualization_dir is not None:
        metrics["visualization"] = {"directory": str(visualization_dir), "records": visualization_count,
                                    "limit": args.visualization_limit, "score_max": visualization_score_max}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
