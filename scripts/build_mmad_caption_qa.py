#!/usr/bin/env python3
"""Build simple caption QA from extracted MMAD files (Python 3.9+, stdlib only).

Expected layout: MMAD/VisA/candle/.../001.png and adjacent 001.txt.
Also accepts a single source directory such as MMAD/VisA as --mmad-root.
Each image yields one QA; captions are copied verbatim except outer whitespace.
No images, captions, or original MMAD annotations are changed.

Example:
    python scripts/build_mmad_caption_qa.py --mmad-root /data/MMAD \
        --output /data/caption_qa.json --exclude-datasets MVTec-AD DS-MVTec

Output image paths are absolute. ``mask`` is deliberately null: MMAD ground-truth
masks are NOT expert predictions. Fill this field with generated MuSc V2 paths
before using the resulting manifest with mask_fusion.train. This script does
not split the data; all QA derived from a source image must share its split.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys


DATASETS = ("DS-MVTec", "MVTec-AD", "MVTec-LOCO", "VisA", "GoodsAD")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
GENERIC_PROMPTS = (
    "Describe the image.",
    "Provide a detailed description of what is visible in the image.",
    "Describe the appearance and condition of the object in the image.",
    "Summarize the main visual features of the object.",
    "What can you observe in this image? Describe the visible details.",
)
INSPECTION_PROMPTS = (
    "Describe the object and indicate whether any visible abnormalities are present.",
    "Inspect the image and describe the object's condition, including any visible defects.",
    "Does the object appear normal or abnormal? Describe what you observe.",
    "Are there any visible defects in the image? Provide a description of your findings.",
    "Give a brief visual inspection report describing the object and any apparent abnormalities.",
    "Determine whether the object has any visible defects and explain the visual evidence for your assessment.",
    "Does the image show any abnormalities? Describe the observations that support your judgment.",
    "Assess the object's visible condition and explain why it appears normal or abnormal.",
    "Describe any visible defects and explain which visual features indicate an abnormality.",
    "Examine the image, describe your findings, and explain the basis for your assessment of the object's condition.",
)
ALL_PROMPTS = GENERIC_PROMPTS + INSPECTION_PROMPTS
# Lightweight routing only; keyword presence does not verify caption correctness.
INSPECTION_WORDS = re.compile(
    r"\b(normal|abnormal|anomal(?:y|ies|ous)|defects?|defective|damage(?:d)?|"
    r"scratch(?:es|ed)?|crack(?:s|ed)?|contaminat(?:ion|ed)|flaws?|intact)\b"
    r"|正常|异常|缺陷|划痕|裂纹|破损|污染",
    re.IGNORECASE,
)
RESERVED_TOKENS = ("<image>", "<im_start>", "<im_end>", "<|im_start|>", "<|im_end|>")


def image_identity(image, root):
    """Return source dataset, product category, and stable MMAD-relative key."""
    parts = image.relative_to(root).parts
    if root.name in DATASETS:
        parts = (root.name,) + parts
    for index, part in enumerate(parts[:-2]):
        if part in DATASETS:
            return part, parts[index + 1], "/".join(parts[index:])
    return None


def choose_prompt(caption, image_key, seed, mode):
    pool = ALL_PROMPTS if mode == "all" else GENERIC_PROMPTS
    if mode == "auto" and INSPECTION_WORDS.search(caption):
        pool = ALL_PROMPTS
    # Stable per-image selection: filtering another source never changes this QA.
    digest = hashlib.sha256(f"{seed}\0{image_key}".encode("utf-8")).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def build_records(root, *, excluded=(), seed=42, prompt_mode="auto"):
    records = []
    counts = Counter()
    source_counts = Counter()
    seen_images = set()
    seen_ids = set()
    scan_errors = []
    for folder, dirs, filenames in os.walk(root, onerror=scan_errors.append):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        parent = Path(folder)
        image_files = [parent / name for name in sorted(filenames)
                       if Path(name).suffix.lower() in IMAGE_SUFFIXES]
        for name in sorted(filenames):
            if Path(name).suffix.lower() != ".txt":
                continue
            caption_path = parent / name
            # Supports both 001.txt and 001.png.txt, without reading image pixels.
            matches = [image for image in image_files
                       if image.stem == caption_path.stem or image.name == caption_path.stem]
            if not matches:
                counts["text_without_matching_image"] += 1
                continue
            if len(matches) != 1:
                raise ValueError(f"Ambiguous image for caption: {caption_path}")
            image = matches[0]
            identity = image_identity(image, root)
            if identity is None:
                counts["unknown_dataset_layout"] += 1
                continue
            dataset, category, image_key = identity
            if dataset in excluded:
                counts["excluded_dataset"] += 1
                continue
            try:
                caption = caption_path.read_text(encoding="utf-8-sig").strip()
            except UnicodeError:
                counts["invalid_utf8_caption"] += 1
                continue
            if not caption:
                counts["empty_caption"] += 1
                continue
            if any(token in caption for token in RESERVED_TOKENS):
                counts["reserved_chat_token_in_caption"] += 1
                continue
            image_path = str(image.resolve())
            record_id = "mmad_caption:" + image_key
            if image_path in seen_images:
                counts["duplicate_image_path"] += 1
                continue
            if record_id in seen_ids:
                raise ValueError(f"Repeated source image key in different folders: {image_key}")
            seen_images.add(image_path)
            seen_ids.add(record_id)
            records.append({
                "id": record_id,
                "category": category,
                "source_dataset": dataset,
                "image": image_path,
                "mask": None,
                "question": choose_prompt(caption, image_key, seed, prompt_mode),
                "answer": caption,
                "caption_path": str(caption_path.resolve()),
            })
            source_counts[dataset] += 1
    if scan_errors:
        raise OSError(f"Could not scan part of the dataset: {scan_errors[0]}")
    if not records:
        raise ValueError(
            "No usable image/caption pairs. Expected source/category/.../001.png + 001.txt. "
            f"Skipped: {dict(counts)}"
        )
    records.sort(key=lambda record: record["id"])
    return records, {"records": len(records), "by_source_dataset": dict(source_counts),
                     "skipped": dict(counts), "missing_expert_masks": len(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mmad-root", required=True, type=Path, help="Extracted MMAD root or one source dataset folder")
    parser.add_argument("--output", required=True, type=Path, help="Output .json (array) or .jsonl (one QA per line)")
    parser.add_argument("--exclude-datasets", nargs="+", choices=DATASETS, default=[], help="E.g. MVTec-AD DS-MVTec when retaining MVTec evaluation")
    parser.add_argument("--prompt-mode", choices=("auto", "generic", "all"), default="auto",
                        help="auto: generic prompts unless caption has inspection keywords; all: all 15 prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output manifest")
    args = parser.parse_args(argv)
    root = args.mmad_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        if not root.is_dir():
            raise ValueError(f"MMAD directory does not exist: {root}")
        if output.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("Output extension must be .json or .jsonl")
        if output.exists() and not args.overwrite:
            raise ValueError(f"Output already exists: {output}; use a new path or --overwrite")
        records, report = build_records(root, excluded=set(args.exclude_datasets),
                                        seed=args.seed, prompt_mode=args.prompt_mode)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as handle:
            if output.suffix.lower() == ".jsonl":
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        report.update({"output": str(output), "seed": args.seed, "prompt_mode": args.prompt_mode,
                       "note": "mask is null. Add expert prediction paths before mask-fusion training."})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
