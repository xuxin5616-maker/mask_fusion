"""Validate QA manifests and category-held-out splits without model weights.

Run ``python -m mask_fusion.validate_data --train train.jsonl --val val.jsonl
--test test.jsonl --check-pixels``. Relative image/mask paths follow QADataset:
the explicit data root, or each manifest's own directory when omitted.
"""

import argparse
import json
import math
from pathlib import Path


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("id is required and must be a nonempty string or an integer")
    result = str(value).strip()
    if not result:
        raise ValueError("id is required and must not be empty")
    return result


def _file_path(record, key, data_root):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required and must be a nonempty file path")
    path = Path(value)
    path = (path if path.is_absolute() else data_root / path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing {key} file: {path}")
    return path


def _new_counts():
    return {"qa_count": 0, "images": set(), "masks": set(), "category_counts": {},
            "category_images": {}, "mcq_count": 0}


def _add_record(counts, image, mask, category, record):
    counts["qa_count"] += 1
    counts["images"].add(image)
    counts["masks"].add(mask)
    counts["category_counts"][category] = counts["category_counts"].get(category, 0) + 1
    counts["category_images"].setdefault(category, set()).add(image)
    counts["mcq_count"] += record["options"] is not None


def _report_counts(counts):
    categories = sorted(counts["category_counts"])
    return {
        "qa_count": counts["qa_count"], "unique_images": len(counts["images"]),
        "unique_masks": len(counts["masks"]), "category_count": len(categories),
        "categories": categories, "mcq_count": counts["mcq_count"],
        "free_text_count": counts["qa_count"] - counts["mcq_count"],
        "per_category": {
            category: {"qa_count": counts["category_counts"][category],
                       "unique_images": len(counts["category_images"][category])}
            for category in categories
        },
        "normality_counts": None,
    }


def validate_splits(
    train, val=None, test=None, *, data_root=None, score_scale=255.0,
    allow_shared_categories=False, check_pixels=False,
):
    """Return a JSON-safe manifest report; raise ValueError for invalid input.

    Category is mandatory because the default target is transfer to unseen
    product categories. Images are grouped by their resolved source path, not
    by QA record: several different questions about one image count as one
    visual example. Different splits must never share an image, including when
    ``allow_shared_categories`` is enabled. Each split accepts a single path or
    an ordered list of paths. Relative image/mask paths use the owning manifest's
    directory unless data_root overrides all sources. IDs must be present but
    may repeat: each manifest and record index identifies a separate sample.
    A resolved manifest may occur only once across all splits.
    Missing images are skipped before QA checks, matching QADataset; the
    report lists skipped source positions and reasons. Each retained split
    must still contain at least one usable sample.

    ``check_pixels=False`` checks manifests and file existence only. True also
    loads each unique mask once, verifies scalar finite nonnegative values via
    load_mask, and rejects scores above score_scale. No normal/abnormal labels
    are inferred from answer text, option letters, or file names.
    """
    from .data import MissingImageError, normalize_data_paths, normalize_record, read_records, resolve_image_path

    try:
        score_scale = float(score_scale)
    except (TypeError, ValueError) as error:
        raise ValueError("score_scale must be a finite positive number") from error
    if not math.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be a finite positive number")
    if train is None:
        raise ValueError("train manifest is required")
    root = Path(data_root).resolve() if data_root is not None else None
    if root is not None and not root.is_dir():
        raise ValueError(f"data_root is not a directory: {root}")

    split_manifests = {}
    manifest_owners = {}
    for split, paths in (("train", train), ("val", val), ("test", test)):
        if paths is None:
            continue
        try:
            manifests = normalize_data_paths(paths)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{split}: {error}") from error
        for manifest in manifests:
            if manifest in manifest_owners:
                raise ValueError(
                    f"Duplicate resolved manifest in {split} and {manifest_owners[manifest]}: {manifest}"
                )
            if not manifest.is_file():
                raise ValueError(f"Missing {split} manifest: {manifest}")
            manifest_owners[manifest] = split
        split_manifests[split] = manifests

    split_reports = {}
    image_owners = {}
    image_categories = {}
    category_owners = {}
    checked_masks = {}
    all_masks = set()
    total_qa = 0
    skipped_records = []

    for split, manifests in split_manifests.items():
        counts = _new_counts()
        source_reports = []
        source_roots = set()
        for manifest in manifests:
            source_root = root if root is not None else manifest.parent
            source_roots.add(source_root)
            try:
                records = read_records(manifest)
            except (OSError, ValueError) as error:
                raise ValueError(f"{split} {manifest}: {error}") from error
            source_counts = _new_counts()
            source_skips = []
            for index, raw in enumerate(records):
                try:
                    image = resolve_image_path(raw, source_root)
                    record = normalize_record(raw, require_answer=True)
                    _identifier(record.get("id"))
                    category = record.get("category")
                    if not isinstance(category, str) or not category:
                        raise ValueError("category is required to validate category-held-out evaluation")
                    mask = _file_path(record, "mask", source_root)
                    if image in image_categories and image_categories[image] != category:
                        raise ValueError(
                            f"Same image has conflicting category labels {image_categories[image]!r} "
                            f"and {category!r}: {image}"
                        )
                    image_categories[image] = category
                    if image in image_owners and image_owners[image] != split:
                        raise ValueError(
                            f"Image leakage: {split} and {image_owners[image]} share resolved image {image}. "
                            "All questions for one source image must stay in one split."
                        )
                    image_owners[image] = split
                    previous_categories = category_owners.setdefault(category, set())
                    if previous_categories - {split} and not allow_shared_categories:
                        previous = ", ".join(sorted(previous_categories - {split}))
                        raise ValueError(
                            f"Category overlap: {category!r} appears in {split} and {previous}. "
                            "Default evaluation requires disjoint product categories; "
                            "use --allow-shared-categories only for a different evaluation protocol."
                        )
                    previous_categories.add(split)

                    if check_pixels and mask not in checked_masks:
                        from .preprocessing import load_mask

                        pixels = load_mask(mask)
                        maximum = float(pixels.max())
                        minimum = float(pixels.min())
                        if maximum > score_scale:
                            raise ValueError(
                                f"Mask score {maximum:g} exceeds score_scale={score_scale:g}: {mask}. "
                                "Set the declared score scale to match the expert's raw score units."
                            )
                        checked_masks[mask] = {"min": minimum, "max": maximum}
                    _add_record(counts, image, mask, category, record)
                    _add_record(source_counts, image, mask, category, record)
                except MissingImageError as error:
                    source_skips.append({
                        "split": split, "source_file": str(manifest), "source_index": index,
                        "id": raw.get("id"), "reason": str(error),
                    })
                except (OSError, ValueError) as error:
                    raise ValueError(f"{split} {manifest}: record {index + 1}: {error}") from error
            skipped_records.extend(source_skips)
            source_reports.append({
                "manifest": str(manifest), "data_root": str(source_root), **_report_counts(source_counts),
                "skipped_missing_images": len(source_skips),
            })

        if counts["qa_count"] == 0:
            raise ValueError(
                f"{split}: No usable samples remain after skipping records with missing images; "
                "check --data-root and image paths."
            )
        split_reports[split] = {
            "manifest": str(manifests[0]) if len(manifests) == 1 else None,
            "manifests": [str(manifest) for manifest in manifests],
            "data_root": str(next(iter(source_roots))) if len(source_roots) == 1 else None,
            "sources": source_reports, **_report_counts(counts),
            "skipped_missing_images": sum(source["skipped_missing_images"] for source in source_reports),
        }
        all_masks.update(counts["masks"])
        total_qa += counts["qa_count"]

    shared_categories = {
        category: sorted(owners) for category, owners in sorted(category_owners.items()) if len(owners) > 1
    }
    return {
        "status": "ok",
        "checks_scope": "retained samples after skipping records with missing images",
        "skipped_missing_images": len(skipped_records),
        "skipped_records": skipped_records,
        "protocol": "shared_categories_allowed" if allow_shared_categories else "category_held_out",
        "checks": {
            "qa_schema": "passed", "required_category_and_id": "passed",
            "image_and_mask_files": "passed", "cross_split_image_disjointness": "passed",
            "consistent_image_categories": "passed",
            "cross_split_category_disjointness": "not_required" if allow_shared_categories else "passed",
            "mask_pixels": "passed" if check_pixels else "not_checked",
        },
        "score_scale": score_scale,
        "checked_mask_count": len(checked_masks),
        "observed_score_range": {
            "min": min(item["min"] for item in checked_masks.values()),
            "max": max(item["max"] for item in checked_masks.values()),
        } if checked_masks else None,
        "total": {"qa_count": total_qa, "unique_images": len(image_owners), "unique_masks": len(all_masks),
                  "category_count": len(category_owners)},
        "normality_note": "Normal/abnormal counts are not inferred from answers, choices, or paths.",
        "id_identity": "resolved_manifest_path + zero_based_record_index; original IDs may repeat",
        "image_identity": "resolved_file_path; copied files with different paths are not content-deduplicated",
        "shared_categories": shared_categories,
        "splits": split_reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, nargs="+", action="extend", help="Training QA JSONL/JSON files; may repeat")
    parser.add_argument("--val", nargs="+", action="extend", help="Optional validation QA files; may repeat")
    parser.add_argument("--test", nargs="+", action="extend", help="Optional test QA files; may repeat")
    parser.add_argument("--data-root", help="Common root for relative image/mask paths; defaults to each manifest directory")
    parser.add_argument("--score-scale", type=float, default=255.0, help="Maximum raw expert score, default 255")
    parser.add_argument("--allow-shared-categories", action="store_true", help="Relax category disjointness, never image disjointness")
    parser.add_argument("--check-pixels", action="store_true", help="Load each mask and check finite scalar scores in [0, score_scale]")
    args = parser.parse_args(argv)
    try:
        report = validate_splits(
            args.train, args.val, args.test, data_root=args.data_root, score_scale=args.score_scale,
            allow_shared_categories=args.allow_shared_categories, check_pixels=args.check_pixels,
        )
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
