"""Generate inspectable, uncalibrated synthetic score maps and grounded MCQs.

This module deliberately uses only NumPy and Pillow. The normal/abnormal label
is an explicit synthetic rule, NOT a calibrated MuSc V2 defect ground truth.
The image field is a placeholder pointing to the mask: these data must only be
used with the mask-only pretraining stage, which removes the visual branch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_CONFIG: dict[str, Any] = {
    "description": "Uncalibrated synthetic score-map curriculum; not real MuSc V2 defect labels.",
    "label_rule": "coverage_patch_mean",
    "label_blur_radius": 3.0,
    "label_peak_threshold": 80.0,
    "abnormal_fraction": 0.5,
    "score_threshold": 128,
    "abnormal_min_coverage": 0.006,
    "patch_size": 14,
    "abnormal_min_patch_mean": 140.0,
    "score_bin_edges": [64, 128, 192],
    "coverage_bin_edges": [0.005, 0.05, 0.2],
    "elongated_aspect_ratio": 3.0,
    "max_attempts": 100,
    "tasks": ["anomaly", "location", "intensity", "coverage", "shape"],
    "normal": {
        "background_range": [5, 65], "background_noise_std": [1, 10],
        "region_count_range": [0, 3], "region_peak_range": [70, 180],
        "region_area_range": [0.0003, 0.003], "elongation_range": [1, 6],
        "blur_radius_range": [0.5, 2.5],
        "shapes": ["ellipse", "rectangle", "streak"],
        "hotspot_count_range": [0, 3], "hotspot_peak_range": [190, 250],
        "hotspot_area_range": [0.00003, 0.0003],
    },
    "abnormal": {
        "background_range": [5, 65], "background_noise_std": [1, 10],
        "region_count_range": [1, 3], "region_peak_range": [165, 250],
        "region_area_range": [0.012, 0.15], "elongation_range": [1, 6],
        "blur_radius_range": [0.5, 3.5],
        "shapes": ["ellipse", "rectangle", "streak"],
        "hotspot_count_range": [0, 3], "hotspot_peak_range": [190, 250],
        "hotspot_area_range": [0.00003, 0.0003],
    },
}
TASKS = set(DEFAULT_CONFIG["tasks"])
DISCLAIMER = (
    "Normal/abnormal answers follow configurable synthetic score-map rules, not "
    "real defect annotations or calibrated MuSc V2 decisions. No per-image min-max "
    "normalization is applied. Image paths are mask placeholders for mask-only "
    "pretraining; do not use these as real photographic SFT examples."
)


def _merge_config(base: dict, changes: dict, prefix: str = "") -> dict:
    result = deepcopy(base)
    for key, value in changes.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be an object")
            result[key] = _merge_config(result[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    changes = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(changes, dict):
        raise ValueError("Generator configuration must be a JSON object")
    return validate_config(_merge_config(DEFAULT_CONFIG, changes))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    def interval(value, name, low, high, integers=False):
        if (not isinstance(value, list) or len(value) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) for x in value)
                or not low <= value[0] <= value[1] <= high
                or (integers and any(int(x) != x for x in value))):
            raise ValueError(f"{name} must be an ordered range within [{low}, {high}]")

    for key, low, high in [
        ("abnormal_fraction", 0, 1), ("score_threshold", 1, 255),
        ("abnormal_min_coverage", 0, 1), ("abnormal_min_patch_mean", 0, 255),
        ("elongated_aspect_ratio", 1, 1000),
        ("label_peak_threshold", 0, 255),
    ]:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    if config["label_rule"] not in ("coverage_patch_mean", "blurred_peak"):
        raise ValueError("label_rule must be coverage_patch_mean or blurred_peak")
    blur_radius = config["label_blur_radius"]
    if (not isinstance(blur_radius, (int, float)) or isinstance(blur_radius, bool)
            or not math.isfinite(blur_radius) or blur_radius < 0):
        raise ValueError("label_blur_radius must be a finite nonnegative number")
    for key in ("patch_size", "max_attempts"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, high in (("score_bin_edges", 255), ("coverage_bin_edges", 1)):
        edges = config[key]
        if (not isinstance(edges, list) or not 1 <= len(edges) <= 7
                or any(not isinstance(x, (int, float)) or not 0 < x < high for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError(f"{key} must contain 1-7 strictly increasing edges inside (0, {high})")
    tasks = config["tasks"]
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or set(tasks) - TASKS:
        raise ValueError(f"tasks must contain distinct entries from {sorted(TASKS)}")
    if "anomaly" not in tasks:
        raise ValueError("The first pretraining curriculum must include the anomaly task")
    for label in ("normal", "abnormal"):
        profile = config[label]
        for key in ("background_range", "region_peak_range", "hotspot_peak_range"):
            interval(profile[key], f"{label}.{key}", 0, 255)
        for key in ("region_area_range", "hotspot_area_range"):
            interval(profile[key], f"{label}.{key}", 0.000001, 1)
        for key in ("region_count_range", "hotspot_count_range"):
            interval(profile[key], f"{label}.{key}", 0, 100, integers=True)
        interval(profile["background_noise_std"], f"{label}.background_noise_std", 0, 100)
        interval(profile["blur_radius_range"], f"{label}.blur_radius_range", 0, 100)
        interval(profile["elongation_range"], f"{label}.elongation_range", 1, 100)
        shapes = profile["shapes"]
        if not isinstance(shapes, list) or not shapes or set(shapes) - {"ellipse", "rectangle", "streak"}:
            raise ValueError(f"{label}.shapes must use ellipse, rectangle or streak")
    return config


def _sample(rng: np.random.Generator, bounds: list) -> float:
    return float(rng.uniform(*bounds))


def _count(rng: np.random.Generator, bounds: list) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _region(size: int, area: float, aspect: float, shape: str, blur: float,
            rng: np.random.Generator) -> np.ndarray:
    # area is the proposal bounding-box fraction, not a promised final coverage.
    width = max(1, min(size - 1, int(round(size * math.sqrt(area * aspect)))))
    height = max(1, min(size - 1, int(round(size * math.sqrt(area / aspect)))))
    if rng.random() < 0.5:
        width, height = height, width
    left, top = int(rng.integers(0, size - width)), int(rng.integers(0, size - height))
    support = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(support)
    box = (left, top, left + width, top + height)
    if shape == "ellipse":
        draw.ellipse(box, fill=255)
    elif shape == "rectangle":
        draw.rectangle(box, fill=255)
    else:
        draw.line((left, top, left + width, top + height), fill=255, width=max(1, min(width, height)))
    if blur > 0:
        support = support.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(support, dtype=np.float32) / 255.0


def propose_mask(size: int, label: str, config: dict, rng: np.random.Generator) -> np.ndarray:
    profile = config[label]
    baseline, noise_std = _sample(rng, profile["background_range"]), _sample(rng, profile["background_noise_std"])
    # Spatially smooth background plus pixel noise; fixed score units are retained.
    coarse = rng.normal(0, noise_std, (8, 8)).astype(np.float32)
    smooth = np.asarray(Image.fromarray(coarse).resize((size, size), Image.Resampling.BILINEAR))
    scores = baseline + smooth + rng.normal(0, noise_std * 0.15, (size, size))
    for hotspot in (False, True):
        prefix = "hotspot" if hotspot else "region"
        for _ in range(_count(rng, profile[f"{prefix}_count_range"])):
            support = _region(
                size, _sample(rng, profile[f"{prefix}_area_range"]),
                1.0 if hotspot else _sample(rng, profile["elongation_range"]),
                "ellipse" if hotspot else str(rng.choice(profile["shapes"])),
                0.0 if hotspot else _sample(rng, profile["blur_radius_range"]), rng,
            )
            peak = _sample(rng, profile[f"{prefix}_peak_range"])
            # Vary the interior gently without renormalizing each image.
            interior = peak + rng.normal(0, 2, (size, size))
            scores = np.maximum(scores, scores * (1 - support) + interior * support)
    return np.rint(np.clip(scores, 0, 255)).astype(np.uint8)


def describe_mask(mask: np.ndarray, config: dict) -> dict[str, Any]:
    """Compute all supervision from the final integer score map, never proposals."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 score map")
    height, width = mask.shape
    patch = config["patch_size"]
    rows, cols = height // patch, width // patch
    if not rows or not cols:
        raise ValueError("Mask must contain at least one complete patch")
    # SigLIP's patch embedding and the mask CNN omit the incomplete right/bottom
    # patch. Supervision must not depend on pixels unavailable to those encoders.
    mask = mask[:rows * patch, :cols * patch]
    visible_height, visible_width = mask.shape
    blocks = mask.reshape(rows, patch, cols, patch)
    max_patch_mean = float(blocks.mean(axis=(1, 3)).max())
    # Blur only visible pixels, so discarded edge content cannot alter labels.
    smoothed = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(config["label_blur_radius"]))
    smoothed_peak = int(np.asarray(smoothed).max())
    high = mask >= config["score_threshold"]
    coverage = float(high.mean())
    if config["label_rule"] == "blurred_peak":
        # A small coherent bright region can be abnormal; isolated single-pixel
        # spikes are attenuated by spatial smoothing without a hard area gate.
        anomaly = int(smoothed_peak >= config["label_peak_threshold"])
    else:
        anomaly = int(coverage >= config["abnormal_min_coverage"]
                      and max_patch_mean >= config["abnormal_min_patch_mean"])
    halves = [mask[:visible_height // 2, :visible_width // 2], mask[:visible_height // 2, visible_width // 2:],
              mask[visible_height // 2:, :visible_width // 2], mask[visible_height // 2:, visible_width // 2:]]
    quadrant_means = [float(part.mean()) for part in halves]
    ys, xs = np.nonzero(high)
    bbox_aspect = None
    if len(xs):
        bbox_w, bbox_h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        bbox_aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
    shape = "No high-score pixels" if not len(xs) else (
        "Elongated bounding box" if bbox_aspect >= config["elongated_aspect_ratio"] else "Compact bounding box"
    )
    return {
        "stored_size": [height, width], "visible_size": [visible_height, visible_width],
        "mean": float(mask.mean()), "maximum": int(mask.max()), "minimum": int(mask.min()),
        "max_patch_mean": max_patch_mean, "high_score_coverage": coverage,
        "smoothed_peak": smoothed_peak,
        "synthetic_anomaly": anomaly, "quadrant_means": quadrant_means,
        "highest_quadrant": int(np.argmax(quadrant_means)), "bbox_aspect_ratio": bbox_aspect,
        "shape": shape,
    }


def _bins(edges: list, *, percentage: bool = False) -> list[str]:
    def fmt(value):
        return f"{value * 100:g}%" if percentage else f"{value:g}"
    return ([f"Below {fmt(edges[0])}"]
            + [f"At least {fmt(a)} but below {fmt(b)}" for a, b in zip(edges, edges[1:])]
            + [f"At least {fmt(edges[-1])}"])


def task_answers(stats: dict, config: dict) -> dict[str, tuple[str, list[str], str]]:
    """Return question, option vocabulary and exact answer text for each task."""
    quadrants = ["Top left", "Top right", "Bottom left", "Bottom right"]
    score_options, coverage_options = _bins(config["score_bin_edges"]), _bins(config["coverage_bin_edges"], percentage=True)
    score_bin = int(np.searchsorted(config["score_bin_edges"], stats["max_patch_mean"], side="right"))
    coverage_bin = int(np.searchsorted(config["coverage_bin_edges"], stats["high_score_coverage"], side="right"))
    threshold, patch = config["score_threshold"], config["patch_size"]
    result = {
        "anomaly": (
            "Under this synthetic score-map dataset's labeling rule, is this map normal or abnormal?",
            ["Normal", "Abnormal"], "Abnormal" if stats["synthetic_anomaly"] else "Normal",
        ),
        "location": (
            "Which quadrant of this synthetic score map has the highest mean score? "
            "For exact ties, use top-to-bottom, then left-to-right order.",
            quadrants, quadrants[stats["highest_quadrant"]],
        ),
        "intensity": (
            f"Using non-overlapping complete {patch} by {patch} patches starting at the top left, "
            "which range contains the highest patch mean score? Scores use the fixed 0-255 scale.",
            score_options, score_options[score_bin],
        ),
        "coverage": (
            f"What fraction of this synthetic score map has a score of at least {threshold:g}?",
            coverage_options, coverage_options[coverage_bin],
        ),
        "shape": (
            f"Consider the bounding box of all pixels with scores of at least {threshold:g}. "
            f"Call it elongated when its longer side divided by its shorter side is at least "
            f"{config['elongated_aspect_ratio']:g}; otherwise call it compact. Which description applies?",
            ["No high-score pixels", "Compact bounding box", "Elongated bounding box"], stats["shape"],
        ),
    }
    height, width = stats["visible_size"]
    prefix = f"Use the top-left {width} by {height} patch-visible area of this synthetic score map. "
    return {task: (prefix + question, choices, answer)
            for task, (question, choices, answer) in result.items()}


class BalancedOptions:
    """Shuffle distractors while balancing answer-letter positions per task."""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.pending: dict[tuple[str, int], list[int]] = {}

    def arrange(self, task: str, texts: list[str], answer: str) -> tuple[dict[str, str], str]:
        key = (task, len(texts))
        if not self.pending.get(key):
            self.pending[key] = self.rng.permutation(len(texts)).tolist()
        correct_index = self.pending[key].pop()
        others = [text for text in texts if text != answer]
        self.rng.shuffle(others)
        others.insert(correct_index, answer)
        return {chr(65 + index): text for index, text in enumerate(others)}, chr(65 + correct_index)


def _preview(samples: list[tuple[np.ndarray, dict, str]], output: Path) -> None:
    if not samples:
        return
    columns, tile, footer = min(4, len(samples)), 192, 60
    rows = math.ceil(len(samples) / columns)
    canvas = Image.new("RGB", (columns * tile, rows * (tile + footer) + 38), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "UNCALIBRATED SYNTHETIC LABELS - fixed grayscale 0..255", fill="black")
    visible_h, visible_w = samples[0][1]["visible_size"]
    draw.text((8, 21), f"Statistics and answers use the top-left {visible_w} x {visible_h} visible pixels.", fill="black")
    for index, (mask, stats, name) in enumerate(samples):
        x, y = (index % columns) * tile, 38 + (index // columns) * (tile + footer)
        canvas.paste(Image.fromarray(mask).resize((tile, tile)), (x, y))
        label = "abnormal" if stats["synthetic_anomaly"] else "normal"
        text = (f"{name}: {label}\nmax={stats['maximum']} patch={stats['max_patch_mean']:.1f}\n"
                f"mean={stats['mean']:.1f} high={stats['high_score_coverage']:.2%}")
        draw.text((x + 4, y + tile + 3), text, fill="black")
    canvas.save(output)


def generate_dataset(output: str | Path, *, train_count: int = 2000, val_count: int = 200,
                     size: int = 384, seed: int = 0, config: dict | None = None,
                     preview_count: int = 16) -> dict[str, Any]:
    config = validate_config(_merge_config(DEFAULT_CONFIG, config or {}))
    if train_count < 1 or val_count < 0 or size < max(28, config["patch_size"]) or preview_count < 0 or seed < 0:
        raise ValueError("Require train_count >= 1, val_count >= 0, size >= max(28, patch_size), preview_count >= 0, seed >= 0")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "generator_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        "schema_version": 1, "generator": "mask_fusion.synthetic_masks", "disclaimer": DISCLAIMER,
        "seed": seed, "size": size, "score_scale": 255, "image_is_mask_placeholder": True,
        "visible_size": size // config["patch_size"] * config["patch_size"],
        "required_training_stage": "mask_pretrain", "recommended_image_mode": "square",
        "label_rule": config["label_rule"],
        "label_parameters": {
            "label_blur_radius": config["label_blur_radius"],
            "label_peak_threshold": config["label_peak_threshold"],
            "score_threshold": config["score_threshold"],
            "abnormal_min_coverage": config["abnormal_min_coverage"],
            "abnormal_min_patch_mean": config["abnormal_min_patch_mean"],
            "patch_size": config["patch_size"],
            "comparison": "greater_than_or_equal",
            "blur_implementation": "Pillow ImageFilter.GaussianBlur on visible uint8 pixels",
        },
        "splits": {},
    }
    digests: set[str] = set()
    previews = []
    split_seeds = np.random.SeedSequence(seed).spawn(2)
    for split, count, split_seed in zip(("train", "val"), (train_count, val_count), split_seeds):
        mask_rng, option_rng = [np.random.default_rng(s) for s in split_seed.spawn(2)]
        arranger = BalancedOptions(option_rng)
        split_dir = output / "masks" / split
        split_dir.mkdir(parents=True)
        labels = np.array([1] * int(round(count * config["abnormal_fraction"]))
                          + [0] * (count - int(round(count * config["abnormal_fraction"]))), dtype=np.uint8)
        mask_rng.shuffle(labels)
        answer_counts, semantic_counts = {}, {}
        anomalies = Counter()
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as records_file, \
                (output / f"{split}_masks.jsonl").open("w", encoding="utf-8") as metadata_file:
            for index, desired in enumerate(labels):
                label = "abnormal" if desired else "normal"
                for _ in range(config["max_attempts"]):
                    mask = propose_mask(size, label, config, mask_rng)
                    stats = describe_mask(mask, config)
                    digest = hashlib.sha256(mask.tobytes()).hexdigest()
                    visible_h, visible_w = stats["visible_size"]
                    visible_digest = hashlib.sha256(mask[:visible_h, :visible_w].tobytes()).hexdigest()
                    if stats["synthetic_anomaly"] == int(desired) and visible_digest not in digests:
                        break
                else:
                    raise ValueError(f"Could not generate a distinct {label} map satisfying the rule after "
                                     f"{config['max_attempts']} proposals. Check profile/rule compatibility.")
                digests.add(visible_digest)
                mask_id = f"{split}_{index:07d}"
                relative = f"masks/{split}/{mask_id}.png"
                Image.fromarray(mask).save(output / relative)
                # Reload so the labels are explicitly grounded in what training reads.
                with Image.open(output / relative) as saved:
                    stats = describe_mask(np.asarray(saved), config)
                anomalies[label] += 1
                metadata = {"mask_id": mask_id, "mask": relative, "sha256_pixels": digest,
                            "sha256_visible_pixels": visible_digest,
                            "label_rule": config["label_rule"],
                            "synthetic_label": label, "statistics": stats}
                metadata_file.write(json.dumps(metadata) + "\n")
                for task, (question, choices, answer) in task_answers(stats, config).items():
                    if task not in config["tasks"]:
                        continue
                    options, letter = arranger.arrange(task, choices, answer)
                    record = {
                        "id": f"{mask_id}_{task}", "image": relative, "mask": relative,
                        "question": question, "options": options, "answer": letter,
                        "category": "synthetic_score_map", "task": task,
                        "mask_id": mask_id, "synthetic": True,
                        "label_source": "uncalibrated_synthetic_rule", "image_is_mask_placeholder": True,
                        "label_rule": config["label_rule"],
                    }
                    records_file.write(json.dumps(record) + "\n")
                    answer_counts.setdefault(task, Counter())[letter] += 1
                    semantic_counts.setdefault(task, Counter())[answer] += 1
                # Reserve preview slots for both splits without editing the scores.
                split_preview_limit = math.ceil(preview_count / 2) if split == "train" and val_count else preview_count
                if len(previews) < split_preview_limit:
                    previews.append((mask, stats, mask_id))
        summary["splits"][split] = {
            "independent_masks": count, "qa_records": count * len(config["tasks"]),
            "labels": dict(anomalies), "answer_letter_counts": answer_counts,
            "answer_text_counts": semantic_counts, "seed_spawn_key": list(split_seed.spawn_key),
        }
    _preview(previews, output / "preview.png")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New or empty directory; never overwritten")
    parser.add_argument("--train-count", type=int, default=2000, help="Independent training masks, not QA count")
    parser.add_argument("--val-count", type=int, default=200, help="Independent validation masks")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "synthetic_masks.json"),
                        help="JSON generator config (default: project configs/synthetic_masks.json)")
    parser.add_argument("--preview-count", type=int, default=16)
    args = parser.parse_args()
    summary = generate_dataset(args.output, train_count=args.train_count, val_count=args.val_count,
                               size=args.size, seed=args.seed, config=load_config(args.config),
                               preview_count=args.preview_count)
    print(DISCLAIMER)
    print(json.dumps({"output": str(Path(args.output).resolve()), "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
