"""CPU supervision, split leakage and fixed score-scale checks for synthesis."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from mask_fusion.synthetic_masks import (
    BalancedOptions, DEFAULT_CONFIG, describe_mask, generate_dataset,
    load_config, main, task_answers, validate_config,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SyntheticMaskTests(unittest.TestCase):
    def test_normal_high_peak_and_abnormal_area_are_distinguishable(self):
        config = load_config()
        normal = np.full((56, 56), 20, dtype=np.uint8)
        normal[2, 2] = 240
        abnormal = normal.copy()
        abnormal[14:28, 14:28] = 180
        self.assertEqual(describe_mask(normal, config)["synthetic_anomaly"], 0)
        self.assertEqual(describe_mask(abnormal, config)["synthetic_anomaly"], 1)
        self.assertEqual(normal.max(), abnormal.max())

    def test_blurred_peak_keeps_small_bright_regions_without_coverage_gate(self):
        legacy = load_config()
        blurred = {**legacy, "label_rule": "blurred_peak"}
        mask = np.full((384, 384), 20, dtype=np.uint8)
        mask[28:42, 28:42] = 200
        self.assertEqual(describe_mask(mask, legacy)["synthetic_anomaly"], 0)
        stats = describe_mask(mask, blurred)
        self.assertGreater(stats["smoothed_peak"], blurred["label_peak_threshold"])
        self.assertLess(stats["high_score_coverage"], blurred["abnormal_min_coverage"])
        self.assertEqual(stats["synthetic_anomaly"], 1)

    def test_blurred_peak_attenuates_an_isolated_hot_pixel(self):
        config = {**load_config(), "label_rule": "blurred_peak"}
        mask = np.full((56, 56), 20, dtype=np.uint8)
        mask[28, 28] = 255
        stats = describe_mask(mask, config)
        self.assertEqual(stats["maximum"], 255)
        self.assertLess(stats["smoothed_peak"], config["label_peak_threshold"])
        self.assertEqual(stats["synthetic_anomaly"], 0)

    def test_blurred_peak_distinguishes_dim_normal_and_midgray_abnormal(self):
        config = {**load_config(), "label_rule": "blurred_peak"}
        normal = np.full((56, 56), 20, dtype=np.uint8)
        normal[14:42, 14:42] = 50
        abnormal = normal.copy()
        abnormal[14:42, 14:42] = 127
        self.assertEqual(describe_mask(normal, config)["synthetic_anomaly"], 0)
        abnormal_stats = describe_mask(abnormal, config)
        self.assertEqual(abnormal_stats["high_score_coverage"], 0)
        self.assertEqual(abnormal_stats["synthetic_anomaly"], 1)

    def test_invisible_border_cannot_change_any_qa_answer(self):
        config = load_config()
        first = np.full((60, 60), 12, dtype=np.uint8)
        first[14:28, 14:28] = 210
        second = first.copy()
        second[56:] = 255
        second[:, 56:] = 255
        left, right = describe_mask(first, config), describe_mask(second, config)
        self.assertEqual(left, right)
        self.assertEqual(left["visible_size"], [56, 56])
        self.assertEqual(task_answers(left, config), task_answers(right, config))
        config["label_rule"] = "blurred_peak"
        self.assertEqual(describe_mask(first, config), describe_mask(second, config))

    def test_quantized_geometry_and_exact_bin_edges_define_answers(self):
        config = load_config()
        mask = np.full((56, 56), 20, dtype=np.uint8)
        mask[28:42, 0:42] = 192
        stats = describe_mask(mask, config)
        tasks = task_answers(stats, config)
        self.assertEqual(stats["highest_quadrant"], 2)
        self.assertEqual(tasks["location"][2], "Bottom left")
        self.assertEqual(tasks["intensity"][2], "At least 192")
        self.assertEqual(tasks["shape"][2], "Elongated bounding box")
        self.assertEqual(tasks["coverage"][2], "At least 5% but below 20%")

    def test_answer_letters_are_balanced_without_fixed_semantic_positions(self):
        arranger = BalancedOptions(np.random.default_rng(0))
        answers = []
        for _ in range(30):
            options, answer = arranger.arrange("shape", ["none", "compact", "elongated"], "compact")
            self.assertEqual(options[answer], "compact")
            answers.append(answer)
        self.assertEqual(Counter(answers), {"A": 10, "B": 10, "C": 10})

    def test_generation_round_trips_labels_and_keeps_mask_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            summary = generate_dataset(output, train_count=12, val_count=8, size=60, seed=4, preview_count=6)
            seen_paths, seen_hashes = set(), set()
            for split, count in (("train", 12), ("val", 8)):
                records = read_jsonl(output / f"{split}.jsonl")
                masks = read_jsonl(output / f"{split}_masks.jsonl")
                self.assertEqual(len(records), 5 * count)
                self.assertEqual(len(masks), count)
                self.assertEqual(summary["splits"][split]["labels"], {"normal": count // 2, "abnormal": count // 2})
                for mask_record in masks:
                    path, digest = mask_record["mask"], mask_record["sha256_pixels"]
                    self.assertNotIn(path, seen_paths)
                    self.assertNotIn(digest, seen_hashes)
                    seen_paths.add(path)
                    seen_hashes.add(digest)
                    with Image.open(output / path) as image:
                        self.assertEqual(image.mode, "L")
                        self.assertEqual(image.size, (60, 60))
                        mask = np.array(image)
                    self.assertGreater(mask.max(), 0)
                    stats = describe_mask(mask, load_config())
                    self.assertEqual(stats, mask_record["statistics"])
                    answers = task_answers(stats, load_config())
                    related = [record for record in records if record["mask_id"] == mask_record["mask_id"]]
                    self.assertEqual(len(related), 5)
                    for record in related:
                        self.assertEqual(record["image"], record["mask"])
                        self.assertEqual(record["options"][record["answer"]], answers[record["task"]][2])
                        self.assertEqual(record["category"], "synthetic_score_map")
                        self.assertNotIn("synthetic_label", record["question"])
                for counts in summary["splits"][split]["answer_letter_counts"].values():
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            self.assertTrue((output / "preview.png").is_file())
            self.assertIn("not real defect", summary["disclaimer"])
            self.assertEqual(summary["visible_size"], 56)

    def test_seed_reproduces_data_and_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second"):
                generate_dataset(root / name, train_count=4, val_count=2, size=56, seed=9, preview_count=0)
            for relative in ("train.jsonl", "val.jsonl", "train_masks.jsonl", "val_masks.jsonl", "summary.json"):
                self.assertEqual((root / "first" / relative).read_bytes(), (root / "second" / relative).read_bytes())
            before = (root / "first" / "summary.json").read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(root / "first", train_count=1, val_count=0, size=56)
            self.assertEqual((root / "first" / "summary.json").read_bytes(), before)

    def test_bad_profiles_and_impossible_rules_fail_clearly(self):
        config = deepcopy(DEFAULT_CONFIG)
        config["normal"]["background_range"] = [100, 10]
        with self.assertRaisesRegex(ValueError, "background_range"):
            validate_config(config)
        config = deepcopy(DEFAULT_CONFIG)
        config["abnormal_fraction"] = 1
        config["abnormal_min_patch_mean"] = 255
        config["max_attempts"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Could not generate"):
                generate_dataset(Path(temporary) / "failed", train_count=1, val_count=0, size=56, config=config)

    def test_invalid_blurred_peak_parameters_are_rejected(self):
        for key, value in (
            ("label_rule", "other"), ("label_blur_radius", -1),
            ("label_blur_radius", float("nan")), ("label_blur_radius", float("inf")),
            ("label_blur_radius", True), ("label_blur_radius", "3"),
            ("label_peak_threshold", -1), ("label_peak_threshold", 256),
            ("label_peak_threshold", float("nan")), ("label_peak_threshold", True),
        ):
            config = {**load_config(), key: value}
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                validate_config(config)

    def test_blurred_rule_and_thresholds_are_saved_for_reproducibility(self):
        config = {**load_config(), "label_rule": "blurred_peak", "label_peak_threshold": 110,
                  "label_blur_radius": 2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            summary = generate_dataset(output, train_count=4, val_count=2, size=56, config=config, preview_count=0)
            self.assertEqual(summary["label_rule"], "blurred_peak")
            self.assertEqual(summary["label_parameters"]["label_peak_threshold"], 110)
            self.assertEqual(summary["label_parameters"]["label_blur_radius"], 2)
            for split in ("train", "val"):
                for record in read_jsonl(output / f"{split}.jsonl"):
                    self.assertEqual(record["label_rule"], "blurred_peak")
                for record in read_jsonl(output / f"{split}_masks.jsonl"):
                    self.assertEqual(record["label_rule"], "blurred_peak")
                    self.assertIn("smoothed_peak", record["statistics"])

    def test_api_defaults_stay_on_legacy_label_rule(self):
        self.assertEqual(DEFAULT_CONFIG["label_rule"], "coverage_patch_mean")
        self.assertEqual(load_config()["label_rule"], "coverage_patch_mean")

    def test_cli_defaults_to_project_config_and_accepts_an_explicit_override(self):
        default = str(Path(__file__).resolve().parents[1] / "configs" / "synthetic_masks.json")
        for arguments, expected in (([], default), (["--config", "custom.json"], "custom.json")):
            with self.subTest(arguments=arguments), \
                    patch("sys.argv", ["synthetic_masks", "--output", "unused"] + arguments), \
                    patch("mask_fusion.synthetic_masks.load_config", return_value={}) as read_config, \
                    patch("mask_fusion.synthetic_masks.generate_dataset", return_value={"splits": {}}), \
                    patch("builtins.print"):
                main()
                read_config.assert_called_once_with(expected)

    def test_checked_in_configuration_is_valid(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        config = load_config(config_dir / "synthetic_masks.json")
        self.assertEqual(config["patch_size"], 14)
        self.assertIn("anomaly", config["tasks"])
        self.assertEqual(config["label_rule"], "blurred_peak")
        self.assertEqual(load_config(config_dir / "synthetic_masks_generic.json")["label_rule"], "coverage_patch_mean")

    def test_reference_config_keeps_normal_dim_and_accepts_weak_abnormal_maps(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "synthetic_masks.json")
        # Force the deliberately weak part of the supported abnormal range to
        # verify rejection sampling does not silently discard every <128 map.
        config["abnormal"]["region_peak_range"] = [100, 120]
        config["abnormal"]["region_area_range"] = [0.15, 0.25]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            generate_dataset(output, train_count=4, val_count=0, size=384, seed=0,
                             config=config, preview_count=0)
            counts = Counter()
            for record in read_jsonl(output / "train_masks.jsonl"):
                stats, label = record["statistics"], record["synthetic_label"]
                counts[label] += 1
                if label == "normal":
                    self.assertLess(stats["maximum"], 80)
                    self.assertEqual(stats["synthetic_anomaly"], 0)
                else:
                    self.assertLess(stats["maximum"], 128)
                    self.assertGreaterEqual(stats["smoothed_peak"], config["label_peak_threshold"])
                    self.assertEqual(stats["synthetic_anomaly"], 1)
            self.assertEqual(counts, {"normal": 2, "abnormal": 2})


if __name__ == "__main__":
    unittest.main()
