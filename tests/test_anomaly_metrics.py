"""Behavioral tests for explicit defect judgments; no models or GPU required."""

import json
import unittest

from mask_fusion.anomaly_metrics import (
    extract_anomaly_prediction,
    summarize_anomaly_predictions,
    validate_anomaly_target,
)


class AnomalyTargetTests(unittest.TestCase):
    def test_explicit_integer_targets_are_required(self):
        self.assertEqual(validate_anomaly_target({"gt": 0}), 0)
        self.assertEqual(validate_anomaly_target({"gt": 1}), 1)
        for value in (None, True, False, 0.0, 1.0, "0", "1", -1, 2, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer gt"):
                validate_anomaly_target({"gt": value})

    def test_labels_are_not_inferred_from_answers_paths_or_scores(self):
        record = {"answer": "Yes", "prediction": "No", "image": "good/001.png", "musc_scores": 0.9}
        with self.assertRaisesRegex(ValueError, "explicit integer gt"):
            validate_anomaly_target(record)
        record["gt"] = 1
        self.assertEqual(validate_anomaly_target(record), 1)


class AnomalyParsingTests(unittest.TestCase):
    def test_explicit_answers_and_descriptions(self):
        for text, expected in (
            ("Yes", 1), (" no ", 0), ("YES. A scratch is visible.", 1),
            ("No, the surface is intact.", 0), ("Yes\nThere is a dent.", 1),
            ("No\r\nNo defects are visible.", 0), ("Yes: the upper edge is chipped.", 1),
            ("No; the part looks normal.", 0), ("Yes，there is a crack.", 1),
            ("No。The surface is intact.", 0), ("Yes \t, a scratch is visible.", 1),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_anomaly_prediction(text), expected)

    def test_simple_paired_markdown_preserves_the_judgment(self):
        for text, expected in (
            ("**Yes**. There is a scratch.", 1), ("**No.** The surface is intact.", 0),
            ("*Yes*, the edge is chipped.", 1), ("_No_\nNothing is damaged.", 0),
            ("__YES__", 1), ("`no`: the part looks normal.", 0),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_anomaly_prediction(text), expected)

    def test_does_not_search_prose_or_guess_options(self):
        for text in (
            "", "There is a defect. Yes.", "Answer: Yes", "I think no.",
            "No doubt there is a scratch.", "No defect detected.", "Noise is visible.",
            "Yesandno", "Yes and no", "Yes or no", "Yes/no", "Yes, and no.",
            "No, or yes.", "Yesterday was fine.", "Nobody can tell.",
            "A", "B", "A. Yes", "B. No", "**Yes", "Yes**", "*No**",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_anomaly_prediction(text))


class AnomalySummaryTests(unittest.TestCase):
    def test_confusion_and_unparsed_rows_use_all_target_denominators(self):
        rows = [
            {"gt": 1, "anomaly_prediction": 1, "category": "large", "correct": False},
            {"gt": 1, "anomaly_prediction": 0, "category": "large"},
            {"gt": 1, "anomaly_prediction": None, "category": "large"},
            {"gt": 0, "anomaly_prediction": 0, "category": "large"},
            {"gt": 0, "anomaly_prediction": 1, "category": "small"},
            {"gt": 0, "anomaly_prediction": None, "category": "small", "correct": True},
            {"anomaly_prediction": 1, "category": "unscored"},
            {"gt": None, "anomaly_prediction": None, "category": "unscored"},
        ]
        report = summarize_anomaly_predictions(rows)
        self.assertEqual((report["qa_records"], report["scored_anomaly"]), (8, 6))
        self.assertEqual({name: report[name] for name in ("tp", "tn", "fp", "fn")},
                         {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(report["unparsed_positive"], 1)
        self.assertEqual(report["unparsed_negative"], 1)
        self.assertEqual(report["unparseable_anomaly"], 2)
        for name in ("accuracy", "positive_recall", "negative_recall", "balanced_accuracy"):
            self.assertAlmostEqual(report[name], 1 / 3)
        self.assertEqual(report["macro_category_accuracy"], 0.25)
        self.assertEqual(report["by_category"]["large"]["count"], 4)
        self.assertEqual(report["by_category"]["large"]["accuracy"], 0.5)
        self.assertNotIn("unscored", report["by_category"])
        self.assertNotIn("auroc", report)
        json.dumps(report, allow_nan=False)

    def test_one_class_has_no_balanced_accuracy_or_missing_class_recall(self):
        for target, available, missing in ((1, "positive_recall", "negative_recall"),
                                           (0, "negative_recall", "positive_recall")):
            with self.subTest(target=target):
                report = summarize_anomaly_predictions([
                    {"gt": target, "anomaly_prediction": target},
                    {"gt": target, "anomaly_prediction": None},
                ])
                self.assertEqual(report["accuracy"], 0.5)
                self.assertEqual(report[available], 0.5)
                self.assertIsNone(report[missing])
                self.assertIsNone(report["balanced_accuracy"])
                self.assertEqual(report["by_category"]["__unspecified__"]["count"], 2)

    def test_all_unparsed_answers_are_errors_not_excluded_samples(self):
        report = summarize_anomaly_predictions([{"gt": 0}, {"gt": 1, "anomaly_prediction": None}])
        self.assertEqual(report["scored_anomaly"], 2)
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(report["balanced_accuracy"], 0)
        self.assertEqual(report["unparseable_anomaly"], 2)

    def test_empty_or_unlabeled_input_has_no_accuracy(self):
        for rows in ([], [{"gt": None}], [{"prediction": "Yes"}]):
            with self.subTest(rows=rows):
                report = summarize_anomaly_predictions(rows)
                self.assertEqual(report["qa_records"], len(rows))
                self.assertEqual(report["scored_anomaly"], 0)
                self.assertEqual(report["unparseable_anomaly"], 0)
                self.assertIsNone(report["accuracy"])
                self.assertIsNone(report["macro_category_accuracy"])
                self.assertIsNone(report["balanced_accuracy"])
                self.assertEqual(report["by_category"], {})

    def test_summary_rejects_invalid_targets_and_prediction_types(self):
        for value in (True, 1.0, "1", 2):
            with self.subTest(target=value), self.assertRaisesRegex(ValueError, "integer gt"):
                summarize_anomaly_predictions([{"gt": value, "anomaly_prediction": 1}])
            with self.subTest(prediction=value), self.assertRaisesRegex(ValueError, "anomaly_prediction"):
                summarize_anomaly_predictions([{"gt": 1, "anomaly_prediction": value}])

    def test_repeated_images_are_separate_qa_judgments(self):
        rows = [{"image": "one.png", "gt": 1, "anomaly_prediction": prediction}
                for prediction in (1, 1, 0)]
        report = summarize_anomaly_predictions(rows)
        self.assertEqual(report["scored_anomaly"], 3)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
