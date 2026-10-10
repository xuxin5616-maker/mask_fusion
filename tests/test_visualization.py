"""Check visual diagnostics against actual preprocessing and inference inputs."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mask_fusion.infer import main
from mask_fusion.preprocessing import prepare_image_mask
from mask_fusion.visualization import restore_rgb_views, save_input_visualization, score_heatmap
from test_data import fake_image_tokenize
from test_infer_anomaly import DecodeTokenizer, RecordingModel
from test_preprocessing import FakeProcessor, load_baseline_without_models


class NormalizingProcessor(FakeProcessor):
    def __init__(self, side=56):
        super().__init__(side)
        self.image_std = (0.5, 0.5, 0.5)
        self.rescale_factor = 1 / 255

    def preprocess(self, images, return_tensors):
        pixels = super().preprocess(images, return_tensors)["pixel_values"]
        return {"pixel_values": (pixels * self.rescale_factor - 0.5) / 0.5}


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_baseline_without_models()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processor = NormalizingProcessor()
        self.config = SimpleNamespace(image_grid_pinpoints=[[112, 56]], image_token_order="base_first",
                                      mm_patch_merge_type="spatial_unpad_add_newl")
        baseline_patch = patch("mask_fusion.preprocessing._load_baseline", return_value=self.baseline)
        baseline_patch.start()
        self.addCleanup(baseline_patch.stop)

    def make_sample(self, image, scores, mode, **kwargs):
        image.save(self.root / "image.png")
        np.save(self.root / "mask.npy", scores)
        prepared = prepare_image_mask(image, scores, self.processor, self.config, mode=mode, **kwargs)
        sample = {key: prepared[key] for key in ("images", "masks", "mask_valid", "image_size")}
        sample["preprocessing_metadata"] = prepared["metadata"]
        record = {"id": "../../duplicate-id", "image": str(self.root / "image.png"),
                  "mask": str(self.root / "mask.npy"), "category": "metal_nut"}
        return sample, record

    def test_roi_views_restore_actual_rgb_and_keep_fractional_mask_and_padding(self):
        self.processor = NormalizingProcessor(384)
        pixels = np.full((384, 384, 3), [24, 140, 251], dtype=np.uint8)
        scores = np.full((384, 384), 91.125, dtype=np.float32)
        scores[50:70, 80:95] = 300.75  # Display saturates, numerical archive must not.
        sample, record = self.make_sample(Image.fromarray(pixels), scores, "randomroi", boxes=[[-14, 0, 369, 383]])
        before = {key: sample[key].clone() for key in ("images", "masks", "mask_valid")}
        np.random.seed(172)
        rng_before = np.random.get_state()
        paths = save_input_visualization(sample, record, {"source_index": 4}, self.processor, self.root / "vis", 0)
        rng_after = np.random.get_state()
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        folder = Path(paths["directory"])
        self.assertEqual(folder.parent, (self.root / "vis").resolve())
        self.assertEqual(folder.name, "00000000")
        for key in before:
            torch.testing.assert_close(sample[key], before[key], rtol=0, atol=0)
        with np.load(paths["mask_values"], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["masks"], sample["masks"].numpy())
            np.testing.assert_array_equal(archive["mask_valid"], sample["mask_valid"].numpy())
            np.testing.assert_array_equal(archive["source_mask"], scores)
            self.assertEqual(archive["masks"].max(), 300.75)
        with Image.open(folder / "view_00_base_image.png") as base:
            np.testing.assert_array_equal(np.asarray(base), pixels)
        with Image.open(folder / "view_01_roi_image.png") as roi:
            self.assertEqual(roi.size, (384, 384))
            np.testing.assert_array_equal(np.asarray(roi)[:, :14], 0)
            np.testing.assert_array_equal(np.asarray(roi)[:, 14:], pixels[:, :370])
        with Image.open(folder / "view_01_roi_mask.png") as roi_mask:
            self.assertTrue(np.all(np.asarray(roi_mask)[:, :14] == 96))
        details = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual(details["preprocessing"]["roi_boxes"], [[-14, 0, 370, 384]])
        self.assertEqual(details["display"]["score_max"], 255)
        self.assertTrue(Path(paths["overview"]).is_file())

    def test_anyres_keeps_row_major_tiles_and_canvas_padding(self):
        pixels = np.full((31, 83, 3), 200, dtype=np.uint8)
        scores = np.full((31, 83), 91.125, dtype=np.float32)
        sample, record = self.make_sample(Image.fromarray(pixels), scores, "anyres_max_9")
        paths = save_input_visualization(sample, record, {}, self.processor, self.root / "vis", 1)
        details = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual(details["preprocessing"]["grid_patches"], [2, 1])
        self.assertEqual([view["kind"] for view in details["views"]], ["base", "tile", "tile"])
        restored = restore_rgb_views(sample["images"], self.processor)
        for index in (1, 2):
            with Image.open(Path(paths["directory"]) / f"view_{index:02d}_tile_image.png") as saved:
                np.testing.assert_array_equal(np.asarray(saved), np.asarray(restored[index]))
            self.assertFalse(sample["mask_valid"][index].all())

    def test_heatmaps_use_same_score_scale_across_images(self):
        first = np.array([[0.125, 50.5, 70.25]], dtype=np.float32)
        second = np.array([[0.125, 50.5, 250.25]], dtype=np.float32)
        one = np.asarray(score_heatmap(first, np.ones_like(first, dtype=bool), 255))
        two = np.asarray(score_heatmap(second, np.ones_like(second, dtype=bool), 255))
        np.testing.assert_array_equal(one[:, :2], two[:, :2])
        self.assertFalse(np.array_equal(one[:, 2], two[:, 2]))

    def test_inference_prepares_once_and_visualization_does_not_change_inputs(self):
        pixels = np.full((420, 512, 3), 140, dtype=np.uint8)
        scores = np.linspace(0.125, 255.125, 420 * 512, dtype=np.float32).reshape(420, 512)
        image_path, mask_path = self.root / "image.png", self.root / "mask.npy"
        Image.fromarray(pixels).save(image_path)
        np.save(mask_path, scores)
        # Repeated IDs must still yield independent, uniquely named diagnostics.
        record = {"id": "duplicate", "image": str(image_path), "mask": str(mask_path),
                  "question": "Is it defective?", "class": "metal_nut", "gt": 1}
        manifest = self.root / "qa.json"
        manifest.write_text(json.dumps([record, record]), encoding="utf-8")
        tokenizer = DecodeTokenizer()
        calls = []
        for label, extra in (("off", []), ("on", ["--visualize"]),
                             ("limited", ["--visualize", "--visualization-limit", "1"])):
            model = RecordingModel(["A", "A"], tokenizer)
            model.config = self.config
            destination = self.root / f"{label}.jsonl"
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("mask_fusion.infer.read_adapter_config", return_value={
                    "base_model": "fake", "image_mode": "randompatch", "fusion_config": {"score_scale": 255},
                }))
                stack.enter_context(patch("mask_fusion.infer.load_model", return_value=(model, tokenizer, self.processor)))
                stack.enter_context(patch("mask_fusion.data._image_tokenize", side_effect=fake_image_tokenize))
                prepare_mock = stack.enter_context(patch("mask_fusion.preprocessing.prepare_image_mask", wraps=prepare_image_mask))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main(["--adapter", "fake", "--data", str(manifest), "--output", str(destination),
                      "--device", "cpu", "--dtype", "float32", "--eval-task", "anomaly",
                      "--anomaly-prompt", "category-mcq", "--seed", "7", *extra])
                self.assertEqual(prepare_mock.call_count, 2)
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            metrics = json.loads(destination.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            calls.append(model.calls)
            self.assertEqual(len(rows), 2)
            self.assertEqual(metrics["accuracy"], 1)
            if label == "off":
                self.assertNotIn("visualization", metrics)
            else:
                count = 1 if label == "limited" else 2
                self.assertEqual(metrics["visualization"]["records"], count)
                for index in range(count):
                    self.assertEqual(Path(rows[index]["visualization"]["directory"]).name, f"{index:08d}")
                    with np.load(rows[index]["visualization"]["mask_values"]) as archive:
                        np.testing.assert_array_equal(archive["masks"], model.calls[index]["masks"][0].numpy())
                if count == 1:
                    self.assertNotIn("visualization", rows[1])
            for actual in model.calls:
                self.assertNotIn("preprocessing_metadata", actual)
        for other in calls[1:]:
            for original, visualized in zip(calls[0], other):
                for key in ("inputs", "attention_mask"):
                    torch.testing.assert_close(original[key], visualized[key], rtol=0, atol=0)
                for key in ("images", "masks", "mask_valid"):
                    torch.testing.assert_close(original[key][0], visualized[key][0], rtol=0, atol=0)

    def test_existing_visualizations_rejected_before_model_loading(self):
        directory = self.root / "run_visualizations"
        directory.mkdir()
        (directory / "existing.txt").write_text("keep", encoding="utf-8")
        with patch("mask_fusion.infer.load_model") as load_model:
            with self.assertRaisesRegex(FileExistsError, "Visualization directory is not empty"):
                main(["--adapter", "fake", "--data", "fake", "--output", str(self.root / "run.jsonl"),
                      "--visualize", "--overwrite"])
            load_model.assert_not_called()
        self.assertEqual((directory / "existing.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
