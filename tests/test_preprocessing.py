"""CPU geometry tests against the actual bundled baseline and a weightless processor."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from mask_fusion.preprocessing import load_mask, prepare_image_mask


RESAMPLE = getattr(Image, "Resampling", Image)


class FakeProcessor:
    """SigLIP's resize-only geometry, without weights or RGB normalization."""

    def __init__(self, side=384):
        self.size = (side, side)
        self.crop_size = {"height": side, "width": side}
        self.image_mean = (0.5, 0.5, 0.5)

    def preprocess(self, images, return_tensors):
        self.assert_tensor_format(return_tensors)
        if isinstance(images, Image.Image):
            images = [images]
        arrays = [
            np.asarray(image.convert("RGB").resize(self.size, RESAMPLE.BILINEAR), dtype=np.float32).copy()
            for image in images
        ]
        return {"pixel_values": torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)}

    @staticmethod
    def assert_tensor_format(value):
        if value != "pt":
            raise ValueError(value)


def load_baseline_without_models():
    """Execute real mm_utils; stub only imports unrelated to image geometry."""
    source = Path(__file__).resolve().parents[1] / "vendor/llava/llava/mm_utils.py"
    spec = importlib.util.spec_from_file_location("_fusion_test_baseline_mm_utils", source)
    module = importlib.util.module_from_spec(spec)
    transformers = ModuleType("transformers")
    transformers.StoppingCriteria = object
    constants = ModuleType("llava.constants")
    constants.IMAGE_TOKEN_INDEX = -200
    replacements = {"transformers": transformers, "llava.constants": constants}
    if importlib.util.find_spec("cv2") is None:
        # Explicit-box ROI and random-patch tests do not call OpenCV.
        replacements["cv2"] = ModuleType("cv2")
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_baseline_without_models()

    def setUp(self):
        self.processor = FakeProcessor()
        self.config = SimpleNamespace(
            image_aspect_ratio="square", image_grid_pinpoints=[[384, 384]],
            mm_patch_merge_type="spatial_unpad_add_newl", image_token_order="base_first",
        )
        self.baseline_patch = patch("mask_fusion.preprocessing._load_baseline", return_value=self.baseline)
        self.baseline_patch.start()
        self.addCleanup(self.baseline_patch.stop)

    def prepare(self, image, scores, **kwargs):
        return prepare_image_mask(image, scores, self.processor, self.config, **kwargs)

    def test_siglip_windows_exclude_last_six_pixels_and_keep_row_major_order(self):
        scores = np.zeros((384, 384), dtype=np.float32)
        scores[0:14, 14:28] = 3.125
        scores[14:28, 0:14] = 7.375
        scores[378:, :] = 1000
        scores[:, 378:] = 1000
        result = self.prepare(Image.new("RGB", (384, 384)), scores, mode="square")
        self.assertEqual(tuple(result["masks"].shape), (1, 1, 384, 384))
        self.assertEqual(result["masks"].dtype, torch.float32)
        self.assertEqual(result["mask_valid"].dtype, torch.bool)
        mean = F.avg_pool2d(result["masks"], 14, 14).flatten()
        maximum = F.max_pool2d(result["masks"], 14, 14).flatten()
        self.assertEqual(mean.numel(), 729)
        self.assertEqual(mean[1].item(), 3.125)
        self.assertEqual(mean[27].item(), 7.375)
        self.assertEqual(maximum.max().item(), 7.375)
        self.assertEqual(maximum[-1].item(), 0)
        self.assertEqual(F.adaptive_max_pool2d(result["masks"], (27, 27))[0, 0, -1, -1].item(), 1000)
        self.assertTrue(result["mask_valid"].all())

    def test_anyres_rectangular_ceil_resize_and_asymmetric_padding(self):
        self.processor = FakeProcessor(56)
        self.config.image_grid_pinpoints = [[56, 56]]
        image = Image.new("RGB", (83, 31), (120, 120, 120))
        result = self.prepare(image, np.full((31, 83), 5.25, dtype=np.float32))
        self.assertEqual(tuple(result["images"].shape), (2, 3, 56, 56))
        self.assertEqual(result["metadata"]["content_resized_size"], [56, 21])
        self.assertEqual(result["metadata"]["padding_ltrb"], [0, 17, 0, 18])
        expected_valid = torch.zeros((56, 56), dtype=torch.bool)
        expected_valid[17:38] = True
        self.assertTrue(torch.equal(result["mask_valid"][1, 0], expected_valid))
        self.assertTrue(torch.equal(result["images"][1, 0] > 0, expected_valid))
        self.assertTrue(torch.all(result["masks"][1, 0][expected_valid] == 5.25))
        self.assertTrue(torch.all(result["masks"][1, 0][~expected_valid] == 0))
        self.assertTrue(result["mask_valid"][0].all())
        self.assertEqual(self.config.image_aspect_ratio, "square")
        json.dumps(result["metadata"])

    def test_anyres_tiles_use_the_actual_row_major_image_order(self):
        self.processor = FakeProcessor(56)
        self.config.image_grid_pinpoints = [[112, 56]]
        pixels = np.full((56, 112, 3), 10, dtype=np.uint8)
        pixels[:, 56:] = 200
        scores = np.full((56, 112), 2.25, dtype=np.float32)
        scores[:, 56:] = 9.75
        result = self.prepare(Image.fromarray(pixels), scores)
        self.assertEqual(result["metadata"]["grid_patches"], [2, 1])
        self.assertEqual(result["metadata"]["views"][1]["crop_box"], [0, 0, 56, 56])
        self.assertEqual(result["metadata"]["views"][2]["crop_box"], [56, 0, 112, 56])
        self.assertTrue(torch.all(result["images"][1] == 10))
        self.assertTrue(torch.all(result["images"][2] == 200))
        self.assertTrue(torch.all(result["masks"][1] == 2.25))
        self.assertTrue(torch.all(result["masks"][2] == 9.75))

    def test_inclusive_roi_box_and_raw_fractional_scores(self):
        rows, columns = np.indices((384, 768))
        scores = (columns + rows * 0.25 + 0.125).astype(np.float32)
        pixels = np.zeros((384, 768, 3), dtype=np.uint8)
        pixels[:, 100:484, 0] = 200
        result = self.prepare(
            Image.fromarray(pixels), scores, mode="randomroi", boxes=[[100, 0, 483, 383]],
        )
        self.assertEqual(result["metadata"]["roi_boxes"], [[100, 0, 484, 384]])
        self.assertTrue(torch.equal(result["masks"][1, 0], torch.from_numpy(scores[:, 100:484])))
        self.assertTrue(torch.all(result["images"][1, 0] == 200))
        self.assertTrue(result["mask_valid"].all())
        self.assertGreater(result["masks"].max().item(), 255)

    def test_outside_roi_padding_is_invalid_and_zero(self):
        result = self.prepare(
            Image.new("RGB", (384, 384), "white"), np.full((384, 384), 2.5, np.float32),
            mode="randomroi", boxes=[[-14, 0, 369, 383]],
        )
        self.assertFalse(result["mask_valid"][1, 0, :, :14].any())
        self.assertTrue(result["mask_valid"][1, 0, :, 14:].all())
        self.assertTrue(torch.all(result["masks"][1, 0, :, :14] == 0))
        self.assertTrue(torch.all(result["masks"][1, 0, :, 14:] == 2.5))
        self.assertTrue(torch.equal(result["images"][1, 0] > 0, result["mask_valid"][1, 0]))

    def test_randompatch_seed_reuses_boxes_and_restores_numpy_rng(self):
        self.processor = FakeProcessor(28)
        self.config.image_grid_pinpoints = [[28, 28]]
        scores = np.broadcast_to(np.arange(512, dtype=np.float32), (420, 512)).copy() + 0.25
        image = Image.new("RGB", (512, 420))
        np.random.seed(391)
        state = np.random.get_state()
        first = self.prepare(image, scores, mode="randompatch", seed=17)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        second = self.prepare(image, scores, mode="randompatch", seed=17)
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertTrue(torch.equal(first["images"], second["images"]))
        self.assertTrue(torch.equal(first["masks"], second["masks"]))
        self.assertEqual(len(first["metadata"]["roi_boxes"]) + 1, first["masks"].shape[0])
        for index, (x0, y0, x1, y1) in enumerate(first["metadata"]["roi_boxes"], start=1):
            self.assertEqual((x1 - x0, y1 - y0), (384, 384))
            # A linear ramp's mean encodes the exact selected crop location.
            self.assertAlmostEqual(first["masks"][index].mean().item(), x0 + 191.75, places=4)
        json.dumps(first["metadata"])

    def test_randomroi_selection_does_not_binarize_fusion_scores(self):
        if not hasattr(self.baseline.cv2, "findContours"):
            self.skipTest("OpenCV is required only for automatic contour-based ROI selection")
        scores = np.full((384, 384), 0.125, dtype=np.float32)
        scores[120:160, 120:160] = 3.75
        original = scores.copy()
        result = self.prepare(Image.new("RGB", (384, 384)), scores, mode="randomroi")
        np.testing.assert_array_equal(scores, original)
        np.testing.assert_array_equal(result["masks"][0, 0].numpy(), original)
        self.assertGreater(len(result["metadata"]["roi_boxes"]), 0)
        self.assertEqual(result["masks"].max().item(), 3.75)

    def test_different_mask_resolution_is_explicit_and_keeps_float_scale(self):
        self.processor = FakeProcessor(28)
        mask = np.array([[0.125, 1.625], [8.125, 16.625]], dtype=np.float32)
        result = self.prepare(Image.new("RGB", (28, 28)), mask, mode="square")
        self.assertEqual(result["metadata"]["mask_original_size"], [2, 2])
        self.assertTrue(result["metadata"]["mask_resized_to_image"])
        self.assertEqual(result["masks"][0, 0, 0, 0].item(), 0.125)
        self.assertEqual(result["masks"][0, 0, -1, -1].item(), 16.625)
        self.assertTrue(torch.any(result["masks"] != result["masks"].round()))

    def test_rejects_bad_masks_and_center_crop_processors(self):
        image = Image.new("RGB", (8, 8))
        for invalid in [np.zeros((8, 8, 3)), np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-0.01]])]:
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    self.prepare(image, invalid, mode="square")
        self.processor.do_center_crop = True
        with self.assertRaisesRegex(ValueError, "center cropping"):
            self.prepare(image, np.ones((8, 8)), mode="square")


class LoadMaskTests(unittest.TestCase):
    def test_scalar_image_and_numpy_files_keep_original_units(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scores = np.array([[0.125, 2.75], [256.125, 500.5]], dtype=np.float32)
            np.save(directory / "scores.npy", scores)
            np.savez(directory / "scores.npz", anomaly_map=scores)
            Image.fromarray(scores).save(directory / "scores.tiff")
            for filename in ["scores.npy", "scores.npz", "scores.tiff"]:
                np.testing.assert_array_equal(load_mask(directory / filename), scores)
            integers = np.array([[0, 17], [1000, 50000]], dtype=np.uint16)
            Image.fromarray(integers).save(directory / "scores.png")
            np.testing.assert_array_equal(load_mask(directory / "scores.png"), integers.astype(np.float32))

    def test_rejects_rgb_heatmaps_invalid_archives_and_invalid_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            Image.new("RGB", (8, 8)).save(directory / "heatmap.png")
            np.savez(directory / "missing.npz", unrelated=np.zeros((2, 2)))
            np.save(directory / "nan.npy", np.array([[np.nan]]))
            np.save(directory / "negative.npy", np.array([[-1.0]]))
            for filename in ["heatmap.png", "missing.npz", "nan.npy", "negative.npy"]:
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    load_mask(directory / filename)


if __name__ == "__main__":
    unittest.main()
