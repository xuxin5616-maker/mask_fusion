"""Fixed Jet encoding and exact view geometry through the real SigLIP processor."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mask_fusion.config import FusionConfig
from mask_fusion.data import QADataset, collate_samples
from mask_fusion.loader import activate_vendor
from mask_fusion.preprocessing import prepare_image_mask, scores_to_jet_rgb
from mask_fusion.visualization import restore_rgb_views, save_input_visualization
from test_data import FakeTokenizer
from test_preprocessing import load_baseline_without_models

activate_vendor()
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


class JetPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.baseline = load_baseline_without_models()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.processor = SigLipImageProcessor()
        self.config = SimpleNamespace(
            image_aspect_ratio="square", image_grid_pinpoints=[[384, 384], [768, 384]],
            mm_patch_merge_type="spatial_unpad_add_newl", image_token_order="base_first",
            mask_fusion=FusionConfig(mask_encoder_type="siglip_jet").to_dict(),
        )
        self.baseline_patch = patch("mask_fusion.preprocessing._load_baseline", return_value=self.baseline)
        self.baseline_patch.start()
        self.addCleanup(self.baseline_patch.stop)

    def test_jet_rgb_fixed_range_endpoints_and_score_identity_across_images(self):
        scores = np.array([[0, 64, 128, 192, 255]], dtype=np.float32)
        rgb = scores_to_jet_rgb(scores)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape, (1, 5, 3))
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 127])
        np.testing.assert_array_equal(rgb[0, -1], [127, 0, 0])
        self.assertGreater(int(rgb[0, 2, 1]), int(rgb[0, 2, 0]))
        np.testing.assert_array_equal(scores_to_jet_rgb(np.array([[128.0]]))[0, 0], rgb[0, 2])
        np.testing.assert_array_equal(scores, [[0, 64, 128, 192, 255]])

    def test_invalid_padding_is_neutral_and_does_not_participate_in_range_validation(self):
        scores = np.array([[float("nan"), 100.0], [float("inf"), -1000.0]])
        valid = np.array([[False, True], [False, False]])
        result = scores_to_jet_rgb(scores, valid)
        np.testing.assert_array_equal(result[~valid], np.full((3, 3), 128))
        np.testing.assert_array_equal(result[0, 1], scores_to_jet_rgb(np.array([[100.0]]))[0, 0])
        for value in (-1.0, 256.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scores_to_jet_rgb(np.array([[value]]))

    def test_square_applies_actual_siglip_rescale_and_normalization_exactly_once(self):
        scores = np.broadcast_to(np.linspace(0, 255, 384, dtype=np.float32), (384, 384)).copy()
        result = prepare_image_mask(Image.new("RGB", (384, 384)), scores,
                                    self.processor, self.config, mode="square")
        raw = scores_to_jet_rgb(scores)
        expected = torch.from_numpy(raw.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        expected = (expected - 0.5) / 0.5
        torch.testing.assert_close(result["mask_images"], expected, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(np.asarray(restore_rgb_views(result["mask_images"], self.processor)[0]), raw)
        np.testing.assert_array_equal(result["masks"][0, 0].numpy(), scores)
        self.assertEqual(result["metadata"]["mask_image_encoding"]["processor_passes_per_view"], 1)

    def test_roi_shares_view_order_and_colorizes_after_scalar_geometry(self):
        image = Image.new("RGB", (200, 100), (25, 50, 75))
        scores = np.tile(np.linspace(0, 255, 200, dtype=np.float32), (100, 1))
        result = prepare_image_mask(image, scores, self.processor, self.config,
                                    mode="randomroi", boxes=[[20, 10, 80, 70]], seed=3)
        old_config = SimpleNamespace(**vars(self.config))
        old_config.mask_fusion = FusionConfig().to_dict()
        legacy = prepare_image_mask(image, scores, self.processor, old_config,
                                    mode="randomroi", boxes=[[20, 10, 80, 70]], seed=3)
        torch.testing.assert_close(result["images"], legacy["images"], rtol=0, atol=0)
        torch.testing.assert_close(result["masks"], legacy["masks"], rtol=0, atol=0)
        self.assertEqual(result["metadata"]["roi_boxes"], legacy["metadata"]["roi_boxes"])
        views = restore_rgb_views(result["mask_images"], self.processor)
        for i, view in enumerate(views):
            np.testing.assert_array_equal(np.asarray(view), scores_to_jet_rgb(
                result["masks"][i, 0].numpy(), result["mask_valid"][i, 0].numpy()))
        self.assertNotIn("mask_images", legacy)

    def test_anyres_padding_is_neutral_and_keeps_invalid_tokens_marked(self):
        result = prepare_image_mask(Image.new("RGB", (768, 128)),
                                    np.full((128, 768), 200, dtype=np.float32),
                                    self.processor, self.config, mode="anyres_max_9")
        self.assertFalse(result["mask_valid"].all())
        for i, view in enumerate(restore_rgb_views(result["mask_images"], self.processor)):
            invalid = ~result["mask_valid"][i, 0].numpy()
            if invalid.any():
                np.testing.assert_array_equal(np.asarray(view)[invalid], np.full((invalid.sum(), 3), 128))

    def test_dataset_collate_and_visualization_preserve_actual_jet_encoder_input(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (384, 384)).save(root / "image.png")
            Image.fromarray(np.full((384, 384), 128, dtype=np.uint8)).save(root / "mask.png")
            manifest = root / "test.jsonl"
            manifest.write_text(json.dumps({"id": "one", "image": "image.png", "mask": "mask.png",
                                            "question": "Any anomaly?"}) + "\n", encoding="utf-8")
            tokenizer = FakeTokenizer()
            tokenizer.bos_token_id = None
            dataset = QADataset(manifest, tokenizer, self.processor, self.config,
                                mode="square", training=False)
            sample = dataset[0]
            batch = collate_samples([sample], 0)
            self.assertEqual(batch["mask_images"][0].shape, (1, 3, 384, 384))
            output = save_input_visualization(sample, dataset.records[0], dataset.sources[0],
                                             self.processor, root / "visualizations", 0)
            metadata = json.loads(Path(output["metadata"]).read_text(encoding="utf-8"))
            filename = metadata["views"][0]["files"]["siglip_mask_input"]
            with Image.open(Path(output["directory"]) / filename) as source:
                np.testing.assert_array_equal(np.asarray(source), scores_to_jet_rgb(
                    np.full((384, 384), 128.0)))


if __name__ == "__main__":
    unittest.main()
