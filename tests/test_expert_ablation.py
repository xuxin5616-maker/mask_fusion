"""Verify direct-channel interventions through fusion, projector and tiny Qwen."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mask_fusion.config import FusionConfig
from mask_fusion.infer_expert_ablation import ExpertInputAblation, main, summarize_metrics
from test_integration import make_batch, make_tiny_model
from test_siglip_jet_model import jet_batch, jet_config


class ExpertAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def capture(self, model, batch):
        core = model.get_model()
        seen = {"projector": [], "attention": []}
        handles = [core.mm_projector.register_forward_pre_hook(
            lambda module, args: seen["projector"].append(args[0].detach().clone())),
            core.mask_fusion.layers[0].register_forward_hook(
                lambda module, args, output: seen["attention"].append(output.detach().clone()))]
        try:
            with torch.no_grad():
                result = model(**batch, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return result, seen

    def test_zero_replaces_full_expert_tail_for_cnn_and_jet_and_all_views_only(self):
        for config, batch in ((None, make_batch((1, 2))), (jet_config(), jet_batch((1, 2)))):
            with self.subTest(encoder="cnn" if config is None else "jet"):
                model = make_tiny_model(fusion_config=config)
                core = model.get_model()
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                ordinary, normal = self.capture(model, batch)
                hook = ExpertInputAblation(model, "zero")
                try:
                    result, replaced = self.capture(model, batch)
                    self.assertEqual(hook.calls, 1)
                    self.assertEqual(hook.views, 3)
                    self.assertEqual(hook.valid_patches, 12)
                finally:
                    hook.close()
                visual_dim = core.mask_fusion.vision_dim
                self.assertEqual(replaced["projector"][0].shape[:2], (3, 4))
                self.assertEqual(replaced["projector"][0][..., visual_dim:].count_nonzero().item(), 0)
                torch.testing.assert_close(replaced["projector"][0][..., :visual_dim],
                                           normal["projector"][0][..., :visual_dim], rtol=0, atol=0)
                torch.testing.assert_close(replaced["attention"][0], normal["attention"][0], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(result.loss))
                self.assertEqual(result.logits.shape, ordinary.logits.shape)
                _, restored = self.capture(model, batch)
                torch.testing.assert_close(restored["projector"][0], normal["projector"][0], rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)

    def test_original_hook_is_exact_identity(self):
        model = make_tiny_model()
        before, normal = self.capture(model, make_batch())
        hook = ExpertInputAblation(model, "original")
        try:
            after, seen = self.capture(model, make_batch())
        finally:
            hook.close()
        torch.testing.assert_close(after.logits, before.logits, rtol=0, atol=0)
        torch.testing.assert_close(seen["projector"][0], normal["projector"][0], rtol=0, atol=0)

    def test_random_is_reproducible_and_does_not_touch_global_rng_or_attention(self):
        def run(seed):
            model = make_tiny_model(fusion_config=jet_config())
            batch = jet_batch((1, 2))
            _, original = self.capture(model, batch)
            hook = ExpertInputAblation(model, "random", seed=seed, mean=2.0, std=0.5)
            rng = torch.get_rng_state().clone()
            try:
                _, seen = self.capture(model, batch)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            finally:
                hook.close()
            torch.testing.assert_close(seen["attention"][0], original["attention"][0], rtol=0, atol=0)
            torch.testing.assert_close(seen["projector"][0][..., :12], original["projector"][0][..., :12],
                                       rtol=0, atol=0)
            return seen["projector"][0][..., 12:]
        first, second, changed = run(7), run(7), run(8)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, changed))
        self.assertGreater(first.std().item(), 0.2)
        self.assertAlmostEqual(first.mean().item(), 2.0, delta=0.2)

    def test_padding_remains_zero_under_random_intervention(self):
        for config, batch in ((None, make_batch((1, 2))), (jet_config(), jet_batch((1, 2)))):
            with self.subTest(encoder="cnn" if config is None else "jet"):
                model = make_tiny_model(fusion_config=config)
                batch["mask_valid"][1][0] = False
                hook = ExpertInputAblation(model, "random")
                try:
                    _, seen = self.capture(model, batch)
                    self.assertEqual(hook.valid_patches, 8)
                finally:
                    hook.close()
                self.assertEqual(seen["projector"][0][1, :, 12:].count_nonzero().item(), 0)
                self.assertGreater(seen["projector"][0][0, :, 12:].count_nonzero().item(), 0)

    def test_random_distribution_uses_requested_scale(self):
        fusion = nn.Identity()
        fusion.vision_dim, fusion.expert_feature_dim, fusion.patch_size = 3, 5, 1
        fusion.config = FusionConfig()
        model = SimpleNamespace(get_model=lambda: SimpleNamespace(mask_fusion=fusion),
                                config=SimpleNamespace(mask_fusion_pretrain=False))
        hook = ExpertInputAblation(model, "random", mean=-1.0, std=2.0)
        try:
            output = fusion(torch.zeros(2, 2000, 8))
        finally:
            hook.close()
        self.assertEqual(output[..., :3].count_nonzero().item(), 0)
        self.assertAlmostEqual(output[..., 3:].mean().item(), -1.0, delta=0.05)
        self.assertAlmostEqual(output[..., 3:].std().item(), 2.0, delta=0.05)

    def test_rejects_attention_only_adapter_and_invalid_noise_options(self):
        config = FusionConfig(mode="attention", direct_to_projector=False)
        with self.assertRaisesRegex(ValueError, "no direct expert"):
            ExpertInputAblation(make_tiny_model(fusion_config=config), "zero")
        for options in ({"seed": -1}, {"std": 0}, {"std": float("inf")}, {"mean": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExpertInputAblation(make_tiny_model(), "random", **options)

    def test_cli_reuses_infer_and_writes_intervention_metadata_without_altering_config(self):
        from mask_fusion import infer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "zero.jsonl"
            metadata = {"fusion_config": FusionConfig().to_dict()}
            model = make_tiny_model()
            old_config = dict(model.config.mask_fusion)
            original_loader = infer.load_model

            def inference(arguments):
                self.assertIn("--dtype", arguments)
                loaded, _, _ = infer.load_model("base", adapter_path="adapter")
                result, seen = self.capture(loaded, make_batch((1, 2)))
                self.assertEqual(seen["projector"][0][..., 12:].count_nonzero().item(), 0)
                output.write_text("{}\n", encoding="utf-8")
                output.with_suffix(".metrics.json").write_text(json.dumps({
                    "qa_records": 1, "accuracy": 1.0, "positive_recall": 1.0, "negative_recall": None,
                    "fp": 0, "fn": 0,
                }), encoding="utf-8")

            with patch("mask_fusion.model.read_adapter_config", return_value=metadata), \
                    patch.object(infer, "load_model", return_value=(model, object(), object())) as loader, \
                    patch.object(infer, "main", side_effect=inference), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = main(["--expert-input", "zero", "--ablation-label", "cnn",
                               "--adapter", "adapter", "--data", "test.jsonl", "--output", str(output),
                               "--dtype", "bfloat16"])
                loader.assert_called_once()
                self.assertIs(infer.load_model, loader)
            self.assertIs(infer.load_model, original_loader)
            self.assertEqual(result["expert_ablation"]["expert_channels"], 10)
            self.assertEqual(result["expert_ablation"]["encoded_views"], 3)
            self.assertEqual(result["expert_ablation"]["label"], "cnn")
            self.assertEqual(model.config.mask_fusion, old_config)
            self.assertEqual(len(model.get_model().mask_fusion._forward_hooks), 0)

    def test_cli_removes_hook_on_inference_failure(self):
        from mask_fusion import infer
        model = make_tiny_model()

        def inference(arguments):
            infer.load_model("base")
            raise RuntimeError("inference sentinel")

        with patch("mask_fusion.model.read_adapter_config", return_value={"fusion_config": FusionConfig().to_dict()}), \
                patch.object(infer, "load_model", return_value=(model, None, None)), \
                patch.object(infer, "main", side_effect=inference):
            with self.assertRaisesRegex(RuntimeError, "inference sentinel"):
                main(["--expert-input", "random", "--adapter", "adapter", "--data", "data", "--output", "out"])
        self.assertEqual(len(model.get_model().mask_fusion._forward_hooks), 0)

    def test_summary_writes_csv_json_with_separate_model_and_noise_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for model in ("jet", "cnn"):
                for mode in ("zero", "random"):
                    path = root / f"{model}_{mode}.metrics.json"
                    path.write_text(json.dumps({
                        "expert_ablation": {"label": model, "expert_input": mode,
                                            "noise": {"seed": 7} if mode == "random" else None},
                        "accuracy": 0.8, "positive_recall": 0.75, "negative_recall": 0.9,
                        "fp": 2, "fn": 5,
                    }), encoding="utf-8")
                    paths.append(path)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize_metrics(paths, root / "comparison")
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[1]["noise_seed"], 7)
            self.assertTrue((root / "comparison.csv").is_file())
            self.assertEqual(len(json.loads((root / "comparison.json").read_text(encoding="utf-8"))), 4)
            with self.assertRaises(FileExistsError):
                summarize_metrics(paths, root / "comparison")

    def test_full_inference_cli_scores_all_records_for_cnn_and_jet(self):
        from mask_fusion import infer

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def decode(self, tokens, skip_special_tokens=True):
                return "A"

        for config, batch in ((None, make_batch((2,))), (jet_config(), jet_batch((2,)))):
            for mode in ("zero", "random"):
                with self.subTest(encoder="cnn" if config is None else "jet", mode=mode), \
                        tempfile.TemporaryDirectory() as directory:
                    model = make_tiny_model(fusion_config=config)
                    metadata = {"base_model": "offline-base", "image_mode": "square",
                                "fusion_config": model.config.mask_fusion}
                    record = {"id": "case", "image": "image.png", "mask": "mask.png",
                              "question": "Any defects?", "options": None, "category": "screw", "gt": 1}

                    class Dataset:
                        def __init__(self, *args, **kwargs):
                            self.records = [dict(record)]
                            self.sources = [{"source_file": "test.jsonl", "source_index": 0}]
                            self.skipped_records = []

                        def __len__(self):
                            return 1

                        def __getitem__(self, index):
                            sample = {"input_ids": batch["input_ids"][0], "images": batch["images"][0],
                                      "masks": batch["masks"][0], "mask_valid": batch["mask_valid"][0],
                                      "image_size": (4, 4)}
                            if "mask_images" in batch:
                                sample["mask_images"] = batch["mask_images"][0]
                            return sample

                    output = Path(directory) / "predictions.jsonl"
                    with patch("mask_fusion.model.read_adapter_config", return_value=metadata), \
                            patch.object(infer, "read_adapter_config", return_value=metadata), \
                            patch.object(infer, "load_model", return_value=(model, Tokenizer(), object())), \
                            patch.object(infer, "QADataset", Dataset), \
                            contextlib.redirect_stdout(io.StringIO()):
                        result = main(["--expert-input", mode, "--adapter", "adapter", "--data", "test.jsonl",
                                       "--output", str(output), "--device", "cpu", "--dtype", "float32",
                                       "--eval-task", "anomaly", "--anomaly-prompt", "category-mcq",
                                       "--max-new-tokens", "1"])
                    prediction = json.loads(output.read_text(encoding="utf-8").strip())
                    self.assertEqual(prediction["anomaly_prediction"], 1)
                    self.assertEqual(result["qa_records"], 1)
                    self.assertEqual(result["expert_ablation"]["encoded_views"], 2)
                    self.assertEqual(result["expert_ablation"]["expert_input"], mode)
                    self.assertEqual(model.config.mask_fusion["mode"], "both")
                    self.assertEqual(len(model.get_model().mask_fusion._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
