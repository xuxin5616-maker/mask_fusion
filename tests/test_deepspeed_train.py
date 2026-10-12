"""CPU contracts for ZeRO integration, not a claim of running NCCL/DeepSpeed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mask_fusion import deepspeed_backend as backend, train
from test_data import FakeTokenizer, TinyTrainingDataset, TinyTrainingModel


def context(rank=0):
    return SimpleNamespace(rank=rank, world_size=2, local_rank=rank,
                           device=torch.device("cpu"), is_main=rank == 0)


class ZeroModel(TinyTrainingModel):
    def __init__(self):
        super().__init__()
        self.config.hidden_size = 8

    def get_model(self):
        return self.core


class FakeEngine(torch.nn.Module):
    """Uses real CPU autograd/AdamW but simulates DeepSpeed's control API only."""
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.module = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.skipped_steps = 0
        self.boundaries = []
        self.raw_losses = []
        self.backward_losses = []
        self.norm = None
        self.optimizer.zero_grad()

    def forward(self, **kwargs):
        output = self.module(**kwargs)
        self.raw_losses.append(float(output.loss.detach()))
        return output

    def set_gradient_accumulation_boundary(self, boundary):
        self.boundaries.append(boundary)

    def is_gradient_accumulation_boundary(self):
        return self.boundaries[-1]

    def backward(self, loss, scale_wrt_gas=True):
        if scale_wrt_gas:
            raise AssertionError("Loss must not be divided by accumulation size twice")
        self.backward_losses.append(float(loss.detach()))
        loss.backward()

    def step(self):
        if self.is_gradient_accumulation_boundary():
            self.norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def get_global_grad_norm(self):
        return self.norm

    def save_checkpoint(self, folder, *, tag, exclude_frozen_parameters, save_latest, client_state):
        assert exclude_frozen_parameters and not save_latest
        target = Path(folder) / tag
        target.mkdir(parents=True, exist_ok=True)
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.lr_scheduler.state_dict(),
                    "client_state": client_state}, target / "fake_rank0.pt")

    def load_checkpoint(self, folder, *, tag, load_module_strict, load_optimizer_states, load_lr_scheduler_states):
        assert not load_module_strict and load_optimizer_states and load_lr_scheduler_states
        target = Path(folder) / tag / "fake_rank0.pt"
        state = torch.load(target, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["scheduler"])
        return str(target), state["client_state"]


class ConfigTests(unittest.TestCase):
    def args(self, path=None):
        return SimpleNamespace(deepspeed_config=path, batch_size=1, grad_accum=4, max_grad_norm=1.0)

    def test_auto_config_matches_cli_and_hidden_size(self):
        config = backend.resolve_config(self.args(), "bfloat16", 2, 3584)
        self.assertEqual(config["train_batch_size"], 8)
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertTrue(config["bf16"]["enabled"])
        self.assertFalse(config["fp16"]["enabled"])
        self.assertEqual(config["zero_optimization"]["reduce_bucket_size"], 3584**2)
        self.assertNotIn('"auto"', json.dumps(config))

    def test_conflicts_wrong_stage_and_offload_fail_before_training(self):
        for change, message in (
            ({"gradient_accumulation_steps": 5}, "conflicts"),
            ({"zero_optimization": {"stage": 2}}, "stage=3"),
            ({"zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu"}}}, "not supported"),
            ({"scheduler": {}}, "optimizer/scheduler"),
            ({"unknown": "auto"}, "Unresolved"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                config = json.loads(backend.DEFAULT_CONFIG.read_text())
                config.update(change)
                source = Path(temporary) / "zero.json"
                source.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, message):
                    backend.resolve_config(self.args(str(source)), "bfloat16", 2, 8)

    def test_optional_backend_help_does_not_import_deepspeed(self):
        with patch.object(backend, "require_deepspeed", side_effect=AssertionError("help must not import")), \
             contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaises(SystemExit) as error:
                train.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--deepspeed-config", log.getvalue())


class BackendContractTests(unittest.TestCase):
    def test_export_loads_through_real_adapter_loader_with_hidden_norm_disabled(self):
        from mask_fusion.config import FusionConfig
        from mask_fusion.model import load_adapter, save_adapter
        from test_integration import make_batch, make_tiny_model

        config = FusionConfig(mask_hidden_size=8, expert_dim=16, position_dim=8,
                              position_hidden_size=16, attention_dim=16, num_heads=4,
                              normalize_hidden=False)
        model = make_tiny_model(fusion_config=config)
        # Simulate native mixed-precision parameter values without a GPU engine.
        model.get_model().mask_fusion.bfloat16()
        model.get_model().mm_projector.bfloat16()
        ds = SimpleNamespace(zero=SimpleNamespace(GatheredParameters=lambda *a, **k: contextlib.nullcontext()))
        with patch.object(backend, "require_deepspeed", return_value=ds):
            weights = backend.collect_adapter_state(SimpleNamespace(module=model), context())
        self.assertFalse(any("hidden_norm" in key for key in weights))
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(model, directory, "offline-tiny-test-backbone", weights=weights)
            restored = make_tiny_model(fusion_config=config)
            load_adapter(restored, directory)
        model.float()
        batch = make_batch()
        with torch.no_grad():
            expected = model(**batch, use_cache=False).logits
            actual = restored(**batch, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_step_handles_overflow_without_double_optimizer_step(self):
        engine = Mock(skipped_steps=0, optimizer=SimpleNamespace(cur_scale=8.0))
        engine.is_gradient_accumulation_boundary.return_value = True
        engine.get_global_grad_norm.return_value = None

        def skip():
            engine.skipped_steps += 1
            engine.optimizer.cur_scale /= 2

        engine.step.side_effect = skip
        updated, _, before, after = backend.engine_step(engine)
        self.assertFalse(updated)
        self.assertEqual((before, after), (8.0, 4.0))
        engine.step.assert_called_once()

    def test_adapter_gather_keeps_frozen_fusion_branches_but_excludes_decoder(self):
        model = ZeroModel()
        model.core.mask_fusion.requires_grad_(False)
        ds = SimpleNamespace(zero=SimpleNamespace(GatheredParameters=Mock(side_effect=lambda *a, **k: contextlib.nullcontext())))
        with patch.object(backend, "require_deepspeed", return_value=ds):
            weights = backend.collect_adapter_state(SimpleNamespace(module=model), context())
            other = backend.collect_adapter_state(SimpleNamespace(module=model), context(1))
        self.assertEqual(other, {})
        self.assertEqual(set(weights), {"mask_fusion.weight", "mask_fusion.bias", "mm_projector.weight", "mm_projector.bias"})
        selected = ds.zero.GatheredParameters.call_args.args[0]
        self.assertFalse(any(parameter is model.decoder.weight for parameter in selected))
        self.assertTrue(all(value.dtype == torch.float32 for value in weights.values()))

    def test_zero_init_precedes_engine_and_registers_image_newline(self):
        model = ZeroModel()
        model.core.image_newline = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
        model.register_buffer("positions", torch.arange(3))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        events = []

        def partition(**kwargs):
            self.assertIs(kwargs["module"], model)
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.bfloat16)
                parameter.ds_id, parameter.ds_numel = 0, parameter.numel()
                parameter.ds_tensor = parameter.detach().flatten()[:1]
            events.append("partition")

        engine = SimpleNamespace(module=model, zero_optimization_stage=lambda: 3)

        def initialize(**kwargs):
            self.assertEqual(events, ["partition"])
            events.append("engine")
            return engine, optimizer, None, scheduler

        ds = SimpleNamespace(init_distributed=Mock(), initialize=initialize,
                             zero=SimpleNamespace(Init=partition, register_external_parameter=Mock()))
        with patch.object(backend, "require_deepspeed", return_value=ds), \
             patch("torch.distributed.get_rank", return_value=1):
            result = backend.initialize_engine(model, optimizer, scheduler, {}, "cpu", torch.bfloat16)
        self.assertIs(result[0], engine)
        ds.zero.register_external_parameter.assert_called_once_with(model, model.core.image_newline)
        self.assertEqual(model.positions.dtype, torch.long)

    def test_incompatible_resume_rejected_before_engine_restore(self):
        with patch("mask_fusion.train._load_training_state", return_value={"signature": {"distributed": "fsdp"}}), \
             patch("torch.distributed.broadcast_object_list"):
            with self.assertRaisesRegex(RuntimeError, "FSDP"):
                backend.load_resume_metadata("unused", context())


class ZeroTrainingLoopTests(unittest.TestCase):
    def test_partial_group_resume_and_eta_match_uninterrupted_training(self):
        engines = []

        def load_model(base, *, adapter_path=None, **kwargs):
            model = ZeroModel()
            if adapter_path:
                model.load_state_dict(torch.load(Path(adapter_path) / "toy_adapter.pt", weights_only=True))
            return model, FakeTokenizer(), object()

        def initialize(model, optimizer, scheduler, *args):
            engine = FakeEngine(model, optimizer, scheduler)
            engines.append(engine)
            return engine, optimizer, scheduler

        def save_adapter(model, output, base_model, tokenizer=None, weights=None):
            torch.save(model.state_dict(), Path(output) / "toy_adapter.pt")

        original_rng = train._rng_state
        ds = SimpleNamespace(zero=SimpleNamespace(GatheredParameters=lambda *a, **k: contextlib.nullcontext()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.jsonl"
            source.write_text("{}\n")
            common = ["--base-model", str(root / "base"), "--data", str(source),
                      "--epochs", "2", "--batch-size", "1", "--grad-accum", "2",
                      "--device", "cpu", "--dtype", "float32", "--seed", "19",
                      "--lr", "0.01", "--projector-lr", "0.02", "--save-every-steps", "1",
                      "--distributed", "deepspeed", "--log-every", "1"]
            with patch("mask_fusion.train.QADataset", TinyTrainingDataset), \
                 patch("mask_fusion.model.load_model", side_effect=load_model), \
                 patch("mask_fusion.model.configure_trainable"), \
                 patch("mask_fusion.model.save_adapter", side_effect=save_adapter), \
                 patch.object(backend, "initialize_engine", side_effect=initialize), \
                 patch.object(backend, "require_deepspeed", return_value=ds), \
                 patch("mask_fusion.train._all_ranks_true", side_effect=lambda value, ctx: bool(value)), \
                 patch("mask_fusion.train._mean_across_ranks", side_effect=lambda value, ctx: value), \
                 patch("mask_fusion.train._rng_state", side_effect=lambda ctx: original_rng()), \
                 patch("mask_fusion.distributed.gather_rank_rng", side_effect=lambda value: [value, value]), \
                 patch("mask_fusion.distributed.rank_rng", side_effect=lambda states: states[0]), \
                 patch("torch.distributed.broadcast_object_list"), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                train._run_training(train.parse_args(common + ["--output", str(root / "full")]), context())
                train._run_training(train.parse_args(common + ["--output", str(root / "resumed"), "--resume",
                                    str(root / "full" / "checkpoint-step-1")]), context())
            full = torch.load(root / "full" / "toy_adapter.pt", weights_only=True)
            resumed = torch.load(root / "resumed" / "toy_adapter.pt", weights_only=True)
            for name in full:
                self.assertTrue(torch.equal(full[name], resumed[name]), name)
            self.assertEqual(engines[0].boundaries, [False, True, True, False, True, True])
            self.assertEqual(engines[1].boundaries, [True, False, True, True])
            for raw, backward, denominator in zip(engines[0].raw_losses, engines[0].backward_losses, [2, 2, 1, 2, 2, 1]):
                self.assertAlmostEqual(backward, raw / denominator)
            events = [json.loads(line) for line in log.getvalue().splitlines()]
            completed = [event for event in events if event["event"] == "train_complete"]
            self.assertEqual([event["steps"] for event in completed], [4, 4])
            self.assertEqual([event["eta"] for event in completed], ["00:00:00", "00:00:00"])
            self.assertTrue(all("eta" in event for event in events if event["event"] == "train_step"))
            self.assertTrue((root / "full" / "deepspeed_config.json").exists())


if __name__ == "__main__":
    unittest.main()
