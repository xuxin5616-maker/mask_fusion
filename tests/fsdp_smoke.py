"""Real two-process FSDP step/export/resume, using tiny Qwen and fake vision.

CUDA (the actual deployment check; no downloads or model weights required)::

    torchrun --standalone --nproc-per-node=2 tests/fsdp_smoke.py --output /tmp/fusion-fsdp-smoke

Optional CPU/Gloo development check on PyTorch 2.5, not CUDA certification::

    torchrun --standalone --nproc-per-node=2 tests/fsdp_smoke.py --cpu --output /tmp/fusion-fsdp-cpu

The output directory must be empty. The GPU path calls the production wrapper;
the CPU path calls its same internal wrapping implementation with CPU device_id
and sync_module_states=False. Both use real FULL_SHARD, not a mocked wrapper.
"""

import argparse
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist

from mask_fusion import distributed as distributed_helpers
from mask_fusion.model import configure_trainable, load_adapter, save_adapter
from test_integration import make_batch, make_tiny_model


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def all_assert(condition, message, device):
    good = torch.tensor(int(bool(condition)), device=device)
    dist.all_reduce(good, op=dist.ReduceOp.MIN)
    if not good.item():
        raise AssertionError(message)


def rank_zero_action(action):
    """Propagate rank-zero filesystem/assertion failures before further collectives."""
    error = [None]
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as exception:
            error[0] = f"{type(exception).__name__}: {exception}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise AssertionError(error[0])


def make_model(cpu):
    model = make_tiny_model().to(dtype=torch.float32 if cpu else torch.bfloat16)
    configure_trainable(model)  # restore FP32 fusion/projector masters
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def wrap(model, device, cpu):
    if cpu:
        return distributed_helpers._wrap_model_impl(model, device, torch.float32, sync_module_states=False)
    return distributed_helpers.wrap_model(model, device, torch.bfloat16)


def make_optimizer(model):
    fusion, projector = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".mask_fusion." in name:
                fusion.append(parameter)
            elif ".mm_projector." in name:
                projector.append(parameter)
            else:
                raise AssertionError(f"Unexpected trainable parameter: {name}")
    return torch.optim.AdamW([{"params": fusion, "lr": 0.001}, {"params": projector, "lr": 0.0003}])


def autocast(device, cpu):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=not cpu)


def cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_snapshot(item) for item in value]
    return value


def train_step(model, optimizer, scheduler, batch, device, cpu, micro_steps=2):
    model.train()
    distributed_helpers.unwrap_model(model).get_vision_tower().eval()
    optimizer.zero_grad(set_to_none=True)
    # Same synchronization strategy as training: reduce every microbatch and
    # accumulate its shard; the second update below is a partial group of one.
    for _ in range(micro_steps):
        with autocast(device, cpu):
            loss = model(**batch, use_cache=False).loss / micro_steps
        all_assert(torch.isfinite(loss).item(), "Nonfinite distributed loss", device)
        loss.backward()
    gradient_checks = (
        "mask_fusion.mask_encoder.patch_conv.weight", "mask_fusion.layers.0.q_proj.weight",
        "mask_fusion.layers.0.alpha", "mm_projector.0.weight",
    )
    for suffix in gradient_checks:
        magnitude = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            normalized = name.replace("_fsdp_wrapped_module.", "")
            if normalized.endswith(suffix) and parameter.grad is not None:
                magnitude += parameter.grad.detach().float().abs().sum()
        dist.all_reduce(magnitude)
        all_assert(torch.isfinite(magnitude).item() and magnitude.item() > 0,
                   f"No finite nonzero global gradient for {suffix}", device)
    trainable_ok = all(parameter.dtype == torch.float32 and
                       (parameter.grad is None or parameter.grad.dtype == torch.float32)
                       for parameter in model.parameters() if parameter.requires_grad)
    all_assert(trainable_ok, "Trainable masters/gradients lost FP32 precision", device)
    trace = {
        "loss": loss.detach().cpu(),
        "gradients": {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                      if parameter.grad is not None},
        "frozen": {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                   if not parameter.requires_grad},
    }
    norm = model.clip_grad_norm_(1.0)
    trace["grad_norm"] = norm.detach().cpu()
    trace["update_lrs"] = [group["lr"] for group in optimizer.param_groups]
    trace["parameters_before_update"] = {name: parameter.detach().cpu().clone()
                                         for name, parameter in model.named_parameters() if parameter.requires_grad}
    trace["optimizer_before_update"] = {name: cpu_snapshot(optimizer.state.get(parameter, {}))
                                        for name, parameter in model.named_parameters() if parameter.requires_grad}
    step_aliases = {}
    for name, parameter in model.named_parameters():
        step = optimizer.state.get(parameter, {}).get("step")
        if torch.is_tensor(step):
            step_aliases.setdefault(step.data_ptr(), []).append(name)
    trace["optimizer_step_aliases"] = sorted(sorted(names) for names in step_aliases.values() if len(names) > 1)
    if trace["optimizer_step_aliases"]:
        print(json.dumps({"rank": dist.get_rank(), "shared_optimizer_steps": trace["optimizer_step_aliases"]}), flush=True)
    all_assert(torch.isfinite(norm).item(), "Nonfinite global gradient norm", device)
    # Compare shards in the same post-backward layout. A preceding no-grad
    # validation forward may temporarily expose different FSDP parameter views.
    frozen_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if not parameter.requires_grad}
    optimizer.step()
    scheduler.step()
    frozen_ok = all(parameter.grad is None and torch.equal(parameter.detach(), frozen_before[name])
                    for name, parameter in model.named_parameters() if name in frozen_before)
    all_assert(frozen_ok, "A frozen backbone shard acquired gradients or changed", device)
    optimizer.zero_grad(set_to_none=True)
    return trace


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def assert_nested_close(actual, expected, path="optimizer"):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: lengths differ")
        for index, item in enumerate(expected):
            assert_nested_close(actual[index], item, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def main(argv=None):
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Development Gloo check only; does not validate CUDA memory/collectives")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("This smoke test requires torchrun --nproc-per-node=2")
    if not args.cpu and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA smoke requires two BF16-capable GPUs. It has NOT run; --cpu is a separate development check.")
    device = torch.device("cpu") if args.cpu else torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo" if args.cpu else "nccl", timeout=timedelta(seconds=120))

    def stage(name):
        print(json.dumps({"rank": dist.get_rank(), "stage": name}), flush=True)

    try:
        folder = Path(args.output).resolve()

        def prepare_output():
            if folder.exists() and any(folder.iterdir()):
                raise ValueError("Smoke output must be a new or empty directory")
            folder.mkdir(parents=True, exist_ok=True)

        rank_zero_action(prepare_output)
        stage("wrap")
        model = wrap(make_model(args.cpu), device, args.cpu)
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        # Unequal crop counts exercise real multimodal packing on different ranks.
        batch = move(make_batch((1 + dist.get_rank(),), seed=37 + dist.get_rank()), device)
        stage("collect_before_first_forward")
        before = distributed_helpers.collect_adapter_state(model)
        stage("accumulated_first_step")
        train_step(model, optimizer, scheduler, batch, device, args.cpu)
        stage("collect_checkpoint")
        after = distributed_helpers.collect_adapter_state(model)
        optimizer_state = distributed_helpers.collect_optimizer_state(model, optimizer)
        ranks_rng = distributed_helpers.gather_rank_rng(rng_state(device))
        model.eval()
        with torch.no_grad(), autocast(device, args.cpu):
            expected_logits = model(**batch, use_cache=False).logits.detach().cpu()

        def export_and_single_device_reload():
            for key in ("mask_fusion.layers.0.alpha", "mm_projector.0.weight"):
                if torch.equal(before[key], after[key]):
                    raise AssertionError(f"Optimizer did not update {key}")
            if after["mask_fusion.layers.0.alpha"].ndim != 0:
                raise AssertionError("Exported alpha must retain the scalar single-device adapter format")
            save_adapter(distributed_helpers.unwrap_model(model), folder, "offline-tiny-backbone", weights=after)
            torch.save({"optimizer": optimizer_state, "scheduler": scheduler.state_dict(),
                        "rank_rng": ranks_rng, "step": 1}, folder / "resume.pt")
            single = make_model(args.cpu).to(device)
            load_adapter(single, folder)
            single.eval()
            with torch.no_grad(), autocast(device, args.cpu):
                actual = single(**batch, use_cache=False).logits.detach().cpu()
            torch.testing.assert_close(actual, expected_logits, rtol=0.003 if not args.cpu else 1e-5,
                                       atol=0.003 if not args.cpu else 1e-6)

        stage("export_and_single_device_reload")
        rank_zero_action(export_and_single_device_reload)
        # Establish the uninterrupted second optimizer update as the reference.
        next_batch = move(make_batch((1 + dist.get_rank(),), seed=57 + dist.get_rank()), device)
        stage("partial_second_step")
        expected_trace = train_step(model, optimizer, scheduler, next_batch, device, args.cpu, micro_steps=1)
        expected_second = distributed_helpers.collect_adapter_state(model)
        expected_lr = [group["lr"] for group in optimizer.param_groups]

        stage("construct_resume")
        resumed_raw = make_model(args.cpu)
        load_adapter(resumed_raw, folder)
        resumed = wrap(resumed_raw, device, args.cpu)
        resumed_optimizer = make_optimizer(resumed)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
        checkpoint = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        stage("restore_optimizer")
        distributed_helpers.restore_optimizer_state(resumed, resumed_optimizer,
                                                   checkpoint["optimizer"] if dist.get_rank() == 0 else None)
        resumed_scheduler.load_state_dict(checkpoint["scheduler"])
        restored_adapter = distributed_helpers.collect_adapter_state(resumed)
        restored_optimizer = distributed_helpers.collect_optimizer_state(resumed, resumed_optimizer)

        def compare_restored_checkpoint():
            assert_nested_close(restored_adapter, after, "adapter_before_resumed_step")
            assert_nested_close(restored_optimizer, optimizer_state, "optimizer_before_resumed_step")

        rank_zero_action(compare_restored_checkpoint)
        local_rng = distributed_helpers.rank_rng(checkpoint["rank_rng"])
        restore_rng(local_rng, device)
        stage("resumed_partial_second_step")
        actual_trace = train_step(resumed, resumed_optimizer, resumed_scheduler, next_batch, device, args.cpu, micro_steps=1)
        trace_error = None
        try:
            assert_nested_close(actual_trace["frozen"], expected_trace["frozen"], "frozen_backbone")
            assert_nested_close(actual_trace["loss"], expected_trace["loss"], "next_step_loss")
            assert_nested_close(actual_trace["grad_norm"], expected_trace["grad_norm"], "next_step_gradient_norm")
            assert_nested_close(actual_trace["gradients"], expected_trace["gradients"], "next_step_gradient_shards")
            assert_nested_close(actual_trace["parameters_before_update"], expected_trace["parameters_before_update"],
                                "next_step_parameter_shards")
            assert_nested_close(actual_trace["update_lrs"], expected_trace["update_lrs"], "update_lrs")
            assert_nested_close(actual_trace["optimizer_before_update"], expected_trace["optimizer_before_update"],
                                "local_optimizer_before_update")
            assert_nested_close(actual_trace["optimizer_step_aliases"], expected_trace["optimizer_step_aliases"],
                                "optimizer_step_aliases")
        except AssertionError as error:
            trace_error = str(error)
            print(json.dumps({"rank": dist.get_rank(), "trace_error": trace_error}), flush=True)
        all_assert(trace_error is None, "Resume forward/frozen/gradient trace differs; see rank diagnostic", device)
        resumed_second = distributed_helpers.collect_adapter_state(resumed)
        all_assert([group["lr"] for group in resumed_optimizer.param_groups] == expected_lr,
                   "Resume scheduler step/LR differs from uninterrupted training", device)

        def compare_resume():
            if expected_second.keys() != resumed_second.keys():
                raise AssertionError("Resume changed adapter state keys")
            for key in expected_second:
                torch.testing.assert_close(resumed_second[key], expected_second[key], rtol=1e-5, atol=1e-6,
                                           msg=lambda msg: f"Resume next-step mismatch: {key}: {msg}")
            report = {"status": "passed", "backend": dist.get_backend(), "world_size": dist.get_world_size(),
                      "device": str(device), "real_fsdp": True, "downloads": False,
                      "checks": ["frozen_shards_unchanged", "cnn_q_alpha_projector_gradients", "fp32_masters",
                                 "scalar_alpha_export", "single_device_adapter_logits", "optimizer_scheduler_resume",
                                 "gradient_accumulation", "partial_accumulation_group"],
                      "cuda_verified": not args.cpu}
            (folder / "smoke_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)

        rank_zero_action(compare_resume)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
