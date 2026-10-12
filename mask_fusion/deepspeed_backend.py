"""Optional ZeRO-3 backend; keep adapter exports compatible with single-GPU inference.

DeepSpeed is imported lazily so FSDP and inference do not require it. The base
checkpoint is read on CPU on each rank, then zero.Init partitions one parameter
at a time before engine construction. All ranks must enter checkpoint helpers.
"""

import json
from pathlib import Path

import torch


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "zero3.json"


def require_deepspeed():
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed is unavailable. In your Linux CUDA environment run: "
            "python -m pip install -r requirements-deepspeed.txt"
        ) from error
    return deepspeed


def resolve_config(args, dtype_name, world_size, hidden_size):
    """Resolve Trainer-style 'auto' values without depending on HF Trainer."""
    source = Path(args.deepspeed_config) if args.deepspeed_config else DEFAULT_CONFIG
    with source.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("DeepSpeed config must be a JSON object")

    def assign(section, key, value, path):
        existing = section.get(key, "auto")
        if existing != "auto" and existing != value:
            raise ValueError(f"DeepSpeed {path}={existing!r} conflicts with training arguments ({value!r})")
        section[key] = value

    for key, value in {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "train_batch_size": args.batch_size * args.grad_accum * world_size,
        "gradient_clipping": args.max_grad_norm,
    }.items():
        assign(config, key, value, key)
    for section, enabled in (("bf16", dtype_name == "bfloat16"), ("fp16", dtype_name == "float16")):
        assign(config.setdefault(section, {}), "enabled", enabled, f"{section}.enabled")
    zero = config.get("zero_optimization", {})
    if not isinstance(zero, dict) or zero.get("stage") != 3:
        raise ValueError("This backend requires zero_optimization.stage=3")
    # Same hidden-size formulas used by HF's Triad training integration.
    for key, value in {
        "reduce_bucket_size": hidden_size * hidden_size,
        "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
        "stage3_param_persistence_threshold": 10 * hidden_size,
    }.items():
        if zero.get(key) == "auto":
            zero[key] = value
    if "optimizer" in config or "scheduler" in config:
        raise ValueError("Set optimizer/scheduler through the training CLI; remove them from DeepSpeed JSON")
    for key in ("offload_optimizer", "offload_param"):
        if zero.get(key, {}).get("device", "none") != "none":
            raise ValueError(f"{key} is not supported by this entrypoint; use device=none (as in Triad)")
    if config.get("torch_autocast", {}).get("enabled") or config.get("amp", {}).get("enabled"):
        raise ValueError("Use --dtype and native bf16/fp16, not additional DeepSpeed autocast/AMP")
    if config.get("managed_gradient_accumulation", True) is not True:
        raise ValueError("This entrypoint manages explicit accumulation boundaries; leave managed_gradient_accumulation=true")

    def reject_auto(value, path="config"):
        if isinstance(value, dict):
            for key, child in value.items():
                reject_auto(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_auto(child, f"{path}[{index}]")
        elif value == "auto":
            raise ValueError(f"Unresolved DeepSpeed setting: {path}; supply an explicit value")

    reject_auto(config)
    return config


def initialize_engine(model, optimizer, scheduler, config, device, dtype):
    deepspeed = require_deepspeed()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Load the model with defer_device=True before ZeRO-3 initialization")
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False, dist_init_required=False)
    # Native BF16/FP16 keeps computation weights in that dtype; ZeRO maintains
    # FP32 master partitions for the optimizer. Cast on CPU BEFORE partitioning.
    model.to(dtype=dtype)
    deepspeed.zero.Init(module=model, config_dict_or_path=config, dtype=dtype)
    # Existing-module zero.Init partitions parameters, but does not move buffers.
    for module in model.modules():
        for name, value in module.named_buffers(recurse=False):
            module._buffers[name] = value.to(device=device)
    core = model.get_model()
    if hasattr(core, "image_newline"):
        # LLaVA uses this core parameter during the outer model's image packing,
        # before the decoder core's own forward hooks have run.
        deepspeed.zero.register_external_parameter(model, core.image_newline)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=config, dist_init_required=False,
    )
    if engine.zero_optimization_stage() != 3:
        raise RuntimeError("DeepSpeed did not enable ZeRO-3")
    parameters = list(engine.module.parameters())
    if any(not hasattr(parameter, "ds_id") for parameter in parameters):
        raise RuntimeError("Some model parameters were not registered with ZeRO-3")
    if torch.distributed.get_rank() == 0:
        print(json.dumps({
            "event": "zero3_initialized", "stage": 3,
            "world_size": torch.distributed.get_world_size(),
            "global_parameter_elements": sum(parameter.ds_numel for parameter in parameters),
            "rank0_partition_elements": sum(parameter.ds_tensor.numel() for parameter in parameters),
            "cuda_allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
            "compute_dtype": str(dtype), "optimizer_master_dtype": "float32",
        }), flush=True)
    return engine, optimizer, scheduler


def engine_step(engine):
    """The engine owns clipping, loss scaling, successful steps and LR updates."""
    boundary = engine.is_gradient_accumulation_boundary()
    skipped_before = engine.skipped_steps
    scale_before = getattr(engine.optimizer, "cur_scale", None)
    engine.step()
    scale_after = getattr(engine.optimizer, "cur_scale", None)
    updated = boundary and engine.skipped_steps == skipped_before
    norm = engine.get_global_grad_norm() if boundary else None
    return updated, norm, scale_before, scale_after


def collect_adapter_state(engine, context):
    deepspeed = require_deepspeed()
    core = engine.module.get_model()
    modules = (("mask_fusion", core.mask_fusion), ("mm_projector", core.mm_projector))
    parameters = [parameter for _, module in modules for parameter in module.parameters()]
    weights = {}
    # Gather only the small adapters, including disabled/frozen fusion branches
    # required by the strict inference loader. Never gather the full backbone.
    with deepspeed.zero.GatheredParameters(parameters, modifier_rank=0):
        if context.is_main:
            for prefix, module in modules:
                for key, value in module.state_dict().items():
                    value = value.detach().cpu()
                    if value.is_floating_point():
                        value = value.float()
                    weights[f"{prefix}.{key}"] = value.contiguous().clone()
    return weights


def save_checkpoint(engine, tokenizer, output, args, signature, *, epoch, next_batch,
                    global_step, best_val_loss, consecutive_overflows, context):
    from .distributed import gather_rank_rng
    from .model import save_adapter
    from .train import _rank_zero_action, _rng_state

    folder = Path(output)
    completion = folder / "training_state.pt"

    def prepare():
        folder.mkdir(parents=True, exist_ok=True)
        if completion.exists():
            completion.unlink()

    _rank_zero_action(prepare, context)
    weights = collect_adapter_state(engine, context)
    state = {
        "format_version": 1, "backend": "deepspeed", "signature": signature,
        "epoch": epoch, "next_batch": next_batch, "global_step": global_step,
        "best_val_loss": best_val_loss, "consecutive_overflows": consecutive_overflows,
        "rng": gather_rank_rng(_rng_state(context)), "deepspeed_tag": "state",
    }
    # Every rank saves its optimizer/master-weight shard. Frozen base weights
    # remain in the original checkpoint and are not copied into every epoch.
    engine.save_checkpoint(str(folder / "deepspeed"), tag="state", save_latest=False,
                           exclude_frozen_parameters=True,
                           client_state={"mask_fusion_global_step": global_step})

    def finish():
        save_adapter(engine.module, folder, args.base_model, tokenizer=tokenizer, weights=weights)
        temporary = folder / "training_state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(completion)

    _rank_zero_action(finish, context)


def load_resume_metadata(folder, context):
    from .train import _load_training_state

    payload = [None, None]
    if context.is_main:
        try:
            state = _load_training_state(Path(folder) / "training_state.pt")
            if state.get("backend") != "deepspeed":
                raise ValueError("--resume requires a DeepSpeed checkpoint; FSDP optimizer state cannot be reused")
            payload[1] = state
        except Exception as error:
            payload[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Could not read DeepSpeed resume metadata: {payload[0]}")
    return payload[1]


def restore_checkpoint(engine, folder, metadata):
    # Called only on a freshly initialized engine. The base and complete adapter
    # have already been loaded on CPU. Frozen parameters were excluded on save.
    loaded, client_state = engine.load_checkpoint(
        str(Path(folder) / "deepspeed"), tag=metadata["deepspeed_tag"],
        load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=True,
    )
    if loaded is None or client_state.get("mask_fusion_global_step") != metadata["global_step"]:
        raise ValueError("DeepSpeed checkpoint shards are missing or do not match training_state.pt")
