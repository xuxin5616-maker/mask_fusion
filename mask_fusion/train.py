"""Single-device, FSDP or DeepSpeed ZeRO-3 SFT with resumable adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import QADataset, collate_samples, normalize_data_paths
from .progress import ProgressTimer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local LLaVA-OneVision Qwen2 checkpoint")
    parser.add_argument("--data", required=True, nargs="+", action="extend", metavar="FILE", help="One or more training JSONL/JSON files; may repeat --data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-stage", choices=("sft", "mask_pretrain"), default="sft",
                        help="mask_pretrain zeroes RGB features; trains the CNN or frozen-SigLIP mask adapter and new projector columns")
    parser.add_argument("--fusion-config", help="Optional fusion configuration JSON")
    parser.add_argument("--vision-tower", help="Optional local SigLIP directory overriding the checkpoint's vision tower path")
    parser.add_argument("--image-mode", default="anyres_max_9")
    parser.add_argument("--data-root", help="Root for relative image and mask paths; otherwise the JSON directory")
    parser.add_argument("--val-data", "--validation", dest="val_data", nargs="+", action="extend", metavar="FILE", help="One or more held-out QA JSONL/JSON files")
    parser.add_argument("--val-data-root")
    parser.add_argument("--require-disjoint-categories", action="store_true", help="Require category annotations and no train/validation category overlap")
    parser.add_argument("--lr", type=float, default=1e-4, help="New fusion module learning rate")
    parser.add_argument("--projector-lr", type=float, default=1e-5, help="Learning rate of the complete pretrained projector")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", "--gradient-accumulation-steps", dest="grad_accum", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-text-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", choices=("none", "fsdp", "deepspeed"), default="none", help="fsdp/deepspeed require torchrun with multiple CUDA processes; use FULL_SHARD/ZeRO-3")
    parser.add_argument("--deepspeed-config", help="ZeRO-3 JSON; defaults to configs/zero3.json for --distributed deepspeed")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10, help="Optimizer steps between loss reports")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Also save resumable checkpoints at this interval; 0 = epoch only")
    parser.add_argument("--resume", help="Explicit checkpoint directory containing adapter and training_state.pt")
    parser.add_argument("--init-adapter", help="Initialize a NEW run from adapter weights only; permits a new image mode or enabling the direct residual MLP, resets optimizer/scheduler, and cannot be combined with --resume")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "grad_accum", "max_text_length", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.save_every_steps < 0:
        parser.error("worker count and checkpoint interval must be nonnegative")
    if args.lr <= 0 or args.projector_lr <= 0 or args.max_grad_norm <= 0:
        parser.error("learning rates and max-grad-norm must be positive")
    if not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0:
        parser.error("warmup-ratio must be in [0, 1), and weight-decay must be nonnegative")
    if args.require_disjoint_categories and not args.val_data:
        parser.error("--require-disjoint-categories requires --val-data")
    if args.deepspeed_config and args.distributed != "deepspeed":
        parser.error("--deepspeed-config requires --distributed deepspeed")
    if args.resume and args.init_adapter:
        parser.error("--init-adapter starts a new run; use either --init-adapter or --resume")
    if args.training_stage == "mask_pretrain" and args.image_mode != "square":
        parser.error("mask_pretrain requires --image-mode square so whole-mask QA labels stay valid")
    # Preserve the existing single-file argument/metadata representation.
    # Multi-file order is meaningful for deterministic shuffle and resume.
    for name in ("data", "val_data"):
        values = getattr(args, name)
        if values is not None and len(values) == 1:
            setattr(args, name, values[0])
    return args


def _seed_everything(seed: int, context=None):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        if context is not None:
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)


def _move(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype if dtype != torch.float32 else torch.bfloat16, enabled=dtype != torch.float32)


def _resolve_dtype(name: str, device: torch.device):
    if device.type not in ("cuda", "cpu"):
        raise ValueError("This training entrypoint supports a single CUDA GPU or CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA PyTorch installation/GPU, or explicitly select --device cpu --dtype float32")
    if name == "auto":
        name = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if name == "float16" and device.type != "cuda":
        raise ValueError("float16 training requires CUDA; select float32 for CPU")
    if name == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; select float32 or float16")
    return name, getattr(torch, name)


def _fingerprint(path):
    """Fingerprint every manifest in order, retaining legacy single-file state."""
    if path is None:
        return None
    paths = normalize_data_paths(path)
    result = [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
              for source in paths]
    return result[0] if len(result) == 1 else result


def _run_signature(args, dataset_size: int, total_steps: int, dtype_name: str, context=None):
    return {
        "training_stage": getattr(args, "training_stage", "sft"),
        "base_model": str(Path(args.base_model).resolve()),
        "train_data": _fingerprint(args.data), "val_data": _fingerprint(args.val_data),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
        "val_data_root": str(Path(args.val_data_root).resolve()) if args.val_data_root else None,
        "image_mode": args.image_mode, "seed": args.seed, "epochs": args.epochs,
        "vision_tower": str(Path(args.vision_tower).resolve()) if args.vision_tower else None,
        "attn_implementation": args.attn_implementation,
        "require_disjoint_categories": args.require_disjoint_categories,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "projector_lr": args.projector_lr, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "max_grad_norm": args.max_grad_norm,
        "max_text_length": args.max_text_length, "dataset_size": dataset_size,
        "total_steps": total_steps, "dtype": dtype_name,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "distributed": args.distributed, "world_size": context.world_size if context else 1,
    }


def _initial_adapter_signature(path):
    """Record the exact initialization without loading optimizer state."""
    folder = Path(path).expanduser().resolve()
    result = {"path": str(folder)}
    for name in ("adapter_config.json", "adapter.safetensors"):
        digest = hashlib.sha256()
        with (folder / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def _validate_pretrain_sizes(dataset, model):
    """Reject label-changing resizes of maps whose QA was computed offline."""
    from PIL import Image
    from .preprocessing import load_mask

    side = int(model.get_vision_tower().config.image_size)
    seen = set()
    for record in dataset.records:
        pair = (record["image"], record["mask"])
        if pair in seen:
            continue
        seen.add(pair)
        with Image.open(record["image"]) as source:
            image_size = source.size
        mask_shape = load_mask(record["mask"]).shape
        if image_size != (side, side) or mask_shape != (side, side):
            raise ValueError(
                f"mask_pretrain requires {side}x{side} image placeholders and masks without resizing; "
                f"got image={image_size}, mask={mask_shape} for {record['mask']}. "
                f"Regenerate masks and their QA with --size {side}."
            )


def _rng_state(context=None):
    if context is not None:
        return {
            "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            # Access only this process's CUDA device; never initialize all visible GPUs.
            "cuda_local": torch.cuda.get_rng_state(context.device),
        }
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state, context=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda_local" in state:
        if context is None:
            raise ValueError("A distributed RNG state requires its rank context")
        torch.cuda.set_rng_state(state["cuda_local"], context.device)
    elif state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _rank_zero_print(payload, context=None):
    if context is None or context.is_main:
        print(json.dumps(payload), flush=True)


def _rank_zero_action(action, context=None):
    """Run shared-filesystem writes once and propagate failures to every rank."""
    if context is None:
        return action()
    status = [None]
    if context.is_main:
        try:
            action()
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Rank 0 operation failed: {status[0]}")


def _all_ranks_true(value, context=None):
    if context is None:
        return bool(value)
    flag = torch.tensor(int(bool(value)), device=context.device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _mean_across_ranks(value, context=None):
    if context is None:
        return float(value)
    total = torch.tensor(float(value), device=context.device, dtype=torch.float64)
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    return float(total.item()) / context.world_size


def _distributed_indices(size, *, rank=0, world_size=1, seed=0, shuffle=True):
    """Equal-length rank shards; mark repeated padding for unbiased validation.

    Like DistributedSampler(drop_last=False), training repeats at most world-1
    examples. Even a dataset smaller than world_size gives every rank a batch.
    """
    if size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset size or distributed rank/world size")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist() if shuffle else list(range(size))
    total_size = math.ceil(size / world_size) * world_size
    padded = (order * math.ceil(total_size / size))[:total_size]
    valid = [index < size for index in range(total_size)]
    return padded[rank:total_size:world_size], valid[rank:total_size:world_size]


def _load_resume_state(path, context=None):
    """Only rank 0 reads the full optimizer state; broadcast small run metadata."""
    if context is None:
        state = _load_training_state(path)
        return state, state["optimizer"]
    payload, optimizer_state = [None, None], None
    if context.is_main:
        try:
            state = _load_training_state(path)
            optimizer_state = state.pop("optimizer")
            payload[1] = state
        except Exception as error:
            payload[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Could not read checkpoint on rank 0: {payload[0]}")
    return payload[1], optimizer_state


def _load_training_state(path: Path):
    if not path.is_file():
        raise ValueError(f"Not a resumable checkpoint: {path} is missing; use an adapter as a base only through an explicit new run")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.1 compatibility
        return torch.load(path, map_location="cpu")


def _save_checkpoint(model, tokenizer, output, args, optimizer, scheduler, scaler, signature, *, epoch, next_batch, global_step, best_val_loss, consecutive_overflows=0, context=None):
    if getattr(args, "distributed", "none") == "deepspeed":
        from .deepspeed_backend import save_checkpoint

        return save_checkpoint(model, tokenizer, output, args, signature,
                               epoch=epoch, next_batch=next_batch, global_step=global_step,
                               best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows,
                               context=context)
    from .model import save_adapter

    output = Path(output)
    weights = None
    base_model = model
    if context is not None:
        from .distributed import collect_adapter_state, collect_optimizer_state, gather_rank_rng, unwrap_model

        # All ranks enter these collectives, including ranks that do not write.
        weights = collect_adapter_state(model)
        optimizer_state = collect_optimizer_state(model, optimizer)
        rng = gather_rank_rng(_rng_state(context))
        base_model = unwrap_model(model)
    else:
        optimizer_state = optimizer.state_dict()
        rng = _rng_state()
    state = {
        "format_version": 1, "signature": signature, "epoch": epoch,
        "next_batch": next_batch, "global_step": global_step, "best_val_loss": best_val_loss,
        "consecutive_overflows": consecutive_overflows,
        "optimizer": optimizer_state, "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "rng": rng,
    }

    def write_checkpoint():
        output.mkdir(parents=True, exist_ok=True)
        completion = output / "training_state.pt"
        if completion.exists():
            completion.unlink()
        if context is None:
            save_adapter(base_model, output, args.base_model, tokenizer=tokenizer)
        else:
            save_adapter(base_model, output, args.base_model, tokenizer=tokenizer, weights=weights)
        # The atomically replaced state is the checkpoint completion marker.
        temporary = output / "training_state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(completion)

    _rank_zero_action(write_checkpoint, context)


def _optimizer_update(optimizer, scheduler, scaler, parameters, max_grad_norm, *, model=None, context=None):
    """Let FP16 GradScaler skip overflowing updates and reduce its loss scale."""
    scaler.unscale_(optimizer)
    # FSDP owns gradient shards; a local clip would compute the wrong norm.
    grad_norm = model.clip_grad_norm_(max_grad_norm) if context is not None else torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not torch.isfinite(grad_norm):
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradients; no optimizer step was taken")
        # GradScaler detects inf/nan entries while unscaling. An overflowing norm
        # of finite entries is a separate error, not a scaler-detected overflow.
        if _all_ranks_true(all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None), context):
            raise RuntimeError("Nonfinite gradient norm despite finite entries; no optimizer step was taken")
    scale_before = scaler.get_scale()
    scaler.step(optimizer)  # skips internally when unscale_ found nonfinite values
    scaler.update()
    scale_after = scaler.get_scale()
    updated = not scaler.is_enabled() or scale_after >= scale_before
    if updated:
        scheduler.step()
    return updated, grad_norm, scale_before, scale_after


def _loader(dataset, indices, args, pad_token_id, epoch):
    generator = torch.Generator().manual_seed(args.seed + 1_000_003 + epoch)
    return DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_samples, pad_token_id=pad_token_id),
        num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
        generator=generator, persistent_workers=False, drop_last=False,
    )


def evaluate(model, dataset, args, pad_token_id, device, dtype, context=None):
    model.eval()
    loss_sum, token_count = 0.0, 0
    indices, valid_samples = _distributed_indices(
        len(dataset), rank=context.rank if context else 0,
        world_size=context.world_size if context else 1, shuffle=False,
    )
    loader = _loader(dataset, indices, args, pad_token_id, 0)
    timer = ProgressTimer(len(loader))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = _move(batch, device)
            valid = torch.tensor(valid_samples[batch_index * args.batch_size : (batch_index + 1) * args.batch_size], device=device, dtype=torch.bool)
            counted_labels = batch["labels"].clone()
            counted_labels[~valid] = -100
            count = int((counted_labels != -100).sum().item())
            # Every rank must execute equally many FSDP forwards. For a rank's
            # all-padding batch, retain dummy labels to avoid all-ignored NaN,
            # then assign zero statistical weight to that forward.
            if count:
                batch["labels"] = counted_labels
            with _autocast(device, dtype):
                loss = model(**batch).loss
            if not _all_ranks_true(torch.isfinite(loss), context):
                raise RuntimeError("Nonfinite validation loss")
            loss_sum += float(loss.item()) * count
            token_count += count
            completed = batch_index + 1
            if completed == 1 or completed % getattr(args, "log_every", 10) == 0 or completed == len(loader):
                _rank_zero_print({"event": "validation_progress", "progress_unit": "batch",
                                 **timer.snapshot(completed)}, context)
    if context is not None:
        totals = torch.tensor([loss_sum, token_count], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        loss_sum, token_count = totals.tolist()
    if token_count == 0:
        raise ValueError("Validation dataset has no supervised answer tokens")
    return loss_sum / token_count


def main(argv=None):
    args = parse_args(argv)
    # argparse exits for --help before CUDA checks or collective initialization.
    context = None
    if args.distributed in ("fsdp", "deepspeed"):
        if torch.device(args.device).type != "cuda":
            raise ValueError(f"--distributed {args.distributed} requires CUDA; use torchrun --nproc_per_node=2")
        if args.distributed == "deepspeed":
            from .deepspeed_backend import require_deepspeed

            require_deepspeed()
        from .distributed import initialize_distributed

        context = initialize_distributed(strategy=args.distributed)
        args.device = str(context.device)
    elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("torchrun multi-process training requires --distributed fsdp or deepspeed; otherwise processes would overwrite one output")
    try:
        _run_training(args, context)
    finally:
        if context is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_training(args, context=None):
    from .model import configure_trainable, load_model

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for an explicit CPU run")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    dtype_name, dtype = _resolve_dtype(args.dtype, device)
    use_deepspeed = args.distributed == "deepspeed"
    if use_deepspeed and context is None:
        raise ValueError("DeepSpeed requires a torchrun distributed context")
    _seed_everything(args.seed, context)
    output = Path(args.output).resolve()

    def check_output():
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise ValueError(f"Output directory is not empty: {output}; choose a new directory or explicitly use --resume")

    _rank_zero_action(check_output, context)
    fusion_config = None
    if args.fusion_config:
        with open(args.fusion_config, "r", encoding="utf-8-sig") as handle:
            fusion_config = json.load(handle)
        if not isinstance(fusion_config, dict):
            raise ValueError("fusion-config must contain a JSON object")
    # Resume explicitly reloads trainable weights as well as optimizer and RNG.
    if use_deepspeed and args.resume:
        from .deepspeed_backend import load_resume_metadata

        resume_state, resume_optimizer = load_resume_metadata(args.resume, context), None
    else:
        resume_state, resume_optimizer = _load_resume_state(Path(args.resume) / "training_state.pt", context) if args.resume else (None, None)
    if resume_state is not None:
        saved_stage = resume_state["signature"].get("training_stage", "sft")
        if saved_stage != args.training_stage:
            raise ValueError(
                f"Cannot --resume {saved_stage} as {args.training_stage}; "
                "use --init-adapter to start the next stage with a fresh optimizer."
            )
    initialization = None
    if args.init_adapter:
        from .model import read_adapter_config

        initialization = _initial_adapter_signature(args.init_adapter)
        if fusion_config is None:
            fusion_config = read_adapter_config(args.init_adapter)["fusion_config"]
    elif resume_state is not None:
        # Resuming never needs to reread the adapter that initialized this run.
        initialization = resume_state["signature"].get("initial_adapter")
    _rank_zero_print({"event": "model_loading", "distributed": args.distributed,
                     "load_device": "cpu" if context else str(device),
                     "message": "Each rank reads the base checkpoint; GPU parameter sharding follows loading." if context else "Loading base checkpoint."}, context)
    deferred = {"defer_device": True} if context is not None else {}
    model, tokenizer, processor = load_model(
        args.base_model, fusion_config=fusion_config, adapter_path=args.resume,
        image_mode=args.image_mode, device=str(device), dtype=dtype_name,
        attn_implementation=args.attn_implementation, vision_tower=args.vision_tower,
        **deferred,
    )
    if args.init_adapter:
        from .model import load_adapter

        load_adapter(model, args.init_adapter, allow_image_mode_change=True,
                     allow_direct_mlp_upgrade=True)
        _rank_zero_print({"event": "adapter_initialization", "source": initialization,
                         "message": "Loaded weights only; optimizer, scheduler and step count start fresh."}, context)
    if args.training_stage == "mask_pretrain":
        from .model import configure_mask_pretraining

        configure_mask_pretraining(model)
    else:
        configure_trainable(model)
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = QADataset(
        args.data, tokenizer, processor, model.config, data_root=args.data_root,
        mode=args.image_mode, seed=args.seed, max_text_length=args.max_text_length,
    )
    validation = None
    if args.val_data:
        validation = QADataset(
            args.val_data, tokenizer, processor, model.config, data_root=args.val_data_root or args.data_root,
            mode=args.image_mode, seed=args.seed, max_text_length=args.max_text_length,
        )
        overlapping_images = {r["image"] for r in dataset.records} & {r["image"] for r in validation.records}
        if overlapping_images:
            raise ValueError(f"Training/validation leakage: {len(overlapping_images)} identical image paths appear in both splits")
        if args.require_disjoint_categories:
            if any(not record.get("category") for record in dataset.records + validation.records):
                raise ValueError("Disjoint-category validation requires a nonempty category for every training and validation record")
            overlap = {record["category"] for record in dataset.records} & {record["category"] for record in validation.records}
            if overlap:
                raise ValueError(f"Training/validation categories overlap: {', '.join(sorted(overlap))}")

    if args.training_stage == "mask_pretrain":
        _validate_pretrain_sizes(dataset, model)
        if validation is not None:
            _validate_pretrain_sizes(validation, model)
            overlapping_masks = {r["mask"] for r in dataset.records} & {r["mask"] for r in validation.records}
            if overlapping_masks:
                raise ValueError(f"Training/validation leakage: {len(overlapping_masks)} identical mask paths appear in both splits")

    # Save the original global count before parameters become local FSDP shards.
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    original_model = model
    if args.distributed == "fsdp":
        from .distributed import wrap_model

        model = wrap_model(model, device=device, dtype=dtype)
        # Initialization is common across ranks; stochastic training is rank-local.
        _seed_everything(args.seed + context.rank, context)
    groups = {"fusion": [], "projector": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".mask_fusion." in name:
            groups["fusion"].append(parameter)
        elif ".mm_projector." in name:
            groups["projector"].append(parameter)
        else:
            raise RuntimeError(f"Unexpected trainable base-model parameter: {name}")
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"Trainable parameter {name} must remain float32; mixed precision belongs in autocast")
    if not groups["projector"]:
        raise RuntimeError("No trainable projector parameters for the selected training stage")
    trainable = groups["fusion"] + groups["projector"]
    parameter_groups = [{"params": groups["fusion"], "lr": args.lr, "name": "fusion"},
                        {"params": groups["projector"], "lr": args.projector_lr, "name": "projector"}]
    if use_deepspeed:
        parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    samples_per_rank = math.ceil(len(dataset) / (context.world_size if context else 1))
    batches_per_epoch = math.ceil(samples_per_rank / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    ds_config = None
    if use_deepspeed:
        from .deepspeed_backend import initialize_engine, resolve_config

        ds_config = resolve_config(args, dtype_name, context.world_size, model.config.hidden_size)
        model, optimizer, scheduler = initialize_engine(model, optimizer, scheduler, ds_config, device, dtype)
        _seed_everything(args.seed + context.rank, context)
        scaler = None  # DeepSpeed owns precision, gradient clipping and optimizer stepping.
    elif context is not None:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler(enabled=dtype == torch.float16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)
    signature = _run_signature(args, len(dataset), total_steps, dtype_name, context)
    signature["initial_adapter"] = initialization
    if use_deepspeed:
        signature["deepspeed_config"] = ds_config
    start_epoch, start_batch, global_step = 0, 0, 0
    best_val_loss = float("inf")
    consecutive_overflows = 0
    if resume_state is not None:
        if resume_state.get("format_version") != 1:
            raise ValueError("Unsupported training checkpoint format")
        saved_signature = dict(resume_state["signature"])
        saved_signature.setdefault("distributed", "none")
        saved_signature.setdefault("world_size", 1)
        saved_signature.setdefault("training_stage", "sft")
        saved_signature.setdefault("initial_adapter", None)
        if saved_signature != signature:
            differences = [key for key in signature if signature[key] != saved_signature.get(key)]
            raise ValueError(f"Resume settings/data differ from checkpoint: {', '.join(differences)}")
        if use_deepspeed:
            from .deepspeed_backend import restore_checkpoint

            restore_checkpoint(model, args.resume, resume_state)
        elif context is not None:
            from .distributed import restore_optimizer_state

            restore_optimizer_state(model, optimizer, resume_optimizer)
        else:
            optimizer.load_state_dict(resume_optimizer)
        resume_state.pop("optimizer", None)
        del resume_optimizer
        if not use_deepspeed:
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch, start_batch = int(resume_state["epoch"]), int(resume_state["next_batch"])
        global_step, best_val_loss = int(resume_state["global_step"]), resume_state["best_val_loss"]
        consecutive_overflows = int(resume_state.get("consecutive_overflows", 0))
        if not 0 <= start_epoch <= args.epochs or not 0 <= start_batch < batches_per_epoch:
            raise ValueError("Invalid checkpoint epoch/batch cursor")
        if start_batch % args.grad_accum != 0:
            raise ValueError("Checkpoint is not at a gradient-accumulation boundary")
        if context is not None:
            from .distributed import rank_rng

            _restore_rng(rank_rng(resume_state["rng"]), context)
        else:
            _restore_rng(resume_state["rng"])

    def write_args():
        output.mkdir(parents=True, exist_ok=True)
        with (output / "training_args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        if ds_config is not None:
            (output / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2), encoding="utf-8")

    _rank_zero_action(write_args, context)
    _rank_zero_print({"event": "train_start", "training_stage": args.training_stage, "qa_records": len(dataset), "unique_images": len({r['image'] for r in dataset.records}), "trainable_parameters": trainable_parameter_count, "total_steps": total_steps, "resume_step": global_step, "dtype": dtype_name, "distributed": args.distributed, "world_size": context.world_size if context else 1, "effective_batch_size": args.batch_size * args.grad_accum * (context.world_size if context else 1), "repeated_samples_per_epoch": samples_per_rank * (context.world_size if context else 1) - len(dataset)}, context)
    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)
    timer = ProgressTimer(args.epochs * batches_per_epoch,
                          start_epoch * batches_per_epoch + start_batch)
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        order, _ = _distributed_indices(len(dataset), rank=context.rank if context else 0, world_size=context.world_size if context else 1, seed=args.seed + epoch)
        cursor = start_batch if epoch == start_epoch else 0
        loader = _loader(dataset, order[cursor * args.batch_size :], args, tokenizer.pad_token_id, epoch)
        model.train()
        original_model.get_vision_tower().eval()
        group_loss = 0.0
        group_start = cursor
        group_size = min(args.grad_accum, batches_per_epoch - group_start)
        for batch_index, batch in enumerate(loader, cursor):
            boundary = (batch_index + 1 - group_start) == group_size
            if use_deepspeed:
                model.set_gradient_accumulation_boundary(boundary)
            batch = _move(batch, device)
            with _autocast(device, dtype):
                loss = model(**batch).loss
            if not _all_ranks_true(torch.isfinite(loss), context):
                raise RuntimeError(f"Nonfinite training loss at epoch={epoch}, batch={batch_index}")
            group_loss += float(loss.detach().item())
            # The last, possibly incomplete accumulation group has its own denominator.
            if use_deepspeed:
                # Average over the ACTUAL group length, including a short last
                # group. DeepSpeed must not divide by grad_accum a second time.
                model.backward(loss / group_size, scale_wrt_gas=False)
                from .deepspeed_backend import engine_step

                updated, grad_norm, scale_before, scale_after = engine_step(model)
            else:
                scaler.scale(loss / group_size).backward()
            if not boundary:
                continue
            if not use_deepspeed:
                updated, grad_norm, scale_before, scale_after = _optimizer_update(
                    optimizer, scheduler, scaler, trainable, args.max_grad_norm, model=model, context=context,
                )
                optimizer.zero_grad(set_to_none=True)
            progress = timer.snapshot(epoch * batches_per_epoch + batch_index + 1)
            if updated:
                global_step += 1
                consecutive_overflows = 0
            else:
                consecutive_overflows += 1
                _rank_zero_print({"event": "optimizer_step_skipped", "reason": "fp16_gradient_overflow", "epoch": epoch + 1, "next_batch": batch_index + 1, "step": global_step, "loss_scale_before": scale_before, "loss_scale_after": scale_after, "consecutive_overflows": consecutive_overflows, **progress}, context)
                if consecutive_overflows >= 32:
                    raise RuntimeError("32 consecutive FP16 gradient overflows prevented optimizer updates; use bfloat16/float32 or inspect the numerical instability")
            if updated and (global_step % args.log_every == 0 or global_step == 1 or batch_index + 1 == batches_per_epoch):
                mean_loss = _mean_across_ranks(group_loss / group_size, context)
                # Older FSDP checkpoints did not name their parameter groups.
                rates = {group.get("name", ("fusion", "projector")[i]): group["lr"]
                         for i, group in enumerate(optimizer.param_groups)}
                _rank_zero_print({"event": "train_step", "epoch": epoch + 1, "step": global_step, "loss": mean_loss, "grad_norm": float(grad_norm) if grad_norm is not None else None, "fusion_lr": rates.get("fusion"), "projector_lr": rates["projector"], "progress_unit": "batch", "eta_scope": "remaining_training_batches", **progress}, context)
            next_batch = batch_index + 1
            if args.save_every_steps and global_step % args.save_every_steps == 0 and next_batch < batches_per_epoch:
                with timer.paused():
                    _save_checkpoint(model, tokenizer, output / f"checkpoint-step-{global_step}", args, optimizer, scheduler, scaler, signature, epoch=epoch, next_batch=next_batch, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows, context=context)
            group_start, group_loss = next_batch, 0.0
            group_size = min(args.grad_accum, batches_per_epoch - group_start)
        with timer.paused():
            if validation is not None:
                val_loss = evaluate(model, validation, args, tokenizer.pad_token_id, device, dtype, context)
                _rank_zero_print({"event": "validation", "epoch": epoch + 1, "step": global_step, "answer_token_loss": val_loss}, context)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(model, tokenizer, output / "best", args, optimizer, scheduler, scaler, signature, epoch=epoch + 1, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows, context=context)
            _save_checkpoint(model, tokenizer, output / f"checkpoint-epoch-{epoch + 1}", args, optimizer, scheduler, scaler, signature, epoch=epoch + 1, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows, context=context)
    if global_step == 0:
        raise RuntimeError("Training ended without a successful optimizer update; inspect overflow events and select bfloat16/float32")
    with timer.paused():
        _save_checkpoint(model, tokenizer, output, args, optimizer, scheduler, scaler, signature, epoch=args.epochs, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows, context=context)
    _rank_zero_print({"event": "train_complete", "output": str(output), "steps": global_step, **timer.snapshot()}, context)


if __name__ == "__main__":
    main()
