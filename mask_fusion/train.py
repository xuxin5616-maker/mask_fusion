"""Single-device supervised training with adapter-only, resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import QADataset, collate_samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local LLaVA-OneVision Qwen2 checkpoint")
    parser.add_argument("--data", required=True, help="Training JSONL or JSON array")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fusion-config", help="Optional fusion configuration JSON")
    parser.add_argument("--vision-tower", help="Optional local SigLIP directory overriding the checkpoint's vision tower path")
    parser.add_argument("--image-mode", default="anyres_max_9")
    parser.add_argument("--data-root", help="Root for relative image and mask paths; otherwise the JSON directory")
    parser.add_argument("--val-data", "--validation", dest="val_data", help="Optional held-out QA dataset")
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
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10, help="Optimizer steps between loss reports")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Also save resumable checkpoints at this interval; 0 = epoch only")
    parser.add_argument("--resume", help="Explicit checkpoint directory containing adapter and training_state.pt")
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
    return args


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def _fingerprint(path: Optional[str]):
    if path is None:
        return None
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _run_signature(args, dataset_size: int, total_steps: int, dtype_name: str):
    return {
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
    }


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_training_state(path: Path):
    if not path.is_file():
        raise ValueError(f"Not a resumable checkpoint: {path} is missing; use an adapter as a base only through an explicit new run")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.1 compatibility
        return torch.load(path, map_location="cpu")


def _save_checkpoint(model, tokenizer, output, args, optimizer, scheduler, scaler, signature, *, epoch, next_batch, global_step, best_val_loss, consecutive_overflows=0):
    from .model import save_adapter

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A previous "best" or final checkpoint must not retain a stale completion
    # marker while its adapter weights are being replaced.
    completion = output / "training_state.pt"
    if completion.exists():
        completion.unlink()
    save_adapter(model, output, args.base_model, tokenizer=tokenizer)
    state = {
        "format_version": 1, "signature": signature, "epoch": epoch,
        "next_batch": next_batch, "global_step": global_step, "best_val_loss": best_val_loss,
        "consecutive_overflows": consecutive_overflows,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "rng": _rng_state(),
    }
    # This file is the completion marker; adapters without it cannot be resumed.
    temporary = output / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(completion)


def _optimizer_update(optimizer, scheduler, scaler, parameters, max_grad_norm):
    """Let FP16 GradScaler skip overflowing updates and reduce its loss scale."""
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not torch.isfinite(grad_norm):
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradients; no optimizer step was taken")
        # GradScaler detects inf/nan entries while unscaling. An overflowing norm
        # of finite entries is a separate error, not a scaler-detected overflow.
        if all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None):
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


def evaluate(model, dataset, args, pad_token_id, device, dtype):
    model.eval()
    loss_sum, token_count = 0.0, 0
    loader = _loader(dataset, list(range(len(dataset))), args, pad_token_id, 0)
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            count = int((batch["labels"] != -100).sum().item())
            with _autocast(device, dtype):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            loss_sum += float(loss.item()) * count
            token_count += count
    if token_count == 0:
        raise ValueError("Validation dataset has no supervised answer tokens")
    return loss_sum / token_count


def main(argv=None):
    args = parse_args(argv)
    from .model import configure_trainable, load_model

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for an explicit CPU run")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    dtype_name, dtype = _resolve_dtype(args.dtype, device)
    _seed_everything(args.seed)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output directory is not empty: {output}; choose a new directory or explicitly use --resume")
    fusion_config = None
    if args.fusion_config:
        with open(args.fusion_config, "r", encoding="utf-8-sig") as handle:
            fusion_config = json.load(handle)
        if not isinstance(fusion_config, dict):
            raise ValueError("fusion-config must contain a JSON object")
    # Resume explicitly reloads trainable weights as well as optimizer and RNG.
    resume_state = _load_training_state(Path(args.resume) / "training_state.pt") if args.resume else None
    model, tokenizer, processor = load_model(
        args.base_model, fusion_config=fusion_config, adapter_path=args.resume,
        image_mode=args.image_mode, device=str(device), dtype=dtype_name,
        attn_implementation=args.attn_implementation, vision_tower=args.vision_tower,
    )
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
        raise RuntimeError("The entire pretrained projector must be trainable")
    trainable = groups["fusion"] + groups["projector"]
    optimizer = torch.optim.AdamW(
        [{"params": groups["fusion"], "lr": args.lr}, {"params": groups["projector"], "lr": args.projector_lr}],
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)
    signature = _run_signature(args, len(dataset), total_steps, dtype_name)
    start_epoch, start_batch, global_step = 0, 0, 0
    best_val_loss = float("inf")
    consecutive_overflows = 0
    if resume_state is not None:
        if resume_state.get("format_version") != 1:
            raise ValueError("Unsupported training checkpoint format")
        if resume_state["signature"] != signature:
            differences = [key for key in signature if signature[key] != resume_state["signature"].get(key)]
            raise ValueError(f"Resume settings/data differ from checkpoint: {', '.join(differences)}")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch, start_batch = int(resume_state["epoch"]), int(resume_state["next_batch"])
        global_step, best_val_loss = int(resume_state["global_step"]), resume_state["best_val_loss"]
        consecutive_overflows = int(resume_state.get("consecutive_overflows", 0))
        if not 0 <= start_epoch <= args.epochs or not 0 <= start_batch < batches_per_epoch:
            raise ValueError("Invalid checkpoint epoch/batch cursor")
        if start_batch % args.grad_accum != 0:
            raise ValueError("Checkpoint is not at a gradient-accumulation boundary")
        _restore_rng(resume_state["rng"])

    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    print(json.dumps({"event": "train_start", "qa_records": len(dataset), "unique_images": len({r['image'] for r in dataset.records}), "trainable_parameters": sum(p.numel() for p in trainable), "total_steps": total_steps, "resume_step": global_step, "dtype": dtype_name}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        cursor = start_batch if epoch == start_epoch else 0
        loader = _loader(dataset, order[cursor * args.batch_size :], args, tokenizer.pad_token_id, epoch)
        model.train()
        model.get_vision_tower().eval()
        group_loss = 0.0
        group_start = cursor
        group_size = min(args.grad_accum, batches_per_epoch - group_start)
        for batch_index, batch in enumerate(loader, cursor):
            batch = _move(batch, device)
            with _autocast(device, dtype):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch={epoch}, batch={batch_index}")
            group_loss += float(loss.detach().item())
            # The last, possibly incomplete accumulation group has its own denominator.
            scaler.scale(loss / group_size).backward()
            boundary = (batch_index + 1 - group_start) == group_size
            if not boundary:
                continue
            updated, grad_norm, scale_before, scale_after = _optimizer_update(
                optimizer, scheduler, scaler, trainable, args.max_grad_norm,
            )
            optimizer.zero_grad(set_to_none=True)
            if updated:
                global_step += 1
                consecutive_overflows = 0
            else:
                consecutive_overflows += 1
                print(json.dumps({"event": "optimizer_step_skipped", "reason": "fp16_gradient_overflow", "epoch": epoch + 1, "next_batch": batch_index + 1, "step": global_step, "loss_scale_before": scale_before, "loss_scale_after": scale_after, "consecutive_overflows": consecutive_overflows}), flush=True)
                if consecutive_overflows >= 32:
                    raise RuntimeError("32 consecutive FP16 gradient overflows prevented optimizer updates; use bfloat16/float32 or inspect the numerical instability")
            if updated and (global_step % args.log_every == 0 or global_step == 1):
                print(json.dumps({"event": "train_step", "epoch": epoch + 1, "step": global_step, "loss": group_loss / group_size, "grad_norm": float(grad_norm), "fusion_lr": optimizer.param_groups[0]["lr"], "projector_lr": optimizer.param_groups[1]["lr"]}), flush=True)
            next_batch = batch_index + 1
            if args.save_every_steps and global_step % args.save_every_steps == 0 and next_batch < batches_per_epoch:
                _save_checkpoint(model, tokenizer, output / f"checkpoint-step-{global_step}", args, optimizer, scheduler, scaler, signature, epoch=epoch, next_batch=next_batch, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows)
            group_start, group_loss = next_batch, 0.0
            group_size = min(args.grad_accum, batches_per_epoch - group_start)
        if validation is not None:
            val_loss = evaluate(model, validation, args, tokenizer.pad_token_id, device, dtype)
            print(json.dumps({"event": "validation", "epoch": epoch + 1, "step": global_step, "answer_token_loss": val_loss}), flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, tokenizer, output / "best", args, optimizer, scheduler, scaler, signature, epoch=epoch + 1, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows)
        _save_checkpoint(model, tokenizer, output / f"checkpoint-epoch-{epoch + 1}", args, optimizer, scheduler, scaler, signature, epoch=epoch + 1, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows)
    if global_step == 0:
        raise RuntimeError("Training ended without a successful optimizer update; inspect overflow events and select bfloat16/float32")
    _save_checkpoint(model, tokenizer, output, args, optimizer, scheduler, scaler, signature, epoch=args.epochs, next_batch=0, global_step=global_step, best_val_loss=best_val_loss, consecutive_overflows=consecutive_overflows)
    print(json.dumps({"event": "train_complete", "output": str(output), "steps": global_step}), flush=True)


if __name__ == "__main__":
    main()
