"""FSDP helpers for torchrun training with a frozen LLaVA-Qwen backbone.

Create the optimizer AFTER ``wrap_model``. Adapter/optimizer collection and
restoration are collective operations: every rank must call them in the same
order, although only rank zero writes the resulting checkpoint. No helper
collects the frozen backbone's full state dict.
"""

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(strategy="FSDP") -> DistributedContext:
    """Initialize a CUDA/NCCL process group from torchrun's environment.

    This is an explicit multi-GPU entrypoint, not an implicit single-process
    fallback. The CLI owns process-group cleanup, including an existing group.
    """
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError(f"{strategy} requires torchrun with RANK, WORLD_SIZE and LOCAL_RANK")
    try:
        rank, world_size, local_rank = (int(os.environ[name]) for name in required)
    except ValueError as error:
        raise ValueError("torchrun rank environment variables must be integers") from error
    if world_size < 2 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(f"{strategy} requires WORLD_SIZE >= 2 and valid torchrun ranks")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError(f"{strategy} training requires a PyTorch build with NCCL")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"No CUDA device for LOCAL_RANK={local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if dist.is_initialized():
        if (dist.get_backend() != "nccl" or dist.get_rank() != rank
                or dist.get_world_size() != world_size):
            raise RuntimeError("Existing process group disagrees with torchrun NCCL/rank settings")
    else:
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank,
            world_size=world_size, timeout=timedelta(minutes=30),
        )
    return DistributedContext(rank, world_size, local_rank, device)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original root model; its FSDP-wrapped children stay wrapped."""
    while isinstance(model, FSDP):
        model = model.module
    return model


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _dtype(value):
    if isinstance(value, str):
        if value not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        value = getattr(torch, value)
    if value is not None and value not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float32, float16, or bfloat16")
    return value


def _wrap_model_impl(model, device, dtype=None, *, sync_module_states=True):
    """Build wrapping units; CPU/Gloo contract tests may exercise this helper.

    Production callers must use ``wrap_model`` for CUDA/NCCL validation. The
    backbone is already cast by load_model; FSDP does not cast FP32 adapters.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    if any(isinstance(module, FSDP) for module in model.modules()):
        raise ValueError("Model is already FSDP-wrapped")
    dtype = _dtype(dtype)
    core = model.get_model()
    tower = model.get_vision_tower()
    if not isinstance(tower, nn.Module):
        raise ValueError("Expected an attached vision tower module")
    if not hasattr(core, "mask_fusion") or not hasattr(core, "mm_projector"):
        raise ValueError("Attach mask fusion before FSDP wrapping")
    decoder_layers = [module for module in core.modules()
                      if isinstance(module, Qwen2DecoderLayer)]
    if not decoder_layers:
        raise ValueError("FSDP wrapping requires Qwen2 decoder layers")
    targets = [tower, core.mask_fusion, core.mm_projector, *decoder_layers]
    target_ids = {id(module) for module in targets}
    if len(target_ids) != len(targets):
        raise ValueError("FSDP wrapping units must be distinct modules")

    # FSDP rejects zero-dimensional Parameters. A one-element alpha broadcasts
    # identically; retain its Parameter identity before constructing optimizer.
    # collect_adapter_state restores the original scalar checkpoint format.
    for layer in core.mask_fusion.layers:
        if layer.alpha.ndim == 0:
            with torch.no_grad():
                layer.alpha.data = layer.alpha.data.reshape(1)

    adapter_ids = {id(parameter) for module in (core.mask_fusion, core.mm_projector)
                   for parameter in module.parameters()}
    for name, parameter in model.named_parameters():
        if parameter.ndim == 0:
            raise ValueError(f"FSDP cannot shard scalar parameter {name}")
        if id(parameter) in adapter_ids:
            if parameter.dtype != torch.float32:
                raise ValueError(f"Trainable adapter parameters must remain FP32: {name}")
        elif parameter.requires_grad:
            raise ValueError(f"The backbone must be frozen before FSDP wrapping: {name}")
        elif dtype is not None and parameter.dtype != dtype:
            raise ValueError(f"Backbone dtype differs from requested dtype at {name}")

    managed_ids = set()
    for module in targets:
        parameters = list(module.parameters())
        parameter_ids = {id(parameter) for parameter in parameters}
        if managed_ids & parameter_ids:
            raise ValueError("Parameters cannot be shared between FSDP wrapping units")
        managed_ids.update(parameter_ids)
        if len({parameter.dtype for parameter in parameters}) > 1:
            raise ValueError("Each FSDP wrapping unit must have a uniform parameter dtype")
    remaining = [parameter for parameter in model.parameters()
                 if id(parameter) not in managed_ids]
    if len({parameter.dtype for parameter in remaining}) > 1:
        raise ValueError("Root FSDP parameters must have a uniform dtype")

    def policy(module, recurse, nonwrapped_numel):
        # Recursion visits all children; only these complete modules are flat
        # units. In particular, FP32 fusion/projector never share a BF16 flat.
        return True if recurse else id(module) in target_ids

    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=torch.device(device),
        sync_module_states=sync_module_states,
        mixed_precision=None,
        limit_all_gathers=True,
        # Avoid keeping a second decoder layer unsharded via prefetch. Full
        # sharding targets memory capacity before communication overlap here.
        backward_prefetch=None,
        forward_prefetch=False,
    )


def wrap_model(model, device, dtype=None):
    """Shard frozen decoder/vision and FP32 fusion/projector independently.

    ``model`` should still reside on CPU (load_model(defer_device=True)) so
    bottom-up FSDP construction moves/shards each unit separately. The returned
    root FSDP must handle forward and clip_grad_norm_; never call the unwrapped
    root's forward during training. Its delegated vision attributes remain
    available through get_vision_tower().
    """
    device = torch.device(device)
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise RuntimeError("Initialize the multi-rank process group before FSDP wrapping")
    if dist.get_backend() != "nccl" or device.type != "cuda":
        raise RuntimeError("Production FSDP wrapping requires a CUDA device and NCCL")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return _wrap_model_impl(model, device, dtype)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _full_state_context(model):
    # CPU/Gloo tests are already on CPU. Offloading CPU parameter storage back
    # to CPU can alias the allocation FSDP frees during unsharding. The optimizer
    # gather also reuses one buffer for successive moments: its offload path
    # assumes .cpu() copies, whereas its non-offload path explicitly clones.
    # CUDA training still offloads both full states to host memory as usual.
    offload_model = next(model.parameters()).device.type != "cpu"
    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=offload_model, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=offload_model, rank0_only=True),
    )


def collect_adapter_state(model) -> Dict[str, torch.Tensor]:
    """ALL ranks call; return full inference-compatible adapter tensors on rank 0.

    Only the two small adapter wrapping units are gathered, never the frozen
    language/vision weights. This also supports an ordinary single-rank model.
    """
    if isinstance(model, FSDP):
        # Initialize the whole tree first, so a child state_dict call before
        # the first forward cannot incorrectly mark that child as an FSDP root.
        model.check_is_root()
    core = unwrap_model(model).get_model()
    output = {}
    for prefix in ("mask_fusion", "mm_projector"):
        module = getattr(core, prefix)
        if isinstance(module, FSDP):
            with _full_state_context(module):
                state = module.state_dict()
        else:
            state = module.state_dict() if _rank() == 0 else {}
        if _rank() == 0:
            for name, value in state.items():
                copied = _cpu_copy(value)
                if (prefix == "mask_fusion" and name.startswith("layers.")
                        and name.endswith(".alpha")):
                    copied = copied.reshape(())
                output[f"{prefix}.{name}"] = copied
    return output


def collect_optimizer_state(model, optimizer) -> Dict[str, Any]:
    """ALL ranks call; return full CPU optimizer state on rank 0 only."""
    if isinstance(model, FSDP):
        with _full_state_context(model):
            state = FSDP.optim_state_dict(model, optimizer)
    else:
        state = optimizer.state_dict() if _rank() == 0 else {}
    return _cpu_copy(state) if _rank() == 0 else {}


def restore_optimizer_state(model, optimizer, state: Optional[Dict[str, Any]]) -> None:
    """ALL ranks call; only rank 0 needs the full saved optimizer state.

    Load adapter weights BEFORE wrapping and build the optimizer AFTER wrapping.
    The optimizer format of an FSDP training checkpoint is used only for FSDP
    resume; the separately saved adapter remains usable for single-GPU inference.
    """
    if isinstance(model, FSDP):
        # rank0_only=True broadcasts/scatters optimizer state to the other ranks;
        # they may supply an empty dictionary instead of loading the full file.
        with _full_state_context(model):
            sharded = FSDP.optim_state_dict_to_load(
                model, optimizer, state if _rank() == 0 else {},
            )
        optimizer.load_state_dict(sharded)
    else:
        if state is None:
            raise ValueError("Missing optimizer state")
        optimizer.load_state_dict(state)


def gather_rank_rng(local_rng):
    """ALL ranks call; collect caller-owned Python/NumPy/Torch RNG payloads."""
    local_rng = _cpu_copy(local_rng)
    if not dist.is_initialized():
        return [local_rng]
    gathered = [None] * dist.get_world_size() if _rank() == 0 else None
    dist.gather_object(local_rng, gathered, dst=0)
    return gathered


def rank_rng(saved_list):
    """Select this rank's RNG payload after the caller broadcasts the list."""
    if not isinstance(saved_list, (list, tuple)) or len(saved_list) != _world_size():
        raise ValueError("Resume requires the same world size as the saved RNG state")
    return saved_list[_rank()]
