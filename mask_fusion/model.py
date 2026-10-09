"""Load the existing checkpoint, attach fusion, and persist trainable weights.

Adapters contain the COMPLETE trained projector and the new fusion module.
The original checkpoint is still needed to supply frozen Qwen/SigLIP weights.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from .config import FusionConfig
from .loader import activate_vendor
from .modules import ExpertMaskFusion, expand_projector


ADAPTER_VERSION = 1


def configure_image_mode(config, mode):
    if mode == "square":
        config.image_aspect_ratio = "square"
        config.mm_patch_merge_type = "flat"
    elif mode == "anyres_max_9":
        config.image_aspect_ratio = mode
        if getattr(config, "mm_patch_merge_type", "") not in {
            "spatial_unpad", "spatial_unpad_add_newl"
        }:
            raise ValueError("anyres_max_9 requires the checkpoint's spatial_unpad packing")
    elif mode in {"randomroi", "randompatch"}:
        config.image_aspect_ratio = "randomroi"
        config.mm_patch_merge_type = "spatial_avgpool_auto_unpad_add_newl"
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
    config.image_token_order = "base_first"
    config.mask_fusion_image_mode = mode


def attach_fusion(model, fusion_config=None):
    """Expand an already-loaded projector without reinitializing its old columns."""
    core = model.get_model()
    if hasattr(core, "mask_fusion"):
        raise ValueError("Fusion has already been attached")
    if fusion_config is None:
        fusion_config = FusionConfig()
    elif isinstance(fusion_config, dict):
        fusion_config = FusionConfig(**fusion_config)
    if not isinstance(fusion_config, FusionConfig):
        raise TypeError("fusion_config must be FusionConfig or a dict")
    tower = model.get_vision_tower()
    vision_dim = int(tower.hidden_size)
    patch_size = int(tower.config.patch_size)
    parameter = next(core.mm_projector.parameters())
    fusion = ExpertMaskFusion(vision_dim, patch_size, fusion_config)
    core.mask_fusion = fusion.to(device=parameter.device, dtype=torch.float32)
    core.mm_projector = expand_projector(core.mm_projector, fusion.expert_feature_dim)
    core.mm_projector.float()
    model.config.mask_fusion = asdict(fusion_config)
    model.config.mask_fusion_vision_dim = vision_dim
    model.config.mask_fusion_patch_size = patch_size
    configure_trainable(model)
    return model


def configure_trainable(model):
    """Freeze the backbone, retaining autograd THROUGH the frozen decoder."""
    model.requires_grad_(False)
    core = model.get_model()
    core.mm_projector.requires_grad_(True)
    core.mm_projector.float()
    core.mask_fusion.requires_grad_(True)
    core.mask_fusion.float()
    # Disabled branches do not need optimizer state or misleading parameter counts.
    mode = model.config.mask_fusion["mode"]
    if mode == "none":
        core.mask_fusion.requires_grad_(False)
    elif mode == "direct":
        for module in (core.mask_fusion.expert_projection, core.mask_fusion.position_projection,
                       core.mask_fusion.prior_norm, core.mask_fusion.layers):
            module.requires_grad_(False)
    model.get_vision_tower().eval()
    model.config.use_cache = False
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def read_adapter_config(path):
    with (Path(path) / "adapter_config.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != ADAPTER_VERSION:
        raise ValueError("Unsupported mask-fusion adapter format")
    return metadata


def load_model(base_model, *, fusion_config=None, adapter_path=None,
               image_mode="anyres_max_9", device="cuda", dtype="bfloat16",
               attn_implementation="sdpa", vision_tower=None, defer_device=False):
    activate_vendor()
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    base_path = Path(base_model).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Supply an existing local LLaVA-Qwen checkpoint: {base_path}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU is supported for small tests with --dtype float32.")
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device)
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if torch_device.type == "cpu" and dtype != "float32":
        raise ValueError("Use float32 for CPU execution")
    if torch_device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; select float16 explicitly")
    if attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("Unsupported decoder attention implementation")
    if adapter_path is not None:
        metadata = read_adapter_config(adapter_path)
        saved_fusion = metadata["fusion_config"]
        proposed = asdict(fusion_config) if isinstance(fusion_config, FusionConfig) else fusion_config
        if proposed is not None:
            proposed = FusionConfig.from_dict(proposed).to_dict()
        if proposed is not None and proposed != saved_fusion:
            raise ValueError("Supplied fusion config differs from the saved adapter")
        if image_mode != metadata["image_mode"]:
            raise ValueError("Image mode must match the saved adapter")
        fusion_config = saved_fusion

    config = LlavaQwenConfig.from_pretrained(str(base_path))
    if getattr(config, "mask_fusion", None):
        raise ValueError("base_model must be the original checkpoint, not an expanded full-model export")
    # Keep checkpoint tower loading semantics; trained tower weights must not be replaced.
    if vision_tower:
        config.mm_vision_tower = str(vision_tower)
    configure_image_mode(config, image_mode)
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = min(
        int(getattr(config, "tokenizer_model_max_length", None) or config.max_position_embeddings),
        int(config.max_position_embeddings),
    )
    tokenizer_path = (Path(adapter_path) / "tokenizer") if adapter_path else base_path
    if not tokenizer_path.is_dir():
        tokenizer_path = base_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = LlavaQwenForCausalLM.from_pretrained(
        str(base_path), config=config, torch_dtype=dtypes[dtype],
        attn_implementation=attn_implementation, low_cpu_mem_usage=True,
    )
    original_vocab = model.get_input_embeddings().weight.shape[0]
    if getattr(config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    if len(tokenizer) > original_vocab:
        raise ValueError("Checkpoint lacks required image-token embeddings; refusing random frozen tokens")
    # Match the baseline loader, which removes padded, unused vocabulary rows.
    model.resize_token_embeddings(len(tokenizer))
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model()
    # This vendor implements the so400m-patch14-384 SigLIP architecture.
    actual = tower.vision_tower.config
    expected = (int(tower.hidden_size), int(tower.config.patch_size), int(tower.config.image_size))
    observed = (int(actual.hidden_size), int(actual.patch_size), int(actual.image_size))
    if observed != expected:
        raise ValueError(f"Unsupported vision tower architecture: {observed}; vendor expects {expected}")
    # FSDP moves and shards individual wrapping units. Keeping the initial
    # model on CPU avoids placing a complete replica on each GPU first.
    initial_device = torch.device("cpu") if defer_device else torch_device
    model.to(device=initial_device, dtype=dtypes[dtype])
    # Pruning is intentionally disabled in this baseline and all fusion modes.
    model.get_model().fastv_enabled = False
    model.get_model().vico_enabled = False
    attach_fusion(model, fusion_config)
    if adapter_path is not None:
        load_adapter(model, adapter_path)
    model.eval()
    return model, tokenizer, tower.image_processor


def save_adapter(model, output, base_model, tokenizer=None, *, weights=None):
    """Save fusion/projector weights, optionally gathered by all FSDP ranks.

    When ``weights`` is supplied, only the caller writes files and no module
    state-dict operation (which may contain distributed collectives) occurs.
    The caller supplies the underlying model for architecture metadata.
    """
    from safetensors.torch import save_file

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if weights is None:
        core = model.get_model()
        weights = {}
        for prefix, module in (("mask_fusion", core.mask_fusion), ("mm_projector", core.mm_projector)):
            weights.update({f"{prefix}.{key}": value.detach().cpu().contiguous().clone()
                            for key, value in module.state_dict().items()})
    else:
        if not weights or any(not key.startswith(("mask_fusion.", "mm_projector.")) for key in weights):
            raise ValueError("Gathered adapter weights must contain only fusion/projector state")
        weights = {key: value.detach().cpu().contiguous().clone() for key, value in weights.items()}
    temporary = folder / "adapter.safetensors.tmp"
    save_file(weights, str(temporary))
    os.replace(temporary, folder / "adapter.safetensors")
    metadata = {
        "format_version": ADAPTER_VERSION,
        "base_model": str(Path(base_model).expanduser().resolve()),
        "fusion_config": model.config.mask_fusion,
        "vision_dim": model.config.mask_fusion_vision_dim,
        "patch_size": model.config.mask_fusion_patch_size,
        "hidden_size": model.config.hidden_size,
        "image_mode": getattr(model.config, "mask_fusion_image_mode", "square"),
        "mm_patch_merge_type": model.config.mm_patch_merge_type,
        "base_model_type": model.config.model_type,
    }
    config_tmp = folder / "adapter_config.json.tmp"
    config_tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(config_tmp, folder / "adapter_config.json")
    if tokenizer is not None:
        tokenizer.save_pretrained(str(folder / "tokenizer"))


def load_adapter(model, path):
    from safetensors.torch import load_file

    folder = Path(path)
    metadata = read_adapter_config(folder)
    for key, actual in (("fusion_config", model.config.mask_fusion),
                        ("vision_dim", model.config.mask_fusion_vision_dim),
                        ("patch_size", model.config.mask_fusion_patch_size),
                        ("hidden_size", model.config.hidden_size),
                        ("mm_patch_merge_type", model.config.mm_patch_merge_type)):
        if metadata[key] != actual:
            raise ValueError(f"Adapter/model mismatch for {key}: {metadata[key]!r} vs {actual!r}")
    weights = load_file(str(folder / "adapter.safetensors"), device="cpu")
    expected = set()
    for prefix, module in (("mask_fusion", model.get_model().mask_fusion),
                           ("mm_projector", model.get_model().mm_projector)):
        expected.update(f"{prefix}.{key}" for key in module.state_dict())
    if set(weights) != expected:
        raise ValueError("Adapter weights do not exactly match the fusion and projector state")
    for prefix, module in (("mask_fusion", model.get_model().mask_fusion),
                           ("mm_projector", model.get_model().mm_projector)):
        module.load_state_dict({key[len(prefix) + 1:]: value for key, value in weights.items()
                                if key.startswith(prefix + ".")}, strict=True)
    return model
