"""Mask CNN, prior-only attention, and baseline-preserving projector expansion.

No pretrained models are loaded here. Grid dimensions and channel widths follow
the supplied vision tower, rather than assuming a particular SigLIP checkpoint.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import FusionConfig


def position_encoding_2d(
    height: int,
    width: int,
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Row-major 2D sine/cosine encoding using integer patch coordinates.

    Half the channels encode rows, half encode columns. Within each axis,
    channels 2k/2k+1 are sin/cos(t / 10000**(2k / axis_dim)).
    """
    if height < 1 or width < 1:
        raise ValueError("Position grid dimensions must be positive")
    if dim < 4 or dim % 4:
        raise ValueError("Position encoding dimension must be divisible by 4")
    axis_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
        / axis_dim
    )

    def encode_axis(length: int) -> Tensor:
        phases = torch.arange(length, device=device, dtype=torch.float32)[:, None] * frequencies
        return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)

    rows = encode_axis(height)[:, None, :].expand(height, width, axis_dim)
    columns = encode_axis(width)[None, :, :].expand(height, width, axis_dim)
    return torch.cat((rows, columns), dim=-1).reshape(height * width, dim).to(dtype=dtype)


class MaskPatchEncoder(nn.Module):
    """Learned local mask features plus exact valid-pixel patch statistics."""

    def __init__(self, patch_size: int, config: FusionConfig) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.score_scale = float(config.score_scale)
        hidden = config.mask_hidden_size
        self.patch_conv = nn.Conv2d(1, hidden, kernel_size=patch_size, stride=patch_size)
        self.context_conv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.hidden_norm = nn.LayerNorm(hidden) if config.normalize_hidden else nn.Identity()

    def forward(
        self, masks: Tensor, mask_valid: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("masks must have shape [batch, 1, height, width]")
        if masks.shape[0] < 1:
            raise ValueError("masks must contain at least one image")
        if masks.is_complex():
            raise ValueError("Mask scores must be real numbers")
        if min(masks.shape[-2:]) < self.patch_size:
            raise ValueError("Mask height and width must each contain at least one vision patch")
        if mask_valid is None:
            valid = torch.ones_like(masks, dtype=torch.bool)
        else:
            if mask_valid.ndim == 3:
                mask_valid = mask_valid.unsqueeze(1)
            if mask_valid.shape != masks.shape:
                raise ValueError("mask_valid must match masks, with an optional omitted channel axis")
            if mask_valid.is_complex() or not torch.all((mask_valid == 0) | (mask_valid == 1)):
                raise ValueError("mask_valid must contain only boolean or binary 0/1 values")
            valid = mask_valid.to(device=masks.device, dtype=torch.bool)

        # Keep expert scores on one global scale; padding never participates in
        # the CNN or the patch mean/max. FP32 accumulation avoids fp16 sums.
        scores = masks.to(dtype=torch.float32).masked_fill(~valid, 0.0)
        if not torch.isfinite(scores).all():
            raise ValueError("Valid mask scores must be finite")
        if torch.any(scores < 0) or torch.any(scores > self.score_scale):
            raise ValueError(f"Valid mask scores must lie in [0, {self.score_scale:g}]")
        normalized = scores / self.score_scale
        patches = F.unfold(normalized, kernel_size=self.patch_size, stride=self.patch_size)
        valid_pixels = F.unfold(
            valid.to(dtype=torch.float32), kernel_size=self.patch_size, stride=self.patch_size
        ).bool()
        counts = valid_pixels.sum(dim=1)
        valid_tokens = counts > 0
        means = patches.sum(dim=1) / counts.clamp_min(1)
        maxima = patches.masked_fill(~valid_pixels, -torch.inf).amax(dim=1)
        maxima = torch.where(valid_tokens, maxima, torch.zeros_like(maxima))

        hidden = F.gelu(self.patch_conv(normalized.to(dtype=self.patch_conv.weight.dtype)))
        grid_shape = (hidden.shape[-2], hidden.shape[-1])
        # Do not let a learned bias at a fully padded patch enter neighboring
        # patches through the context convolution.
        grid_valid = valid_tokens.reshape(masks.shape[0], 1, *grid_shape)
        hidden = hidden.masked_fill(~grid_valid, 0.0)
        hidden = F.gelu(self.context_conv(hidden))
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = self.hidden_norm(hidden)
        statistics = torch.stack((means, maxima), dim=-1).to(dtype=hidden.dtype)
        features = torch.cat((hidden, statistics), dim=-1)
        features = features.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        return features, valid_tokens, grid_shape


class PriorAttentionLayer(nn.Module):
    """Attention whose Q/K depend only on mask/location, never visual content."""

    def __init__(self, vision_dim: int, config: FusionConfig, prior_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.attention_dim // config.num_heads
        prior_dim = config.prior_dim if prior_dim is None else prior_dim
        if isinstance(prior_dim, bool) or not isinstance(prior_dim, int) or prior_dim < 1:
            raise ValueError("prior_dim must be a positive integer")
        self.q_proj = nn.Linear(prior_dim, config.attention_dim)
        self.k_proj = nn.Linear(prior_dim, config.attention_dim)
        self.v_proj = nn.Linear(vision_dim, config.attention_dim)
        self.out_proj = nn.Linear(config.attention_dim, vision_dim)
        self.alpha = nn.Parameter(torch.tensor(float(config.alpha_init)))

    def forward(self, features: Tensor, prior: Tensor, valid_tokens: Tensor) -> Tensor:
        batch, length, _ = features.shape

        def heads(value: Tensor) -> Tensor:
            return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query = heads(self.q_proj(prior))
        key = heads(self.k_proj(prior))
        value = heads(self.v_proj(features))
        # A dummy key makes entirely padded samples safe on every SDPA backend.
        # Their query outputs are explicitly discarded below, including biases.
        safe_keys = valid_tokens.clone()
        safe_keys[:, 0] = safe_keys[:, 0] | ~safe_keys.any(dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=safe_keys[:, None, None, :], dropout_p=0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        delta = self.out_proj(attended)
        delta = delta.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        # FSDP stores this scalar as shape [1]. Keep scalar type-promotion
        # semantics under BF16 autocast identical to single-device execution.
        return features + self.alpha.reshape(()) * delta


class ExpertMaskFusion(nn.Module):
    """Produce [updated visual features; direct expert features] for a projector.

    ``both`` enables attention and direct features. ``direct`` skips attention;
    ``attention`` zeroes the direct feature slots; ``none`` returns [X; zeros].
    Keeping these slots in every mode makes checkpoint shapes mode-independent.
    """

    def __init__(self, vision_dim: int, patch_size: int, config: FusionConfig) -> None:
        super().__init__()
        for name, value in (("vision_dim", vision_dim), ("patch_size", patch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(config, FusionConfig):
            raise TypeError("config must be a FusionConfig")
        self.vision_dim = vision_dim
        self.patch_size = patch_size
        self.config = config
        if config.mask_encoder_type == "cnn":
            self.mask_encoder = MaskPatchEncoder(patch_size, config)
            self.expert_projection = nn.Sequential(
                nn.Linear(config.expert_feature_dim, config.expert_dim),
                nn.GELU(),
                nn.Linear(config.expert_dim, config.expert_dim),
            )
            self.position_projection = nn.Linear(config.position_dim, config.position_hidden_size)
            self.prior_dim = config.prior_dim
        else:
            # SigLIP features are extracted by the shared, frozen vision tower
            # outside this module. No second tower or scalar-mask CNN is owned.
            self.mask_encoder = None
            self.expert_projection = None
            self.position_projection = None
            self.prior_dim = vision_dim
        self.prior_norm = nn.LayerNorm(self.prior_dim) if config.normalize_e else nn.Identity()
        self.layers = nn.ModuleList(
            PriorAttentionLayer(vision_dim, config, prior_dim=self.prior_dim)
            for _ in range(config.num_layers)
        )
        # Initialize after all existing modules so enabling this branch leaves
        # their random initialization unchanged for the same seed.
        self.direct_mlp = None
        self.direct_norm = None
        if config.direct_residual_mlp:
            direct_dim = vision_dim if config.mask_encoder_type == "siglip_jet" else config.mask_hidden_size
            if config.mask_encoder_type == "siglip_jet":
                self.direct_norm = nn.LayerNorm(direct_dim)
            self.direct_mlp = nn.Sequential(
                nn.Linear(direct_dim, config.direct_mlp_hidden_size),
                nn.GELU(),
                nn.Linear(config.direct_mlp_hidden_size, direct_dim),
            )
            nn.init.zeros_(self.direct_mlp[2].weight)
            nn.init.zeros_(self.direct_mlp[2].bias)

    @property
    def expert_feature_dim(self) -> int:
        return self.vision_dim if self.config.mask_encoder_type == "siglip_jet" else self.config.expert_feature_dim

    def encode_masks(
        self, masks: Tensor, mask_valid: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
        if self.mask_encoder is None:
            raise ValueError("siglip_jet requires externally extracted expert_features")
        return self.mask_encoder(masks, mask_valid)

    def encode_direct(
        self, expert_features: Tensor, valid_tokens: Optional[Tensor] = None
    ) -> Tensor:
        """Adapt the direct projector path without changing the Q/K input.

        CNN mode adapts H and preserves mean/max; Jet mode adapts all of A with
        an independent pre-LN. A zero final layer initially preserves either.
        """
        if expert_features.ndim != 3 or expert_features.shape[-1] != self.expert_feature_dim:
            raise ValueError("expert_features must have shape [batch, patches, expert_feature_dim]")
        direct = expert_features
        if self.direct_mlp is not None:
            if self.config.mask_encoder_type == "siglip_jet":
                direct = expert_features + self.direct_mlp(self.direct_norm(expert_features))
            else:
                hidden = expert_features[..., :self.config.mask_hidden_size]
                statistics = expert_features[..., self.config.mask_hidden_size:]
                direct = torch.cat((hidden + self.direct_mlp(hidden), statistics), dim=-1)
        if valid_tokens is not None:
            if valid_tokens.shape != expert_features.shape[:2]:
                raise ValueError("valid_tokens must match batch and patch dimensions")
            # Learned MLP biases must never turn padding into expert features.
            direct = direct.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        return direct

    def encode_prior(
        self,
        expert_features: Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
        valid_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        if expert_features.ndim != 3 or expert_features.shape[-1] != self.expert_feature_dim:
            raise ValueError("expert_features must have shape [batch, patches, expert_feature_dim]")
        if self.config.mask_encoder_type == "siglip_jet":
            # Learned SigLIP positions are already present in A. The direct
            # residual MLP must not affect this mask-only attention prior.
            prior = self.prior_norm(expert_features)
        else:
            if grid_shape is None or grid_shape[0] * grid_shape[1] != expert_features.shape[1]:
                raise ValueError("Position grid must match the expert feature sequence")
            position = position_encoding_2d(
                *grid_shape, self.config.position_dim,
                device=expert_features.device, dtype=expert_features.dtype,
            )
            expert = self.expert_projection(expert_features)
            spatial = self.position_projection(position).unsqueeze(0).expand(expert.shape[0], -1, -1)
            # E really is [U; R]. There is no extra projection/compression before Q/K.
            prior = self.prior_norm(torch.cat((expert, spatial), dim=-1))
        if valid_tokens is not None:
            if valid_tokens.shape != expert_features.shape[:2]:
                raise ValueError("valid_tokens must match batch and patch dimensions")
            prior = prior.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        return prior

    def forward(
        self,
        image_features: Tensor,
        masks: Optional[Tensor] = None,
        mask_valid: Optional[Tensor] = None,
        *,
        mask_only: bool = False,
        expert_features: Optional[Tensor] = None,
        valid_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        if image_features.ndim != 3 or image_features.shape[-1] != self.vision_dim:
            raise ValueError("image_features must have shape [batch, patches, vision_dim]")
        if min(image_features.shape[:2]) < 1:
            raise ValueError("image_features must contain at least one image and patch")
        if not image_features.is_floating_point():
            raise ValueError("image_features must be floating-point features")
        if self.config.mode == "none" and not mask_only:
            direct = image_features.new_zeros(*image_features.shape[:2], self.expert_feature_dim)
            return torch.cat((image_features, direct), dim=-1)
        if self.config.mask_encoder_type == "siglip_jet":
            if (not isinstance(expert_features, Tensor) or expert_features.ndim != 3
                    or expert_features.shape != image_features.shape
                    or not expert_features.is_floating_point()):
                raise ValueError("siglip_jet expert_features must be floating-point [batch, patches, vision_dim] matching image_features")
            if (not isinstance(valid_tokens, Tensor) or valid_tokens.dtype != torch.bool
                    or valid_tokens.shape != image_features.shape[:2]):
                raise ValueError("siglip_jet valid_tokens must be boolean [batch, patches]")
            valid_tokens = valid_tokens.to(device=image_features.device)
            expert = expert_features.to(device=image_features.device, dtype=image_features.dtype)
            expert = expert.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
            if not torch.isfinite(expert).all():
                raise ValueError("Valid siglip_jet expert_features must be finite")
            grid_shape = None
        else:
            if expert_features is not None or valid_tokens is not None:
                raise ValueError("External expert_features/valid_tokens are only supported by siglip_jet")
            if masks is None:
                raise ValueError("Mask inputs are required when fusion is enabled")
            masks = masks.to(device=image_features.device)
            if mask_valid is not None:
                mask_valid = mask_valid.to(device=image_features.device)
            expert, valid_tokens, grid_shape = self.encode_masks(masks, mask_valid)
        if expert.shape[:2] != image_features.shape[:2]:
            raise ValueError(
                "Mask patch grid must match vision features: "
                f"mask={tuple(expert.shape[:2])}, vision={tuple(image_features.shape[:2])}"
            )
        if mask_only:
            # Runtime phase-1 branch, not a persistent architecture ablation.
            # Keep the complete fusion forward as the FSDP/ZeRO entrypoint;
            # calling mask_encoder directly would bypass its parent's hooks.
            direct = self.encode_direct(expert, valid_tokens)
            return torch.cat((torch.zeros_like(image_features), direct), dim=-1)
        updated = image_features
        if self.config.mode in ("both", "attention"):
            prior = self.encode_prior(expert, grid_shape, valid_tokens)
            for layer in self.layers:
                # E remains purely mask/position-based at every layer; only V's
                # visual stream is updated by preceding layers.
                updated = layer(updated, prior, valid_tokens)
        if self.config.mode == "attention":
            direct = torch.zeros_like(expert)
        else:
            direct = self.encode_direct(expert, valid_tokens)
        return torch.cat((updated, direct.to(dtype=updated.dtype)), dim=-1)


class SplitInputLinear(nn.Module):
    """A linear projector with physically separate visual and expert weights.

    Phase 1 optimizes only expert_weight. Frozen visual_weight and bias are
    different Parameters, so AdamW decay/momentum cannot change their columns.
    Concatenating weights inside forward preserves the ordinary Linear's
    numerical path, including autocast; exported adapters merge these weights.
    """

    def __init__(self, linear: nn.Linear, visual_dim: int) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("SplitInputLinear requires an nn.Linear")
        if (isinstance(visual_dim, bool) or not isinstance(visual_dim, int)
                or not 0 < visual_dim < linear.in_features):
            raise ValueError("visual_dim must leave nonempty visual and expert columns")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.visual_dim = visual_dim
        self.visual_weight = nn.Parameter(linear.weight[:, :visual_dim].detach().clone(), requires_grad=False)
        self.expert_weight = nn.Parameter(linear.weight[:, visual_dim:].detach().clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.train(linear.training)

    def forward(self, features: Tensor) -> Tensor:
        weight = torch.cat((self.visual_weight, self.expert_weight), dim=1)
        return F.linear(features, weight, self.bias)


def expand_projector(projector: nn.Module, extra_dim: int) -> nn.Module:
    """Copy a linear/MLP projector and add zero-initialized input columns.

    No supplied module is mutated. Original weights, bias, dtype, device,
    train/eval mode, and requires_grad flags are preserved; subsequent layers
    are deep-copied unchanged. ``mm_hidden_size`` remains a vision property.
    """
    if isinstance(extra_dim, bool) or not isinstance(extra_dim, int) or extra_dim < 0:
        raise ValueError("extra_dim must be a nonnegative integer")
    if isinstance(projector, nn.Linear):
        first = projector
    elif isinstance(projector, nn.Sequential) and len(projector) and isinstance(projector[0], nn.Linear):
        first = projector[0]
    else:
        raise TypeError("Projector must be Linear or Sequential whose first layer is Linear")
    expanded_first = nn.Linear(
        first.in_features + extra_dim,
        first.out_features,
        bias=first.bias is not None,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        expanded_first.weight.zero_()
        expanded_first.weight[:, :first.in_features].copy_(first.weight)
        if first.bias is not None:
            expanded_first.bias.copy_(first.bias)
    expanded_first.weight.requires_grad_(first.weight.requires_grad)
    if first.bias is not None:
        expanded_first.bias.requires_grad_(first.bias.requires_grad)
    expanded_first.train(first.training)
    if isinstance(projector, nn.Linear):
        return expanded_first
    expanded = copy.deepcopy(projector)
    expanded[0] = expanded_first
    return expanded
