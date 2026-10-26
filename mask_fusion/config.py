"""Serializable configuration for the mask-conditioned visual adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class FusionConfig:
    """Sizes and ablations independent of a checkpoint's vision architecture.

    In CNN mode, ``expert_dim`` and ``position_hidden_size`` form E. Jet mode
    uses the actual vision-tower width for its expert features and prior.
    All modes retain the same extended projector input width.
    """

    mask_hidden_size: int = 128
    expert_dim: int = 256
    position_dim: int = 128
    position_hidden_size: int = 256
    attention_dim: int = 256
    num_heads: int = 8
    num_layers: int = 1
    alpha_init: float = 0.01
    normalize_hidden: bool = True
    normalize_e: bool = True
    score_scale: float = 255.0
    mode: str = "both"
    direct_residual_mlp: bool = False
    direct_mlp_hidden_size: int = 256
    mask_encoder_type: str = "cnn"

    def __post_init__(self) -> None:
        for name in (
            "mask_hidden_size", "expert_dim", "position_dim",
            "position_hidden_size", "attention_dim", "num_heads", "num_layers",
            "direct_mlp_hidden_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.position_dim % 4:
            raise ValueError("position_dim must be divisible by 4 (sin/cos for two axes)")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        for name in ("alpha_init", "score_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")
        for name in ("normalize_hidden", "normalize_e", "direct_residual_mlp"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.mode not in ("both", "direct", "attention", "none"):
            raise ValueError("mode must be one of: both, direct, attention, none")
        if self.mask_encoder_type not in ("cnn", "siglip_jet"):
            raise ValueError("mask_encoder_type must be one of: cnn, siglip_jet")

    @property
    def expert_feature_dim(self) -> int:
        """CNN feature width; use ExpertMaskFusion.expert_feature_dim in Jet mode."""
        return self.mask_hidden_size + 2

    @property
    def prior_dim(self) -> int:
        """CNN prior width; Jet mode obtains its width from the vision tower."""
        return self.expert_dim + self.position_hidden_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FusionConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Fusion configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown fusion configuration fields: {sorted(unknown)}")
        return cls(**dict(values))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "FusionConfig":
        return cls.from_dict(json.loads(value))
