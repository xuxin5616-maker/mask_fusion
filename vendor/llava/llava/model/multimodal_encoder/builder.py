"""Construct the SigLIP vision tower used by LLaVA checkpoints."""

from .siglip_encoder import SigLipVisionTower


def build_vision_tower(config, **kwargs):
    name = getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    if not name:
        raise ValueError("Checkpoint config must specify mm_vision_tower")
    return SigLipVisionTower(name, vision_tower_cfg=config, **kwargs)
