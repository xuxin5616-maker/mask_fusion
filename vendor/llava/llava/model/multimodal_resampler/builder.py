"""LLaVA checkpoints in this project use no additional vision resampler."""

import torch


class IdentityMap(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(config, delay_load=False, **kwargs):
    kind = getattr(config, "mm_resampler_type", None)
    if kind is not None:
        raise ValueError(f"Unsupported vision resampler for this LLaVA backend: {kind}")
    return IdentityMap()
