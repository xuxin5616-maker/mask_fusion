"""Logging helpers used by the retained LLaVA model code."""

import torch.distributed as dist


def rank0_print(*args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}:", *args)
    else:
        print(*args)
