"""Opt-in wall-clock accounting for instrumented pruning blocks.

CUDA work submitted before a block is drained before starting its clock. The
ending synchronization completes the block's work before stopping that clock,
so these durations can be subtracted from an enclosing wall-clock measurement.
Synchronization perturbs execution; the subtraction is a diagnostic estimate.
"""

import math
from contextlib import contextmanager, nullcontext
from time import perf_counter

import torch


class PruningTimer:
    """Accumulate one generation's pruning time on a single CPU/CUDA device.

    Nested sections using the same timer are included in their outer section,
    without a second clock or synchronization and without double counting.
    The caller owns this timer's lifecycle; model sample resets do not clear it.
    """

    def __init__(self):
        self.seconds = 0.0
        self.sections = 0
        self._active = False
        self._device = None

    @contextmanager
    def measure(self, device):
        if self._active:
            yield
            return
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Pruning timing supports CPU or a single CUDA device")
        if self._device is not None and device != self._device:
            raise ValueError("Pruning timing requires all measured blocks on one device")
        self._device = device
        self._active = True
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            try:
                yield
            finally:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                total = self.seconds + elapsed
                if elapsed < 0 or not math.isfinite(elapsed) or not math.isfinite(total):
                    raise RuntimeError("Invalid pruning wall-clock interval")
                self.seconds = total
                self.sections += 1
        finally:
            self._active = False


def measure_pruning(core, device):
    """Return a no-op context unless a generation-scoped timer is attached."""
    timer = getattr(core, "_pruning_timer", None)
    return nullcontext() if timer is None else timer.measure(device)
