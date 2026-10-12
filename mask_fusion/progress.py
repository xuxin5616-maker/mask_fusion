"""Wall-clock progress estimates, excluding initialization and paused stages."""

from contextlib import contextmanager
import math
import time


def format_duration(seconds):
    if seconds is None:
        return "--:--:--"
    hours, remainder = divmod(max(0, math.ceil(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressTimer:
    """Estimate remaining work using only work completed in this invocation.

    Counts are absolute (including a resume offset); the rate excludes that
    offset. Pause around validation/checkpointing when counting train batches.
    This is an estimate: differing crop/token counts can change batch costs.
    """

    def __init__(self, total, completed=0, *, clock=None):
        if not 0 <= completed <= total:
            raise ValueError("Progress must satisfy 0 <= completed <= total")
        self.total = total
        self.completed = self.initial = completed
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.paused_seconds = 0.0
        self.pause_started = None

    @contextmanager
    def paused(self):
        outermost = self.pause_started is None
        if outermost:
            self.pause_started = self.clock()
        try:
            yield
        finally:
            if outermost:
                self.paused_seconds += self.clock() - self.pause_started
                self.pause_started = None

    def snapshot(self, completed=None):
        if completed is not None:
            if not self.completed <= completed <= self.total:
                raise ValueError("Progress must advance monotonically up to total")
            self.completed = completed
        now = self.clock() if self.pause_started is None else self.pause_started
        elapsed = max(0.0, now - self.started - self.paused_seconds)
        processed = self.completed - self.initial
        rate = processed / elapsed if elapsed > 0 and processed > 0 else None
        remaining = self.total - self.completed
        eta = 0.0 if remaining == 0 else (remaining / rate if rate else None)
        return {
            "completed": self.completed, "total": self.total,
            "elapsed": format_duration(elapsed), "eta": format_duration(eta),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta, 3) if eta is not None else None,
            "units_per_second": round(rate, 4) if rate is not None else None,
        }
