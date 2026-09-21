"""Slicing a serving stream into monitoring windows.

The UCI extract has no calendar timestamps (DATASET_ANALYSIS.md), so a window
is a contiguous block of the encounter-ordered stream -- the same chronology
proxy `ml/data/split.py` uses for the train/eval/future split.

Window bounds are therefore derived from the window's *position in the stream*,
not from the wall clock: replaying scenario S1 tomorrow produces the same
`window_start` and `window_end` it produced today, which is what makes the
drift benchmark re-runnable. When a measurement was actually taken is
`drift_events.created_at`'s job, and that is a separate column for exactly this
reason.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from loop.monitor.config import (
    BENCHMARK_EPOCH,
    DROP_PARTIAL_WINDOW,
    WINDOW_DURATION,
    WINDOW_ROWS,
)


@dataclass(frozen=True)
class MonitoringWindow:
    """One window of serving traffic, with the bounds it will be recorded under."""

    index: int
    start: datetime
    end: datetime
    frame: pd.DataFrame

    @property
    def rows(self) -> int:
        return len(self.frame)


def window_bounds(
    index: int,
    *,
    epoch: datetime = BENCHMARK_EPOCH,
    duration: timedelta = WINDOW_DURATION,
) -> tuple[datetime, datetime]:
    """Bounds for window `index`, counting from zero. Contiguous, never overlapping."""
    if index < 0:
        raise ValueError("window index must be non-negative")
    start = epoch + index * duration
    return start, start + duration


def iter_windows(
    stream: pd.DataFrame,
    *,
    window_rows: int = WINDOW_ROWS,
    max_windows: int | None = None,
    epoch: datetime = BENCHMARK_EPOCH,
    duration: timedelta = WINDOW_DURATION,
    drop_partial: bool = DROP_PARTIAL_WINDOW,
) -> list[MonitoringWindow]:
    """Consecutive windows over the stream, in stream order.

    A trailing block shorter than `window_rows` is dropped by default: PSI on a
    short sample is noisy, and a noisy final window would register in the
    benchmark as a false trigger rather than as the sampling artefact it is.
    """
    if window_rows < 1:
        raise ValueError("window_rows must be at least 1")
    if max_windows is not None and max_windows < 0:
        raise ValueError("max_windows must be non-negative")

    windows: list[MonitoringWindow] = []
    for index, offset in enumerate(range(0, len(stream), window_rows)):
        if max_windows is not None and len(windows) >= max_windows:
            break
        frame = stream.iloc[offset : offset + window_rows]
        if len(frame) < window_rows and drop_partial:
            break
        start, end = window_bounds(index, epoch=epoch, duration=duration)
        windows.append(MonitoringWindow(index=index, start=start, end=end, frame=frame))
    return windows
