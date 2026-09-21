"""The two evaluation sets §3.12 gates on.

    "Compares challenger vs champion on (a) a frozen holdout and (b) the most
     recent labeled window"

(a) is `datasets/processed/eval_frozen.csv` -- the slice the champion's
published metrics were established on, which no training or calibration code
opens. (b) is the monitoring window the Decision Card was written about,
re-cut from the serving stream with the card's own scenario applied.

Using the card's own window rather than "whatever is newest now" is what makes
a gate result re-derivable: the card names a window, the window is a
deterministic slice of a seeded stream (`loop/monitor/windows.py`), so
re-running the gate on the same card a month later compares the same rows.

RISK_ANALYSIS.md §2 is why both sets are required rather than just the frozen
one: "Overfitting to the frozen eval set through repeated gating -- Gate
evaluates on frozen holdout **and** most-recent labeled window."

Nothing here is cached or copied anywhere: the frames are read, scored, reduced
to aggregate metrics, and dropped. No patient row reaches a gate result.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from loop.gate.criteria import FROZEN_HOLDOUT, RECENT_LABELED_WINDOW
from loop.monitor.config import (
    BENCHMARK_EPOCH,
    LIVE_SCENARIO,
    SERVING_STREAM_PATH,
    WINDOW_DURATION,
    WINDOW_ROWS,
)
from loop.monitor.runner import load_serving_stream
from loop.monitor.windows import iter_windows
from ml.config import EVAL_PATH
from ml.data.clean import TARGET_COLUMN
from scenarios.injection import SCENARIO_SEED, apply_scenario


class EvaluationSetError(RuntimeError):
    """An evaluation set could not be assembled. Never swallowed."""


def frozen_holdout(path: Path = EVAL_PATH) -> pd.DataFrame:
    """The frozen evaluation slice, exactly as Week 3 published against."""
    if not path.exists():
        raise EvaluationSetError(
            f"{path} not found -- run `dvc repro` to build the Week 2 datasets."
        )
    frame = pd.read_csv(path, low_memory=False)
    if TARGET_COLUMN not in frame.columns:
        raise EvaluationSetError(f"{path.name} carries no {TARGET_COLUMN!r} column")
    return frame


def labeled_window(
    scenario: str,
    window_start: datetime,
    *,
    stream: pd.DataFrame | None = None,
    stream_path: Path = SERVING_STREAM_PATH,
    seed: int = SCENARIO_SEED,
    window_rows: int = WINDOW_ROWS,
    epoch: datetime = BENCHMARK_EPOCH,
    duration=WINDOW_DURATION,
) -> pd.DataFrame:
    """The rows of one monitoring window, with their labels.

    The scenario is re-applied with the same seed Week 6 injected it with, so
    the window is the one the drift event measured rather than a similar slice
    of untouched traffic.
    """
    frame = stream if stream is not None else load_serving_stream(stream_path)
    if str(scenario).lower() != LIVE_SCENARIO:
        frame = apply_scenario(frame, scenario, seed=seed)

    windows = iter_windows(frame, window_rows=window_rows, epoch=epoch, duration=duration)
    for window in windows:
        if window.start == window_start:
            return window.frame
    raise EvaluationSetError(
        f"no window starting {window_start.isoformat()} in scenario {scenario!r}; "
        f"the stream yields {len(windows)} window(s) from {epoch.isoformat()}"
    )


def evaluation_frames(
    scenario: str,
    window_start: datetime,
    *,
    sets: tuple[str, ...] = (FROZEN_HOLDOUT, RECENT_LABELED_WINDOW),
    stream: pd.DataFrame | None = None,
    stream_path: Path = SERVING_STREAM_PATH,
    eval_path: Path = EVAL_PATH,
    seed: int = SCENARIO_SEED,
) -> dict[str, pd.DataFrame]:
    """Both sets, keyed by the names `configs/gate-v1.yaml` uses."""
    frames: dict[str, pd.DataFrame] = {}
    for name in sets:
        if name == FROZEN_HOLDOUT:
            frames[name] = frozen_holdout(eval_path)
        elif name == RECENT_LABELED_WINDOW:
            frames[name] = labeled_window(
                scenario, window_start, stream=stream, stream_path=stream_path, seed=seed
            )
        else:
            raise EvaluationSetError(f"unknown evaluation set {name!r}")
    return frames
