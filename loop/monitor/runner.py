"""One window, or a whole scenario, end to end.

This is WORKFLOW.md steps 11-12 in code: take the serving stream, apply a
scenario, cut it into windows, measure each window against the training
reference, write the Evidently report, and persist a `drift_events` row --
including for the windows where nothing happened.

`run_scenario` takes its reference, model, stream and session as arguments so
the same function that the APScheduler worker calls is the one the tests call,
with synthetic pieces and no files, no database service and no trained model.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from loop.monitor.config import (
    BENCHMARK_EPOCH,
    DRIFT_REPORTS_DIR,
    LIVE_SCENARIO,
    SERVING_STREAM_PATH,
    WINDOW_DURATION,
    WINDOW_ROWS,
)
from loop.monitor.drift import WindowDrift, compute_window_drift
from loop.monitor.persistence import record_window
from loop.monitor.reference import MonitoringReference, engineered_features
from loop.monitor.scoring import ScoringModel
from loop.monitor.windows import MonitoringWindow, iter_windows
from scenarios.injection import SCENARIO_SEED, apply_scenario


class StreamUnavailableError(RuntimeError):
    """The serving stream could not be read. Never swallowed."""


@dataclass(frozen=True)
class MonitorOutcome:
    """What one `run_scenario` call measured."""

    scenario: str
    results: tuple[WindowDrift, ...]
    persisted_event_ids: tuple[str, ...] = ()

    @property
    def quiet(self) -> bool:
        """True when no window in the run saw drift -- the S5 control's expected shape."""
        return all(result.quiet for result in self.results)


def load_serving_stream(path: Path = SERVING_STREAM_PATH) -> pd.DataFrame:
    """The Week 2 `future_stream.csv`, which no training or evaluation code opens."""
    if not path.exists():
        raise StreamUnavailableError(
            f"{path} not found -- run `dvc repro` to build the Week 2 datasets."
        )
    return pd.read_csv(path, low_memory=False)


def report_path_for(scenario: str, window_index: int, reports_dir: Path) -> Path:
    """`reports/drift/<scenario>/window_000.html`.

    Under `reports/`, which is git-ignored and DVC-tracked territory: the HTML
    embeds the window's distribution charts, so it is an artifact to link from
    the dashboard, never something to commit.
    """
    return reports_dir / scenario.lower() / f"window_{window_index:03d}.html"


def measure_window(
    window: MonitoringWindow,
    *,
    scenario: str,
    reference: MonitoringReference,
    model: ScoringModel,
    reports_dir: Path | None = DRIFT_REPORTS_DIR,
) -> WindowDrift:
    """Measures a single window. No database, no side effects beyond the report file."""
    report_path = (
        report_path_for(scenario, window.index, reports_dir) if reports_dir is not None else None
    )
    return compute_window_drift(
        scenario=scenario,
        window_index=window.index,
        window_start=window.start,
        window_end=window.end,
        reference_features=reference.features,
        reference_scores=reference.scores,
        current_features=engineered_features(window.frame),
        current_scores=model.score(window.frame),
        report_path=report_path,
        model_version=model.version,
        data_version=reference.data_version,
    )


def run_scenario(
    scenario: str,
    *,
    reference: MonitoringReference,
    model: ScoringModel,
    stream: pd.DataFrame | None = None,
    session: Session | None = None,
    windows: int | None = None,
    window_rows: int = WINDOW_ROWS,
    reports_dir: Path | None = DRIFT_REPORTS_DIR,
    epoch: datetime = BENCHMARK_EPOCH,
    duration: timedelta = WINDOW_DURATION,
    seed: int = SCENARIO_SEED,
) -> MonitorOutcome:
    """Applies a scenario to the stream and measures every window in it.

    `session=None` measures without persisting, which is how the tests and a
    dry run use it. With a session, **every** window is written, including the
    quiet ones: PROJECT_DESIGN.md §6 makes the persisted history the input to
    Week 7's consecutive-window persistence rule, and a missing quiet window
    would read as a gap in the evidence rather than as evidence of calm.
    """
    if stream is None:
        stream = load_serving_stream()

    injected = stream if scenario == LIVE_SCENARIO else apply_scenario(stream, scenario, seed=seed)

    results: list[WindowDrift] = []
    event_ids: list[str] = []
    for window in iter_windows(
        injected,
        window_rows=window_rows,
        max_windows=windows,
        epoch=epoch,
        duration=duration,
    ):
        result = measure_window(
            window,
            scenario=scenario,
            reference=reference,
            model=model,
            reports_dir=reports_dir,
        )
        results.append(result)
        if session is not None:
            event_ids.append(record_window(session, result).event_id)

    return MonitorOutcome(
        scenario=scenario,
        results=tuple(results),
        persisted_event_ids=tuple(event_ids),
    )


def summarise(outcome: MonitorOutcome) -> str:
    """A one-line-per-window summary, for the CLI and the worker log."""
    lines = []
    for result in outcome.results:
        breaching = ", ".join(stat.feature for stat in result.breaching_features) or "none"
        lines.append(
            f"[{result.scenario}] window {result.window_index}: "
            f"max_psi={result.max_psi:.4f} "
            f"breaching={result.breaching_feature_count} ({breaching}) "
            f"prediction_psi={result.prediction_psi:.4f} "
            f"prediction_drift={result.prediction_drift}"
        )
    return "\n".join(lines)
