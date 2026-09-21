"""The Evidently run: one monitoring window in, aggregate statistics out.

ARCHITECTURE.md §3.7 makes Evidently the drift engine, so every number that
reaches a `drift_events` row is produced here and nowhere else. The hand-rolled
PSI in `psi.py` is a cross-check for the tests, not a second source of truth.

**What leaves this module is aggregate-only, by construction.** A
`FeatureDrift` carries a feature *name* and three statistics; there is no field
on it that can hold a patient value, so `feature_stats` cannot accidentally
become a copy of the window. The Evidently HTML report is written to a path the
row references -- the report is an artifact, not a column.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from loop.monitor.config import (
    KS_P_VALUE_THRESHOLD,
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_NUMERIC_FEATURES,
    PREDICTION_COLUMN,
    PREDICTION_PSI_THRESHOLD,
    PSI_BREACH_THRESHOLD,
    PSI_SEVERE_THRESHOLD,
)
from ml.config import PROJECT_ROOT

NUMERIC = "numerical"
CATEGORICAL = "categorical"

PSI_METHOD = "psi"
KS_METHOD = "ks"


class DriftComputationError(RuntimeError):
    """Evidently could not produce a statistic the row requires."""


@dataclass(frozen=True)
class FeatureDrift:
    """One feature's measurement for one window. Statistics only, never values."""

    feature: str
    kind: str
    psi: float
    ks_p_value: float | None
    breaching: bool
    severe: bool

    def to_record(self) -> dict:
        """The JSON shape persisted in drift_events.feature_stats.

        An explicit dict rather than `asdict`, so adding a field to this class
        can never silently widen what gets written to the database.
        """
        return {
            "feature": self.feature,
            "kind": self.kind,
            "psi": self.psi,
            "ks_p_value": self.ks_p_value,
            "breaching": self.breaching,
            "severe": self.severe,
        }


@dataclass(frozen=True)
class WindowDrift:
    """Everything measured about one window, ready to become a `drift_events` row."""

    scenario: str
    window_index: int
    window_start: datetime
    window_end: datetime
    feature_stats: tuple[FeatureDrift, ...]
    prediction_psi: float
    prediction_drift: bool
    max_psi: float
    breaching_feature_count: int
    reference_rows: int
    current_rows: int
    policy_thresholds: dict
    report_uri: str | None = None
    model_version: str | None = None
    data_version: str | None = None

    @property
    def breaching_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(stat for stat in self.feature_stats if stat.breaching)

    @property
    def quiet(self) -> bool:
        """No feature breached and the score distribution held -- a `NO_OP` window."""
        return self.breaching_feature_count == 0 and not self.prediction_drift

    def with_report(self, report_uri: str | None) -> "WindowDrift":
        return replace(self, report_uri=report_uri)


def current_thresholds() -> dict:
    """The thresholds this measurement was taken under, recorded on the row.

    Stored per row rather than assumed, so a window measured last month can
    still be re-read correctly after a reviewed threshold change.
    """
    return {
        "psi_breach": PSI_BREACH_THRESHOLD,
        "psi_severe": PSI_SEVERE_THRESHOLD,
        "ks_p_value": KS_P_VALUE_THRESHOLD,
        "prediction_psi": PREDICTION_PSI_THRESHOLD,
    }


def repository_relative_uri(path: Path) -> str:
    """`reports/drift/s1/window_000.html`, not `F:/somebody/checkout/reports/...`.

    An absolute path would put one developer's filesystem layout into an
    append-only audit row, and would not resolve for the monitor container, the
    dashboard, or anyone reading the trail on another machine. A path outside
    the repository is kept as given rather than mangled.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _build_dataset(features: pd.DataFrame, scores: np.ndarray, columns) -> Dataset:
    """Feature frame plus the champion's scores, typed for Evidently."""
    numeric, categorical = columns
    frame = features.loc[:, [*numeric, *categorical]].copy()
    frame[PREDICTION_COLUMN] = np.asarray(scores, dtype=float)
    definition = DataDefinition(
        numerical_columns=[*numeric, PREDICTION_COLUMN],
        categorical_columns=list(categorical),
    )
    return Dataset.from_pandas(frame, data_definition=definition)


def _metric_values(run) -> dict[tuple[str, str], float]:
    """(column, method) -> value, read off the Evidently run's own config blocks.

    Keyed by what Evidently says it computed rather than by list position, so a
    reordered or deduplicated metric list cannot silently mislabel a number.
    """
    values: dict[tuple[str, str], float] = {}
    for metric in run.dict().get("metrics", []):
        config = metric.get("config") or {}
        column, method = config.get("column"), config.get("method")
        value = metric.get("value")
        if column is None or method is None or not isinstance(value, int | float):
            continue
        values[(str(column), str(method))] = float(value)
    return values


def compute_window_drift(
    *,
    scenario: str,
    window_index: int,
    window_start: datetime,
    window_end: datetime,
    reference_features: pd.DataFrame,
    reference_scores: np.ndarray,
    current_features: pd.DataFrame,
    current_scores: np.ndarray,
    numeric_features=MONITORED_NUMERIC_FEATURES,
    categorical_features=MONITORED_CATEGORICAL_FEATURES,
    report_path: Path | None = None,
    model_version: str | None = None,
    data_version: str | None = None,
) -> WindowDrift:
    """Measures one window against the reference. Writes the HTML report if asked.

    Every returned statistic comes from the single Evidently run below: PSI for
    every monitored feature, KS for the numeric ones (KS is undefined on
    categories, so those carry `ks_p_value = None`), and PSI on the champion's
    score distribution for ARCHITECTURE.md §3.7's prediction drift.
    """
    numeric = [c for c in numeric_features if c in current_features.columns]
    categorical = [c for c in categorical_features if c in current_features.columns]
    if not numeric and not categorical:
        raise DriftComputationError("no monitored features present in the window")
    if len(current_features) == 0 or len(reference_features) == 0:
        raise DriftComputationError("reference and window must both be non-empty")

    columns = (numeric, categorical)
    reference_dataset = _build_dataset(reference_features, reference_scores, columns)
    current_dataset = _build_dataset(current_features, current_scores, columns)

    metrics = [ValueDrift(column=column, method=PSI_METHOD) for column in (*numeric, *categorical)]
    metrics += [ValueDrift(column=column, method=KS_METHOD) for column in numeric]
    metrics.append(ValueDrift(column=PREDICTION_COLUMN, method=PSI_METHOD))

    run = Report(metrics=metrics, include_tests=False).run(
        current_data=current_dataset, reference_data=reference_dataset
    )
    values = _metric_values(run)

    report_uri = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        run.save_html(str(report_path))
        report_uri = repository_relative_uri(report_path)

    feature_stats: list[FeatureDrift] = []
    for column in (*numeric, *categorical):
        psi = values.get((column, PSI_METHOD))
        if psi is None:
            raise DriftComputationError(f"Evidently returned no PSI for {column!r}")
        feature_stats.append(
            FeatureDrift(
                feature=column,
                kind=NUMERIC if column in numeric else CATEGORICAL,
                psi=psi,
                ks_p_value=values.get((column, KS_METHOD)) if column in numeric else None,
                breaching=psi >= PSI_BREACH_THRESHOLD,
                severe=psi >= PSI_SEVERE_THRESHOLD,
            )
        )

    prediction_psi = values.get((PREDICTION_COLUMN, PSI_METHOD))
    if prediction_psi is None:
        raise DriftComputationError("Evidently returned no PSI for the prediction column")

    return WindowDrift(
        scenario=scenario,
        window_index=window_index,
        window_start=window_start,
        window_end=window_end,
        feature_stats=tuple(feature_stats),
        prediction_psi=prediction_psi,
        prediction_drift=prediction_psi >= PREDICTION_PSI_THRESHOLD,
        max_psi=max(stat.psi for stat in feature_stats),
        breaching_feature_count=sum(1 for stat in feature_stats if stat.breaching),
        reference_rows=len(reference_features),
        current_rows=len(current_features),
        policy_thresholds={**current_thresholds(), "monitored_features": len(feature_stats)},
        report_uri=report_uri,
        model_version=model_version,
        data_version=data_version,
    )
