"""The gate's input: one metric set per model per evaluation set.

ARCHITECTURE.md §3.12 calls the gate "a pure function of two metric sets".
This module is what produces those sets, and it is deliberately the *only*
impure part of `loop/gate/`: it touches dataframes and a fitted model, so that
`loop/gate/gate.py` can touch neither.

**No new metric mathematics lives here.** ROC-AUC, recall at the top decile,
Brier, ECE and the per-subgroup breakdown all come from `ml/evaluate.py`, which
Week 3 wrote and whose subgroup block exists, in its own words, "because Week
8's promotion gate blocks on subgroup regression, so the baseline has to be on
record now". A gate that recomputed those numbers its own way would be gating
against a bar the published model card never measured.

**It carries statistics only.** A `MetricSet` holds counts and aggregates; no
row, identifier or score vector survives into it, which is what lets a gate
result be persisted next to a Decision Card under the same PHI posture.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml.config import SUBGROUP_COLUMNS, TOP_DECILE_FRACTION
from ml.data.clean import TARGET_COLUMN
from ml.data.features import NON_FEATURE_COLUMNS
from ml.evaluate import (
    expected_calibration_error,
    recall_at_top_fraction,
    subgroup_metrics,
)

METRIC_PRECISION = 6


class MetricSetError(RuntimeError):
    """A metric set could not be computed. Never swallowed."""


@dataclass(frozen=True)
class MetricSet:
    """One model's performance on one labeled evaluation set.

    `subgroup_roc_auc` maps a column to `{value: auc-or-None}`. A subgroup with
    a single outcome class has no defined AUROC; it is kept with a null rather
    than dropped, so a shrinking subgroup stays visible to the gate instead of
    silently ceasing to be checked.
    """

    evaluation_set: str
    rows: int
    positive_rate: float
    roc_auc: float
    recall_at_top_decile: float
    brier_score: float
    expected_calibration_error: float
    subgroup_roc_auc: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def to_record(self) -> dict:
        """The JSON shape stored inside `retrain_runs.gate_result`."""
        return {
            "evaluation_set": self.evaluation_set,
            "rows": self.rows,
            "positive_rate": self.positive_rate,
            "roc_auc": self.roc_auc,
            "recall_at_top_decile": self.recall_at_top_decile,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "subgroup_roc_auc": {
                column: dict(groups) for column, groups in self.subgroup_roc_auc.items()
            },
        }


def _round(value: float) -> float:
    return round(float(value), METRIC_PRECISION)


def metric_set(
    evaluation_set: str,
    frame: pd.DataFrame,
    y_prob,
    *,
    subgroup_columns: tuple[str, ...] = SUBGROUP_COLUMNS,
    top_decile_fraction: float = TOP_DECILE_FRACTION,
    target_column: str = TARGET_COLUMN,
) -> MetricSet:
    """Scores one already-predicted evaluation set.

    Takes probabilities rather than a model so the champion and the challenger
    are measured by identical code on identical rows -- the comparison the gate
    makes is only meaningful if nothing but the model differs.
    """
    if target_column not in frame.columns:
        raise MetricSetError(
            f"{evaluation_set!r} carries no {target_column!r} column; the gate compares "
            "models on labeled data and cannot score an unlabeled window"
        )
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) != len(frame):
        raise MetricSetError(f"{evaluation_set!r}: {len(y_prob)} predictions for {len(frame)} rows")
    y_true = np.asarray(frame[target_column], dtype=int)
    if len(np.unique(y_true)) < 2:
        raise MetricSetError(
            f"{evaluation_set!r} has a single outcome class; AUROC is undefined, so the "
            "gate cannot judge non-inferiority on it"
        )

    return MetricSet(
        evaluation_set=evaluation_set,
        rows=int(len(frame)),
        positive_rate=_round(y_true.mean()),
        roc_auc=_round(roc_auc_score(y_true, y_prob)),
        recall_at_top_decile=_round(recall_at_top_fraction(y_true, y_prob, top_decile_fraction)),
        brier_score=_round(brier_score_loss(y_true, y_prob)),
        expected_calibration_error=_round(expected_calibration_error(y_true, y_prob)),
        subgroup_roc_auc={
            column: {value: stats["roc_auc"] for value, stats in groups.items()}
            for column, groups in subgroup_metrics(frame, y_true, y_prob, subgroup_columns).items()
        },
    )


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The feature frame a fitted pipeline expects: no identifiers, no target.

    Same exclusion list `ml.data.features.split_features_target` applies, and
    the same one `loop/monitor/scoring.py` applies, so the gate scores rows the
    way training and serving do rather than inventing a third convention.
    """
    return frame.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in frame.columns])


def score_frame(model, frame: pd.DataFrame):
    """Calibrated probability of 30-day readmission, one per row."""
    return np.asarray(model.predict_proba(model_inputs(frame))[:, 1], dtype=float)


def metric_sets_for(
    model,
    frames: dict[str, pd.DataFrame],
    *,
    subgroup_columns: tuple[str, ...] = SUBGROUP_COLUMNS,
    top_decile_fraction: float = TOP_DECILE_FRACTION,
) -> dict[str, MetricSet]:
    """One model, every evaluation set §3.12 asks for."""
    return {
        name: metric_set(
            name,
            frame,
            score_frame(model, frame),
            subgroup_columns=subgroup_columns,
            top_decile_fraction=top_decile_fraction,
        )
        for name, frame in frames.items()
    }
