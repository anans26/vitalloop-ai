"""The gate's input: metric sets built from labeled data.

The contract asserted here is that `loop/gate/metrics.py` introduces no metric
mathematics of its own. Week 3's `ml/evaluate.py` is the single definition of
ROC-AUC, top-decile recall, Brier, ECE and the subgroup breakdown, and a gate
that recomputed any of them its own way would be defending a bar the champion's
published numbers were never measured against.
"""

import numpy as np
import pytest

from loop.gate.criteria import FROZEN_HOLDOUT
from loop.gate.metrics import (
    MetricSetError,
    metric_set,
    metric_sets_for,
    model_inputs,
    score_frame,
)
from ml.config import SUBGROUP_COLUMNS
from ml.data.clean import TARGET_COLUMN
from ml.data.features import NON_FEATURE_COLUMNS
from ml.evaluate import (
    expected_calibration_error,
    recall_at_top_fraction,
    score_predictions,
    subgroup_metrics,
)

from .conftest import ConstantChallenger


@pytest.fixture
def scored(synthetic_clean_df, champion_model):
    return synthetic_clean_df, score_frame(champion_model, synthetic_clean_df)


# ---------------------------------------------------------------------------
# The numbers are Week 3's numbers
# ---------------------------------------------------------------------------
def test_every_headline_metric_matches_ml_evaluate(scored):
    frame, probabilities = scored
    published = score_predictions(frame[TARGET_COLUMN], probabilities)
    measured = metric_set(FROZEN_HOLDOUT, frame, probabilities)

    assert measured.roc_auc == published["roc_auc"]
    assert measured.brier_score == published["brier_score"]
    assert measured.recall_at_top_decile == published["recall_at_top_decile"]
    assert measured.expected_calibration_error == published["expected_calibration_error"]
    assert measured.positive_rate == published["positive_rate"]


def test_the_subgroup_breakdown_matches_ml_evaluate(scored):
    frame, probabilities = scored
    published = subgroup_metrics(frame, frame[TARGET_COLUMN], probabilities, SUBGROUP_COLUMNS)
    measured = metric_set(FROZEN_HOLDOUT, frame, probabilities)

    assert set(measured.subgroup_roc_auc) == set(published)
    for column, groups in published.items():
        assert measured.subgroup_roc_auc[column] == {
            value: stats["roc_auc"] for value, stats in groups.items()
        }


def test_the_top_decile_fraction_is_honoured(scored):
    """The criteria file's operating point, not a constant buried here."""
    frame, probabilities = scored
    measured = metric_set(FROZEN_HOLDOUT, frame, probabilities, top_decile_fraction=0.25)
    assert measured.recall_at_top_decile == round(
        recall_at_top_fraction(frame[TARGET_COLUMN], probabilities, 0.25), 6
    )


def test_the_ece_uses_week_threes_binning(scored):
    frame, probabilities = scored
    measured = metric_set(FROZEN_HOLDOUT, frame, probabilities)
    assert measured.expected_calibration_error == round(
        expected_calibration_error(frame[TARGET_COLUMN], probabilities), 6
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_an_unlabeled_frame_is_refused(synthetic_clean_df, champion_model):
    """The gate compares models on labeled data; serving traffic is not that."""
    unlabeled = synthetic_clean_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(MetricSetError, match="carries no"):
        metric_set(FROZEN_HOLDOUT, unlabeled, np.zeros(len(unlabeled)))


def test_a_length_mismatch_is_refused(scored):
    frame, probabilities = scored
    with pytest.raises(MetricSetError, match="predictions for"):
        metric_set(FROZEN_HOLDOUT, frame, probabilities[:-1])


def test_a_single_class_frame_is_refused(synthetic_clean_df):
    """AUROC is undefined, so non-inferiority cannot be judged on it."""
    frame = synthetic_clean_df.copy()
    frame[TARGET_COLUMN] = 1
    with pytest.raises(MetricSetError, match="single outcome class"):
        metric_set(FROZEN_HOLDOUT, frame, np.linspace(0, 1, len(frame)))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_model_inputs_drop_identifiers_and_the_target(synthetic_clean_df):
    """The same exclusion list training and the monitor apply -- not a third one."""
    columns = set(model_inputs(synthetic_clean_df).columns)
    assert columns.isdisjoint(NON_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in columns


def test_scoring_is_deterministic(synthetic_clean_df, champion_model):
    first = score_frame(champion_model, synthetic_clean_df)
    second = score_frame(champion_model, synthetic_clean_df)
    assert np.array_equal(first, second)


def test_metric_sets_are_built_for_every_frame(gate_frames, champion_model):
    sets = metric_sets_for(champion_model, gate_frames)
    assert set(sets) == set(gate_frames)
    for name, value in sets.items():
        assert value.evaluation_set == name
        assert value.rows == len(gate_frames[name])


def test_a_constant_challenger_has_no_discrimination(gate_frames):
    """The degenerate model the gate must refuse: every patient scores alike."""
    sets = metric_sets_for(ConstantChallenger(), gate_frames)
    for value in sets.values():
        assert value.roc_auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def test_the_record_holds_statistics_only(scored):
    """§6: no patient row survives into anything that gets persisted."""
    frame, probabilities = scored
    record = metric_set(FROZEN_HOLDOUT, frame, probabilities).to_record()

    assert record["rows"] == len(frame)
    for key, value in record.items():
        if key == "subgroup_roc_auc":
            continue
        assert isinstance(value, (str, int, float))
