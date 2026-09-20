"""Week 3 metric suite: deterministic, hand-checkable behaviour."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import (
    expected_calibration_error,
    recall_at_top_fraction,
    reliability_table,
    score_predictions,
    subgroup_metrics,
    threshold_counts,
    top_decile_threshold,
)

REQUIRED_METRICS = {
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "recall_at_top_decile",
    "counts_at_default_threshold",
    "counts_at_top_decile_threshold",
}


@pytest.fixture
def separable_scores():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
    return y_true, y_prob


def test_score_predictions_reports_every_required_metric(separable_scores):
    metrics = score_predictions(*separable_scores)
    assert REQUIRED_METRICS.issubset(metrics.keys())


def test_perfectly_separable_scores_give_auc_one(separable_scores):
    assert score_predictions(*separable_scores)["roc_auc"] == 1.0


def test_expected_calibration_error_is_zero_for_perfect_calibration():
    """Half the rows at p=0.25 with a 25% observed rate, half at p=0.75 with 75%."""
    y_prob = np.array([0.25] * 4 + [0.75] * 4)
    y_true = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_detects_overconfidence():
    y_prob = np.full(10, 0.9)
    y_true = np.zeros(10, dtype=int)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.9, abs=1e-9)


def test_recall_at_top_fraction_counts_positives_in_the_top_slice():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.99, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06])
    # top 10% of 10 rows = 1 row, which holds 1 of the 2 positives
    assert recall_at_top_fraction(y_true, y_prob, 0.10) == pytest.approx(0.5)
    # top 20% = 2 rows, capturing both
    assert recall_at_top_fraction(y_true, y_prob, 0.20) == pytest.approx(1.0)


def test_threshold_counts_matches_a_hand_computed_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    counts = threshold_counts(y_true, y_prob, 0.5)

    assert (counts["true_negatives"], counts["false_positives"]) == (1, 1)
    assert (counts["false_negatives"], counts["true_positives"]) == (1, 1)
    assert counts["precision"] == pytest.approx(0.5)
    assert counts["recall"] == pytest.approx(0.5)


def test_top_decile_threshold_flags_about_a_tenth_of_rows():
    y_prob = np.linspace(0.0, 1.0, 100)
    cut = top_decile_threshold(y_prob, 0.10)
    assert (y_prob >= cut).sum() == pytest.approx(10, abs=1)


def test_reliability_table_bins_account_for_every_row():
    rng = np.random.default_rng(0)
    y_prob = rng.random(500)
    y_true = (rng.random(500) < y_prob).astype(int)

    table = reliability_table(y_true, y_prob, n_bins=10)
    assert len(table) == 10
    assert table["count"].sum() == 500


def test_subgroup_metrics_returns_null_auc_for_single_class_groups():
    frame = pd.DataFrame({"gender": ["F", "F", "F", "M", "M", "M"]})
    y_true = np.array([0, 1, 0, 1, 1, 1])  # M is all-positive
    y_prob = np.array([0.1, 0.9, 0.2, 0.7, 0.8, 0.6])

    result = subgroup_metrics(frame, y_true, y_prob, ["gender"])
    assert result["gender"]["M"]["roc_auc"] is None
    assert result["gender"]["M"]["count"] == 3
    assert result["gender"]["F"]["roc_auc"] is not None


def test_metrics_are_deterministic(separable_scores):
    assert score_predictions(*separable_scores) == score_predictions(*separable_scores)
