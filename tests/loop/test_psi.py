"""The hand-rolled PSI and KS, and the cross-check against Evidently.

IMPLEMENTATION_ROADMAP.md Week 6 asks for "hand-rolled PSI for 2 features as a
cross-check test". `test_cross_check_*` are those two features; the rest pin the
formula itself against cases whose answer can be worked out on paper.
"""

import math

import numpy as np
import pandas as pd
import pytest

from loop.monitor.psi import (
    categorical_psi,
    ks_p_value,
    numeric_bin_shares,
    numeric_psi,
    population_stability_index,
    psi_from_shares,
)


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------
def test_identical_distributions_give_zero_psi():
    assert psi_from_shares([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]) == 0.0


def test_psi_matches_the_value_worked_out_by_hand():
    """70/30 -> 50/50 is 0.2*ln(5/7) + 0.2*ln(5/3), which is 0.16946."""
    expected = (0.5 - 0.7) * math.log(0.5 / 0.7) + (0.5 - 0.3) * math.log(0.5 / 0.3)
    assert psi_from_shares([0.7, 0.3], [0.5, 0.5]) == pytest.approx(expected, abs=1e-12)
    assert psi_from_shares([0.7, 0.3], [0.5, 0.5]) == pytest.approx(0.169459, abs=1e-6)


def test_psi_is_symmetric_in_its_two_arguments():
    """The (c - r) * ln(c / r) form is symmetric; a test guards against a typo."""
    forward = psi_from_shares([0.7, 0.2, 0.1], [0.3, 0.4, 0.3])
    reverse = psi_from_shares([0.3, 0.4, 0.3], [0.7, 0.2, 0.1])
    assert forward == pytest.approx(reverse, abs=1e-12)


def test_counts_and_proportions_give_the_same_psi():
    assert psi_from_shares([700, 300], [500, 500]) == pytest.approx(
        psi_from_shares([0.7, 0.3], [0.5, 0.5]), abs=1e-12
    )


def test_an_empty_bin_on_one_side_stays_finite():
    """A category present in one sample and absent from the other must not be inf."""
    value = psi_from_shares([0.5, 0.5], [1.0, 0.0])
    assert math.isfinite(value)
    assert value > 1.0


def test_psi_grows_monotonically_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0.0, 1.0, 20_000))
    values = [
        numeric_psi(reference, pd.Series(rng.normal(shift, 1.0, 20_000)))
        for shift in (0.0, 0.25, 0.5, 1.0)
    ]
    assert values == sorted(values)
    assert values[0] < 0.01


@pytest.mark.parametrize(
    "reference, current",
    [([], [1.0]), ([1.0], []), ([None, None], [1.0])],
)
def test_empty_samples_are_rejected_rather_than_scored(reference, current):
    with pytest.raises(ValueError):
        numeric_psi(pd.Series(reference, dtype="float64"), pd.Series(current, dtype="float64"))


def test_mismatched_bin_counts_are_rejected():
    with pytest.raises(ValueError):
        psi_from_shares([0.5, 0.5], [0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def test_numeric_bins_have_infinite_outer_edges_so_nothing_is_dropped():
    """A current value beyond the reference range must land in a bin, not vanish."""
    reference = pd.Series(np.arange(100.0))
    current = pd.Series([-500.0, 500.0])
    reference_counts, current_counts = numeric_bin_shares(reference, current)
    assert current_counts.sum() == len(current)
    assert reference_counts.sum() == len(reference)


def test_a_constant_reference_does_not_raise():
    value = numeric_psi(pd.Series([5.0] * 100), pd.Series([5.0] * 50 + [9.0] * 50))
    assert math.isfinite(value)
    assert value > 0.0


def test_categorical_psi_ignores_the_order_values_first_appear_in():
    first = categorical_psi(pd.Series(["a", "b", "b"]), pd.Series(["b", "a", "a"]))
    second = categorical_psi(pd.Series(["b", "b", "a"]), pd.Series(["a", "a", "b"]))
    assert first == pytest.approx(second, abs=1e-12)


def test_population_stability_index_dispatches_on_the_feature_kind():
    numbers = pd.Series(np.linspace(0.0, 1.0, 500))
    assert population_stability_index(numbers, numbers, numeric=True) == pytest.approx(0.0)

    labels = pd.Series(["x"] * 70 + ["y"] * 30)
    shifted = pd.Series(["x"] * 50 + ["y"] * 50)
    assert population_stability_index(labels, shifted, numeric=False) == pytest.approx(
        0.169459, abs=1e-6
    )


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------
def test_ks_p_value_is_high_for_samples_from_the_same_distribution():
    rng = np.random.default_rng(3)
    assert ks_p_value(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)) > 0.05


def test_ks_p_value_collapses_for_a_shifted_distribution():
    rng = np.random.default_rng(3)
    assert ks_p_value(rng.normal(0, 1, 2000), rng.normal(1.0, 1, 2000)) < 0.001


def test_ks_is_deterministic_for_the_same_inputs():
    rng = np.random.default_rng(11)
    reference, current = rng.normal(0, 1, 500), rng.normal(0.3, 1, 500)
    assert ks_p_value(reference, current) == ks_p_value(reference, current)


# ---------------------------------------------------------------------------
# The cross-check: two features, two independent implementations
# ---------------------------------------------------------------------------
# `psi.py` bins numeric features at reference quantiles; Evidently bins them
# with Sturges edges over the pooled sample. The two will not agree to the
# decimal -- they are not supposed to. What must agree is the verdict, because
# that is what the policy reads.
def _evidently_psi(reference: pd.Series, current: pd.Series, column: str) -> float:
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    definition = DataDefinition(numerical_columns=[column])
    run = Report(metrics=[ValueDrift(column=column, method="psi")], include_tests=False).run(
        current_data=Dataset.from_pandas(
            pd.DataFrame({column: current}), data_definition=definition
        ),
        reference_data=Dataset.from_pandas(
            pd.DataFrame({column: reference}), data_definition=definition
        ),
    )
    return float(run.dict()["metrics"][0]["value"])


@pytest.mark.parametrize("column", ["num_lab_procedures", "num_medications"])
def test_cross_check_agrees_with_evidently_that_an_undrifted_feature_is_quiet(
    synthetic_clean_df, column
):
    rng = np.random.default_rng(5)
    reference = synthetic_clean_df[column].astype(float)
    current = pd.Series(rng.choice(reference.to_numpy(), size=len(reference), replace=True))

    hand_rolled = numeric_psi(reference, current)
    evidently = _evidently_psi(reference, current, column)
    assert hand_rolled < 0.10
    assert evidently < 0.10


@pytest.mark.parametrize("column", ["num_lab_procedures", "num_medications"])
def test_cross_check_agrees_with_evidently_that_a_shifted_feature_breaches(
    synthetic_clean_df, column
):
    """The S1 mechanism -- a 20% upward shift -- seen by both implementations."""
    reference = synthetic_clean_df[column].astype(float)
    current = (reference * 1.6).round()

    hand_rolled = numeric_psi(reference, current)
    evidently = _evidently_psi(reference, current, column)
    assert hand_rolled >= 0.25
    assert evidently >= 0.25
