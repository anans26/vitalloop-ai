"""The Evidently engine: statistics, the breach policy, and what may leave it.

These run on synthetic frames and a small synthetic model, so CI needs no
dataset, no trained artifact and no database.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from loop.monitor.config import (
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_FEATURES,
    MONITORED_NUMERIC_FEATURES,
    PREDICTION_COLUMN,
    PSI_BREACH_THRESHOLD,
    PSI_SEVERE_THRESHOLD,
)
from loop.monitor.drift import DriftComputationError, compute_window_drift, current_thresholds
from loop.monitor.reference import engineered_features
from scenarios.injection import apply_scenario

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(days=1)


def _measure(reference_frame, current_frame, *, scenario="test", **kwargs):
    reference = engineered_features(reference_frame)
    current = engineered_features(current_frame)
    rng = np.random.default_rng(0)
    return compute_window_drift(
        scenario=scenario,
        window_index=0,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        reference_features=reference,
        reference_scores=kwargs.pop("reference_scores", rng.uniform(0, 0.3, len(reference))),
        current_features=current,
        current_scores=kwargs.pop("current_scores", rng.uniform(0, 0.3, len(current))),
        **kwargs,
    )


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    from tests.conftest import make_synthetic_clean_df

    frame = make_synthetic_clean_df(n_rows=3000, seed=11)
    frame.loc[frame.index[::3], "A1Cresult"] = None
    return frame


@pytest.fixture(scope="module")
def reference(stream) -> pd.DataFrame:
    return stream.iloc[:1500]


@pytest.fixture(scope="module")
def window(stream) -> pd.DataFrame:
    return stream.iloc[1500:]


# Evidently runs 25 columns per call, so the measurements every test reads are
# taken once for the module rather than once per test.
@pytest.fixture(scope="module")
def baseline(reference, window):
    return _measure(reference, window)


@pytest.fixture(scope="module")
def s1_result(reference, window):
    return _measure(reference, apply_scenario(window, "S1"), scenario="S1")


# ---------------------------------------------------------------------------
# What is measured
# ---------------------------------------------------------------------------
def test_every_monitored_feature_gets_a_statistic(baseline):
    result = baseline
    assert [stat.feature for stat in result.feature_stats] == [
        *MONITORED_NUMERIC_FEATURES,
        *MONITORED_CATEGORICAL_FEATURES,
    ]
    assert len(result.feature_stats) == len(MONITORED_FEATURES)


def test_numeric_features_carry_a_ks_p_value_and_categorical_ones_do_not(baseline):
    """KS is undefined on categories; a fabricated p-value there would be a lie."""
    by_name = {stat.feature: stat for stat in baseline.feature_stats}
    assert by_name["num_lab_procedures"].ks_p_value is not None
    assert 0.0 <= by_name["num_lab_procedures"].ks_p_value <= 1.0
    assert by_name["race"].ks_p_value is None


def test_row_counts_and_window_bounds_are_carried_through(baseline, reference, window):
    result = baseline
    assert result.reference_rows == len(reference)
    assert result.current_rows == len(window)
    assert result.window_start == WINDOW_START
    assert result.window_end == WINDOW_END


def test_the_thresholds_used_are_recorded_on_the_result(baseline):
    result = baseline
    assert result.policy_thresholds["psi_breach"] == PSI_BREACH_THRESHOLD
    assert result.policy_thresholds["psi_severe"] == PSI_SEVERE_THRESHOLD
    assert result.policy_thresholds["monitored_features"] == len(MONITORED_FEATURES)
    assert current_thresholds()["prediction_psi"] == pytest.approx(0.10)


def test_measuring_a_window_against_itself_is_quiet(reference):
    """The floor: identical distributions must not breach anything."""
    result = _measure(reference, reference)
    assert result.breaching_feature_count == 0
    assert result.max_psi < PSI_BREACH_THRESHOLD
    assert not result.prediction_drift
    assert result.quiet


def test_the_same_inputs_measure_to_the_same_numbers(reference, window):
    """Determinism: the benchmark is only re-runnable if the engine is."""
    first, second = _measure(reference, window), _measure(reference, window)
    assert first.max_psi == second.max_psi
    assert first.prediction_psi == second.prediction_psi
    assert [s.psi for s in first.feature_stats] == [s.psi for s in second.feature_stats]


# ---------------------------------------------------------------------------
# The breach policy
# ---------------------------------------------------------------------------
def test_a_feature_breaches_exactly_at_the_threshold(baseline):
    for stat in baseline.feature_stats:
        assert stat.breaching == (stat.psi >= PSI_BREACH_THRESHOLD)
        assert stat.severe == (stat.psi >= PSI_SEVERE_THRESHOLD)


def test_severe_implies_breaching(baseline):
    assert all(stat.breaching for stat in baseline.feature_stats if stat.severe)


def test_the_breach_count_and_max_agree_with_the_per_feature_statistics(s1_result):
    result = s1_result
    assert result.breaching_feature_count == sum(1 for s in result.feature_stats if s.breaching)
    assert result.max_psi == max(s.psi for s in result.feature_stats)
    assert set(result.breaching_features) == {s for s in result.feature_stats if s.breaching}


def test_an_injected_covariate_shift_breaches_the_features_it_shifted(s1_result):
    """S1 end to end: the two shifted columns must be the ones that light up."""
    result = s1_result
    breaching = {stat.feature for stat in result.breaching_features}
    assert "num_lab_procedures" in breaching
    assert result.max_psi >= PSI_BREACH_THRESHOLD
    assert not result.quiet


def test_an_injected_coding_change_breaches_the_recoded_feature(reference, window):
    result = _measure(reference, apply_scenario(window, "S2"), scenario="S2")
    by_name = {stat.feature: stat for stat in result.feature_stats}
    assert by_name["A1Cresult"].breaching
    assert by_name["A1Cresult"].severe


def test_the_untouched_control_does_not_breach_its_own_reference(reference):
    """S5's contract: an untouched stream measured against its own era is quiet."""
    result = _measure(reference, apply_scenario(reference, "S5"), scenario="S5")
    assert result.quiet
    assert result.breaching_feature_count == 0


def test_label_drift_is_invisible_to_input_drift(reference, window):
    """S4 moves labels only, so it must measure identically to the control."""
    control = _measure(reference, apply_scenario(window, "S5"), scenario="S5")
    labels = _measure(reference, apply_scenario(window, "S4"), scenario="S4")
    assert [s.psi for s in labels.feature_stats] == [s.psi for s in control.feature_stats]


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------
def test_prediction_drift_is_false_when_the_score_distribution_holds(reference, window):
    rng = np.random.default_rng(4)
    scores = rng.beta(2, 8, len(reference))
    result = _measure(
        reference,
        window,
        reference_scores=scores,
        current_scores=rng.beta(2, 8, len(window)),
    )
    assert not result.prediction_drift
    assert result.prediction_psi < 0.10


def test_prediction_drift_is_true_when_the_score_distribution_moves(reference, window):
    rng = np.random.default_rng(4)
    result = _measure(
        reference,
        window,
        reference_scores=rng.beta(2, 8, len(reference)),
        current_scores=rng.beta(8, 2, len(window)),
    )
    assert result.prediction_drift
    assert result.prediction_psi >= 0.10
    assert not result.quiet


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_an_evidently_report_is_written_when_a_path_is_given(reference, window, tmp_path):
    path = tmp_path / "drift" / "window_000.html"
    result = _measure(reference, window, report_path=path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert result.report_uri == path.as_posix()


def test_no_report_is_written_when_no_path_is_given(baseline, tmp_path):
    assert baseline.report_uri is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
def test_the_monitored_frame_cannot_contain_identifiers_or_labels(stream):
    """Selection by name is the control: there is no code path that adds them back."""
    monitored = engineered_features(stream)
    assert list(monitored.columns) == list(MONITORED_FEATURES)
    for forbidden in ("encounter_id", "patient_nbr", "readmitted_30d", "readmitted"):
        assert forbidden not in monitored.columns


def test_a_feature_record_holds_statistics_and_nothing_else(baseline):
    expected = {"feature", "kind", "psi", "ks_p_value", "breaching", "severe"}
    for stat in baseline.feature_stats:
        record = stat.to_record()
        assert set(record) == expected
        assert record["feature"] in MONITORED_FEATURES
        assert isinstance(record["psi"], float)


def test_nothing_but_names_statistics_and_flags_survives_into_the_record(baseline):
    """A structural check, not a substring search.

    Every string a record can carry is either a monitored feature name or a
    feature kind; everything else is a number, a bool or None. A patient value
    has nowhere to be.
    """
    allowed_strings = {*MONITORED_FEATURES, "numerical", "categorical"}
    for stat in baseline.feature_stats:
        for key, value in stat.to_record().items():
            if isinstance(value, str):
                assert value in allowed_strings, key
            else:
                assert value is None or isinstance(value, float | bool), key


def test_the_prediction_column_is_not_reported_as_a_monitored_feature(baseline):
    assert PREDICTION_COLUMN not in {s.feature for s in baseline.feature_stats}


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_an_empty_window_is_refused_rather_than_measured(reference):
    with pytest.raises(DriftComputationError):
        _measure(reference, reference.iloc[:0])


def test_a_frame_with_no_monitored_features_is_refused(reference):
    with pytest.raises(DriftComputationError):
        compute_window_drift(
            scenario="test",
            window_index=0,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            reference_features=pd.DataFrame({"unrelated": [1.0, 2.0]}),
            reference_scores=np.array([0.1, 0.2]),
            current_features=pd.DataFrame({"unrelated": [1.0, 2.0]}),
            current_scores=np.array([0.1, 0.2]),
        )


def test_the_report_uri_is_repository_relative(tmp_path):
    """An absolute path would put a local filesystem layout into the audit trail."""
    from loop.monitor.drift import repository_relative_uri
    from ml.config import PROJECT_ROOT

    inside = PROJECT_ROOT / "reports" / "drift" / "s1" / "window_000.html"
    assert repository_relative_uri(inside) == "reports/drift/s1/window_000.html"

    outside = tmp_path / "elsewhere.html"
    assert repository_relative_uri(outside) == outside.as_posix()
