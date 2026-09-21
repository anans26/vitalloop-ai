"""S1-S5: seeded, schema-valid, and each one changing only what it claims to.

The benchmark is only a benchmark if a reviewer re-running a scenario gets the
stream this repository measured. Determinism is therefore tested per scenario,
not once.
"""

import numpy as np
import pandas as pd
import pytest

from ml.data.clean import TARGET_COLUMN
from scenarios.injection import (
    S1_COLUMNS,
    S2_COLUMN,
    S2_INTRODUCED_VALUES,
    S3_ELDERLY_AGE_BANDS,
    SCENARIO_SEED,
    SCENARIOS,
    apply_scenario,
)


@pytest.fixture
def stream(synthetic_clean_df) -> pd.DataFrame:
    """A stream with the missing HbA1c values S2 needs to find."""
    frame = synthetic_clean_df.copy()
    frame.loc[frame.index[::3], "A1Cresult"] = None
    return frame


# ---------------------------------------------------------------------------
# Properties every scenario must have
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_same_seed_reproduces_the_same_stream(stream, scenario):
    first = apply_scenario(stream, scenario, seed=SCENARIO_SEED)
    second = apply_scenario(stream, scenario, seed=SCENARIO_SEED)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_no_scenario_mutates_the_caller_frame(stream, scenario):
    before = stream.copy()
    apply_scenario(stream, scenario)
    pd.testing.assert_frame_equal(stream, before)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_scenario_preserves_the_row_count_and_columns(stream, scenario):
    injected = apply_scenario(stream, scenario)
    assert len(injected) == len(stream)
    assert list(injected.columns) == list(stream.columns)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_encounter_ids_stay_unique(stream, scenario):
    """encounter_id is the chronology proxy; a duplicate would break windowing."""
    injected = apply_scenario(stream, scenario)
    assert not injected["encounter_id"].duplicated().any()


def test_an_unknown_scenario_is_rejected(stream):
    with pytest.raises(KeyError):
        apply_scenario(stream, "S9")


def test_scenario_names_are_case_insensitive(stream):
    pd.testing.assert_frame_equal(apply_scenario(stream, "s1"), apply_scenario(stream, "S1"))


# ---------------------------------------------------------------------------
# S1 -- covariate shift
# ---------------------------------------------------------------------------
def test_s1_shifts_the_two_named_columns_upward(stream):
    injected = apply_scenario(stream, "S1")
    for column in S1_COLUMNS:
        assert injected[column].mean() > stream[column].mean() * 1.15


def test_s1_leaves_every_other_column_alone(stream):
    injected = apply_scenario(stream, "S1")
    untouched = [c for c in stream.columns if c not in S1_COLUMNS]
    pd.testing.assert_frame_equal(injected[untouched], stream[untouched])


def test_s1_stays_inside_the_schema_domain(stream):
    """`ml/data/schema.py` requires these counts to be non-negative integers."""
    injected = apply_scenario(stream, "S1")
    for column in S1_COLUMNS:
        assert injected[column].min() >= 0
        assert pd.api.types.is_integer_dtype(injected[column])


# ---------------------------------------------------------------------------
# S2 -- coding change
# ---------------------------------------------------------------------------
def test_s2_records_hba1c_for_previously_blank_encounters(stream):
    injected = apply_scenario(stream, "S2")
    assert injected[S2_COLUMN].isna().sum() < stream[S2_COLUMN].isna().sum()


def test_s2_only_fills_blanks_and_never_rewrites_a_recorded_result(stream):
    injected = apply_scenario(stream, "S2")
    already_recorded = stream[S2_COLUMN].notna()
    pd.testing.assert_series_equal(
        injected.loc[already_recorded, S2_COLUMN], stream.loc[already_recorded, S2_COLUMN]
    )


def test_s2_introduces_only_schema_valid_hba1c_values(stream):
    injected = apply_scenario(stream, "S2")
    assert set(injected[S2_COLUMN].dropna()) <= set(S2_INTRODUCED_VALUES)


def test_s2_leaves_every_other_column_alone(stream):
    injected = apply_scenario(stream, "S2")
    untouched = [c for c in stream.columns if c != S2_COLUMN]
    pd.testing.assert_frame_equal(injected[untouched], stream[untouched])


# ---------------------------------------------------------------------------
# S3 -- prevalence shift
# ---------------------------------------------------------------------------
def test_s3_raises_the_elderly_share(stream):
    injected = apply_scenario(stream, "S3")
    before = stream["age"].isin(S3_ELDERLY_AGE_BANDS).mean()
    after = injected["age"].isin(S3_ELDERLY_AGE_BANDS).mean()
    assert after > before


def test_s3_raises_prior_utilisation(stream):
    injected = apply_scenario(stream, "S3")
    utilisation = ["number_inpatient", "number_emergency", "number_outpatient"]
    assert injected[utilisation].sum(axis=1).mean() > stream[utilisation].sum(axis=1).mean()


def test_s3_keeps_the_window_the_same_size(stream):
    """A changed row count would confound case-mix drift with a traffic change."""
    assert len(apply_scenario(stream, "S3")) == len(stream)


# ---------------------------------------------------------------------------
# S4 -- label drift
# ---------------------------------------------------------------------------
def test_s4_changes_labels(stream):
    injected = apply_scenario(stream, "S4")
    assert not injected[TARGET_COLUMN].equals(stream[TARGET_COLUMN])


def test_s4_leaves_every_feature_untouched(stream):
    """S4 must be invisible to input drift -- that is the whole point of it."""
    injected = apply_scenario(stream, "S4")
    features = [c for c in stream.columns if c != TARGET_COLUMN]
    pd.testing.assert_frame_equal(injected[features], stream[features])


def test_s4_roughly_preserves_the_class_prevalence(stream):
    """Labels are redrawn at the base rate, so the imbalance does not move."""
    injected = apply_scenario(stream, "S4")
    assert injected[TARGET_COLUMN].mean() == pytest.approx(stream[TARGET_COLUMN].mean(), abs=0.05)


def test_s4_degrades_the_relationship_a_model_would_have_learned(stream):
    """The point of S4: the signal the champion learned gets weaker.

    Measured on a frame with a strong, deliberate feature-label relationship.
    The shared fixture's relationship is faint by design, and 30% relabelling of
    300 noisy rows moves a 0.06 correlation in either direction.
    """
    rng = np.random.default_rng(0)
    frame = stream.sample(n=len(stream), replace=True, random_state=0).reset_index(drop=True)
    frame["encounter_id"] = range(len(frame))
    frame["number_inpatient"] = rng.integers(0, 6, len(frame))
    frame[TARGET_COLUMN] = (frame["number_inpatient"] >= 3).astype(int)

    injected = apply_scenario(frame, "S4")
    before = abs(np.corrcoef(frame["number_inpatient"], frame[TARGET_COLUMN])[0, 1])
    after = abs(np.corrcoef(injected["number_inpatient"], injected[TARGET_COLUMN])[0, 1])
    assert after < before


def test_s4_needs_a_labelled_stream(stream):
    with pytest.raises(KeyError):
        apply_scenario(stream.drop(columns=[TARGET_COLUMN]), "S4")


# ---------------------------------------------------------------------------
# S5 -- the control
# ---------------------------------------------------------------------------
def test_s5_returns_the_stream_untouched(stream):
    """The thresholds are calibrated on this stream; touching it invalidates S1-S4."""
    pd.testing.assert_frame_equal(apply_scenario(stream, "S5"), stream)


def test_s5_ignores_the_seed(stream):
    pd.testing.assert_frame_equal(
        apply_scenario(stream, "S5", seed=1), apply_scenario(stream, "S5", seed=999)
    )
