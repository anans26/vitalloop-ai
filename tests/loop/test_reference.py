"""The monitoring reference: what is selected, how it is sampled, what is refused."""

import numpy as np
import pandas as pd
import pytest

from loop.monitor.config import (
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_FEATURES,
    MONITORED_NUMERIC_FEATURES,
    REFERENCE_PATH,
)
from loop.monitor.reference import (
    ReferenceUnavailableError,
    build_reference,
    engineered_features,
    load_reference,
    reference_data_version,
    sample_reference,
)
from loop.monitor.scoring import ScoringModel
from ml.data.clean import MISSING_CATEGORY
from ml.data.features import MEDICATION_COLUMNS


class _ConstantModel:
    """Stands in for the champion; the reference does not care what it scores."""

    def predict_proba(self, frame):
        return np.column_stack([np.full(len(frame), 0.8), np.full(len(frame), 0.2)])


@pytest.fixture
def model() -> ScoringModel:
    return ScoringModel(model=_ConstantModel(), version="test-model", source="local")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def test_only_the_monitored_features_survive(synthetic_clean_df):
    monitored = engineered_features(synthetic_clean_df)
    assert list(monitored.columns) == list(MONITORED_FEATURES)


def test_engineered_columns_come_from_the_fitted_pipelines_own_transformer(synthetic_clean_df):
    """`service_utilization` here must be the same computation the model was trained on."""
    monitored = engineered_features(synthetic_clean_df)
    expected = (
        synthetic_clean_df["number_outpatient"]
        + synthetic_clean_df["number_emergency"]
        + synthetic_clean_df["number_inpatient"]
    )
    pd.testing.assert_series_equal(
        monitored["service_utilization"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_identifiers_and_targets_are_not_monitored(synthetic_clean_df):
    monitored = engineered_features(synthetic_clean_df)
    for forbidden in ("encounter_id", "patient_nbr", "readmitted_30d", "readmitted"):
        assert forbidden not in monitored.columns


def test_the_individual_medication_columns_are_not_monitored():
    """23 near-constant columns would dilute the Week 7 breadth term for no signal."""
    assert not set(MONITORED_FEATURES) & set(MEDICATION_COLUMNS)
    assert "num_med_changes" in MONITORED_NUMERIC_FEATURES
    assert "insulin_changed" in MONITORED_NUMERIC_FEATURES


def test_nulls_become_the_same_missing_category_the_model_is_given(synthetic_clean_df):
    """The pipeline imputes categoricals with "missing"; monitoring must see that too."""
    frame = synthetic_clean_df.copy()
    frame["A1Cresult"] = None

    monitored = engineered_features(frame)
    assert monitored["A1Cresult"].isna().sum() == 0
    assert set(monitored["A1Cresult"]) == {MISSING_CATEGORY}


def test_no_monitored_categorical_is_left_null(synthetic_clean_df):
    monitored = engineered_features(synthetic_clean_df)
    for column in MONITORED_CATEGORICAL_FEATURES:
        assert monitored[column].isna().sum() == 0


def test_a_frame_missing_a_monitored_feature_is_refused(synthetic_clean_df):
    with pytest.raises(ReferenceUnavailableError):
        engineered_features(synthetic_clean_df.drop(columns=["num_lab_procedures"]))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_a_frame_smaller_than_the_cap_is_taken_whole(synthetic_clean_df):
    assert len(sample_reference(synthetic_clean_df, rows=10_000)) == len(synthetic_clean_df)


def test_sampling_is_seeded_and_reproducible(synthetic_clean_df):
    first = sample_reference(synthetic_clean_df, rows=100, seed=7)
    second = sample_reference(synthetic_clean_df, rows=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100


def test_a_different_seed_draws_a_different_sample(synthetic_clean_df):
    first = sample_reference(synthetic_clean_df, rows=100, seed=1)
    second = sample_reference(synthetic_clean_df, rows=100, seed=2)
    assert not first.index.equals(second.index)


def test_the_sample_keeps_stream_order(synthetic_clean_df):
    sampled = sample_reference(synthetic_clean_df, rows=100, seed=7)
    assert sampled["encounter_id"].is_monotonic_increasing


def test_the_sample_draws_without_replacement(synthetic_clean_df):
    sampled = sample_reference(synthetic_clean_df, rows=100, seed=7)
    assert not sampled["encounter_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def test_a_reference_carries_features_scores_and_lineage(synthetic_clean_df, model):
    reference = build_reference(synthetic_clean_df, model, rows=150, data_version="dvc:abc")
    assert reference.rows == 150
    assert len(reference.scores) == 150
    assert list(reference.features.columns) == list(MONITORED_FEATURES)
    assert reference.model_version == "test-model"
    assert reference.data_version == "dvc:abc"


def test_building_a_reference_twice_gives_the_same_reference(synthetic_clean_df, model):
    first = build_reference(synthetic_clean_df, model, rows=150)
    second = build_reference(synthetic_clean_df, model, rows=150)
    pd.testing.assert_frame_equal(first.features, second.features)
    assert np.array_equal(first.scores, second.scores)


def test_a_missing_reference_file_is_reported_not_guessed(model, tmp_path):
    with pytest.raises(ReferenceUnavailableError):
        load_reference(model, path=tmp_path / "absent.csv")


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------
def test_the_reference_is_the_frozen_evaluation_slice():
    """The launch reference, per the calibration recorded in loop/monitor/config.py."""
    assert REFERENCE_PATH.name == "eval_frozen.csv"


def test_an_unknown_reference_file_has_no_dvc_hash(tmp_path):
    assert reference_data_version(tmp_path / "somethingelse.csv") is None
