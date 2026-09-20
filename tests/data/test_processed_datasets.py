"""Regression guards on the built datasets themselves.

These assert properties of `datasets/processed/*.csv`, which are DVC outputs
rather than git contents. They skip when the data has not been built, so CI
(which runs without the dataset) stays green while a local `dvc repro` is
still held to the full data contract.
"""

from pathlib import Path

import pandas as pd
import pytest

from ml.data.clean import LEAKAGE_DISCHARGE_DISPOSITIONS, RAW_TARGET_COLUMN, TARGET_COLUMN

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "datasets" / "processed"
SPLIT_FILES = ("train.csv", "eval_frozen.csv", "future_stream.csv")

pytestmark = pytest.mark.skipif(
    not all((PROCESSED_DIR / name).exists() for name in SPLIT_FILES),
    reason="processed datasets not built (run `dvc repro`)",
)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(PROCESSED_DIR / name, low_memory=False) for name in SPLIT_FILES}


@pytest.mark.parametrize("name", SPLIT_FILES)
def test_processed_split_has_no_raw_target_column(splits, name):
    """The leakage guard: readmitted_30d is derived from readmitted, so the raw
    column must not survive into anything a model can read."""
    assert RAW_TARGET_COLUMN not in splits[name].columns
    assert TARGET_COLUMN in splits[name].columns


@pytest.mark.parametrize("name", SPLIT_FILES)
def test_processed_split_has_no_leakage_discharges(splits, name):
    assert not splits[name]["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_DISPOSITIONS).any()


@pytest.mark.parametrize("name", SPLIT_FILES)
def test_processed_split_has_one_encounter_per_patient(splits, name):
    assert splits[name]["patient_nbr"].is_unique


def test_no_patient_appears_in_more_than_one_split(splits):
    train, evaluation, future = (set(splits[n]["patient_nbr"]) for n in SPLIT_FILES)
    assert train.isdisjoint(evaluation)
    assert train.isdisjoint(future)
    assert evaluation.isdisjoint(future)


def test_splits_are_chronological_and_70_15_15(splits):
    train, evaluation, future = (splits[n] for n in SPLIT_FILES)

    assert train["encounter_id"].max() < evaluation["encounter_id"].min()
    assert evaluation["encounter_id"].max() < future["encounter_id"].min()

    total = sum(len(df) for df in (train, evaluation, future))
    assert len(train) / total == pytest.approx(0.70, abs=0.005)
    assert len(evaluation) / total == pytest.approx(0.15, abs=0.005)
    assert len(future) / total == pytest.approx(0.15, abs=0.005)
