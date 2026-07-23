import pandas as pd
import pytest

from ml.data.split import time_sliced_patient_split


def _deduped_df(n_patients: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": range(1, n_patients + 1),
            "patient_nbr": range(1000, 1000 + n_patients),
        }
    )


def test_split_produces_no_patient_overlap_across_any_pair_of_splits():
    """The exact leakage guard RISK_ANALYSIS.md calls for: a unit test, not vigilance."""
    train, eval_frozen, future = time_sliced_patient_split(_deduped_df())

    train_ids = set(train["patient_nbr"])
    eval_ids = set(eval_frozen["patient_nbr"])
    future_ids = set(future["patient_nbr"])

    assert train_ids.isdisjoint(eval_ids)
    assert train_ids.isdisjoint(future_ids)
    assert eval_ids.isdisjoint(future_ids)


def test_split_is_chronological_by_encounter_id():
    train, eval_frozen, future = time_sliced_patient_split(_deduped_df())
    assert train["encounter_id"].max() < eval_frozen["encounter_id"].min()
    assert eval_frozen["encounter_id"].max() < future["encounter_id"].min()


def test_split_covers_all_rows_with_no_duplication():
    df = _deduped_df()
    train, eval_frozen, future = time_sliced_patient_split(df)
    assert len(train) + len(eval_frozen) + len(future) == len(df)


def test_split_rejects_data_with_duplicate_patients():
    dupe_df = pd.DataFrame({"encounter_id": [1, 2], "patient_nbr": [1, 1]})
    with pytest.raises(ValueError, match="deduplicated"):
        time_sliced_patient_split(dupe_df)
