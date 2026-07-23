import pandas as pd

from ml.data.clean import (
    LEAKAGE_DISCHARGE_DISPOSITIONS,
    apply_missing_value_policy,
    binarize_target,
    clean,
    deduplicate_patients,
    remove_leakage_discharges,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "patient_nbr": [100, 100, 200, 300, 300],
            "discharge_disposition_id": [1, 1, 11, 1, 1],
            "weight": [None, None, None, "[75-100)", None],
            "payer_code": [None, "MC", None, None, None],
            "medical_specialty": [None, None, None, "Cardiology", None],
            "readmitted": ["<30", ">30", "NO", "<30", "NO"],
        }
    )


def test_remove_leakage_discharges_drops_expired_hospice_codes():
    df = remove_leakage_discharges(_sample_df())
    assert not df["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_DISPOSITIONS).any()
    assert len(df) == 4  # only row with disposition_id=11 (encounter 3) removed


def test_deduplicate_patients_keeps_first_encounter_only():
    df = deduplicate_patients(_sample_df())
    assert df["patient_nbr"].is_unique
    # patient 100 has encounters 1,2 -> keep encounter_id 1; patient 300 has 4,5 -> keep 4
    assert set(df["encounter_id"]) == {1, 3, 4}


def test_missing_value_policy_drops_weight_and_fills_categories():
    df = apply_missing_value_policy(_sample_df())
    assert "weight" not in df.columns
    assert df["payer_code"].isna().sum() == 0
    assert df["medical_specialty"].isna().sum() == 0
    assert (df["payer_code"] == "missing").sum() == 4  # 4 of 5 rows were None


def test_binarize_target_maps_only_less_than_30_to_positive():
    df = binarize_target(_sample_df())
    assert df["readmitted_30d"].tolist() == [1, 0, 0, 1, 0]


def test_clean_end_to_end_removes_leakage_and_binarizes():
    df = clean(_sample_df())
    assert not df["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_DISPOSITIONS).any()
    assert df["patient_nbr"].is_unique
    assert "weight" not in df.columns
    assert "readmitted_30d" in df.columns
