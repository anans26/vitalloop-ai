import pandas as pd
import pandera.errors
import pytest

from ml.data.schema import validate_raw


def _valid_row(**overrides) -> dict:
    row = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "weight": None,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "payer_code": None,
        "medical_specialty": None,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250.83",
        "diag_2": None,
        "diag_3": None,
        "number_diagnoses": 5,
        "max_glu_serum": None,
        "A1Cresult": None,
        "change": "No",
        "diabetesMed": "Yes",
        "readmitted": "NO",
    }
    row.update(overrides)
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    from ml.data.schema import MEDICATION_COLUMNS

    df = pd.DataFrame(rows)
    for col in MEDICATION_COLUMNS:
        if col not in df.columns:
            df[col] = "No"
    return df


def test_valid_row_passes_validation():
    df = _make_df([_valid_row()])
    validated = validate_raw(df)
    assert len(validated) == 1


def test_invalid_gender_value_is_rejected():
    df = _make_df([_valid_row(gender="Not A Gender")])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)


def test_time_in_hospital_out_of_range_is_rejected():
    df = _make_df([_valid_row(time_in_hospital=20)])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)


def test_duplicate_encounter_id_is_rejected():
    df = _make_df([_valid_row(encounter_id=1), _valid_row(encounter_id=1, patient_nbr=200)])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)
