"""`scripts/send_traffic.py` sends the API's contract and nothing else."""

import numpy as np

from api.schemas import PredictionRequest
from scripts.send_traffic import contract_fields, payloads


def test_payloads_carry_no_identifier_or_label(synthetic_clean_df):
    frame = synthetic_clean_df.assign(readmitted=0)
    for body in payloads(frame.head(5)):
        for forbidden in ("encounter_id", "patient_nbr", "readmitted"):
            assert forbidden not in body


def test_payloads_validate_against_the_request_schema(synthetic_clean_df):
    for body in payloads(synthetic_clean_df.head(5)):
        PredictionRequest.model_validate(body)


def test_missing_values_are_sent_as_null(synthetic_clean_df):
    frame = synthetic_clean_df.head(1).copy()
    frame["payer_code"] = np.nan
    (body,) = payloads(frame)
    assert body["payer_code"] is None


def test_the_contract_uses_the_wire_names_for_hyphenated_drugs():
    assert "glyburide-metformin" in contract_fields()


def test_diagnosis_codes_are_read_as_text(tmp_path):
    """Numeric-looking ICD-9 codes must reach the API as strings, not floats."""
    import pandas as pd

    from scripts.send_traffic import TEXT_COLUMNS

    path = tmp_path / "stream.csv"
    path.write_text("diag_1,diag_2,diag_3\n428,V27,250.83\n", encoding="utf-8")
    frame = pd.read_csv(path, dtype=TEXT_COLUMNS)
    assert frame.iloc[0].to_dict() == {"diag_1": "428", "diag_2": "V27", "diag_3": "250.83"}
