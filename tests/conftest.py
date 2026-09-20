"""Shared synthetic fixtures.

Week 3 tests must run in CI, where `datasets/processed/*.csv` does not exist.
These fixtures produce a frame with the same *shape of contract* as a cleaned
Week 2 split -- every column the feature pipeline consumes, the binary target,
and identifier columns -- so model, calibration, and SHAP behaviour can be
tested without the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from ml.data.clean import TARGET_COLUMN
from ml.data.features import MEDICATION_COLUMNS

RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Other"]
AGES = ["[40-50)", "[50-60)", "[60-70)", "[70-80)"]
SPECIALTIES = ["InternalMedicine", "Cardiology", "Surgery", "missing"]
DIAGNOSES = ["250.83", "428", "486", "V27", "789", "401"]


def make_synthetic_clean_df(n_rows: int = 300, seed: int = 7) -> pd.DataFrame:
    """A deterministic stand-in for the output of ml.data.clean.clean()."""
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(1, n_rows + 1),
            "patient_nbr": np.arange(10_000, 10_000 + n_rows),
            "time_in_hospital": rng.integers(1, 14, n_rows),
            "num_lab_procedures": rng.integers(1, 90, n_rows),
            "num_procedures": rng.integers(0, 6, n_rows),
            "num_medications": rng.integers(1, 40, n_rows),
            "number_outpatient": rng.integers(0, 4, n_rows),
            "number_emergency": rng.integers(0, 3, n_rows),
            "number_inpatient": rng.integers(0, 5, n_rows),
            "number_diagnoses": rng.integers(1, 16, n_rows),
            "race": rng.choice(RACES, n_rows),
            "gender": rng.choice(["Female", "Male"], n_rows),
            "age": rng.choice(AGES, n_rows),
            "payer_code": rng.choice(["MC", "HM", "missing"], n_rows),
            "medical_specialty": rng.choice(SPECIALTIES, n_rows),
            "max_glu_serum": rng.choice([None, "Norm", ">200"], n_rows),
            "A1Cresult": rng.choice([None, "Norm", ">7"], n_rows),
            "change": rng.choice(["Ch", "No"], n_rows),
            "diabetesMed": rng.choice(["Yes", "No"], n_rows),
            "diag_1": rng.choice(DIAGNOSES, n_rows),
            "diag_2": rng.choice(DIAGNOSES, n_rows),
            "diag_3": rng.choice(DIAGNOSES, n_rows),
            "admission_source_id": rng.choice([1, 4, 7, 8], n_rows),
        }
    )
    for column in MEDICATION_COLUMNS:
        frame[column] = rng.choice(["No", "Steady", "Up", "Down"], n_rows)

    # A learnable but noisy signal, so a fitted model is not degenerate.
    logit = -1.2 + 0.18 * frame["number_inpatient"] - 0.05 * frame["time_in_hospital"]
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame[TARGET_COLUMN] = (rng.random(n_rows) < probability).astype(int)
    return frame


@pytest.fixture(scope="session")
def synthetic_clean_df() -> pd.DataFrame:
    return make_synthetic_clean_df()
