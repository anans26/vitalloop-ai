"""Cleaning steps applied after schema validation, before the feature pipeline.

Policy decisions here mirror DATASET_ANALYSIS.md / notebooks/01_eda.ipynb:
  - drop expired/hospice discharges (leakage: these patients cannot be readmitted)
  - one encounter per patient (leakage: identity overlap across splits)
  - drop `weight` (~97% missing); payer_code/medical_specialty missingness is
    informative, so it becomes an explicit category rather than being dropped
  - binarize the target to <30-day readmission vs rest, and drop the raw
    `readmitted` column so it can never reach a model as a feature
"""

import pandas as pd

LEAKAGE_DISCHARGE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
MISSING_CATEGORY = "missing"

# The data contract for the target. `RAW_TARGET_COLUMN` is the dataset's
# three-class column; it is consumed here and must never survive into the
# processed datasets, because `TARGET_COLUMN` is a deterministic function of
# it -- keeping both would hand any model a perfect predictor.
RAW_TARGET_COLUMN = "readmitted"
TARGET_COLUMN = "readmitted_30d"

# The dataset carries no timestamps, so encounter_id is the chronology proxy
# (the same assumption ml/data/split.py relies on).
ENCOUNTER_ORDER_COLUMN = "encounter_id"


def remove_leakage_discharges(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_DISPOSITIONS)].copy()


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the chronologically earliest encounter per patient.

    Sorting by encounter_id first makes that intent explicit. The UCI export
    happens to already list each patient's encounters in ascending order, so
    this reproduces the previous output exactly -- but it no longer depends on
    an undocumented property of the input file, which the mock-FHIR and
    drift-scenario feeds in later weeks are not obliged to preserve.
    """
    ordered = df.sort_values(ENCOUNTER_ORDER_COLUMN, kind="mergesort")
    return ordered.drop_duplicates(subset="patient_nbr", keep="first").copy()


def apply_missing_value_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["weight"])
    for col in ("payer_code", "medical_specialty"):
        df[col] = df[col].fillna(MISSING_CATEGORY)
    return df


def binarize_target(df: pd.DataFrame, target_col: str = RAW_TARGET_COLUMN) -> pd.DataFrame:
    """Derives the binary target and drops the raw multi-class column it came from."""
    df = df.copy()
    df[TARGET_COLUMN] = (df[target_col] == "<30").astype(int)
    return df.drop(columns=[target_col])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full Week-2 cleaning pipeline, in the documented order."""
    df = remove_leakage_discharges(df)
    df = deduplicate_patients(df)
    df = apply_missing_value_policy(df)
    df = binarize_target(df)
    return df
