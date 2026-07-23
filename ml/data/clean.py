"""Cleaning steps applied after schema validation, before the feature pipeline.

Policy decisions here mirror DATASET_ANALYSIS.md / notebooks/01_eda.ipynb:
  - drop expired/hospice discharges (leakage: these patients cannot be readmitted)
  - one encounter per patient (leakage: identity overlap across splits)
  - drop `weight` (~97% missing); payer_code/medical_specialty missingness is
    informative, so it becomes an explicit category rather than being dropped
  - binarize the target to <30-day readmission vs rest
"""

import pandas as pd

LEAKAGE_DISCHARGE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
MISSING_CATEGORY = "missing"


def remove_leakage_discharges(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_DISPOSITIONS)].copy()


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the first encounter per patient_nbr (by row order)."""
    return df.drop_duplicates(subset="patient_nbr", keep="first").copy()


def apply_missing_value_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["weight"])
    for col in ("payer_code", "medical_specialty"):
        df[col] = df[col].fillna(MISSING_CATEGORY)
    return df


def binarize_target(df: pd.DataFrame, target_col: str = "readmitted") -> pd.DataFrame:
    df = df.copy()
    df["readmitted_30d"] = (df[target_col] == "<30").astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full Week-2 cleaning pipeline, in the documented order."""
    df = remove_leakage_discharges(df)
    df = deduplicate_patients(df)
    df = apply_missing_value_policy(df)
    df = binarize_target(df)
    return df
