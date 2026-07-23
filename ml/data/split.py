"""Time-sliced, patient-level split.

The dataset has no calendar timestamps (DATASET_ANALYSIS.md), so encounter_id
order is used as the chronology proxy -- this is the same assumption the
seeded drift scenarios (S1-S5) rely on. Splitting is patient-level: once
ml/data/clean.py has deduplicated to one encounter per patient, each patient
belongs to exactly one contiguous chronological block, never split across
train/eval/future.
"""

import pandas as pd


def time_sliced_patient_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    eval_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assumes df has already been deduplicated to one row per patient_nbr.

    Returns (train_df, eval_frozen_df, future_stream_df) as contiguous
    chronological blocks, ordered by encounter_id.
    """
    if df["patient_nbr"].duplicated().any():
        raise ValueError("df must be deduplicated to one encounter per patient before splitting")

    ordered = df.sort_values("encounter_id").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    eval_end = train_end + int(n * eval_frac)

    train_df = ordered.iloc[:train_end]
    eval_frozen_df = ordered.iloc[train_end:eval_end]
    future_stream_df = ordered.iloc[eval_end:]
    return train_df, eval_frozen_df, future_stream_df
