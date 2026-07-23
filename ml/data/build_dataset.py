"""DVC pipeline entry point: raw CSV -> validated -> cleaned -> split -> processed/*.csv.

Run via `dvc repro`, not directly -- see dvc.yaml for the tracked stage.
"""

from pathlib import Path

import pandas as pd

from ml.data.clean import clean
from ml.data.schema import validate_raw
from ml.data.split import time_sliced_patient_split

RAW_PATH = Path("datasets/raw/diabetic_data.csv")
PROCESSED_DIR = Path("datasets/processed")


def build():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = validate_raw(df)
    df = clean(df)

    train_df, eval_frozen_df, future_stream_df = time_sliced_patient_split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    eval_frozen_df.to_csv(PROCESSED_DIR / "eval_frozen.csv", index=False)
    future_stream_df.to_csv(PROCESSED_DIR / "future_stream.csv", index=False)

    print(f"train: {len(train_df)} rows")
    print(f"eval_frozen: {len(eval_frozen_df)} rows")
    print(f"future_stream: {len(future_stream_df)} rows")


if __name__ == "__main__":
    build()
