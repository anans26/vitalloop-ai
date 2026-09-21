"""The training reference every monitoring window is compared against.

ARCHITECTURE.md §3.7 defines input drift as PSI and KS "per feature vs the
training reference", so the reference is the Week 2 training slice -- the exact
data the champion was fitted on, not the frozen evaluation slice and not a
rolling window of recent traffic.

Two things are read off it: the monitored feature distributions, and the
champion's score distribution, which is what prediction drift is measured
against.

Note what is *not* here: no cleaning, no encoding, no second preprocessing
path. The engineered columns come from `ml.data.features.FeatureEngineer`, the
same object that sits at the head of the fitted pipeline, so a monitored
`service_utilization` is computed by the same code that computed the one the
model was trained on.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from loop.monitor.config import (
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_FEATURES,
    REFERENCE_PATH,
    REFERENCE_SAMPLE_ROWS,
    REFERENCE_SEED,
)
from loop.monitor.scoring import ScoringModel
from ml.data.clean import MISSING_CATEGORY
from ml.data.features import FeatureEngineer


class ReferenceUnavailableError(RuntimeError):
    """The training reference could not be read. Never swallowed."""


def engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The monitored columns of the engineered frame, and only those.

    Selecting by name is the privacy control as well as the contract: the frame
    handed to Evidently structurally cannot contain `encounter_id`,
    `patient_nbr`, or the label, because none of them is in
    `MONITORED_FEATURES`.

    Nulls in a categorical feature become the explicit `missing` category,
    because that is precisely what the fitted pipeline's
    `SimpleImputer(strategy="constant", fill_value="missing")` does before the
    model sees the row. Dropping them instead -- which is what both Evidently
    and the hand-rolled PSI do with NaN -- would monitor a different variable
    than the one being scored, and on a column like `max_glu_serum` (~94%
    unrecorded) it would reduce a 2000-row window to ~25 observations and turn
    sampling noise into a permanent drift alarm.
    """
    engineered = FeatureEngineer().transform(frame)
    missing = [column for column in MONITORED_FEATURES if column not in engineered.columns]
    if missing:
        raise ReferenceUnavailableError(
            f"monitored features absent from the frame: {', '.join(sorted(missing))}"
        )
    monitored = engineered.loc[:, list(MONITORED_FEATURES)].copy()
    for column in MONITORED_CATEGORICAL_FEATURES:
        monitored[column] = monitored[column].astype("object").fillna(MISSING_CATEGORY)
    return monitored


def sample_reference(
    frame: pd.DataFrame,
    *,
    rows: int = REFERENCE_SAMPLE_ROWS,
    seed: int = REFERENCE_SEED,
) -> pd.DataFrame:
    """A seeded sample of the reference, kept in stream order.

    Seeded and order-preserving so every machine measures against byte-identical
    reference histograms; the sample only exists because the champion has to
    score the reference once, and that is the expensive half of a window.
    """
    if rows <= 0 or len(frame) <= rows:
        return frame
    selected = np.sort(np.random.default_rng(seed).choice(len(frame), size=rows, replace=False))
    return frame.iloc[selected]


@dataclass(frozen=True)
class MonitoringReference:
    """Reference feature distributions plus the champion's reference scores."""

    features: pd.DataFrame
    scores: np.ndarray
    model_version: str | None = None
    data_version: str | None = None
    source_path: str | None = field(default=None)

    @property
    def rows(self) -> int:
        return len(self.features)


def build_reference(
    frame: pd.DataFrame,
    model: ScoringModel,
    *,
    rows: int = REFERENCE_SAMPLE_ROWS,
    seed: int = REFERENCE_SEED,
    data_version: str | None = None,
    source_path: str | None = None,
) -> MonitoringReference:
    """Turns a cleaned training frame into a reference. No file access."""
    sampled = sample_reference(frame, rows=rows, seed=seed)
    return MonitoringReference(
        features=engineered_features(sampled),
        scores=model.score(sampled),
        model_version=model.version,
        data_version=data_version,
        source_path=source_path,
    )


# dvc.lock keys, per ml.tracking.DVC_TRACKED_PATHS.
_DVC_KEYS = {"train.csv": "dvc_train_md5", "eval_frozen.csv": "dvc_eval_frozen_md5"}


def reference_data_version(path: Path = REFERENCE_PATH) -> str | None:
    """The DVC hash of the reference split, so a drift row names what it compared against."""
    try:
        from ml.tracking import dvc_lineage

        return dvc_lineage().get(_DVC_KEYS.get(path.name, ""))
    except Exception:
        return None


def load_reference(
    model: ScoringModel,
    *,
    path: Path = REFERENCE_PATH,
    rows: int = REFERENCE_SAMPLE_ROWS,
    seed: int = REFERENCE_SEED,
) -> MonitoringReference:
    """Reads the training slice from disk and builds the reference."""
    if not path.exists():
        raise ReferenceUnavailableError(
            f"{path} not found -- run `dvc repro` to build the Week 2 datasets."
        )
    frame = pd.read_csv(path, low_memory=False)
    return build_reference(
        frame,
        model,
        rows=rows,
        seed=seed,
        data_version=reference_data_version(path),
        source_path=path.name,
    )
