"""Loading the champion so the monitor can measure prediction drift.

The monitor needs strictly less than the serving API does: the calibrated model
and nothing else. No SHAP explainer, no base model, and -- importantly -- no
JWT secret, which `api.config.Settings` requires and would otherwise force a
monitoring worker to hold a credential it has no use for. So this module
resolves the model itself rather than reusing `api.model_loader.ModelBundle`.

What it does *not* do is load a different model. `local` reads the same
DVC-tracked joblib bundle the API reads; `mlflow` resolves the same registered
name and alias. There is one champion.
"""

import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import MODEL_PATH
from ml.data.features import NON_FEATURE_COLUMNS

DEFAULT_REGISTERED_MODEL_NAME = "vitalloop-readmission"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_TRACKING_URI = "http://localhost:5000"


class ScoringModelUnavailableError(RuntimeError):
    """The champion could not be loaded. Never swallowed."""


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The feature frame the champion expects: everything except identifiers and targets.

    Same exclusion list `ml.data.features.split_features_target` applies at
    training time, but without requiring a label column -- serving traffic does
    not carry one.
    """
    return frame.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in frame.columns])


@dataclass(frozen=True)
class ScoringModel:
    """A loaded champion plus the version string that goes on every drift row."""

    model: Any
    version: str
    source: str

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of 30-day readmission, one per row."""
        return np.asarray(self.model.predict_proba(model_inputs(frame))[:, 1], dtype=float)


def _load_local() -> ScoringModel:
    if not MODEL_PATH.exists():
        raise ScoringModelUnavailableError(
            f"Model artifact not found at {MODEL_PATH.name}. Run `dvc repro` to build it."
        )
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as error:  # pragma: no cover - corrupt artifact
        raise ScoringModelUnavailableError(f"Could not load {MODEL_PATH.name}: {error}") from error
    return ScoringModel(model=bundle["calibrated_model"], version="local-artifact", source="local")


def _load_mlflow() -> ScoringModel:
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    tracking_uri = os.environ.get("VITALLOOP_MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    name = os.environ.get("VITALLOOP_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME)
    alias = os.environ.get("VITALLOOP_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)

    # Both URIs, explicitly. `models:/name@alias` resolves against the *global*
    # registry URI, not the client's, and MLflow's global default can be a local
    # store -- which would silently resolve the champion alias against a
    # different registry than the one this worker was pointed at.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    try:
        version = MlflowClient(tracking_uri, tracking_uri).get_model_version_by_alias(name, alias)
        model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    except Exception as error:
        raise ScoringModelUnavailableError(
            f"Could not load {name}@{alias} from {tracking_uri}: {error}"
        ) from error
    return ScoringModel(model=model, version=str(version.version), source="mlflow")


def load_scoring_model(source: str | None = None) -> ScoringModel:
    """Resolves the champion. `source` defaults to $VITALLOOP_MODEL_SOURCE, then "local".

    "local" is the default here, unlike the API, because the monitor is also the
    thing a reviewer runs offline to replay the drift benchmark; a tracking
    server should be opt-in for that, not required.
    """
    source = source or os.environ.get("VITALLOOP_MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_mlflow()
    if source == "local":
        return _load_local()
    raise ScoringModelUnavailableError(f"Unknown model source {source!r}; use 'local' or 'mlflow'.")
