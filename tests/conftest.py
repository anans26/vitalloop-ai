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


@pytest.fixture(autouse=True)
def _isolated_registry_audit(tmp_path, monkeypatch):
    """No test may append to the real alias audit trail (`mlflow/registry_audit.jsonl`).

    Week 11 found two Week 4 tracking tests writing `test-model` rows into it on
    every run: the live stack's audit log collected test noise. Every test now
    gets a private file; a test that needs its own path still sets it.
    """
    monkeypatch.setattr("ml.registry.REGISTRY_AUDIT_PATH", tmp_path / "registry_audit.jsonl")


# ---------------------------------------------------------------------------
# Week 5 API fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """Isolated serving configuration: throwaway secret and SQLite audit store.

    SQLite rather than Postgres so the suite needs no database service; the real
    Postgres path is exercised by the documented manual verification.
    """
    from api.config import get_settings
    from db import session as db_session

    monkeypatch.setenv("VITALLOOP_JWT_SECRET", "test-secret-not-a-real-key")
    monkeypatch.setenv("VITALLOOP_JWT_EXPIRY_MINUTES", "30")
    monkeypatch.setenv("VITALLOOP_MODEL_SOURCE", "local")
    monkeypatch.setenv("VITALLOOP_DATABASE_URL", f"sqlite:///{(tmp_path / 'audit.db').as_posix()}")
    get_settings.cache_clear()
    db_session.reset_engine()
    yield get_settings()
    get_settings.cache_clear()
    db_session.reset_engine()


@pytest.fixture(scope="session")
def synthetic_bundle(synthetic_clean_df):
    """A ModelBundle built from a small model fitted on synthetic data.

    Lets the prediction path be tested without the DVC-tracked artifact, which
    does not exist in CI.
    """
    from api.model_loader import ModelBundle
    from ml.data.features import split_features_target
    from ml.explain import build_explainer, transformed_feature_names
    from ml.train import build_base_model, build_calibrated_model

    X, y = split_features_target(synthetic_clean_df)
    base = build_base_model(n_estimators=5)
    base.fit(X, y)
    calibrated = build_calibrated_model(build_base_model(n_estimators=5))
    calibrated.fit(X, y)

    return ModelBundle(
        calibrated_model=calibrated,
        base_model=base,
        explainer=build_explainer(base),
        feature_names=transformed_feature_names(base),
        model_name="test-readmission",
        model_version="1",
        model_source="local",
        data_version="testdatahash",
    )


@pytest.fixture
def api_client(api_env, synthetic_bundle):
    """TestClient with lifespan run and the synthetic model injected."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        app.state.model_bundle = synthetic_bundle
        yield client


@pytest.fixture
def valid_payload(synthetic_clean_df) -> dict:
    """A syntactically valid, entirely synthetic prediction request."""
    from api.schemas import PredictionRequest

    row = synthetic_clean_df.iloc[0].to_dict()
    fields = {f.alias or name for name, f in PredictionRequest.model_fields.items()}
    payload = {k: v for k, v in row.items() if k in fields}
    for key, value in list(payload.items()):
        if hasattr(value, "item"):
            payload[key] = value.item()
        if value is None or (isinstance(value, float) and value != value):
            payload[key] = None
    return payload


@pytest.fixture
def auth_headers(api_env) -> dict:
    from api.auth import create_access_token

    token = create_access_token("dr-synthetic", role="clinician", settings=api_env)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Week 6 monitoring fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def drift_db(tmp_path):
    """A throwaway SQLite database brought up through the real `init_db` path.

    Deliberately `init_db` rather than `DriftEvent.__table__.create`: the point
    of these tests is that the documented initialisation call is what creates
    `drift_events`, not that SQLAlchemy can create a table it is handed.
    """
    from db.session import configure_engine, get_session, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{(tmp_path / 'monitor.db').as_posix()}")
    init_db()
    session = get_session()
    try:
        yield session
    finally:
        session.close()
        reset_engine()
