"""Shadow scoring on `/predict` (ARCHITECTURE.md §3.6, §3.13).

Three properties, each asserted rather than assumed:

* the response is byte-for-byte the champion's, shadow or no shadow;
* the shadow's score is logged, against the request's own audit id;
* nothing the shadow does -- fail to score, fail to write -- reaches the caller.

`TestClient` runs background tasks after the response, synchronously, before
the call returns -- so a test can read the shadow row immediately.
"""

import numpy as np
import pytest
from sqlalchemy import select

from api.shadow import ShadowBundle, load_shadow_bundle
from db.models import Prediction, ShadowPrediction


class ConstantModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, frame):
        column = np.full(len(frame), self.score)
        return np.column_stack([1 - column, column])


class BrokenModel:
    def predict_proba(self, frame):
        raise ValueError("feature mismatch")


def _shadow_rows():
    from db.session import get_session

    session = get_session()
    try:
        return session.scalars(select(ShadowPrediction)).all()
    finally:
        session.close()


def _with_shadow(api_client, model, version="2"):
    api_client.app.state.shadow_bundle = ShadowBundle(
        calibrated_model=model, model_name="test-readmission", model_version=version
    )


def test_without_a_shadow_nothing_is_shadow_scored(api_client, valid_payload, auth_headers):
    api_client.app.state.shadow_bundle = None
    assert api_client.post("/predict", json=valid_payload, headers=auth_headers).status_code == 200
    assert _shadow_rows() == []


def test_the_shadow_score_is_logged_against_the_request(api_client, valid_payload, auth_headers):
    _with_shadow(api_client, ConstantModel(0.83))
    body = api_client.post("/predict", json=valid_payload, headers=auth_headers).json()

    (row,) = _shadow_rows()
    assert row.request_id == body["request_id"]
    assert row.champion_version == "1"
    assert row.shadow_version == "2"
    assert row.champion_score == pytest.approx(body["readmission_probability"], abs=1e-6)
    assert row.shadow_score == pytest.approx(0.83)
    assert row.decision_threshold == body["decision_threshold"]
    assert row.status == "scored"


def test_the_response_is_the_champions_alone(api_client, valid_payload, auth_headers):
    api_client.app.state.shadow_bundle = None
    plain = api_client.post("/predict", json=valid_payload, headers=auth_headers).json()

    _with_shadow(api_client, ConstantModel(0.99))
    shadowed = api_client.post("/predict", json=valid_payload, headers=auth_headers).json()

    for body in (plain, shadowed):
        body.pop("request_id")
    assert shadowed == plain
    assert "shadow" not in str(shadowed).lower()


def test_the_audit_row_records_only_the_champion(api_client, valid_payload, auth_headers):
    from db.session import get_session

    _with_shadow(api_client, ConstantModel(0.99))
    api_client.post("/predict", json=valid_payload, headers=auth_headers)
    session = get_session()
    try:
        (audit,) = session.scalars(select(Prediction)).all()
    finally:
        session.close()
    assert audit.model_version == "1"
    assert audit.risk_score != pytest.approx(0.99)


def test_a_failing_shadow_never_reaches_the_caller(api_client, valid_payload, auth_headers):
    _with_shadow(api_client, BrokenModel())
    response = api_client.post("/predict", json=valid_payload, headers=auth_headers)

    assert response.status_code == 200
    (row,) = _shadow_rows()
    assert row.status == "shadow_failed"
    assert row.shadow_score is None
    assert row.error_category == "ValueError"


def test_a_failed_shadow_write_never_reaches_the_caller(
    api_client, valid_payload, auth_headers, monkeypatch
):
    def fail(session, row):
        raise RuntimeError("database went away")

    monkeypatch.setattr("loop.shadow.persistence.persist_shadow_row", fail)
    _with_shadow(api_client, ConstantModel(0.5))
    assert api_client.post("/predict", json=valid_payload, headers=auth_headers).status_code == 200


def test_a_failed_champion_prediction_is_not_shadowed(
    api_client, valid_payload, auth_headers, monkeypatch
):
    _with_shadow(api_client, ConstantModel(0.5))
    bundle = api_client.app.state.model_bundle
    monkeypatch.setattr(bundle, "calibrated_model", BrokenModel())

    assert api_client.post("/predict", json=valid_payload, headers=auth_headers).status_code == 503
    assert _shadow_rows() == []


def test_the_shadow_row_holds_no_payload(api_client, valid_payload, auth_headers):
    _with_shadow(api_client, ConstantModel(0.4))
    api_client.post("/predict", json=valid_payload, headers=auth_headers)
    (row,) = _shadow_rows()
    stored = " ".join(str(getattr(row, c)) for c in ShadowPrediction.__table__.columns.keys())
    for field in ("time_in_hospital", "num_lab_procedures", "Caucasian", "diag_1"):
        assert field not in stored


# ---------------------------------------------------------------------------
# Resolving the shadow model
# ---------------------------------------------------------------------------
def test_the_local_artifact_has_no_shadow(api_env):
    assert load_shadow_bundle(api_env, "1") is None


def test_shadowing_can_be_switched_off(api_env):
    settings = api_env.model_copy(update={"model_source": "mlflow", "shadow_enabled": False})
    assert load_shadow_bundle(settings, "1") is None


class _Version:
    def __init__(self, version):
        self.version = version


class _Client:
    def __init__(self, version=None, error=None):
        self._version, self._error = version, error

    def __call__(self, *args, **kwargs):
        return self

    def get_model_version_by_alias(self, name, alias):
        assert alias == "shadow"
        if self._error:
            raise self._error
        return _Version(self._version)


def test_no_shadow_alias_means_no_shadow(api_env, monkeypatch):
    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client(error=RuntimeError("no alias")))
    settings = api_env.model_copy(update={"model_source": "mlflow"})
    assert load_shadow_bundle(settings, "1") is None


def test_a_shadow_pointing_at_the_champion_is_not_loaded(api_env, monkeypatch):
    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client(version=1))
    settings = api_env.model_copy(update={"model_source": "mlflow"})
    assert load_shadow_bundle(settings, "1") is None


def test_the_shadow_is_loaded_by_version_not_alias(api_env, monkeypatch):
    loaded = []
    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client(version=2))
    monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: loaded.append(uri) or "model")
    settings = api_env.model_copy(update={"model_source": "mlflow"})

    bundle = load_shadow_bundle(settings, "1")

    assert bundle.model_version == "2"
    assert loaded == [f"models:/{settings.registered_model_name}/2"]


def test_an_unloadable_shadow_does_not_stop_the_api(api_env, monkeypatch):
    def fail(uri):
        raise OSError("artifact missing")

    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client(version=2))
    monkeypatch.setattr("mlflow.sklearn.load_model", fail)
    settings = api_env.model_copy(update={"model_source": "mlflow"})
    assert load_shadow_bundle(settings, "1") is None
