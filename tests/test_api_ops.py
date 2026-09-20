"""Health, readiness, OpenAPI surface, and configuration hygiene."""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------
def test_health_needs_no_authentication(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "vitalloop-api"}


def test_health_reveals_no_configuration(api_client):
    body = api_client.get("/health").text
    for marker in ("secret", "password", "sqlite", "postgres", "Traceback", "/", "\\"):
        assert marker not in body.replace("vitalloop-api", "")


def test_ready_reports_model_and_database(api_client):
    response = api_client.get("/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["database_available"] is True
    assert body["model_version"] == "1"


def test_ready_needs_no_authentication(api_client):
    assert api_client.get("/ready").status_code == 200


def test_ready_reports_not_ready_when_the_model_is_missing(api_client):
    api_client.app.state.model_bundle = None

    response = api_client.get("/ready")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    assert response.json()["model_version"] is None


def test_ready_reports_not_ready_when_the_database_is_down(api_client, monkeypatch):
    monkeypatch.setattr("api.routers.health.check_connection", lambda *a, **k: False)

    response = api_client.get("/ready")
    assert response.status_code == 503
    assert response.json()["database_available"] is False


def test_ready_never_exposes_connection_details(api_client, monkeypatch):
    monkeypatch.setattr("api.routers.health.check_connection", lambda *a, **k: False)
    body = api_client.get("/ready").text

    for marker in ("sqlite:///", "postgresql", "password", "Traceback", "audit.db"):
        assert marker not in body


def test_predict_is_refused_when_no_model_is_loaded(api_client, valid_payload, auth_headers):
    api_client.app.state.model_bundle = None

    response = api_client.post("/predict", json=valid_payload, headers=auth_headers)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------
def test_openapi_documents_the_endpoints(api_client):
    schema = api_client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


def test_openapi_marks_predict_as_authenticated(api_client):
    schema = api_client.get("/openapi.json").json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert schema["paths"]["/predict"]["post"].get("security")


def test_openapi_exposes_the_request_and_response_schemas(api_client):
    schemas = api_client.get("/openapi.json").json()["components"]["schemas"]
    assert "PredictionRequest" in schemas
    assert "PredictionResponse" in schemas
    assert "time_in_hospital" in schemas["PredictionRequest"]["properties"]


def test_openapi_contains_no_identifier_fields(api_client):
    """The documented contract must not invite callers to send identifiers."""
    properties = api_client.get("/openapi.json").json()["components"]["schemas"][
        "PredictionRequest"
    ]["properties"]
    for forbidden in ("patient_nbr", "encounter_id", "readmitted"):
        assert forbidden not in properties


# ---------------------------------------------------------------------------
# Configuration hygiene
# ---------------------------------------------------------------------------
def test_jwt_secret_has_no_default_in_the_source():
    """An unset secret must break startup, not fall back to a repository value."""
    source = Path("api/config.py").read_text(encoding="utf-8")
    assert "jwt_secret: str\n" in source
    assert "jwt_secret: str =" not in source


def test_settings_refuse_to_load_without_a_secret(monkeypatch):
    from api.config import get_settings

    monkeypatch.delenv("VITALLOOP_JWT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="configuration is incomplete"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_no_secret_literals_are_committed_in_the_api_package():
    suspicious = ("password=", "secret=", "api_key=", "BEGIN PRIVATE KEY")
    for path in Path("api").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for marker in suspicious:
            assert marker not in text, f"{path} contains {marker!r}"


def test_env_files_are_gitignored():
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore


def test_database_url_is_composed_not_interpolated_into_sql(api_env):
    """The URL carries credentials; it must never reach a response or a query string."""
    url = api_env.resolved_database_url()
    assert url.startswith("sqlite:///")
    assert "SELECT" not in url.upper()


def test_responses_never_include_the_configured_secret(api_client, valid_payload, auth_headers):
    secret = os.environ["VITALLOOP_JWT_SECRET"]
    for response in (
        api_client.get("/health"),
        api_client.get("/ready"),
        api_client.post("/predict", json=valid_payload, headers=auth_headers),
        api_client.post("/predict", json={}, headers=auth_headers),
    ):
        assert secret not in response.text


def test_log_fields_exclude_request_content():
    """structlog output is restricted to identifiers and the payload hash."""
    from api.logging_config import request_log_fields

    fields = request_log_fields(
        request_id="r1",
        caller="dr-synthetic",
        input_hash="abc123",
        model_version="1",
        status="success",
        latency_ms=12.345,
    )
    assert set(fields) == {
        "request_id",
        "caller",
        "input_hash",
        "model_version",
        "status",
        "latency_ms",
    }
