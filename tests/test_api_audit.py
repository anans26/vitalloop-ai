"""The audit trail: what is written, what is deliberately not, and fail-closed."""

import json

import pytest
from sqlalchemy import select

from api.audit import AuditWriteError, hash_payload, persist_audit_row
from db.models import Prediction
from db.session import get_session


def _predict(client, payload, headers):
    return client.post("/predict", json=payload, headers=headers)


def _rows() -> list[Prediction]:
    with get_session() as session:
        return list(session.scalars(select(Prediction)))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_payload_hash_is_stable_regardless_of_key_order():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_payload_hash_changes_with_content():
    assert hash_payload({"a": 1}) != hash_payload({"a": 2})


def test_payload_hash_is_a_sha256_digest():
    digest = hash_payload({"a": 1})
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# What gets written
# ---------------------------------------------------------------------------
def test_one_successful_prediction_writes_exactly_one_row(api_client, valid_payload, auth_headers):
    assert _rows() == []
    response = _predict(api_client, valid_payload, auth_headers)
    assert response.status_code == 200

    rows = _rows()
    assert len(rows) == 1
    assert rows[0].request_id == response.json()["request_id"]


def test_audit_row_captures_the_required_metadata(api_client, valid_payload, auth_headers):
    body = _predict(api_client, valid_payload, auth_headers).json()
    row = _rows()[0]

    assert row.caller == "dr-synthetic"
    assert row.caller_role == "clinician"
    assert row.model_name == "test-readmission"
    assert row.model_version == "1"
    assert row.data_version == "testdatahash"
    assert row.risk_score == pytest.approx(body["readmission_probability"], abs=1e-6)
    assert row.predicted_class == body["predicted_class"]
    assert row.status == "success"
    assert row.error_category is None
    assert row.latency_ms is not None and row.latency_ms >= 0
    assert row.ts is not None
    assert row.input_hash == hash_payload(valid_payload)


def test_audit_row_stores_shap_names_without_feature_values(
    api_client, valid_payload, auth_headers
):
    _predict(api_client, valid_payload, auth_headers)
    top_shap = _rows()[0].top_shap

    assert len(top_shap) == 3
    for entry in top_shap:
        assert set(entry) == {"feature", "contribution"}


def test_failed_predictions_are_still_audited(api_client, valid_payload, auth_headers, monkeypatch):
    """A prediction that errors must leave a trace, and must not return a score."""

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(api_client.app.state.model_bundle.calibrated_model, "predict_proba", boom)

    response = _predict(api_client, valid_payload, auth_headers)
    assert response.status_code == 503

    row = _rows()[0]
    assert row.status == "prediction_failed"
    assert row.error_category == "RuntimeError"
    assert row.risk_score is None


# ---------------------------------------------------------------------------
# What must never be written
# ---------------------------------------------------------------------------
def test_the_request_payload_is_never_persisted(api_client, valid_payload, auth_headers):
    """ARCHITECTURE.md §3.6: the hash exists so the payload need not be stored."""
    _predict(api_client, valid_payload, auth_headers)
    row = _rows()[0]

    stored = json.dumps(
        {column.name: str(getattr(row, column.name)) for column in Prediction.__table__.columns}
    )
    # No clinical value from the request may appear anywhere in the row.
    for field in ("age", "gender", "race", "medical_specialty", "diag_1"):
        value = valid_payload.get(field)
        if value and isinstance(value, str) and len(str(value)) > 3:
            assert str(value) not in stored, f"{field} value leaked into the audit row"

    assert not hasattr(row, "payload")
    assert not hasattr(row, "request_body")


def test_audit_table_has_no_column_for_raw_request_content():
    columns = {c.name for c in Prediction.__table__.columns}
    for forbidden in ("payload", "request_body", "features", "patient_nbr", "encounter_id"):
        assert forbidden not in columns


def test_no_token_or_password_column_exists():
    columns = {c.name for c in Prediction.__table__.columns}
    assert not any(
        any(marker in name for marker in ("token", "password", "secret")) for name in columns
    )


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------
def test_prediction_is_refused_when_the_audit_write_fails(
    api_client, valid_payload, auth_headers, monkeypatch
):
    """Fail closed: no audit row means no score reaches the caller."""

    def failing_commit(self):
        raise RuntimeError("database is down")

    monkeypatch.setattr("sqlalchemy.orm.Session.commit", failing_commit)

    response = _predict(api_client, valid_payload, auth_headers)

    assert response.status_code == 503
    assert "audited" in response.json()["detail"].lower()
    assert "readmission_probability" not in response.text


def test_persist_audit_row_raises_a_typed_error(api_env, monkeypatch):
    from db.session import configure_engine, init_db

    configure_engine(api_env.resolved_database_url())
    init_db()
    session = get_session()

    monkeypatch.setattr(
        "sqlalchemy.orm.Session.commit", lambda self: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    with pytest.raises(AuditWriteError):
        persist_audit_row(session, Prediction(request_id="x"))


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def test_audit_writes_go_through_the_orm_not_string_sql():
    """No SQL is assembled from request data anywhere in the audit path."""
    from pathlib import Path

    source = Path("api/audit.py").read_text(encoding="utf-8")
    for pattern in ('execute(f"', "execute('", 'execute("', "% (", ".format("):
        assert pattern not in source


def test_a_hostile_string_is_stored_as_data_not_executed(api_client, valid_payload, auth_headers):
    """Parameterised binding: an injection attempt is just a value."""
    payload = dict(valid_payload, medical_specialty="Robert'); DROP TABLE predictions;--")

    response = _predict(api_client, payload, auth_headers)
    assert response.status_code == 200

    rows = _rows()
    assert len(rows) == 1  # the table still exists and holds the row
