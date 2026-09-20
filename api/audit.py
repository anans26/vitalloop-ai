"""Building and persisting the per-request audit row.

`ARCHITECTURE.md` §3.6 defines the record: a SHA-256 hash of the input payload,
model version, data version, score, top SHAP features, timestamp, and caller
identity -- with the explicit note that the hash "allows later verification
without storing the record twice". So the payload is hashed, never stored, and
the SHAP entries carry feature names and contributions but not the encounter's
values for those features.

**Audit failure is fail-closed.** If the row cannot be written, the caller gets
a 503 and no score. A prediction that reached a clinician without leaving an
audit trail would be exactly the silent, unreviewable event this project exists
to prevent; returning the score anyway would make the audit trail a best-effort
log rather than a record.
"""

import hashlib
import json

from sqlalchemy.orm import Session

from db.models import Prediction

STATUS_SUCCESS = "success"
STATUS_PREDICTION_FAILED = "prediction_failed"


class AuditWriteError(RuntimeError):
    """The audit row could not be persisted. Never swallowed."""


def hash_payload(payload: dict) -> str:
    """Stable SHA-256 of a request body.

    Canonical JSON (sorted keys, no incidental whitespace) so the same clinical
    input always hashes the same way regardless of field order on the wire.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit_row(
    *,
    request_id: str,
    caller: str,
    caller_role: str,
    model_name: str,
    model_version: str,
    model_source: str,
    data_version: str | None,
    input_hash: str,
    risk_score: float | None,
    predicted_class: int | None,
    decision_threshold: float | None,
    top_shap: list[dict] | None,
    status: str,
    error_category: str | None,
    latency_ms: float | None,
) -> Prediction:
    """Assembles the row. Takes only values that are safe to persist."""
    return Prediction(
        request_id=request_id,
        caller=caller,
        caller_role=caller_role,
        model_name=model_name,
        model_version=model_version,
        model_source=model_source,
        data_version=data_version,
        input_hash=input_hash,
        risk_score=risk_score,
        predicted_class=predicted_class,
        decision_threshold=decision_threshold,
        top_shap=top_shap,
        status=status,
        error_category=error_category,
        latency_ms=latency_ms,
    )


def persist_audit_row(session: Session, row: Prediction) -> Prediction:
    """Writes the row, or raises AuditWriteError.

    Uses the ORM, so values are bound as parameters; no SQL is assembled from
    request data anywhere in this module.
    """
    try:
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    except Exception as error:
        session.rollback()
        raise AuditWriteError(str(error)) from error
