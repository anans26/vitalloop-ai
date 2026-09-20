"""SQLAlchemy models for the audit tables.

Week 5 creates the first of the five tables in `project_docs/ARCHITECTURE.md`
§4.6: `predictions`, the per-request audit row. The remaining four
(`drift_events`, `decision_cards`, `retrain_runs`, `approvals`) belong to
Weeks 6-9 and are deliberately absent.

The table is **append-only in application code**: nothing in this repository
issues UPDATE or DELETE against it, which is how §4.6's immutability claim is
kept without database-level machinery.

What is *not* stored is as deliberate as what is. The request payload is
reduced to a SHA-256 hash, so a later investigation can prove which input
produced a score without the audit trail becoming a second copy of the clinical
record.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON as GenericJSON

try:  # pragma: no cover - exercised only with a real Postgres driver
    from sqlalchemy.dialects.postgresql import JSONB

    JSON_TYPE = JSONB().with_variant(GenericJSON(), "sqlite")
except ImportError:  # pragma: no cover
    JSON_TYPE = JSON


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per scored request.

    Columns follow the ERD in ARCHITECTURE.md §4.6 (`input_hash`,
    `model_version`, `data_version`, `risk_score`, `top_shap`, `caller`, `ts`),
    plus the operational fields a served endpoint needs to be debuggable:
    outcome status, latency, and an error category.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)

    # Client-facing audit identifier, also returned in the response so a caller
    # can quote it when querying the trail.
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # Who asked. Identity comes from the verified JWT, never from the body.
    caller: Mapped[str] = mapped_column(String(255), index=True)
    caller_role: Mapped[str] = mapped_column(String(64))

    # What scored it.
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    model_source: Mapped[str] = mapped_column(String(32))
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # The input, as a hash only -- never the payload itself.
    input_hash: Mapped[str] = mapped_column(String(64), index=True)

    # The result.
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Feature names and SHAP contributions only; no per-encounter feature values.
    top_shap: Mapped[list | None] = mapped_column(JSON_TYPE, nullable=True)

    # How it went.
    status: Mapped[str] = mapped_column(String(32), index=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


Index("ix_predictions_ts_caller", Prediction.ts, Prediction.caller)
