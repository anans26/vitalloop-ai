"""SQLAlchemy models for the audit tables.

Three of the five tables in `project_docs/ARCHITECTURE.md` §4.6 exist here:
`predictions` (Week 5, the per-request audit row), `drift_events` (Week 6, one
monitoring window's measurement) and `decision_cards` (Week 7, what the policy
made of that measurement). The remaining two (`retrain_runs`, `approvals`)
belong to Weeks 8-9 and are deliberately absent.

`drift_events -> decision_cards` is the first link of the relational lineage
§4.6 calls "the audit trail": a card names the window that triggered it, so the
evidence behind any decision is one join away.

All three tables are **append-only in application code**: nothing in this
repository issues UPDATE or DELETE against any of them, which is how §4.6's
immutability claim is kept without database-level machinery.

What is *not* stored is as deliberate as what is. A prediction's request
payload is reduced to a SHA-256 hash, so a later investigation can prove which
input produced a score without the audit trail becoming a second copy of the
clinical record; a drift event holds per-feature *statistics* and a path to the
Evidently report, never the window's rows; a decision card holds those same
statistics and the policy's verdict on them.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
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


class DriftEvent(Base):
    """One monitoring window's drift measurement.

    The second of the five tables in `project_docs/ARCHITECTURE.md` §4.6.
    Columns follow that ERD (`event_id`, `feature_stats`, `prediction_drift`,
    `report_uri`, `window_start`, `window_end`); the rest are the operational
    fields needed to tell one window from another when reading the trail back.

    **Every window is persisted, including quiet ones.** PROJECT_DESIGN.md §6
    calls the monitor "the system's senses; every window persisted", and Week 7's
    Decision Engine needs consecutive-window history to distinguish a one-off
    blip from a persistent breach. A window with no breaching feature is a row
    with `breaching_feature_count = 0`, not an absent row.

    Append-only in application code, like `predictions`.
    """

    __tablename__ = "drift_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Which stream produced this window: a seeded scenario (S1-S5) or live traffic.
    scenario: Mapped[str] = mapped_column(String(32), index=True)

    # Per-feature PSI/KS, as measured. No feature values, only statistics.
    feature_stats: Mapped[list | None] = mapped_column(JSON_TYPE, nullable=True)
    prediction_drift: Mapped[bool] = mapped_column(Boolean, default=False)
    prediction_psi: Mapped[float | None] = mapped_column(Float, nullable=True)

    max_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    breaching_feature_count: Mapped[int] = mapped_column(Integer, default=0)
    reference_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Path to the Evidently HTML artifact, not its contents.
    report_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)

    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    policy_thresholds: Mapped[dict | None] = mapped_column(JSON_TYPE, nullable=True)


Index("ix_drift_events_scenario_created", DriftEvent.scenario, DriftEvent.created_at)


class DecisionCard(Base):
    """One policy decision about one monitoring window.

    The third of the five tables in §4.6. Columns follow that ERD (`card_id`,
    `drift_event_id`, `policy_version`, `action`, `confidence`, `card_json`,
    `status`, `created_at`); the rest are the fields a dashboard or an examiner
    would otherwise have to dig out of `card_json` to answer "which windows
    escalated, under which policy, on which scenario".

    `card_json` is the full Pydantic Decision Card (`loop.engine.card`), which
    is the frozen contract Weeks 8-9 read. The columns beside it are a
    queryable projection of it, never a second source of truth.

    **Idempotent by construction.** `card_id` is derived from the drift event
    and the policy version, and `(drift_event_id, policy_version)` is unique, so
    evaluating the same window twice under the same policy cannot append a
    second decision about the same evidence. Re-evaluating history under a new
    policy version *is* allowed and produces a distinct card -- that is the
    governance story in ARCHITECTURE.md §3.8, not a duplicate.

    Append-only in application code, like the two tables above it.
    """

    __tablename__ = "decision_cards"

    card_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # The window this decision is about. §4.6: DRIFT_EVENTS ||--o{ DECISION_CARDS.
    drift_event_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("drift_events.event_id"), index=True
    )
    scenario: Mapped[str] = mapped_column(String(32), index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Which policy decided, and what it decided.
    policy_version: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    disposition: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    # The row of the §3.8 table that fired: 1-6, or "uncovered".
    rule_id: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)

    # A summary of the trigger, so the common queries need no JSON access.
    breaching_feature_count: Mapped[int] = mapped_column(Integer, default=0)
    max_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_drift: Mapped[bool] = mapped_column(Boolean, default=False)

    # Lineage onto the model and data planes.
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # The whole card. Statistics and metadata only -- never a patient row.
    card_json: Mapped[dict] = mapped_column(JSON_TYPE)

    __table_args__ = (
        UniqueConstraint("drift_event_id", "policy_version", name="uq_decision_cards_event_policy"),
    )


Index("ix_decision_cards_scenario_created", DecisionCard.scenario, DecisionCard.created_at)
Index("ix_decision_cards_action_created", DecisionCard.action, DecisionCard.created_at)
