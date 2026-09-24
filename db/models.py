"""SQLAlchemy models for the audit tables.

All five tables in `project_docs/ARCHITECTURE.md` §4.6 exist here:
`predictions` (Week 5, the per-request audit row), `drift_events` (Week 6, one
monitoring window's measurement), `decision_cards` (Week 7, what the policy
made of that measurement), `retrain_runs` (Week 8, the challenger that
decision produced and the gate's verdict on it) and `approvals` (Week 9, the
human decision that lets an escalated card retrain or a gated challenger become
champion). Week 9 adds one table the ERD does not draw, `shadow_predictions`:
§3.6 says the shadow model's score is logged, and a score that is never
returned to anyone needs a row of its own rather than a column on the audit
row the clinician's response was built from.

`drift_events -> decision_cards -> retrain_runs -> approvals` is the relational
lineage §4.6 calls "the audit trail": a card names the window that triggered it,
a retrain run names the card that authorised it, and an approval names the run
whose challenger a person promoted, so the evidence behind any champion is
three joins away.

All six tables are **append-only in application code**: nothing in this
repository issues UPDATE or DELETE against any of them, which is how §4.6's
immutability claim is kept without database-level machinery. The status
transitions §3.12 describes are appended as `retrain_runs` rows rather than
written back onto a card, which is §4.6's own instruction ("status transitions
append history rows").

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


class RetrainRun(Base):
    """One retrain attempt for one Decision Card, and the gate's verdict on it.

    The fourth of the five tables in §4.6. Columns follow that ERD (`run_id`,
    `card_id`, `mlflow_run`, `data_version`, `gate_result`, `outcome`); the rest
    are the queryable projection this schema has used since `predictions` --
    the numbers a dashboard or an examiner would otherwise have to dig out of
    `gate_result` to answer "which challengers were blocked, on which card, and
    did anything move".

    `decision_cards -> retrain_runs` closes the second link of the relational
    lineage §4.6 calls the audit trail: a drift window explains a card, and a
    card explains the retrain it authorised.

    **The card's status transition lives here, not on the card.** §3.12 says a
    blocked challenger leaves the card "closed as BLOCKED", while §4.6 requires
    every table to be append-only in application code -- "no UPDATE on card
    contents; status transitions append history rows". This row *is* that
    history row: nothing in this repository updates a `decision_cards` row after
    it is written, and a card's effective state is read by joining to its
    retrain runs.

    **Idempotent by construction**, like `decision_cards` before it. `run_id` is
    derived from the card, the mode and the challenger being judged, and
    `(card_id, mode, challenger_version)` carries a unique constraint, so
    re-running the same retrain cannot append a second verdict about the same
    challenger. A *different* challenger on the same card is a distinct row,
    which is what the deliberately-bad-challenger demonstration needs.

    Append-only in application code, like the three tables above it. Statistics
    and metadata only -- `gate_result` holds aggregate metric sets, never rows.
    """

    __tablename__ = "retrain_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # The decision that authorised this retrain. §4.6: DECISION_CARDS ||--o{ RETRAIN_RUNS.
    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("decision_cards.card_id"), index=True
    )
    # Carried from the card so the common queries need no join.
    scenario: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    policy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # "live" trains; "replay" re-registers a pre-trained challenger (§3.11's
    # demo acceleration). Recorded rather than inferred, because the roadmap
    # requires replay to be "a first-class, labeled feature" -- a replayed run
    # must never be mistaken for a run that actually trained.
    mode: Mapped[str] = mapped_column(String(16), index=True)

    # Lineage onto the model and data planes.
    mlflow_run: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    champion_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    challenger_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # The gate's verdict. `outcome` is PASS or BLOCK (§3.12).
    outcome: Mapped[str] = mapped_column(String(16), index=True)
    criteria_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failed_criteria_count: Mapped[int] = mapped_column(Integer, default=0)

    # Whether the PASS actually moved the `shadow` alias. Separated from
    # `outcome` because §3.12's "PASS -> shadow" is an effect, and an effect
    # that did not happen (dry run, registry unavailable) must not read as one
    # that did.
    shadow_alias_moved: Mapped[bool] = mapped_column(Boolean, default=False)

    # Who authorised a retrain the policy escalated. Null on the automated
    # path, where the card's own AUTO_PROCEED_SHADOW disposition is the
    # authority (WORKFLOW.md §4: escalated evidence retrains only once an ops
    # user acts).
    authorized_by: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # The whole gate result: criteria, every check, both metric sets.
    gate_result: Mapped[dict] = mapped_column(JSON_TYPE)

    __table_args__ = (
        UniqueConstraint(
            "card_id", "mode", "challenger_version", name="uq_retrain_runs_card_mode_challenger"
        ),
    )


Index("ix_retrain_runs_outcome_created", RetrainRun.outcome, RetrainRun.created_at)
Index("ix_retrain_runs_card_created", RetrainRun.card_id, RetrainRun.created_at)


class ShadowPrediction(Base):
    """One request scored a second time, by the `shadow` model, after the response.

    ARCHITECTURE.md §3.6: "when a `shadow` alias exists, middleware also scores
    the request with the shadow model and logs it. The response *only ever*
    contains the champion score." This is that log. It is written after the
    clinician already has the champion's answer, so nothing here can reach a
    response, and a shadow failure is a row with a status rather than an error
    anyone sees.

    `request_id` ties the row to its `predictions` audit row, which already
    holds the input hash, the caller and the champion's lineage -- so this row
    carries only what is new: which shadow version scored, what it said, and
    the champion score it is compared with. No payload, no feature value, no
    SHAP.

    §3.13's agreement and stability statistics are computed over these rows,
    grouped by the `(champion_version, shadow_version)` pair, so a window that
    straddles a promotion never mixes two comparisons.

    Append-only in application code, like every table above.
    """

    __tablename__ = "shadow_predictions"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    model_name: Mapped[str] = mapped_column(String(255))
    champion_version: Mapped[str] = mapped_column(String(64), index=True)
    shadow_version: Mapped[str] = mapped_column(String(64), index=True)

    champion_score: Mapped[float] = mapped_column(Float)
    shadow_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_threshold: Mapped[float] = mapped_column(Float)

    status: Mapped[str] = mapped_column(String(32), index=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


Index(
    "ix_shadow_predictions_pair_ts",
    ShadowPrediction.champion_version,
    ShadowPrediction.shadow_version,
    ShadowPrediction.ts,
)


class Approval(Base):
    """One human decision, final, with the evidence the person was shown.

    The fifth table of §4.6. Columns follow that ERD (`id`, `card_id`,
    `approver`, `decision`, `ts`); the rest are what makes the row answer "who
    let this model reach clinicians, on what evidence" without reconstructing
    the screen they saw.

    Two decisions a person makes, one table, told apart by `kind`:

    * `RETRAIN` -- a card the policy escalated (WORKFLOW.md §4: "nothing
      retrains until an ops user acts"). An `APPROVE` here is the
      `authorized_by` name Week 8's runner records on the retrain.
    * `PROMOTION` -- a gated challenger in shadow (§3.13: "Promotion to
      `champion` requires a human click ... which writes an approval row (who,
      when, card reference)"). An `APPROVE` here is the only thing in the
      repository that moves `champion`.

    `subject` is what the decision is about -- the card for a retrain, the
    retrain run for a promotion -- and `(kind, subject)` is unique, so each is
    decided once. A person who changes their mind does so about a new
    challenger, not by overwriting what they said about the old one.

    `evidence` is a snapshot of what the decision was made on: the gate's
    outcome and headline numbers, the shadow window's agreement statistics,
    the aliases at that moment. RISK_ANALYSIS.md §3 names rubber-stamping as a
    risk and answers it with "the full evidence chain ... not a bare 'Approve'
    button"; recording that chain on the row is what lets a later reviewer
    check the button was not bare.

    Append-only in application code, like every table above.
    """

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    approval_id: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    # §4.6: DECISION_CARDS ||--o{ APPROVALS.
    card_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("decision_cards.card_id"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16), index=True)
    subject: Mapped[str] = mapped_column(String(64))
    # The gated run a promotion is about; null for a retrain authorisation.
    run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("retrain_runs.run_id"), nullable=True, index=True
    )

    # Who, from the verified token -- never from the request body.
    approver: Mapped[str] = mapped_column(String(255), index=True)
    approver_role: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str] = mapped_column(String(1000))

    # What moved. Recorded as effects, like `retrain_runs.shadow_alias_moved`:
    # an approval whose alias move did not happen must not read as one that did.
    challenger_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    champion_version_before: Mapped[str | None] = mapped_column(String(64), nullable=True)
    champion_version_after: Mapped[str | None] = mapped_column(String(64), nullable=True)
    champion_alias_moved: Mapped[bool] = mapped_column(Boolean, default=False)
    shadow_alias_cleared: Mapped[bool] = mapped_column(Boolean, default=False)

    # Which promotion rules the decision was checked against, and the evidence
    # it was made on. Aggregates only.
    rules_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence: Mapped[dict] = mapped_column(JSON_TYPE)

    __table_args__ = (UniqueConstraint("kind", "subject", name="uq_approvals_kind_subject"),)


Index("ix_approvals_card_ts", Approval.card_id, Approval.ts)
