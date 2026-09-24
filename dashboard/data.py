"""Everything the dashboard reads, as plain functions over a session.

ARCHITECTURE.md §5 draws the dashboard reading Postgres and MLflow directly.
This module is that read side, and it is **read-only**: nothing here adds,
changes or removes a row. Every figure a page shows comes from one of these
functions, so the pages hold layout and nothing else, and the numbers can be
tested without a browser.

**A card's lifecycle state is derived, never stored.** §4.6 keeps every table
append-only ("status transitions append history rows"), so a card's current
state is a function of the card and the rows appended after it --
`lifecycle_state` is that function, and the dashboard, the audit log and the
audit PDF all call the same one.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from db.models import Approval, DecisionCard, DriftEvent, Prediction, RetrainRun, ShadowPrediction
from loop.engine.rules import ALERT_ONLY, ESCALATE_HUMAN, NO_OP, RETRAIN_ACTIONS
from ml.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Lifecycle state
# ---------------------------------------------------------------------------
STATE_NO_OP = "NO_OP"
STATE_ALERT = "ALERT"
STATE_ALERT_ESCALATED = "ALERT (escalated)"
STATE_AWAITING_AUTHORISATION = "AWAITING AUTHORISATION"
STATE_RETRAIN_REJECTED = "RETRAIN REJECTED"
STATE_AWAITING_RETRAIN = "AWAITING RETRAIN"
STATE_BLOCKED = "BLOCKED"
STATE_IN_SHADOW = "IN SHADOW"
STATE_PASSED = "PASSED (not in shadow)"
STATE_PROMOTED = "PROMOTED"
STATE_PROMOTION_REJECTED = "PROMOTION REJECTED"


def lifecycle_state(card: DecisionCard, runs: list[RetrainRun], approvals: list[Approval]) -> str:
    """Where one card is in WORKFLOW.md §1's flow, read off the rows after it.

    The latest retrain run decides, because that is what happened most
    recently to the card: a card promoted and then re-run with the deliberately
    bad challenger is `BLOCKED` now, and its audit history shows both.
    """
    if card.action == NO_OP:
        return STATE_NO_OP
    if card.action == ALERT_ONLY:
        return STATE_ALERT_ESCALATED if card.disposition == ESCALATE_HUMAN else STATE_ALERT
    if card.action not in RETRAIN_ACTIONS:
        return card.action

    by_run = {a.run_id: a for a in approvals if a.kind == "PROMOTION"}
    retrain_decision = next((a for a in approvals if a.kind == "RETRAIN"), None)

    if not runs:
        if retrain_decision is not None and retrain_decision.decision == "REJECT":
            return STATE_RETRAIN_REJECTED
        if card.disposition == ESCALATE_HUMAN and retrain_decision is None:
            return STATE_AWAITING_AUTHORISATION
        return STATE_AWAITING_RETRAIN

    latest = max(runs, key=lambda run: (_aware(run.created_at), run.run_id))
    if latest.outcome != "PASS":
        return STATE_BLOCKED
    decision = by_run.get(latest.run_id)
    if decision is not None:
        return STATE_PROMOTED if decision.decision == "APPROVE" else STATE_PROMOTION_REJECTED
    return STATE_IN_SHADOW if latest.shadow_alias_moved else STATE_PASSED


def _aware(value: datetime | None) -> datetime:
    """SQLite hands back naive datetimes; compare everything as UTC."""
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Overview:
    predictions: int
    successful_predictions: int
    predictions_by_version: dict[str, int]
    last_prediction_at: datetime | None
    median_latency_ms: float | None
    drift_windows: int
    breaching_windows: int
    cards: int
    cards_by_state: dict[str, int]
    shadow_scores: int


def overview(session: Session) -> Overview:
    ok = Prediction.status == "success"
    by_version = dict(
        session.execute(
            select(Prediction.model_version, func.count())
            .where(ok)
            .group_by(Prediction.model_version)
            .order_by(Prediction.model_version)
        ).all()
    )
    latencies = sorted(
        value
        for value in session.scalars(
            select(Prediction.latency_ms).where(ok).order_by(Prediction.ts.desc()).limit(500)
        )
        if value is not None
    )
    states: dict[str, int] = {}
    for _, state in cards_with_state(session):
        states[state] = states.get(state, 0) + 1

    return Overview(
        predictions=session.scalar(select(func.count()).select_from(Prediction)) or 0,
        successful_predictions=session.scalar(select(func.count()).where(ok)) or 0,
        predictions_by_version={str(k): int(v) for k, v in by_version.items()},
        last_prediction_at=session.scalar(select(func.max(Prediction.ts))),
        median_latency_ms=latencies[len(latencies) // 2] if latencies else None,
        drift_windows=session.scalar(select(func.count()).select_from(DriftEvent)) or 0,
        breaching_windows=session.scalar(
            select(func.count()).where(DriftEvent.breaching_feature_count > 0)
        )
        or 0,
        cards=session.scalar(select(func.count()).select_from(DecisionCard)) or 0,
        cards_by_state=dict(sorted(states.items())),
        shadow_scores=session.scalar(select(func.count()).select_from(ShadowPrediction)) or 0,
    )


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def scenarios(session: Session) -> list[str]:
    return list(
        session.scalars(select(DriftEvent.scenario).distinct().order_by(DriftEvent.scenario))
    )


def drift_events(session: Session, scenario: str | None = None) -> list[DriftEvent]:
    statement = select(DriftEvent).order_by(
        DriftEvent.scenario, DriftEvent.window_start, DriftEvent.created_at
    )
    if scenario:
        statement = statement.where(DriftEvent.scenario == scenario)
    return list(session.scalars(statement))


def report_path(event: DriftEvent, root: Path = PROJECT_ROOT) -> Path | None:
    """The Evidently HTML for a window, if it exists on this machine.

    `report_uri` is stored relative to the project; anything that would
    resolve outside it is refused rather than read.
    """
    if not event.report_uri:
        return None
    path = (root / event.report_uri).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def _runs_by_card(session: Session) -> dict[str, list[RetrainRun]]:
    grouped: dict[str, list[RetrainRun]] = {}
    for run in session.scalars(
        select(RetrainRun).order_by(RetrainRun.created_at, RetrainRun.run_id)
    ):
        grouped.setdefault(run.card_id, []).append(run)
    return grouped


def _approvals_by_card(session: Session) -> dict[str, list[Approval]]:
    grouped: dict[str, list[Approval]] = {}
    for row in session.scalars(select(Approval).order_by(Approval.ts, Approval.id)):
        grouped.setdefault(row.card_id, []).append(row)
    return grouped


def cards_with_state(
    session: Session, *, scenario: str | None = None, action: str | None = None
) -> list[tuple[DecisionCard, str]]:
    statement = select(DecisionCard).order_by(
        DecisionCard.window_start.desc(), DecisionCard.card_id
    )
    if scenario:
        statement = statement.where(DecisionCard.scenario == scenario)
    if action:
        statement = statement.where(DecisionCard.action == action)
    runs, approvals = _runs_by_card(session), _approvals_by_card(session)
    return [
        (card, lifecycle_state(card, runs.get(card.card_id, []), approvals.get(card.card_id, [])))
        for card in session.scalars(statement)
    ]


@dataclass(frozen=True)
class CardHistory:
    card: DecisionCard
    event: DriftEvent | None
    runs: list[RetrainRun]
    approvals: list[Approval]
    alias_moves: list[dict]
    state: str


def card_history(
    session: Session, card_id: str, alias_rows: list[dict] | None = None
) -> CardHistory | None:
    """One card and everything appended after it: the §4.6 lineage, three joins deep."""
    card = session.get(DecisionCard, card_id)
    if card is None:
        return None
    runs = list(
        session.scalars(
            select(RetrainRun)
            .where(RetrainRun.card_id == card_id)
            .order_by(RetrainRun.created_at, RetrainRun.run_id)
        )
    )
    approvals = list(
        session.scalars(
            select(Approval).where(Approval.card_id == card_id).order_by(Approval.ts, Approval.id)
        )
    )
    return CardHistory(
        card=card,
        event=session.get(DriftEvent, card.drift_event_id),
        runs=runs,
        approvals=approvals,
        alias_moves=alias_moves_for(card_id, [r.run_id for r in runs], alias_rows or []),
        state=lifecycle_state(card, runs, approvals),
    )


def alias_moves_for(card_id: str, run_ids: list[str], alias_rows: list[dict]) -> list[dict]:
    """Registry audit rows whose recorded reason names this card or one of its runs."""
    keys = [card_id, *run_ids]
    return [row for row in alias_rows if any(key in str(row.get("reason", "")) for key in keys)]


# ---------------------------------------------------------------------------
# Champion vs challenger
# ---------------------------------------------------------------------------
def retrain_runs(session: Session) -> list[RetrainRun]:
    return list(
        session.scalars(
            select(RetrainRun).order_by(RetrainRun.created_at.desc(), RetrainRun.run_id)
        )
    )


def headline_metrics(run: RetrainRun) -> list[dict]:
    """Champion beside challenger for every evaluation set the gate used."""
    stored = dict(run.gate_result or {})
    champion = stored.get("champion_metrics") or {}
    challenger = stored.get("challenger_metrics") or {}
    rows = []
    for evaluation_set in stored.get("evaluation_sets") or sorted(champion):
        ours, theirs = champion.get(evaluation_set) or {}, challenger.get(evaluation_set) or {}
        for metric in (
            "roc_auc",
            "recall_at_top_decile",
            "brier_score",
            "expected_calibration_error",
        ):
            a, b = ours.get(metric), theirs.get(metric)
            rows.append(
                {
                    "evaluation_set": evaluation_set,
                    "metric": metric,
                    "champion": a,
                    "challenger": b,
                    "delta": round(b - a, 6)
                    if isinstance(a, int | float) and isinstance(b, int | float)
                    else None,
                }
            )
    return rows


def gate_checks(run: RetrainRun, *, failed_only: bool = False) -> list[dict]:
    checks = list((run.gate_result or {}).get("checks") or [])
    return [c for c in checks if not c.get("passed")] if failed_only else checks


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
EVENT_KINDS = ("drift", "card", "retrain", "approval", "alias")


def audit_events(
    session: Session,
    alias_rows: list[dict] | None = None,
    *,
    query: str = "",
    kinds: tuple[str, ...] = EVENT_KINDS,
    limit: int = 500,
) -> list[dict]:
    """One time-ordered log across every audit table, newest first, text-searchable."""
    events: list[dict] = []
    if "drift" in kinds:
        for e in session.scalars(select(DriftEvent)):
            events.append(
                _event(
                    e.created_at,
                    "drift",
                    e.event_id,
                    e.scenario,
                    f"window {e.window_start.date()}: {e.breaching_feature_count} breaching, "
                    f"max PSI {_fmt(e.max_psi)}, prediction drift {e.prediction_drift}",
                )
            )
    if "card" in kinds:
        for card, state in cards_with_state(session):
            events.append(
                _event(
                    card.created_at,
                    "card",
                    card.card_id,
                    card.scenario,
                    f"{card.policy_version} rule {card.rule_id}: "
                    f"{card.action} / {card.disposition} "
                    f"(confidence {_fmt(card.confidence)}) -> {state}",
                )
            )
    if "retrain" in kinds:
        for run in session.scalars(select(RetrainRun)):
            events.append(
                _event(
                    run.created_at,
                    "retrain",
                    run.run_id,
                    run.card_id,
                    f"{run.mode} challenger {run.challenger_version} "
                    f"vs champion {run.champion_version}: "
                    f"{run.outcome} ({run.failed_criteria_count} failed checks)"
                    + (f", authorised by {run.authorized_by}" if run.authorized_by else ""),
                )
            )
    if "approval" in kinds:
        for a in session.scalars(select(Approval)):
            events.append(
                _event(
                    a.ts,
                    "approval",
                    a.approval_id,
                    a.card_id,
                    f"{a.kind} {a.decision} by {a.approver}: {a.reason}",
                )
            )
    if "alias" in kinds:
        for row in alias_rows or []:
            events.append(
                _event(
                    _parse_ts(row.get("timestamp_utc")),
                    "alias",
                    str(row.get("alias")),
                    str(row.get("to_version")),
                    f"{row.get('action')} {row.get('alias')}: {row.get('from_version')} -> "
                    f"{row.get('to_version')} by {row.get('actor')} ({row.get('reason')})",
                )
            )

    needle = query.strip().lower()
    if needle:
        events = [e for e in events if needle in " ".join(str(v) for v in e.values()).lower()]
    events.sort(key=lambda e: _aware(e["ts"]), reverse=True)
    return events[:limit]


def _event(ts, kind, ref, subject, summary) -> dict:
    return {"ts": ts, "kind": kind, "ref": ref, "subject": subject, "summary": summary}


def _parse_ts(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def find_predictions(session: Session, key: str, limit: int = 50) -> list[Prediction]:
    """Prediction audit rows by request id or input hash (exact), or caller (exact).

    The row holds a hash of the input, never the input, so this proves which
    request produced a score without the dashboard ever showing clinical data.
    """
    key = key.strip()
    if not key:
        return []
    statement = (
        select(Prediction)
        .where(
            or_(
                Prediction.request_id == key, Prediction.input_hash == key, Prediction.caller == key
            )
        )
        .order_by(Prediction.ts.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))
