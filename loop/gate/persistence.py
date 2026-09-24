"""Writing a gate verdict to `retrain_runs`, at most once per challenger.

Append-only, like the three tables before it: nothing here issues UPDATE or
DELETE. That is also how §3.12's "card status set to `BLOCKED`" is honoured
without contradicting §4.6's "no UPDATE on card contents; status transitions
append history rows" -- the row this module writes *is* the history row, and a
card's effective state is read by joining to its retrain runs.

**Idempotency is enforced in two places, on purpose**, exactly as
`loop/engine/persistence.py` does it. `run_id` is derived from the card, the
mode and the challenger being judged, so a re-run produces the same primary
key; and `(card_id, mode, challenger_version)` carries a unique constraint, so
the guarantee survives any future change to how ids are shaped.
`record_retrain_run` checks for the existing row first and returns it, which
makes re-running a gate after a crash safe rather than merely non-destructive.

A *different* challenger on the same card is a distinct row. That is not a
loophole: it is what the roadmap's deliberately-bad-challenger demonstration
needs, and what a second attempt after a BLOCK would legitimately be.
"""

import hashlib

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from db.models import Approval, RetrainRun
from db.models import DecisionCard as DecisionCardRow
from loop.engine.rules import AUTO_PROCEED_SHADOW, RETRAIN_ACTIONS
from loop.gate.gate import GateResult

RUN_ID_HASH_LENGTH = 8


class RetrainPersistenceError(RuntimeError):
    """The retrain run could not be written. Never swallowed."""


def retrain_run_id(card_id: str, mode: str, challenger_version: str | None) -> str:
    """`rr-2026-01-03-1a2b3c4d-9f8e7d6c` -- deterministic, and traceable by eye.

    The card's own id is kept in the middle so a reader can tie a run to its
    decision without a join, and the suffix distinguishes challengers judged
    under the same card and mode. Re-gating the same challenger collides on the
    primary key instead of appending a second verdict about the same model.
    """
    digest = hashlib.sha256(f"{card_id}|{mode}|{challenger_version}".encode()).hexdigest()
    return f"rr-{card_id.removeprefix('dc-')}-{digest[:RUN_ID_HASH_LENGTH]}"


def build_retrain_row(
    *,
    run_id: str,
    card_id: str,
    mode: str,
    gate_result: GateResult,
    scenario: str | None = None,
    policy_version: str | None = None,
    action: str | None = None,
    mlflow_run: str | None = None,
    data_version: str | None = None,
    champion_version: str | None = None,
    challenger_version: str | None = None,
    shadow_alias_moved: bool = False,
    authorized_by: str | None = None,
) -> RetrainRun:
    """Projects a gate verdict onto its table row.

    Every column is read off the `GateResult` rather than recomputed -- a row
    whose `outcome` disagreed with its own `gate_result` would make the audit
    trail unusable.
    """
    return RetrainRun(
        run_id=run_id,
        card_id=card_id,
        scenario=scenario,
        policy_version=policy_version,
        action=action,
        mode=mode,
        mlflow_run=mlflow_run,
        data_version=data_version,
        champion_version=champion_version,
        challenger_version=challenger_version,
        outcome=gate_result.outcome,
        criteria_version=gate_result.criteria_version,
        failed_criteria_count=len(gate_result.failed_checks()),
        shadow_alias_moved=shadow_alias_moved,
        authorized_by=authorized_by,
        gate_result=gate_result.to_record(),
    )


def find_existing(
    session: Session, card_id: str, mode: str, challenger_version: str | None
) -> RetrainRun | None:
    """The verdict already recorded for this challenger under this card, if any."""
    statement = (
        select(RetrainRun)
        .where(RetrainRun.card_id == card_id)
        .where(RetrainRun.mode == mode)
        .where(RetrainRun.challenger_version == challenger_version)
    )
    return session.scalars(statement).one_or_none()


def runs_for_card(session: Session, card_id: str) -> tuple[RetrainRun, ...]:
    """Every retrain attempt against one card, oldest first.

    This is how a card's effective status is read: a card with a `BLOCK` run is
    §3.12's card "closed as BLOCKED", without anything having been written back
    onto the immutable card row.
    """
    statement = (
        select(RetrainRun)
        .where(RetrainRun.card_id == card_id)
        .order_by(RetrainRun.created_at, RetrainRun.run_id)
    )
    return tuple(session.scalars(statement).all())


def latest_run_for(
    session: Session, card_id: str, *, mode: str | None = None, data_version: str | None = None
) -> RetrainRun | None:
    """The most recent attempt matching the filters -- the restart check.

    A worker that died between training and persisting re-enters here: if a
    verdict for this card, mode and data version is already on record, the
    expensive half does not run again.
    """
    statement = select(RetrainRun).where(RetrainRun.card_id == card_id)
    if mode is not None:
        statement = statement.where(RetrainRun.mode == mode)
    if data_version is not None:
        statement = statement.where(RetrainRun.data_version == data_version)
    statement = statement.order_by(RetrainRun.created_at.desc(), RetrainRun.run_id.desc())
    return session.scalars(statement).first()


def cards_awaiting_retrain(
    session: Session,
    *,
    scenario: str | None = None,
    policy_version: str | None = None,
    automated_only: bool = True,
    include_approved: bool = False,
    limit: int | None = None,
) -> tuple[DecisionCardRow, ...]:
    """Cards that call for a retrain and have no run recorded yet, oldest first.

    `automated_only` is the default because WORKFLOW.md §4 is explicit about
    the other path: "Ambiguous evidence (confidence < 0.75) | Disposition =
    `ESCALATE_HUMAN`; nothing retrains until an ops user acts." An escalated
    card is therefore never picked up by a backlog sweep on its own; it is
    retrained when a caller names it and supplies the authorisation to record,
    or -- Week 9 -- when `include_approved` is set and an ops user has recorded
    an `APPROVE` decision for it in `approvals`. The person acting is the
    approval row; the sweep only notices it.
    """
    decided = select(RetrainRun.card_id)
    statement = (
        select(DecisionCardRow)
        .where(DecisionCardRow.action.in_(RETRAIN_ACTIONS))
        .where(DecisionCardRow.card_id.not_in(decided))
    )
    if automated_only:
        automated = DecisionCardRow.disposition == AUTO_PROCEED_SHADOW
        if include_approved:
            approved = (
                select(Approval.subject)
                .where(Approval.kind == "RETRAIN")
                .where(Approval.decision == "APPROVE")
            )
            statement = statement.where(or_(automated, DecisionCardRow.card_id.in_(approved)))
        else:
            statement = statement.where(automated)
    if scenario is not None:
        statement = statement.where(DecisionCardRow.scenario == scenario)
    if policy_version is not None:
        statement = statement.where(DecisionCardRow.policy_version == policy_version)
    statement = statement.order_by(DecisionCardRow.window_start, DecisionCardRow.card_id)
    if limit is not None:
        statement = statement.limit(limit)
    return tuple(session.scalars(statement).all())


def persist_run(session: Session, row: RetrainRun) -> RetrainRun:
    """Writes the row, or raises `RetrainPersistenceError`.

    Uses the ORM, so values are bound as parameters; no SQL is assembled from
    measured or decided data anywhere in this module.
    """
    try:
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    except Exception as error:
        session.rollback()
        raise RetrainPersistenceError(str(error)) from error


def record_retrain_run(session: Session, row: RetrainRun) -> tuple[RetrainRun, bool]:
    """Writes the verdict unless this challenger was already judged on this card.

    Returns `(row, created)`. `created` is False when an identical verdict was
    already on record, which is what makes re-running the gate over an
    already-gated backlog a no-op instead of a duplicate.
    """
    existing = find_existing(session, row.card_id, row.mode, row.challenger_version)
    if existing is not None:
        return existing, False
    return persist_run(session, row), True
