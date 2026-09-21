"""Writing a Decision Card to `decision_cards`, at most once per window and policy.

Append-only, like the two tables before it: nothing here issues UPDATE or
DELETE. The status transitions §3.12 describes (`EXECUTED`, `BLOCKED`) are
Week 8's, and §4.6 says they append history rows rather than editing a card.

**Idempotency is enforced in two places, on purpose.** `card_id` is derived
from the drift event and the policy version, so a re-evaluation produces the
same primary key; and `(drift_event_id, policy_version)` carries a unique
constraint, so the guarantee survives any future change to how ids are shaped.
`record_decision` checks for the existing card first and returns it, which
makes re-running the evaluator over a backlog safe rather than merely
non-destructive.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DecisionCard as DecisionCardRow
from loop.engine.card import DecisionCard


class DecisionPersistenceError(RuntimeError):
    """The decision could not be written. Never swallowed."""


def build_card_row(card: DecisionCard) -> DecisionCardRow:
    """Projects the Pydantic card onto its table row.

    The columns are a queryable view of `card_json`, so every one of them is
    read off the card rather than recomputed -- a row whose `action` column
    disagreed with its own `card_json` would make the audit trail unusable.
    """
    return DecisionCardRow(
        card_id=card.card_id,
        created_at=card.created_at,
        drift_event_id=card.trigger.drift_event_id,
        scenario=card.trigger.scenario,
        window_start=card.trigger.window_start,
        window_end=card.trigger.window_end,
        policy_version=card.policy_version,
        action=card.action,
        disposition=card.disposition,
        confidence=card.confidence,
        rule_id=str(card.rule_id),
        status=card.status,
        breaching_feature_count=len(card.trigger.breaching_features),
        max_psi=card.trigger.max_psi,
        prediction_drift=card.trigger.prediction_drift,
        model_version=card.model_version,
        data_version=card.data_version,
        candidate_data_version=card.candidate_data_version,
        card_json=card.to_json_dict(),
    )


def find_existing(
    session: Session, drift_event_id: str, policy_version: str
) -> DecisionCardRow | None:
    """The card this policy version already emitted for this window, if any."""
    statement = (
        select(DecisionCardRow)
        .where(DecisionCardRow.drift_event_id == drift_event_id)
        .where(DecisionCardRow.policy_version == policy_version)
    )
    return session.scalars(statement).one_or_none()


def persist_card(session: Session, row: DecisionCardRow) -> DecisionCardRow:
    """Writes the row, or raises `DecisionPersistenceError`.

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
        raise DecisionPersistenceError(str(error)) from error


def record_decision(session: Session, card: DecisionCard) -> tuple[DecisionCardRow, bool]:
    """Writes the card unless this policy version already decided this window.

    Returns `(row, created)`. `created` is False when an identical decision was
    already on record, which is what makes re-running the evaluator over an
    already-evaluated backlog a no-op instead of a duplicate.
    """
    existing = find_existing(session, card.trigger.drift_event_id, card.policy_version)
    if existing is not None:
        return existing, False
    return persist_card(session, build_card_row(card)), True
