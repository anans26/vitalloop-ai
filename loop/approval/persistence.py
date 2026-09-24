"""Writing a human decision to `approvals`, exactly once per subject.

Append-only, like every audit table: nothing here updates or deletes. A
decision is final -- `(kind, subject)` is unique, and `approval_id` is derived
from it -- so a second click on the same challenger collides instead of
quietly overwriting what was said the first time.

Unlike the engine and the gate, a repeat is **refused** rather than returned
as "already on record". Re-running a deterministic evaluation is harmless and
produces the same answer; a second human decision about the same thing is a
different answer, and silently keeping the first would tell the second person
they had decided something they had not.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Approval

KIND_RETRAIN = "RETRAIN"
KIND_PROMOTION = "PROMOTION"
KINDS = (KIND_RETRAIN, KIND_PROMOTION)

DECISION_APPROVE = "APPROVE"
DECISION_REJECT = "REJECT"
DECISIONS = (DECISION_APPROVE, DECISION_REJECT)

# `approvals.reason` is String(1000).
MAX_REASON_LENGTH = 1000


class ApprovalError(RuntimeError):
    """Base for every refusal. `status` is the HTTP code the API maps it to."""

    status = 409


class ApprovalNotFound(ApprovalError):
    status = 404


class ApprovalInvalid(ApprovalError):
    status = 422


class ApprovalConflict(ApprovalError):
    """A precondition does not hold: the decision cannot be taken right now."""

    status = 409


class AlreadyDecided(ApprovalError):
    status = 409


def approval_id(kind: str, subject: str) -> str:
    """`ap-promotion-rr-2026-01-01-...` -- one id per thing that can be decided."""
    return f"ap-{kind.lower()}-{subject}"


def find_decision(session: Session, kind: str, subject: str) -> Approval | None:
    statement = select(Approval).where(Approval.kind == kind).where(Approval.subject == subject)
    return session.scalars(statement).one_or_none()


def decisions_for_card(session: Session, card_id: str) -> tuple[Approval, ...]:
    statement = (
        select(Approval).where(Approval.card_id == card_id).order_by(Approval.ts, Approval.id)
    )
    return tuple(session.scalars(statement).all())


def record_decision(session: Session, row: Approval) -> Approval:
    """Writes the decision, or raises `AlreadyDecided` if one is on record."""
    if find_decision(session, row.kind, row.subject) is not None:
        raise AlreadyDecided(f"{row.kind.lower()} of {row.subject} has already been decided")
    try:
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    except Exception as error:
        session.rollback()
        # A concurrent click that won the unique constraint is still "already
        # decided", not a server error.
        if find_decision(session, row.kind, row.subject) is not None:
            raise AlreadyDecided(
                f"{row.kind.lower()} of {row.subject} has already been decided"
            ) from error
        raise


def validate_decision(decision: str, reason: str | None, min_reason_length: int) -> tuple[str, str]:
    """Normalises the two things a person supplies, or raises `ApprovalInvalid`."""
    normalised = (decision or "").strip().upper()
    if normalised not in DECISIONS:
        raise ApprovalInvalid(f"decision must be one of {DECISIONS}")
    text = " ".join((reason or "").split())
    if len(text) < min_reason_length:
        raise ApprovalInvalid(
            f"a reason of at least {min_reason_length} characters is required for every decision"
        )
    if len(text) > MAX_REASON_LENGTH:
        raise ApprovalInvalid(f"a reason may be at most {MAX_REASON_LENGTH} characters")
    return normalised, text
