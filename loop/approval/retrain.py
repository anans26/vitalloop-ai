"""Authorising a retrain the policy escalated.

WORKFLOW.md §4: "Ambiguous evidence (confidence < 0.75) | Disposition =
`ESCALATE_HUMAN`; nothing retrains until an ops user acts." Week 8 built the
refusal -- `loop.gate.runner.check_authorised` will not retrain an escalated
card without a name to record -- and said Week 9's approval screen would
supply the name. This module is that supply: an `APPROVE` decision of kind
`RETRAIN` is the authorisation, and `authorising_approver` is how the runner
reads it back.

The decision authorises; it does not train. Training takes minutes and belongs
to the runner (`python -m loop.gate.runner`), which now sweeps approved
escalated cards alongside automated ones. Keeping the two apart means an API
request never holds a retrain open, and the retrain still goes through every
Week 8 check -- the pinned data version, the gate, the shadow-only effect.

A `REJECT` closes the card: the evidence was reviewed and judged not worth a
retrain, and that judgement is on record with its reason.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Approval, RetrainRun
from db.models import DecisionCard as DecisionCardRow
from loop.approval.persistence import (
    DECISION_APPROVE,
    KIND_RETRAIN,
    ApprovalConflict,
    ApprovalNotFound,
    approval_id,
    find_decision,
    record_decision,
    validate_decision,
)
from loop.approval.rules import PromotionRules
from loop.engine.rules import ESCALATE_HUMAN, RETRAIN_ACTIONS


def _card_summary(card_row: DecisionCardRow) -> dict:
    card = dict(card_row.card_json or {})
    trigger = card.get("trigger") or {}
    return {
        "card_id": card_row.card_id,
        "scenario": card_row.scenario,
        "window_start": str(card_row.window_start),
        "policy_version": card_row.policy_version,
        "rule_id": card_row.rule_id,
        "action": card_row.action,
        "disposition": card_row.disposition,
        "confidence": card_row.confidence,
        "confidence_breakdown": card.get("confidence_breakdown"),
        "rationale": card.get("rationale"),
        "breaching_features": trigger.get("breaching_features") or [],
        "prediction_drift": card_row.prediction_drift,
        "label_maturity": trigger.get("label_maturity"),
        "candidate_data_version": card_row.candidate_data_version,
        "acceptance_criteria": card.get("acceptance_criteria"),
        "narrative": card.get("narrative"),
        "narrative_source": card.get("narrative_source"),
    }


def cards_awaiting_authorisation(session: Session) -> tuple[DecisionCardRow, ...]:
    """Escalated retrain cards nobody has decided and nothing has retrained, oldest first."""
    decided = select(Approval.subject).where(Approval.kind == KIND_RETRAIN)
    retrained = select(RetrainRun.card_id)
    statement = (
        select(DecisionCardRow)
        .where(DecisionCardRow.action.in_(RETRAIN_ACTIONS))
        .where(DecisionCardRow.disposition == ESCALATE_HUMAN)
        .where(DecisionCardRow.card_id.not_in(decided))
        .where(DecisionCardRow.card_id.not_in(retrained))
        .order_by(DecisionCardRow.window_start, DecisionCardRow.card_id)
    )
    return tuple(session.scalars(statement).all())


def retrain_evidence(card_row: DecisionCardRow) -> dict:
    """What the person deciding sees, and what the approval row records."""
    return {"card": _card_summary(card_row)}


def authorising_approver(session: Session, card_id: str) -> str | None:
    """The approver whose `APPROVE` authorises this card's retrain, if any."""
    decision = find_decision(session, KIND_RETRAIN, card_id)
    if decision is not None and decision.decision == DECISION_APPROVE:
        return decision.approver
    return None


def decide_retrain(
    session: Session,
    card_id: str,
    *,
    decision: str,
    approver: str,
    approver_role: str,
    reason: str,
    rules: PromotionRules,
) -> Approval:
    """Records a person's decision on an escalated card. Moves nothing."""
    decision, reason = validate_decision(decision, reason, rules.min_reason_length)

    card_row = session.get(DecisionCardRow, card_id)
    if card_row is None:
        raise ApprovalNotFound(f"no decision card {card_id!r}")
    if card_row.action not in RETRAIN_ACTIONS:
        raise ApprovalConflict(
            f"card {card_id} decided {card_row.action}; only {RETRAIN_ACTIONS} can be "
            "authorised to retrain, and a person does not override the policy's action"
        )
    if card_row.disposition != ESCALATE_HUMAN:
        raise ApprovalConflict(
            f"card {card_id} has disposition {card_row.disposition}; only escalated cards "
            "wait for a person -- an automated card already carries its own authority"
        )
    existing_run = session.scalars(
        select(RetrainRun.run_id).where(RetrainRun.card_id == card_id).limit(1)
    ).first()
    if existing_run is not None:
        raise ApprovalConflict(f"card {card_id} has already been retrained ({existing_run})")

    row = Approval(
        approval_id=approval_id(KIND_RETRAIN, card_id),
        card_id=card_id,
        kind=KIND_RETRAIN,
        subject=card_id,
        run_id=None,
        approver=approver,
        approver_role=approver_role,
        decision=decision,
        reason=reason,
        rules_version=rules.version,
        evidence=retrain_evidence(card_row),
    )
    return record_decision(session, row)
