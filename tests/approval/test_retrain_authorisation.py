"""Authorising a retrain the policy escalated (WORKFLOW.md §4).

The decision authorises; it never trains and never overrides the policy's
action. Every refusal is tested with nothing written.
"""

import pytest
from sqlalchemy import select

from db.models import Approval
from loop.approval.persistence import AlreadyDecided, ApprovalConflict, ApprovalNotFound
from loop.approval.retrain import (
    authorising_approver,
    cards_awaiting_authorisation,
    decide_retrain,
)
from tests.gate.conftest import seed_card

from .conftest import seed_run

REASON = "two lab features moved sharply; worth a challenger"


def _decide(session, card_id, decision, rules, approver="ops-alice"):
    return decide_retrain(
        session,
        card_id,
        decision=decision,
        approver=approver,
        approver_role="ops",
        reason=REASON,
        rules=rules,
    )


@pytest.fixture
def escalated(approval_db):
    return seed_card(approval_db, disposition="ESCALATE_HUMAN", confidence=0.42)


def test_an_escalated_retrain_card_is_awaiting(approval_db, escalated):
    assert [c.card_id for c in cards_awaiting_authorisation(approval_db)] == [escalated.card_id]


def test_automated_and_non_retrain_cards_are_not_awaiting(approval_db):
    seed_card(approval_db, card_id="dc-2026-01-01-00000001")
    seed_card(
        approval_db,
        card_id="dc-2026-01-01-00000002",
        action="ALERT_ONLY",
        disposition="ESCALATE_HUMAN",
    )
    assert cards_awaiting_authorisation(approval_db) == ()


def test_an_approval_authorises_and_names_the_approver(approval_db, escalated, rules):
    row = _decide(approval_db, escalated.card_id, "APPROVE", rules)

    assert row.kind == "RETRAIN"
    assert row.subject == escalated.card_id
    assert row.run_id is None
    assert row.evidence["card"]["card_id"] == escalated.card_id
    assert authorising_approver(approval_db, escalated.card_id) == "ops-alice"
    assert cards_awaiting_authorisation(approval_db) == ()


def test_a_rejection_authorises_nothing(approval_db, escalated, rules):
    _decide(approval_db, escalated.card_id, "REJECT", rules)
    assert authorising_approver(approval_db, escalated.card_id) is None
    assert cards_awaiting_authorisation(approval_db) == ()


def test_an_automated_card_needs_no_person(approval_db, rules):
    card = seed_card(approval_db)
    with pytest.raises(ApprovalConflict, match="already carries its own authority"):
        _decide(approval_db, card.card_id, "APPROVE", rules)


def test_a_person_cannot_turn_an_alert_into_a_retrain(approval_db, rules):
    card = seed_card(approval_db, action="ALERT_ONLY", disposition="ESCALATE_HUMAN")
    with pytest.raises(ApprovalConflict, match="does not override the policy"):
        _decide(approval_db, card.card_id, "APPROVE", rules)


def test_an_already_retrained_card_cannot_be_authorised_again(approval_db, escalated, rules):
    seed_run(approval_db, escalated.card_id)
    with pytest.raises(ApprovalConflict, match="already been retrained"):
        _decide(approval_db, escalated.card_id, "APPROVE", rules)


def test_a_second_decision_is_refused(approval_db, escalated, rules):
    _decide(approval_db, escalated.card_id, "APPROVE", rules)
    with pytest.raises(AlreadyDecided):
        _decide(approval_db, escalated.card_id, "REJECT", rules, approver="ops-bob")
    (row,) = approval_db.scalars(select(Approval)).all()
    assert row.decision == "APPROVE"


def test_an_unknown_card_is_not_found(approval_db, rules):
    with pytest.raises(ApprovalNotFound):
        _decide(approval_db, "dc-nope", "APPROVE", rules)
