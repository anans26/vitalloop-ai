"""Week 9 closes Week 8's open seam: an escalated card retrains on a recorded approval.

Week 8 built the refusal ("nothing retrains until an ops user acts") and said
Week 9 would supply the name. These tests pin how the runner reads that name
back from `approvals`, and that a person's rejection is final -- a CLI flag
cannot overrule it.
"""

import pytest

from loop.approval.persistence import DECISION_APPROVE, DECISION_REJECT
from loop.approval.retrain import decide_retrain
from loop.approval.rules import load_rules
from loop.gate.gate import PASS
from loop.gate.persistence import cards_awaiting_retrain
from loop.gate.runner import GateRunnerError, run_card, run_pending

from .conftest import seed_card
from .test_runner import retrainer_for

ESCALATED = {"disposition": "ESCALATE_HUMAN", "confidence": 0.42}


@pytest.fixture
def wiring(gate_frames, champion_model):
    def frames_builder(scenario, window_start, sets):
        return {name: gate_frames[name] for name in sets}

    return {
        "champion_loader": lambda: (champion_model, "1"),
        "frames_builder": frames_builder,
        "alias_setter": lambda version, run_id, reason: {"alias": "shadow", "version": version},
        "retrainer": retrainer_for(champion_model),
    }


def _decide(session, card_id, decision, approver="ops-alice"):
    return decide_retrain(
        session,
        card_id,
        decision=decision,
        approver=approver,
        approver_role="ops",
        reason="reviewed the drift evidence on the card",
        rules=load_rules(),
    )


def test_an_unapproved_escalated_card_is_not_swept(gate_db, wiring):
    seed_card(gate_db, **ESCALATED)
    assert run_pending(gate_db, **wiring) == []


def test_an_approved_escalated_card_is_swept_and_records_the_approver(gate_db, wiring):
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_APPROVE)

    (run,) = run_pending(gate_db, **wiring)

    assert run.card_id == card.card_id
    assert run.outcome == PASS
    assert run.authorized_by == "ops-alice"
    assert run.row.authorized_by == "ops-alice"


def test_a_named_card_picks_up_its_approval_without_a_flag(gate_db, wiring):
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_APPROVE, approver="ops-bob")

    run = run_card(gate_db, card, **wiring)
    assert run.authorized_by == "ops-bob"


def test_a_rejected_card_is_not_swept(gate_db, wiring):
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_REJECT)
    assert cards_awaiting_retrain(gate_db, include_approved=True) == ()
    assert run_pending(gate_db, **wiring) == []


def test_a_rejection_cannot_be_overruled_from_the_command_line(gate_db, wiring):
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_REJECT)

    with pytest.raises(GateRunnerError, match="rejected by ops-alice"):
        run_card(gate_db, card, authorized_by="someone-else", **wiring)


def test_automated_cards_are_still_swept_without_any_approval(gate_db, wiring):
    card = seed_card(gate_db)
    (run,) = run_pending(gate_db, **wiring)
    assert run.card_id == card.card_id
    assert run.authorized_by is None


def test_the_default_backlog_query_is_unchanged(gate_db):
    """`include_approved` is opt-in: Week 8 callers see exactly what they saw."""
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_APPROVE)
    assert cards_awaiting_retrain(gate_db) == ()
    assert [c.card_id for c in cards_awaiting_retrain(gate_db, include_approved=True)] == [
        card.card_id
    ]


def test_an_approved_card_is_swept_only_once(gate_db, wiring):
    card = seed_card(gate_db, **ESCALATED)
    _decide(gate_db, card.card_id, DECISION_APPROVE)
    run_pending(gate_db, **wiring)
    assert run_pending(gate_db, **wiring) == []
