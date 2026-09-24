"""Promotion: the one code path that moves `champion`.

ARCHITECTURE.md §6: "Alias change requires gate PASS + shadow window + logged
human approval." Each precondition gets a test that shows it *refusing*, with
nothing moved and nothing written -- a promotion-safety test that only checked
the happy path would be checking the wrong thing.
"""

import pytest
from sqlalchemy import select

from db.models import Approval
from loop.approval import persistence
from loop.approval.persistence import (
    AlreadyDecided,
    ApprovalConflict,
    ApprovalInvalid,
    ApprovalNotFound,
)
from loop.approval.promotion import (
    decide_promotion,
    pending_promotions,
    promotion_evidence,
)

from .conftest import CHALLENGER, CHAMPION, seed_run, seed_shadow_rows

REASON = "shadow agreement is high and the gate passed on both sets"


def _decide(session, run_id, decision, registry, rules, approver="ops-alice", reason=REASON):
    return decide_promotion(
        session,
        run_id,
        decision=decision,
        approver=approver,
        approver_role="ops",
        reason=reason,
        registry=registry,
        rules=rules,
    )


def _approvals(session):
    return session.scalars(select(Approval)).all()


# ---------------------------------------------------------------------------
# What is waiting
# ---------------------------------------------------------------------------
def test_a_passed_run_in_shadow_is_pending(approval_db, gated):
    _, run = gated
    assert [r.run_id for r in pending_promotions(approval_db)] == [run.run_id]


def test_a_blocked_run_is_never_pending(approval_db, blocked):
    assert pending_promotions(approval_db) == ()


def test_a_decided_run_is_no_longer_pending(approval_db, gated, registry, rules):
    _, run = gated
    _decide(approval_db, run.run_id, "REJECT", registry, rules)
    assert pending_promotions(approval_db) == ()


# ---------------------------------------------------------------------------
# The evidence chain the approver is shown
# ---------------------------------------------------------------------------
def test_the_evidence_carries_the_full_chain(approval_db, gated, registry, rules):
    card, run = gated
    seed_shadow_rows(approval_db, 10)
    evidence = promotion_evidence(approval_db, run, registry, rules)

    assert evidence["card"]["card_id"] == card.card_id
    assert evidence["card"]["disposition"] == "ESCALATE_HUMAN"
    assert evidence["gate"]["outcome"] == "PASS"
    assert evidence["gate"]["authorized_by"] == "ops-alice"
    assert evidence["aliases"] == {"champion": CHAMPION, "shadow": CHALLENGER}
    assert evidence["shadow_window"]["requests"] == 10
    assert evidence["shadow_window"]["required_requests"] == rules.min_shadow_requests
    assert {p["name"] for p in evidence["preconditions"]} == {
        "gate_pass",
        "challenger_in_shadow",
        "champion_unchanged_since_gate",
        "shadow_window_complete",
    }


def test_the_evidence_puts_champion_and_challenger_side_by_side(
    approval_db, gated, registry, rules
):
    _, run = gated
    gate = promotion_evidence(approval_db, run, registry, rules)["gate"]
    auc = gate["headline"]["frozen_holdout"]["roc_auc"]
    assert auc == {"champion": 0.60, "challenger": 0.61, "delta": 0.01}


def test_the_evidence_names_the_worst_subgroup_drop(approval_db, gated, registry, rules):
    """RISK_ANALYSIS.md §3 asks for "subgroup deltas", not only headline numbers."""
    _, run = gated
    worst = promotion_evidence(approval_db, run, registry, rules)["gate"][
        "worst_subgroup_auroc_drop"
    ]
    assert worst["frozen_holdout"] == {"subgroup": "age=[50-60)", "auroc_drop": 0.005}


def test_the_evidence_is_json_serialisable(approval_db, gated, registry, rules):
    import json

    _, run = gated
    seed_shadow_rows(approval_db, 3)
    json.dumps(promotion_evidence(approval_db, run, registry, rules))


# ---------------------------------------------------------------------------
# APPROVE: the happy path
# ---------------------------------------------------------------------------
def test_an_approval_moves_champion_and_clears_shadow(approval_db, gated, registry, rules):
    card, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)

    result = _decide(approval_db, run.run_id, "APPROVE", registry, rules)

    assert result.champion_alias_moved
    assert result.shadow_alias_cleared
    assert registry.aliases["champion"] == CHALLENGER
    assert "shadow" not in registry.aliases
    assert [m["action"] for m in registry.moves] == ["set_alias", "delete_alias"]


def test_every_alias_move_is_made_in_the_approvers_name(approval_db, gated, registry, rules):
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)
    _decide(approval_db, run.run_id, "APPROVE", registry, rules, approver="ops-carol")
    assert {m["actor"] for m in registry.moves} == {"ops-carol"}
    assert run.card_id in registry.moves[0]["reason"]


def test_an_approval_writes_the_row_section_3_13_asks_for(approval_db, gated, registry, rules):
    """§3.13: the row records who, when, and the card reference."""
    card, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)
    _decide(approval_db, run.run_id, "APPROVE", registry, rules)

    (row,) = _approvals(approval_db)
    assert row.kind == "PROMOTION"
    assert row.approver == "ops-alice"
    assert row.approver_role == "ops"
    assert row.ts is not None
    assert row.card_id == card.card_id
    assert row.run_id == run.run_id
    assert row.decision == "APPROVE"
    assert row.reason == REASON
    assert (row.champion_version_before, row.champion_version_after) == (CHAMPION, CHALLENGER)
    assert row.champion_alias_moved and row.shadow_alias_cleared
    assert row.rules_version == rules.version
    assert row.evidence["ready_to_approve"] is True


# ---------------------------------------------------------------------------
# APPROVE: every precondition refuses on its own
# ---------------------------------------------------------------------------
def test_an_incomplete_shadow_window_refuses(approval_db, gated, registry, rules):
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests - 1)

    with pytest.raises(ApprovalConflict, match="requests dual-scored"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)
    assert registry.moves == []
    assert _approvals(approval_db) == []


def test_requests_scored_against_another_champion_do_not_count(approval_db, gated, registry, rules):
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests, champion="7")
    with pytest.raises(ApprovalConflict):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)


def test_a_challenger_no_longer_in_shadow_refuses(approval_db, gated, registry, rules):
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)
    registry.aliases["shadow"] = "3"  # a later PASS moved shadow on

    with pytest.raises(ApprovalConflict, match="shadow alias -> 3"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)
    assert registry.aliases["champion"] == CHAMPION


def test_a_champion_that_changed_since_the_gate_refuses(approval_db, gated, registry, rules):
    """The gate's verdict is "better than *that* model"; a new champion voids it."""
    _, run = gated
    registry.aliases["champion"] = "5"
    seed_shadow_rows(approval_db, rules.min_shadow_requests, champion="5")

    with pytest.raises(ApprovalConflict, match="gate compared against 1"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)
    assert registry.aliases["champion"] == "5"


def test_a_blocked_run_cannot_be_approved(approval_db, blocked, registry, rules):
    _, run = blocked
    with pytest.raises(ApprovalConflict, match="only a gate PASS"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)
    assert registry.moves == []


def test_a_pass_that_did_not_reach_shadow_cannot_be_approved(approval_db, gated, registry, rules):
    card, _ = gated
    run = seed_run(approval_db, card.card_id, challenger="4", shadow_moved=False)
    with pytest.raises(ApprovalConflict):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)


def test_an_unknown_run_is_not_found(approval_db, registry, rules):
    with pytest.raises(ApprovalNotFound):
        _decide(approval_db, "rr-nope", "APPROVE", registry, rules)


# ---------------------------------------------------------------------------
# REJECT
# ---------------------------------------------------------------------------
def test_a_rejection_needs_no_shadow_window(approval_db, gated, registry, rules):
    """Stopping a model from reaching clinicians needs no evidence threshold."""
    _, run = gated
    result = _decide(approval_db, run.run_id, "REJECT", registry, rules)

    assert not result.champion_alias_moved
    assert registry.aliases["champion"] == CHAMPION
    (row,) = _approvals(approval_db)
    assert row.decision == "REJECT"
    assert row.champion_version_before == row.champion_version_after == CHAMPION


def test_a_rejection_stops_the_rejected_model_scoring(approval_db, gated, registry, rules):
    _, run = gated
    result = _decide(approval_db, run.run_id, "REJECT", registry, rules)
    assert result.shadow_alias_cleared
    assert "shadow" not in registry.aliases


def test_a_rejection_leaves_a_newer_shadow_alone(approval_db, gated, registry, rules):
    _, run = gated
    registry.aliases["shadow"] = "3"
    result = _decide(approval_db, run.run_id, "REJECT", registry, rules)
    assert not result.shadow_alias_cleared
    assert registry.aliases["shadow"] == "3"


# ---------------------------------------------------------------------------
# One decision, with a reason, recorded before anything is left inconsistent
# ---------------------------------------------------------------------------
def test_a_second_decision_is_refused_and_moves_nothing(approval_db, gated, registry, rules):
    _, run = gated
    _decide(approval_db, run.run_id, "REJECT", registry, rules)
    registry.aliases["shadow"] = CHALLENGER
    seed_shadow_rows(approval_db, rules.min_shadow_requests)
    moves_before = list(registry.moves)

    with pytest.raises(AlreadyDecided):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules, approver="ops-bob")
    assert registry.moves == moves_before
    assert registry.aliases["champion"] == CHAMPION


@pytest.mark.parametrize("reason", ["", "   ", "ok", "lgtm"])
def test_a_decision_without_a_real_reason_is_refused(reason, approval_db, gated, registry, rules):
    _, run = gated
    with pytest.raises(ApprovalInvalid, match="reason"):
        _decide(approval_db, run.run_id, "REJECT", registry, rules, reason=reason)


def test_an_unknown_decision_is_refused(approval_db, gated, registry, rules):
    _, run = gated
    with pytest.raises(ApprovalInvalid, match="decision"):
        _decide(approval_db, run.run_id, "MAYBE", registry, rules)


def test_a_failed_approval_write_reverses_the_promotion(
    approval_db, gated, registry, rules, monkeypatch
):
    """No champion move may survive without its approval row (RISK_ANALYSIS.md §3)."""
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)

    def fail(session, row):
        raise RuntimeError("database went away")

    monkeypatch.setattr("loop.approval.promotion.record_decision", fail)
    with pytest.raises(RuntimeError, match="database went away"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)

    assert registry.aliases["champion"] == CHAMPION
    assert registry.aliases["shadow"] == CHALLENGER
    assert [m["to_version"] for m in registry.moves] == [CHALLENGER, CHAMPION]
    assert "compensation" in registry.moves[-1]["reason"]


def test_a_registry_failure_writes_no_approval(approval_db, gated, registry, rules):
    _, run = gated
    seed_shadow_rows(approval_db, rules.min_shadow_requests)
    registry.fail_on_set = "champion"
    with pytest.raises(RuntimeError, match="registry unavailable"):
        _decide(approval_db, run.run_id, "APPROVE", registry, rules)
    assert _approvals(approval_db) == []


def test_approval_ids_are_derived_from_what_is_decided():
    assert persistence.approval_id("PROMOTION", "rr-x") == "ap-promotion-rr-x"
    assert persistence.approval_id("RETRAIN", "dc-x") == "ap-retrain-dc-x"


def test_only_promotion_and_initial_registration_can_move_champion():
    """RISK_ANALYSIS.md §3, "Silent clinical model swap": there is no other code path.

    A source scan for any alias setter called with the champion alias. Week 4's
    `ensure_initial_champion` (in `ml/registry.py`) sets it once, on an empty
    registry; everything after that must go through `decide_promotion`.
    """
    import re
    from pathlib import Path

    setter = re.compile(
        r"(set_alias|\.set)\((?:(?!\)\n).){0,400}?(CHAMPION_ALIAS|\"champion\"|INITIAL_ALIAS)",
        re.S,
    )
    found = {
        path.as_posix()
        for package in ("api", "loop", "ml", "scripts", "scenarios", "db")
        for path in Path(package).rglob("*.py")
        if setter.search(path.read_text(encoding="utf-8"))
    }
    assert found == {"loop/approval/promotion.py", "ml/registry.py"}
