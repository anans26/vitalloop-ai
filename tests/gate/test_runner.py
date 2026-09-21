"""Week 8's deliverable, end to end: card -> retrain -> gate -> persist -> alias.

    "drift -> card -> retrain -> gate PASS path fully automated; bad challenger
     BLOCKED with reasons"  (IMPLEMENTATION_ROADMAP.md, Week 8)

Every impure step is injected, so this file exercises the real orchestration
against real models and a real database without an MLflow server, a Postgres
service, or a trained artifact on disk. What is *not* injected is the gate
itself, the persistence, the authorisation check and the alias decision -- those
are the behaviour under test.

The bad challenger is `InvertedChallenger`, which reverses the champion's
ranking. It is bad by construction rather than by seed, which is what the
roadmap's "deliberately-bad-challenger fixture" has to be if the BLOCK is going
to mean anything.
"""

import pytest

from db.models import DecisionCard as DecisionCardRow
from db.models import RetrainRun
from loop.gate.criteria import load_criteria
from loop.gate.gate import BLOCK, PASS
from loop.gate.persistence import runs_for_card
from loop.gate.runner import (
    SHADOW_ALIAS,
    GateRunnerError,
    check_authorised,
    run_card,
    run_pending,
    summarise,
)
from ml.retrain import MODE_LIVE, MODE_REPLAY

from .conftest import ConstantChallenger, challenger_run, seed_card


class AliasRecorder:
    """Stands in for `ml.registry.set_alias`, and remembers what it was asked to move."""

    def __init__(self):
        self.moves: list[dict] = []

    def __call__(self, version, run_id, reason):
        move = {"alias": SHADOW_ALIAS, "version": version, "run_id": run_id, "reason": reason}
        self.moves.append(move)
        return move


@pytest.fixture
def alias_setter():
    return AliasRecorder()


@pytest.fixture
def wiring(gate_frames, champion_model, alias_setter):
    """The injected half: a champion, two evaluation frames, and an alias recorder."""

    def frames_builder(scenario, window_start, sets):
        return {name: gate_frames[name] for name in sets}

    return {
        "champion_loader": lambda: (champion_model, "1"),
        "frames_builder": frames_builder,
        "alias_setter": alias_setter,
    }


def retrainer_for(model, **kwargs):
    def retrainer(card, *, mode, **_):
        return challenger_run(model, mode=mode, **kwargs)

    return retrainer


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------
def test_the_fixture_champion_clears_the_real_criteria(gate_frames, champion_model):
    """A guard, so a PASS test can never quietly become a test of the fixture.

    Every PASS below is judged by the shipped `gate-v1` numbers, not by a bar
    relaxed to suit a toy model. If the synthetic champion ever stops meeting
    §3.12's absolute calibration ceiling on its own, this fails first and says
    so, instead of every PASS turning into a confusing BLOCK.
    """
    from loop.gate.metrics import metric_sets_for

    criteria = load_criteria()
    for metrics in metric_sets_for(champion_model, gate_frames).values():
        assert metrics.expected_calibration_error <= criteria.max_ece
        assert metrics.roc_auc > 0.5


# ---------------------------------------------------------------------------
# The automated PASS path
# ---------------------------------------------------------------------------
def test_an_equivalent_challenger_passes_and_reaches_shadow(
    gate_db, card_row, champion_model, wiring, alias_setter
):
    """The money shot's happy half: drift -> card -> retrain -> gate PASS -> shadow."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)

    assert run.outcome == PASS
    assert run.passed
    assert run.shadow_alias_moved
    assert alias_setter.moves == [
        {
            "alias": SHADOW_ALIAS,
            "version": "2",
            "run_id": "run-challenger",
            "reason": run.alias_audit["reason"],
        }
    ]


def test_a_pass_is_persisted_with_its_full_lineage(gate_db, card_row, champion_model, wiring):
    run = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    stored = gate_db.get(RetrainRun, run.run_id)

    assert stored.outcome == PASS
    assert stored.card_id == card_row.card_id
    assert stored.mode == MODE_LIVE
    assert stored.mlflow_run == "run-challenger"
    assert stored.challenger_version == "2"
    assert stored.champion_version == "1"
    assert stored.data_version == "dvc-train-hash"
    assert stored.shadow_alias_moved is True
    assert stored.failed_criteria_count == 0


def test_the_champion_alias_is_never_moved(gate_db, card_row, champion_model, wiring, alias_setter):
    """§3.13: autonomy stops at shadow. No code path here can promote."""
    run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    assert {move["alias"] for move in alias_setter.moves} == {SHADOW_ALIAS}


# ---------------------------------------------------------------------------
# The BLOCK path -- the roadmap's "money shot"
# ---------------------------------------------------------------------------
def test_the_deliberately_bad_challenger_is_blocked(
    gate_db, card_row, bad_challenger_model, wiring
):
    run = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)

    assert run.outcome == BLOCK
    assert not run.passed
    assert run.gate_result.reasons


def test_a_blocked_challenger_reports_why(gate_db, card_row, bad_challenger_model, wiring):
    """ "bad challenger BLOCKED with reasons" -- named criteria, not a bare refusal."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)
    failed = {check.criterion for check in run.gate_result.failed_checks()}

    assert "auroc_non_inferiority" in failed
    assert len(run.gate_result.reasons) == len(run.gate_result.failed_checks())


def test_a_block_moves_no_alias(gate_db, card_row, bad_challenger_model, wiring, alias_setter):
    """§3.12: "nothing changes in serving"."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)

    assert alias_setter.moves == []
    assert run.shadow_alias_moved is False
    assert gate_db.get(RetrainRun, run.run_id).shadow_alias_moved is False


def test_a_block_leaves_the_decision_card_immutable(
    gate_db, card_row, bad_challenger_model, wiring
):
    """§4.6: the transition appends a history row; the card itself is untouched."""
    before = (card_row.status, dict(card_row.card_json))
    run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)
    gate_db.expire_all()

    after = gate_db.get(DecisionCardRow, card_row.card_id)
    assert (after.status, after.card_json) == before
    assert runs_for_card(gate_db, card_row.card_id)[0].outcome == BLOCK


def test_a_degenerate_challenger_is_blocked(gate_db, card_row, wiring):
    """A model with no discrimination is refused as firmly as a backwards one."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(ConstantChallenger()), **wiring)
    assert run.outcome == BLOCK


def test_the_block_is_persisted_with_its_reasons(gate_db, card_row, bad_challenger_model, wiring):
    run = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)
    stored = gate_db.get(RetrainRun, run.run_id)

    assert stored.outcome == BLOCK
    assert stored.failed_criteria_count == len(run.gate_result.failed_checks())
    assert stored.gate_result["reasons"] == list(run.gate_result.reasons)


# ---------------------------------------------------------------------------
# Which cards authorise a retrain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", ["NO_OP", "ALERT_ONLY"])
def test_a_card_that_did_not_call_for_a_retrain_is_refused(action):
    with pytest.raises(GateRunnerError, match="authorise a retrain"):
        check_authorised({"card_id": "dc-x", "action": action, "disposition": "NONE"}, None)


def test_an_auto_proceed_card_needs_no_authorisation():
    card = {"card_id": "dc-x", "action": "FULL_RETRAIN", "disposition": "AUTO_PROCEED_SHADOW"}
    assert check_authorised(card, None) is None


def test_an_escalated_card_refuses_to_retrain_unattended(gate_db, champion_model, wiring):
    """WORKFLOW.md §4: "nothing retrains until an ops user acts"."""
    escalated = seed_card(
        gate_db, card_id="dc-2026-01-02-ccccdddd", disposition="ESCALATE_HUMAN", confidence=0.46
    )
    with pytest.raises(GateRunnerError, match="escalated to a human"):
        run_card(gate_db, escalated, retrainer=retrainer_for(champion_model), **wiring)


def test_an_escalated_card_runs_once_a_person_is_recorded(gate_db, champion_model, wiring):
    escalated = seed_card(
        gate_db, card_id="dc-2026-01-02-ccccdddd", disposition="ESCALATE_HUMAN", confidence=0.46
    )
    run = run_card(
        gate_db,
        escalated,
        authorized_by="ops-alice",
        retrainer=retrainer_for(champion_model),
        **wiring,
    )

    assert run.outcome == PASS
    assert gate_db.get(RetrainRun, run.run_id).authorized_by == "ops-alice"


def test_an_unknown_disposition_authorises_nothing():
    card = {"card_id": "dc-x", "action": "FULL_RETRAIN", "disposition": "NONE"}
    with pytest.raises(GateRunnerError, match="authorises nothing"):
        check_authorised(card, None)


def test_an_unknown_mode_is_refused(gate_db, card_row):
    with pytest.raises(GateRunnerError, match="unknown retrain mode"):
        run_card(gate_db, card_row, mode="teleport")


# ---------------------------------------------------------------------------
# Replay (§3.11's demo acceleration)
# ---------------------------------------------------------------------------
def test_a_replayed_challenger_is_labeled_as_replay(gate_db, card_row, champion_model, wiring):
    run = run_card(
        gate_db,
        card_row,
        mode=MODE_REPLAY,
        retrainer=retrainer_for(champion_model),
        **wiring,
    )

    assert run.mode == MODE_REPLAY
    assert gate_db.get(RetrainRun, run.run_id).mode == MODE_REPLAY
    assert "replay" in summarise(run)


def test_replay_is_gated_by_exactly_the_same_bar(gate_db, card_row, bad_challenger_model, wiring):
    """A labeled shortcut to producing a challenger, never to approving one."""
    run = run_card(
        gate_db,
        card_row,
        mode=MODE_REPLAY,
        retrainer=retrainer_for(bad_challenger_model),
        **wiring,
    )
    assert run.outcome == BLOCK


# ---------------------------------------------------------------------------
# Idempotency and restart
# ---------------------------------------------------------------------------
def test_re_running_a_gated_card_does_not_retrain_again(gate_db, card_row, champion_model, wiring):
    """The restart check: the expensive half never runs twice for the same work."""
    calls = []

    def counting_retrainer(card, *, mode, **_):
        calls.append(card["card_id"])
        return challenger_run(champion_model, mode=mode)

    first = run_card(gate_db, card_row, retrainer=counting_retrainer, **wiring)
    second = run_card(gate_db, card_row, retrainer=counting_retrainer, **wiring)

    assert calls == [card_row.card_id]
    assert second.resumed and not second.created
    assert second.run_id == first.run_id
    assert gate_db.query(RetrainRun).count() == 1


def test_a_resumed_run_reports_the_stored_verdict(gate_db, card_row, bad_challenger_model, wiring):
    first = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)
    resumed = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)

    assert resumed.outcome == first.outcome == BLOCK
    assert resumed.gate_result.reasons == first.gate_result.reasons
    assert resumed.gate_result.criteria_version == first.gate_result.criteria_version


def test_the_restart_check_can_be_turned_off(gate_db, card_row, champion_model, wiring):
    run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    again = run_card(
        gate_db, card_row, reuse_existing=False, retrainer=retrainer_for(champion_model), **wiring
    )
    assert not again.resumed
    assert gate_db.query(RetrainRun).count() == 1


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def test_a_dry_run_writes_nothing_and_moves_nothing(
    gate_db, card_row, champion_model, wiring, alias_setter
):
    run = run_card(
        gate_db,
        card_row,
        persist=False,
        move_shadow_alias=False,
        retrainer=retrainer_for(champion_model),
        **wiring,
    )

    assert run.outcome == PASS
    assert run.row is None and not run.created
    assert alias_setter.moves == []
    assert gate_db.query(RetrainRun).count() == 0


def test_a_pass_without_a_registered_version_moves_no_alias(
    gate_db, card_row, champion_model, wiring, alias_setter
):
    """An effect that did not happen must not be recorded as one that did."""
    run = run_card(
        gate_db,
        card_row,
        retrainer=retrainer_for(champion_model, version=None),
        **wiring,
    )
    assert run.outcome == PASS
    assert run.shadow_alias_moved is False
    assert alias_setter.moves == []


# ---------------------------------------------------------------------------
# The criteria a run is judged by
# ---------------------------------------------------------------------------
def test_a_run_is_judged_by_the_criteria_its_card_pinned(gate_db, card_row, champion_model, wiring):
    """§3.9 pins the bar before training starts; the run records that it used it."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    assert run.gate_result.criteria_version == f"card:{card_row.card_id}"
    assert gate_db.get(RetrainRun, run.run_id).criteria_version == f"card:{card_row.card_id}"


def test_explicit_criteria_override_the_card(gate_db, card_row, champion_model, wiring):
    criteria = load_criteria()
    run = run_card(
        gate_db, card_row, criteria=criteria, retrainer=retrainer_for(champion_model), **wiring
    )
    assert run.gate_result.criteria_version == criteria.version


def test_a_card_with_an_impossible_bar_blocks_an_identical_challenger(
    gate_db, champion_model, wiring
):
    """A stricter card is honoured, which is what pinning the bar is for."""
    strict = seed_card(
        gate_db,
        card_id="dc-2026-01-02-ccccdddd",
        acceptance_criteria={
            "auroc_non_inferiority_margin": 0.0,
            "recall_top_decile_min_ratio": 2.0,
            "max_brier_increase": 0.0,
            "max_ece": 0.000001,
            "max_subgroup_auroc_drop": 0.0,
        },
    )
    run = run_card(gate_db, strict, retrainer=retrainer_for(champion_model), **wiring)
    assert run.outcome == BLOCK


# ---------------------------------------------------------------------------
# The backlog
# ---------------------------------------------------------------------------
def test_the_backlog_gates_every_pending_card(gate_db, card_row, champion_model, wiring):
    seed_card(gate_db, card_id="dc-2026-01-02-ccccdddd")
    runs = run_pending(gate_db, retrainer=retrainer_for(champion_model), **wiring)

    assert len(runs) == 2
    assert all(run.outcome == PASS for run in runs)
    assert gate_db.query(RetrainRun).count() == 2


def test_the_backlog_empties_once_gated(gate_db, card_row, champion_model, wiring):
    run_pending(gate_db, retrainer=retrainer_for(champion_model), **wiring)
    assert run_pending(gate_db, retrainer=retrainer_for(champion_model), **wiring) == []


def test_the_backlog_can_be_limited(gate_db, card_row, champion_model, wiring):
    seed_card(gate_db, card_id="dc-2026-01-02-ccccdddd")
    runs = run_pending(gate_db, limit=1, retrainer=retrainer_for(champion_model), **wiring)
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_the_summary_line_names_the_card_the_outcome_and_the_bar(
    gate_db, card_row, champion_model, wiring
):
    line = summarise(run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring))
    assert card_row.card_id in line
    assert PASS in line
    assert "shadow -> challenger" in line


def test_the_summary_line_counts_the_block_reasons(gate_db, card_row, bad_challenger_model, wiring):
    run = run_card(gate_db, card_row, retrainer=retrainer_for(bad_challenger_model), **wiring)
    assert f"reasons: {len(run.gate_result.reasons)}" in summarise(run)


def test_a_gate_result_carries_no_patient_data(gate_db, card_row, champion_model, wiring):
    """§6, again, at the point where the row is actually written."""
    run = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    stored = gate_db.get(RetrainRun, run.run_id).gate_result

    assert set(stored) == {
        "outcome",
        "criteria_version",
        "evaluation_sets",
        "criteria",
        "checks",
        "reasons",
        "champion_metrics",
        "challenger_metrics",
    }


def test_a_resumed_run_still_names_the_challenger_it_judged(
    gate_db, card_row, champion_model, wiring
):
    """An operator re-running a gated card must still see which version was judged."""
    first = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)
    resumed = run_card(gate_db, card_row, retrainer=retrainer_for(champion_model), **wiring)

    assert resumed.challenger_version == first.challenger_version == "2"
    assert "v2" in summarise(resumed)


def test_an_unregistered_challenger_is_named_as_such(gate_db, card_row, champion_model, wiring):
    run = run_card(
        gate_db, card_row, retrainer=retrainer_for(champion_model, version=None), **wiring
    )
    assert "unregistered" in summarise(run)
