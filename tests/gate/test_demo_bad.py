"""WORKFLOW.md §5 step 6, clickable: the deliberately bad challenger through the real gate.

Week 8 proved BLOCK with a test fixture and handed the clickable version to
Week 10. These tests pin what the `demo-bad` mode must and must not be: it
goes through the *same* gate with the card's own criteria, it is labelled
`demo-bad` on its row, it never trains, never registers, never moves an alias,
and clicking it twice returns the verdict already on record.
"""

import pytest

from loop.gate.demo import BAD_CHALLENGER_PREFIX, InvertedModel, bad_challenger
from loop.gate.gate import (
    BLOCK,
    CRITERION_AUROC,
    CRITERION_BRIER,
    CRITERION_ECE,
    CRITERION_RECALL,
    CRITERION_SUBGROUP,
)
from loop.gate.runner import GateRunnerError, run_card, summarise
from ml.retrain import GATED_MODES, MODE_DEMO_BAD, MODES, RetrainError, retrain_from_card

from .conftest import seed_card


@pytest.fixture
def wiring(gate_frames, champion_model):
    moves = []

    def frames_builder(scenario, window_start, sets):
        return {name: gate_frames[name] for name in sets}

    def never_retrain(card, *, mode, **_):
        raise AssertionError("demo-bad must never call the retrainer")

    return {
        "champion_loader": lambda: (champion_model, "1"),
        "frames_builder": frames_builder,
        "alias_setter": lambda *args: moves.append(args) or {},
        "retrainer": never_retrain,
        "_moves": moves,
    }


def _run(session, card, wiring, **kwargs):
    options = {k: v for k, v in wiring.items() if not k.startswith("_")}
    return run_card(session, card, mode=MODE_DEMO_BAD, **options, **kwargs)


def test_the_bad_challenger_is_blocked_by_the_real_gate(gate_db, wiring):
    run = _run(gate_db, seed_card(gate_db), wiring)

    assert run.outcome == BLOCK
    assert run.gate_result.reasons
    # Every criterion the gate knows is tripped, not a hand-picked one.
    failed = {check.criterion for check in run.gate_result.failed_checks()}
    assert {
        CRITERION_AUROC,
        CRITERION_RECALL,
        CRITERION_BRIER,
        CRITERION_ECE,
        CRITERION_SUBGROUP,
    } <= failed


def test_it_is_labelled_and_never_touches_the_registry(gate_db, wiring):
    run = _run(gate_db, seed_card(gate_db), wiring)

    assert run.row.mode == MODE_DEMO_BAD
    assert run.challenger_version == f"{BAD_CHALLENGER_PREFIX}1"
    assert run.row.mlflow_run is None
    assert run.shadow_alias_moved is False
    assert wiring["_moves"] == []


def test_clicking_twice_returns_the_verdict_on_record(gate_db, wiring):
    card = seed_card(gate_db)
    first = _run(gate_db, card, wiring)
    second = _run(gate_db, card, wiring)
    assert second.resumed
    assert second.run_id == first.run_id


def test_an_escalated_card_still_needs_its_authorisation(gate_db, wiring):
    card = seed_card(gate_db, disposition="ESCALATE_HUMAN", confidence=0.4)
    with pytest.raises(GateRunnerError, match="escalated"):
        _run(gate_db, card, wiring)


def test_the_retrain_pipeline_itself_refuses_the_demo_mode():
    """Nothing is trained or registered in this mode: `retrain_from_card` never sees it."""
    assert MODE_DEMO_BAD not in MODES
    assert MODE_DEMO_BAD in GATED_MODES
    with pytest.raises(RetrainError, match="unknown retrain mode"):
        retrain_from_card({"card_id": "dc-x"}, mode=MODE_DEMO_BAD)


def test_the_inverted_model_reverses_the_ranking(champion_model, gate_frames):
    import numpy as np

    frame = next(iter(gate_frames.values())).head(20)
    good = champion_model.predict_proba(frame)[:, 1]
    bad = InvertedModel(champion_model).predict_proba(frame)
    assert np.allclose(bad[:, 1], 1 - good)
    assert np.allclose(bad.sum(axis=1), 1)


def test_the_constructed_run_records_what_it_is(champion_model):
    run = bad_challenger(champion_model, "3", data_version="dvc-x")
    assert run.mode == MODE_DEMO_BAD
    assert run.registered_version == "inverted-v3"
    assert run.lineage["trained"] == "false"
    assert run.lineage["registered"] == "false"
    assert run.data_version == "dvc-x"


def test_the_summary_labels_the_mode_without_calling_it_replay(gate_db, wiring):
    line = summarise(_run(gate_db, seed_card(gate_db), wiring))
    assert MODE_DEMO_BAD in line
    assert "(replay)" not in line
