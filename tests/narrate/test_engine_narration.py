"""Narration attached where WORKFLOW.md step 13 puts it: after the decision, before the write.

The Week 7 card contract left `narrative` and `narrative_source` present and
null "so the contract does not move". These tests pin what Week 9 does with
them: a card emitted from now on is stored *with* its narrative, and nothing
else about it differs from what `decide` produced.
"""

from sqlalchemy import select

from db.models import DecisionCard as DecisionCardRow
from loop.engine.engine import decide
from loop.engine.evaluate import evaluate_event, evaluate_pending
from loop.engine.history import evidence_for
from loop.engine.policy import load_policy
from loop.narrate.grounding import check_grounding
from loop.narrate.template import TEMPLATE_SOURCE
from tests.engine.conftest import FIXED_NOW
from tests.engine.test_evaluate import COVARIATE_WINDOWS, make_event


def test_an_emitted_card_is_stored_with_its_narrative(drift_db, monkeypatch):
    monkeypatch.delenv("VITALLOOP_NARRATION_BACKEND", raising=False)
    policy = load_policy()
    event = make_event(drift_db, index=0, scenario="S1", breaches=COVARIATE_WINDOWS[0])

    card, written = evaluate_event(drift_db, event, policy, now=FIXED_NOW)

    assert written
    assert card.narrative_source == TEMPLATE_SOURCE
    stored = drift_db.scalars(select(DecisionCardRow)).one().card_json
    assert stored["narrative"] == card.narrative
    assert stored["narrative_source"] == TEMPLATE_SOURCE
    assert check_grounding(stored["narrative"], stored).grounded


def test_narration_changes_nothing_but_the_narrative(drift_db):
    policy = load_policy()
    event = make_event(drift_db, index=0, scenario="S1", breaches=COVARIATE_WINDOWS[0])
    undecorated = decide(evidence_for(drift_db, event, policy), policy, now=FIXED_NOW)

    card, _ = evaluate_event(drift_db, event, policy, now=FIXED_NOW, persist=False)

    before, after = undecorated.to_json_dict(), card.to_json_dict()
    assert {k for k in before if before[k] != after[k]} == {"narrative", "narrative_source"}


def test_a_narrator_receives_a_card_whose_decision_is_already_fixed(drift_db):
    policy = load_policy()
    event = make_event(drift_db, index=0, scenario="S1", breaches=COVARIATE_WINDOWS[0])
    seen = []

    def narrator(card):
        seen.append((card.action, card.disposition, card.confidence, card.narrative))
        return card.model_copy(update={"narrative": "x", "narrative_source": "test"})

    card, _ = evaluate_event(drift_db, event, policy, now=FIXED_NOW, narrator=narrator)
    assert seen == [(card.action, card.disposition, card.confidence, None)]


def test_an_already_decided_window_is_not_narrated_again(drift_db):
    """Idempotency must not cost an LLM call per backlog replay."""
    policy = load_policy()
    event = make_event(drift_db, index=0, scenario="S1", breaches=COVARIATE_WINDOWS[0])
    evaluate_event(drift_db, event, policy, now=FIXED_NOW)

    calls = []
    _, written = evaluate_event(
        drift_db, event, policy, now=FIXED_NOW, narrator=lambda c: calls.append(c) or c
    )
    assert not written
    assert calls == []


def test_a_down_llm_still_emits_every_card(drift_db, monkeypatch):
    """ARCHITECTURE.md §3.10: "The demo never depends on the LLM."."""
    monkeypatch.setenv("VITALLOOP_NARRATION_BACKEND", "ollama")
    monkeypatch.setenv("VITALLOOP_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VITALLOOP_OLLAMA_TIMEOUT_SECONDS", "0.5")
    policy = load_policy()
    for index, breaches in enumerate(COVARIATE_WINDOWS):
        make_event(drift_db, index=index, scenario="S1", breaches=breaches)

    results = evaluate_pending(drift_db, policy)

    assert len(results) == len(COVARIATE_WINDOWS)
    assert all(written for _, written in results)
    assert all(card.narrative_source == TEMPLATE_SOURCE for card, _ in results)
