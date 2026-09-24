"""The narrator: Ollama if it is on and honest, the template otherwise.

No test here needs Ollama. The LLM is replaced either by a `generator` callable
or, for the HTTP client itself, by an `httpx.MockTransport` -- so the default
CI path is the template, exactly as the roadmap's Week 9 risk line requires,
and the LLM path is still exercised end to end.
"""

import json

import httpx
import pytest

from loop.narrate.config import BACKEND_OLLAMA, BACKEND_TEMPLATE, NarrationSettings
from loop.narrate.narrator import (
    OUTCOME_ACCEPTED,
    OUTCOME_UNAVAILABLE,
    OUTCOME_UNGROUNDED,
    attach,
    narrate,
    narrate_card,
)
from loop.narrate.ollama import (
    SYSTEM_PROMPT,
    OllamaUnavailable,
    build_prompt,
    card_for_prompt,
    generate,
)
from loop.narrate.template import TEMPLATE_SOURCE, render_template

TEMPLATE_SETTINGS = NarrationSettings(narration_backend=BACKEND_TEMPLATE)
OLLAMA_SETTINGS = NarrationSettings(
    narration_backend=BACKEND_OLLAMA, ollama_url="http://ollama.invalid:11434"
)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_the_template_is_the_default_backend(monkeypatch):
    monkeypatch.delenv("VITALLOOP_NARRATION_BACKEND", raising=False)
    settings = NarrationSettings()
    assert settings.narration_backend == BACKEND_TEMPLATE
    assert settings.llm_source() is None


def test_the_template_backend_never_calls_an_llm(escalated_card):
    narration = narrate(escalated_card, settings=TEMPLATE_SETTINGS)
    assert narration.source == TEMPLATE_SOURCE
    assert narration.attempts == ()
    assert narration.text == render_template(escalated_card)
    assert narration.grounding.grounded


# ---------------------------------------------------------------------------
# The LLM path, with the three outcomes §3.10 names
# ---------------------------------------------------------------------------
def test_a_grounded_llm_narrative_is_used_and_labelled(escalated_card):
    text = (
        f"The policy chose {escalated_card['action']} with confidence "
        f"{escalated_card['confidence']:.2f}, below the 0.75 line, so it waits for a person."
    )
    narration = narrate(escalated_card, settings=OLLAMA_SETTINGS, generator=lambda card: text)

    assert narration.text == text
    assert narration.source == "ollama/llama3.1:8b"
    assert narration.grounding.grounded
    assert [a.outcome for a in narration.attempts] == [OUTCOME_ACCEPTED]


def test_an_ungrounded_llm_narrative_falls_back_to_the_template(escalated_card):
    """The check with teeth: one invented figure and the LLM's text is discarded."""
    narration = narrate(
        escalated_card,
        settings=OLLAMA_SETTINGS,
        generator=lambda card: "Confidence was 0.93, so retraining is urgent.",
    )

    assert narration.source == TEMPLATE_SOURCE
    assert narration.text == render_template(escalated_card)
    assert narration.fell_back
    (attempt,) = narration.attempts
    assert attempt.outcome == OUTCOME_UNGROUNDED
    assert "0.93" in attempt.detail


def test_an_unavailable_llm_falls_back_to_the_template(escalated_card):
    def down(card):
        raise OllamaUnavailable("ConnectError: connection refused")

    narration = narrate(escalated_card, settings=OLLAMA_SETTINGS, generator=down)
    assert narration.source == TEMPLATE_SOURCE
    assert narration.attempts[0].outcome == OUTCOME_UNAVAILABLE
    assert "connection refused" in narration.attempts[0].detail


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyError("response"), None, 42])
def test_nothing_an_llm_does_can_stop_a_card_being_narrated(failure, escalated_card):
    def misbehave(card):
        if isinstance(failure, Exception):
            raise failure
        return failure

    narration = narrate(escalated_card, settings=OLLAMA_SETTINGS, generator=misbehave)
    assert narration.source == TEMPLATE_SOURCE
    assert narration.grounding.grounded


def test_the_llm_is_never_shown_a_previous_narrative(escalated_card):
    seen = {}

    def spy(card):
        seen.update(card)
        return "The inputs moved."

    card = {**escalated_card, "narrative": "old words 0.99", "narrative_source": "x"}
    narrate(card, settings=OLLAMA_SETTINGS, generator=spy)
    prompt = build_prompt(seen)
    assert "old words" not in prompt


# ---------------------------------------------------------------------------
# Narration never changes a decision
# ---------------------------------------------------------------------------
def test_attach_changes_only_the_narrative_fields():
    from loop.engine.engine import decide
    from loop.engine.policy import load_policy
    from tests.engine.conftest import FIXED_NOW, evidence_for, feature_stat, severe_psi

    policy = load_policy()
    card = decide(
        evidence_for(policy, stats=(feature_stat("a", severe_psi(policy)),)), policy, now=FIXED_NOW
    )
    narrated, narration = narrate_card(card, settings=TEMPLATE_SETTINGS)

    before, after = card.to_json_dict(), narrated.to_json_dict()
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"narrative", "narrative_source"}
    assert narrated.narrative == narration.text
    assert card.narrative is None  # the original is frozen and untouched
    assert attach(card, narration) == narrated


# ---------------------------------------------------------------------------
# The Ollama client, over a mocked transport
# ---------------------------------------------------------------------------
def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_generate_posts_one_non_streaming_deterministic_request(escalated_card):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "  The inputs moved.  "})

    text = generate(
        escalated_card, url="http://ollama:11434/", model="llama3.1:8b", client=_client(handler)
    )

    assert text == "The inputs moved."
    assert captured["url"] == "http://ollama:11434/api/generate"
    body = captured["body"]
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0, "seed": 0}
    assert body["system"] == SYSTEM_PROMPT


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="model not loaded"),
        httpx.Response(200, json={"response": ""}),
        httpx.Response(200, json={"error": "no such model"}),
        httpx.Response(200, text="not json"),
    ],
)
def test_every_ollama_failure_is_one_exception(response, escalated_card):
    with pytest.raises(OllamaUnavailable):
        generate(escalated_card, url="http://x", model="m", client=_client(lambda r: response))


def test_a_connection_failure_is_ollama_unavailable(escalated_card):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OllamaUnavailable, match="ConnectError"):
        generate(escalated_card, url="http://x", model="m", client=_client(refuse))


# ---------------------------------------------------------------------------
# PHI: the prompt is the card and nothing else
# ---------------------------------------------------------------------------
def test_the_prompt_is_built_from_the_card_alone(escalated_card):
    prompt = build_prompt(escalated_card)
    assert json.dumps(card_for_prompt(escalated_card), indent=2, sort_keys=True) in prompt
    for forbidden in ("patient_nbr", "encounter_id", "readmitted", "diag_1", "payload"):
        assert forbidden not in prompt


def test_generate_accepts_no_argument_that_could_carry_a_row():
    """Structural, per §3.10: the only data parameter is the card."""
    import inspect

    parameters = inspect.signature(generate).parameters
    assert list(parameters) == ["card", "url", "model", "timeout", "client"]


def test_ollama_is_the_only_llm_backend():
    """Project decision: free and self-hosted only -- no cloud or paid LLM backend."""
    from pathlib import Path

    from loop.narrate.config import BACKENDS

    assert BACKENDS == ("template", "ollama")
    with pytest.raises(ValueError):
        NarrationSettings(narration_backend="gemini")
    sources = " ".join(
        p.read_text(encoding="utf-8").lower() for p in Path("loop/narrate").rglob("*.py")
    )
    for cloud in ("generativelanguage.googleapis.com", "openai.com", "api.anthropic.com"):
        assert cloud not in sources
