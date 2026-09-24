"""`narrate(card) -> Narration`: the LLM if it is on and honest, the template otherwise.

ARCHITECTURE.md §3.10 in one function:

1. With the Ollama backend configured, ask the LLM for a narrative.
2. Grounding-check it: every number it quotes must be in the card.
3. If the LLM is down, or the check fails, render the Jinja2 template instead.

The result is labelled with its source -- `ollama/<model>` or
`template/<version>` -- which PROJECT_DESIGN.md §6.9 lists as the fifth
guardrail ("every narrative labeled with its source"). A narrative that was
generated and then rejected is not lost: `Narration.attempts` records what was
tried and why it was refused, which is what RESEARCH_NOVELTY.md C2's
"narrative faithfulness rate" is computed from.

**Narration never changes a decision.** The function returns text and a label;
the caller attaches them to a card whose action, disposition and confidence are
already fixed. `attach` makes that explicit by copying *only* the two narrative
fields onto the card.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from loop.engine.card import DecisionCard
from loop.narrate.config import BACKEND_OLLAMA, NarrationSettings, narration_settings
from loop.narrate.grounding import GroundingResult, check_grounding
from loop.narrate.prompt import LLMUnavailable
from loop.narrate.template import TEMPLATE_SOURCE, render_template

OUTCOME_ACCEPTED = "accepted"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class NarrationAttempt:
    """One try at an LLM narrative, and what became of it."""

    source: str
    outcome: str
    detail: str | None = None
    grounding: GroundingResult | None = None


@dataclass(frozen=True)
class Narration:
    text: str
    source: str
    grounding: GroundingResult
    attempts: tuple[NarrationAttempt, ...] = field(default=())

    @property
    def fell_back(self) -> bool:
        return bool(self.attempts) and self.source == TEMPLATE_SOURCE


Generator = Callable[[dict], str]


def _configured_generator(settings: NarrationSettings) -> Generator | None:
    """The configured LLM backend as a `card -> text` callable, or None for the template."""
    if settings.narration_backend == BACKEND_OLLAMA:
        from loop.narrate import ollama

        return lambda card: ollama.generate(
            card,
            url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
        )
    return None


def narrate(
    card: dict,
    *,
    settings: NarrationSettings | None = None,
    generator: Generator | None = None,
    source: str | None = None,
) -> Narration:
    """The narrative for one card, never raising for anything the LLM does.

    `generator` replaces the configured LLM call (tests, or another model);
    passing one implies the LLM path, labelled with `source` (default: the
    configured backend's label, else `llm/custom`).
    """
    settings = settings or narration_settings()
    attempts: list[NarrationAttempt] = []

    if generator is None:
        generator = _configured_generator(settings)
    if generator is not None:
        source = source or settings.llm_source() or "llm/custom"
        try:
            text = generator(card)
            if not isinstance(text, str):
                raise LLMUnavailable(f"generator returned {type(text).__name__}, not text")
        except Exception as error:
            # Deliberately broad: nothing an LLM does may stop a card from
            # being narrated, and so nothing it does may stop a card at all.
            detail = str(error) if isinstance(error, LLMUnavailable) else type(error).__name__
            attempts.append(NarrationAttempt(source, OUTCOME_UNAVAILABLE, detail))
        else:
            grounding = check_grounding(text, card)
            if grounding.grounded:
                return Narration(
                    text=text,
                    source=source,
                    grounding=grounding,
                    attempts=(NarrationAttempt(source, OUTCOME_ACCEPTED, None, grounding),),
                )
            attempts.append(
                NarrationAttempt(
                    source, OUTCOME_UNGROUNDED, "; ".join(grounding.problems), grounding
                )
            )

    text = render_template(card)
    return Narration(
        text=text,
        source=TEMPLATE_SOURCE,
        grounding=check_grounding(text, card),
        attempts=tuple(attempts),
    )


def attach(card: DecisionCard, narration: Narration) -> DecisionCard:
    """The same decision, with its narrative filled in. Nothing else changes."""
    return card.model_copy(
        update={"narrative": narration.text, "narrative_source": narration.source}
    )


def narrate_card(
    card: DecisionCard,
    *,
    settings: NarrationSettings | None = None,
    generator: Generator | None = None,
) -> tuple[DecisionCard, Narration]:
    """Narrates a freshly decided card and returns it with the narrative attached."""
    narration = narrate(card.to_json_dict(), settings=settings, generator=generator)
    return attach(card, narration), narration
