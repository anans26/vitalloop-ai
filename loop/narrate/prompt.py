"""Everything the LLM is allowed to see, and the one error the Ollama client raises.

ARCHITECTURE.md §3.10: "Input: the Decision Card JSON (aggregate statistics and
metadata only -- structurally incapable of seeing PHI)." `card_for_prompt` is
where that is enforced: the prompt is built from a card dict and nothing else,
the narrative fields are removed so the model cannot copy a previous narrative,
and there is no parameter through which a patient row could arrive.

Kept apart from the Ollama client so the PHI argument is made in one small
module: this is everything the LLM can receive, and it is built from the card.
"""

import json

from loop.narrate.grounding import NARRATIVE_FIELDS

SYSTEM_PROMPT = """You explain decisions made by a hospital model-monitoring system to \
clinicians and auditors. You are given one Decision Card as JSON. The decision has \
already been made by a fixed rule table; you do not make or change it.

Write one or two short plain-English paragraphs that say what drifted, how strong \
the evidence is, which action the policy chose, and what happens next.

Rules:
- Use only numbers that appear in the JSON, written with digits, copied exactly or \
rounded. Never compute, estimate or invent a number.
- Do not add recommendations, and do not suggest a different action from the card's.
- Mention that the data is aggregate statistics only when relevant; never speculate \
about individual patients.
- No headings, no bullet points, no preamble."""


class LLMUnavailable(RuntimeError):
    """The LLM could not produce a narrative. The caller falls back; never fatal."""


def card_for_prompt(card: dict) -> dict:
    """The card as the LLM sees it: every field except the narrative slots."""
    return {key: value for key, value in card.items() if key not in NARRATIVE_FIELDS}


def build_prompt(card: dict) -> str:
    body = json.dumps(card_for_prompt(card), indent=2, sort_keys=True, default=str)
    return f"Decision Card:\n{body}\n\nExplain this decision."
