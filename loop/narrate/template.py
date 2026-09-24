"""The Jinja2 narrative: deterministic, offline, and always available.

ARCHITECTURE.md §3.10: "If Ollama is down, the Jinja2 template renders the
narrative deterministically. The demo never depends on the LLM." The roadmap's
Week 9 risk line makes it the default path in CI, and PROJECT_DESIGN.md §6.9
says the template "proves it every CI run": the system is complete without any
LLM.

The template is versioned like the policy and the gate criteria --
`templates/decision_card-v1.j2` -- and that version is what a card's
`narrative_source` records, so a narrative can always be traced to the exact
wording that produced it.

`StrictUndefined` is deliberate: a template that referenced a field the card
does not have would otherwise render an empty string and read as a sentence
with a hole in it. Here it raises.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_VERSION = "decision_card-v1"
TEMPLATE_SOURCE = f"template/{TEMPLATE_VERSION}"

_HEADLINES = {
    ("NO_OP", "NONE"): "no action",
    ("ALERT_ONLY", "NONE"): "alert only, no retrain",
    ("ALERT_ONLY", "ESCALATE_HUMAN"): "alert only, escalated to an ops user",
    ("INCREMENTAL_RETRAIN", "AUTO_PROCEED_SHADOW"): "incremental retrain, proceeding to shadow",
    ("INCREMENTAL_RETRAIN", "ESCALATE_HUMAN"): "incremental retrain, awaiting an ops user",
    ("FULL_RETRAIN", "AUTO_PROCEED_SHADOW"): "full retrain, proceeding to shadow",
    ("FULL_RETRAIN", "ESCALATE_HUMAN"): "full retrain, awaiting an ops user",
}


def format_number(value) -> str:
    """At most four decimals, trailing zeros dropped: `0.417667` -> `0.4177`.

    Only ever rounds, which is what keeps the template's output inside the
    grounding check's rule ("the card value rounded to the precision written").
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def headline(card: dict) -> str:
    action, disposition = card.get("action"), card.get("disposition")
    return _HEADLINES.get((action, disposition), f"{action} / {disposition}")


def _environment(template_dir: Path = TEMPLATE_DIR) -> Environment:
    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,
    )
    environment.filters["num"] = format_number
    return environment


def _tidy(rendered: str) -> str:
    """Collapses each paragraph onto one line; paragraphs stay separated."""
    paragraphs = [" ".join(block.split()) for block in rendered.split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def render_template(card: dict, *, version: str = TEMPLATE_VERSION) -> str:
    """The narrative for `card`, as plain text. Pure: same card, same words."""
    trigger = card["trigger"]
    context = {
        "card": card,
        "trigger": trigger,
        "breaches": list(trigger.get("breaching_features") or ()),
        "breakdown": card["confidence_breakdown"],
        "thresholds": card.get("policy_thresholds") or {},
        "criteria": card.get("acceptance_criteria") or {},
        "headline": headline(card),
        "window_start": str(trigger["window_start"])[:10],
        "window_end": str(trigger["window_end"])[:10],
    }
    template = _environment().get_template(f"{version}.j2")
    return _tidy(template.render(**context))
