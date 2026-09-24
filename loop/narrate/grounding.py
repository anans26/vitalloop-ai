"""The grounding check: every number a narrative quotes must be in the card.

ARCHITECTURE.md §3.10: "A post-generation **grounding check** verifies every
number quoted in the narrative appears in the card; a failed check falls back
to the Jinja2 template." RESEARCH_NOVELTY.md C2 calls this what gives the
decision/narration separation "teeth": the LLM may choose the words, but it
cannot introduce a figure the policy did not produce.

A pure function of two values, like the gate: a narrative string and a card
dict in, a `GroundingResult` out. No model, no network, no clock.

**What counts as "in the card".**

* Every numeric leaf of the card (booleans excluded -- `True` is not a figure).
* The length of every list in the card. "2 features breached" restates the
  card's list of two breaching features; it is not a new number.
* Every number written inside the card's own `rationale` sentence. The engine
  writes that sentence from the evidence -- rule 6, for one, names the days
  since the last retrain -- so it is part of the card, and a narrative quoting
  it is quoting the card.
* Every calendar date written in the card's timestamps.

**What counts as "quoting a number".** A narrative number matches a card number
when it equals the card value rounded to the precision the narrative wrote
(`0.4177` quotes `0.417667`; `0.42` does too), or when it is that value as a
percentage (`42%` or `41.8%` quotes `0.417667`). Rounding is how people write
numbers; inventing them is what this refuses.

Digits that are part of an identifier -- `policy-v2`, `S1`, `dc-2026-01-01-…`,
a DVC hash -- are not quantities and are not checked as numbers. Dates are
checked separately, against the card's own dates. Numbers spelled as words
("two") are not checked; the prompt asks for digits, and the limitation is
documented rather than hidden.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

# The fields a narrative is written *into*; never evidence for itself.
NARRATIVE_FIELDS = ("narrative", "narrative_source")

# A quantity: optional sign, digits (optionally comma-grouped), optional
# decimals, optional percent sign. Not preceded or followed by a letter, digit,
# hyphen or underscore, so `policy-v2`, `S1`, `2026-01-01` and hex hashes are
# not read as quantities.
_NUMBER = re.compile(
    r"(?<![\w\-.])(?P<sign>-)?(?P<body>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?P<pct>%)?(?![\w\-]|\.\d)"
)
_DATE = re.compile(r"(?<![\w\-])\d{4}-\d{2}-\d{2}(?![\w\-])")
# In the card a date is usually the front of an ISO timestamp (`...T00:00:00Z`).
_CARD_DATE = re.compile(r"(?<![\w\-])(\d{4}-\d{2}-\d{2})(?:T|(?![\w\-]))")

# Two floats are "the same number" within this tolerance after rounding.
_EPSILON = 1e-9


@dataclass(frozen=True)
class QuotedNumber:
    """One number as written in the narrative."""

    text: str
    value: float
    decimals: int
    percent: bool


@dataclass(frozen=True)
class GroundingResult:
    """The verdict, with the evidence for it."""

    grounded: bool
    checked_numbers: tuple[str, ...] = ()
    checked_dates: tuple[str, ...] = ()
    ungrounded_numbers: tuple[str, ...] = ()
    ungrounded_dates: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default=())

    def to_record(self) -> dict:
        return {
            "grounded": self.grounded,
            "checked_numbers": list(self.checked_numbers),
            "checked_dates": list(self.checked_dates),
            "ungrounded_numbers": list(self.ungrounded_numbers),
            "ungrounded_dates": list(self.ungrounded_dates),
            "problems": list(self.problems),
        }


# ---------------------------------------------------------------------------
# Reading the narrative
# ---------------------------------------------------------------------------
def quoted_numbers(text: str) -> tuple[QuotedNumber, ...]:
    """Every quantity the narrative writes, in order."""
    found = []
    for match in _NUMBER.finditer(text):
        body = match.group("body").replace(",", "")
        frac = match.group("frac") or ""
        literal = f"{body}.{frac}" if frac else body
        value = float(literal)
        if match.group("sign"):
            value = -value
        found.append(
            QuotedNumber(
                text=match.group(0),
                value=value,
                decimals=len(frac),
                percent=bool(match.group("pct")),
            )
        )
    return tuple(found)


def quoted_dates(text: str) -> tuple[str, ...]:
    """Every ISO calendar date the narrative writes."""
    return tuple(match.group(0) for match in _DATE.finditer(text))


# ---------------------------------------------------------------------------
# Reading the card
# ---------------------------------------------------------------------------
def _walk(value) -> Iterable:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in NARRATIVE_FIELDS:
                continue
            yield from _walk(item)
    elif isinstance(value, list | tuple):
        yield len(value)
        for item in value:
            yield from _walk(item)
    else:
        yield value


def card_numbers(card: dict) -> tuple[float, ...]:
    """Every number the card states, plus the numbers inside its own rationale."""
    numbers = [
        float(value)
        for value in _walk(card)
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    rationale = card.get("rationale")
    if isinstance(rationale, str):
        numbers.extend(quoted.value for quoted in quoted_numbers(rationale) if not quoted.percent)
    return tuple(numbers)


def card_dates(card: dict) -> frozenset[str]:
    """Every calendar date the card's strings carry, as `YYYY-MM-DD`."""
    dates = set()
    for value in _walk(card):
        if isinstance(value, str):
            dates.update(match.group(1) for match in _CARD_DATE.finditer(value))
        elif isinstance(value, date):
            dates.add(value.isoformat()[:10])
    return frozenset(dates)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _matches(quoted: QuotedNumber, source: float) -> bool:
    candidates = (source * 100.0,) if quoted.percent else (source,)
    for candidate in candidates:
        if abs(round(candidate, quoted.decimals) - quoted.value) < _EPSILON:
            return True
        if abs(candidate - quoted.value) < _EPSILON:
            return True
    return False


def is_grounded_number(quoted: QuotedNumber, sources: Iterable[float]) -> bool:
    return any(_matches(quoted, source) for source in sources)


def check_grounding(narrative: str, card: dict) -> GroundingResult:
    """Every number and date in `narrative` must be one the card states."""
    if not isinstance(narrative, str) or not narrative.strip():
        return GroundingResult(grounded=False, problems=("narrative is empty",))

    sources = card_numbers(card)
    dates = card_dates(card)

    numbers = quoted_numbers(narrative)
    ungrounded = tuple(q.text for q in numbers if not is_grounded_number(q, sources))

    written_dates = quoted_dates(narrative)
    unknown_dates = tuple(d for d in written_dates if d not in dates)

    problems = []
    if ungrounded:
        problems.append(f"numbers not in the card: {', '.join(ungrounded)}")
    if unknown_dates:
        problems.append(f"dates not in the card: {', '.join(unknown_dates)}")

    return GroundingResult(
        grounded=not problems,
        checked_numbers=tuple(q.text for q in numbers),
        checked_dates=written_dates,
        ungrounded_numbers=ungrounded,
        ungrounded_dates=unknown_dates,
        problems=tuple(problems),
    )
