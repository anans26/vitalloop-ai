"""Narrate stored Decision Cards, and measure how often the LLM stays grounded.

    python -m loop.narrate --card dc-2026-01-01-894fe793          # one card
    python -m loop.narrate --card dc-... --backend ollama         # ask the LLM
    python -m loop.narrate --all --backend ollama                 # faithfulness rate

Read-only. A stored card is immutable (§4.6), so this never writes a narrative
back: it shows the one the card was emitted with, or renders one for a card
emitted before Week 9. `--all` is RESEARCH_NOVELTY.md C2's evaluation: the
fraction of generated narratives that pass the grounding check.
"""

import argparse
import sys

from sqlalchemy import select

from db.models import DecisionCard as DecisionCardRow
from loop.narrate.config import BACKENDS, narration_settings
from loop.narrate.narrator import OUTCOME_ACCEPTED, OUTCOME_UNGROUNDED, narrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrate stored Decision Cards (read-only).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--card", default=None, help="one card id")
    target.add_argument("--all", action="store_true", help="every stored card")
    parser.add_argument("--scenario", default=None, help="with --all: restrict to one stream")
    parser.add_argument(
        "--backend",
        default=None,
        choices=list(BACKENDS),
        help="override VITALLOOP_NARRATION_BACKEND for this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = narration_settings()
    if args.backend:
        settings = settings.model_copy(update={"narration_backend": args.backend})

    from loop.monitor.database import open_session

    session = open_session()
    try:
        if args.card:
            row = session.get(DecisionCardRow, args.card)
            if row is None:
                print(f"no decision card {args.card!r}", file=sys.stderr)
                return 2
            rows = [row]
        else:
            statement = select(DecisionCardRow).order_by(
                DecisionCardRow.scenario, DecisionCardRow.window_start
            )
            if args.scenario:
                statement = statement.where(DecisionCardRow.scenario == args.scenario)
            rows = list(session.scalars(statement).all())
    finally:
        session.close()

    accepted = rejected = unavailable = 0
    for row in rows:
        card = dict(row.card_json)
        if args.card and card.get("narrative") and not args.backend:
            print(f"{row.card_id}  [stored, {card.get('narrative_source')}]\n")
            print(card["narrative"])
            continue

        narration = narrate(card, settings=settings)
        for attempt in narration.attempts:
            if attempt.outcome == OUTCOME_ACCEPTED:
                accepted += 1
            elif attempt.outcome == OUTCOME_UNGROUNDED:
                rejected += 1
            else:
                unavailable += 1

        if args.card:
            print(f"{row.card_id}  [{narration.source}]\n")
            print(narration.text)
            for attempt in narration.attempts:
                if attempt.outcome != OUTCOME_ACCEPTED:
                    print(f"\n  {attempt.source} {attempt.outcome}: {attempt.detail}")
        else:
            status = narration.attempts[-1].outcome if narration.attempts else "template"
            print(f"  {row.card_id}: {status}")

    if args.all:
        generated = accepted + rejected
        rate = f"{accepted / generated:.4f}" if generated else "n/a"
        print(
            f"\n{len(rows)} card(s): {accepted} grounded, {rejected} rejected, "
            f"{unavailable} unavailable -- faithfulness rate {rate}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
