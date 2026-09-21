"""`scripts/replay_retrain.py` -- the demo's fast path, named in ARCHITECTURE.md §10.

IMPLEMENTATION_ROADMAP.md Week 8 lists the risk and the mitigation together:
"retrain wall-clock kills demos -> replay mode is a first-class, labeled
feature". This script is that feature's front door. It re-registers a
pre-trained challenger instead of fitting one (§3.11), gates it exactly as a
live retrain is gated, and records the run with `mode='replay'` so nothing
downstream can mistake it for work that actually happened.

    python -m scripts.replay_retrain --card dc-2026-01-03-1a2b3c4d
    python -m scripts.replay_retrain --card dc-... --version 3
    python -m scripts.replay_retrain --card dc-... --authorized-by ops-alice

The gate is identical in both modes. Replay accelerates *producing* the
challenger; it never relaxes the bar the challenger has to clear, which is the
only reason a labeled shortcut is defensible in a promotion path at all.
"""

import argparse
import sys

from db.models import DecisionCard as DecisionCardRow
from loop.gate.criteria import load_criteria
from loop.gate.runner import run_card, summarise
from ml.retrain import MODE_REPLAY, retrain_from_card


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a cached challenger against a Decision Card and gate it."
    )
    parser.add_argument("--card", required=True, help="the decision card authorising the retrain")
    parser.add_argument(
        "--version",
        default=None,
        help="registered model version to reuse; default follows the `challenger` alias",
    )
    parser.add_argument(
        "--authorized-by",
        default=None,
        help="who authorised a retrain the policy escalated; recorded on the run",
    )
    parser.add_argument("--criteria", default=None, help="gate criteria version to force")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="gate and print without writing retrain_runs or moving any alias",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # See `loop/gate/runner.py`: MLflow's emoji run URL kills a cp1252 console
    # from inside `end_run`, after the work has already succeeded.
    from ml.tracking import make_console_encoding_safe

    make_console_encoding_safe()

    from loop.monitor.database import open_session

    def replay(card: dict, *, mode: str, **kwargs):
        return retrain_from_card(card, mode=mode, replay_version=args.version, **kwargs)

    session = open_session()
    try:
        card_row = session.get(DecisionCardRow, args.card)
        if card_row is None:
            print(f"no decision card {args.card!r}", file=sys.stderr)
            return 2
        run = run_card(
            session,
            card_row,
            mode=MODE_REPLAY,
            authorized_by=args.authorized_by,
            criteria=load_criteria(args.criteria) if args.criteria else None,
            retrainer=replay,
            persist=not args.dry_run,
            move_shadow_alias=not args.dry_run,
        )
    finally:
        session.close()

    print(f"REPLAY{' (dry run)' if args.dry_run else ''}: {summarise(run)}")
    for reason in run.gate_result.reasons:
        print(f"  BLOCK: {reason}")
    return 0 if run.passed else 1


if __name__ == "__main__":
    sys.exit(main())
