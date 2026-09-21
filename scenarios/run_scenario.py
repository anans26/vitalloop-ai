"""CLI: replay one seeded drift scenario through the monitor.

    python -m scenarios.run_scenario S1 --windows 3
    python -m scenarios.run_scenario S5 --windows 3 --no-persist

This is the Week 6 deliverable a reviewer runs: an injected scenario produces a
`drift_events` row and an Evidently report per window, and the S5 control stays
quiet. Without `--no-persist` it writes to the database named by the usual
`VITALLOOP_DATABASE_URL` / `POSTGRES_*` environment, creating the schema first
via the same `init_db` the API uses.
"""

import argparse
import sys

from loop.monitor.config import DRIFT_REPORTS_DIR, LIVE_SCENARIO, WINDOW_ROWS
from loop.monitor.reference import load_reference
from loop.monitor.runner import run_scenario, summarise
from loop.monitor.scoring import load_scoring_model
from scenarios.injection import SCENARIO_DESCRIPTIONS, SCENARIO_SEED, SCENARIOS


def build_parser() -> argparse.ArgumentParser:
    epilog = "\n".join(f"  {key}  {text}" for key, text in SCENARIO_DESCRIPTIONS.items())
    parser = argparse.ArgumentParser(
        description="Replay a seeded drift scenario through the Evidently monitor.",
        epilog=f"scenarios:\n{epilog}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenario",
        choices=[*SCENARIOS, LIVE_SCENARIO],
        help="which scenario to inject; 'live' measures the untransformed stream",
    )
    parser.add_argument("--windows", type=int, default=None, help="how many windows to measure")
    parser.add_argument("--window-rows", type=int, default=WINDOW_ROWS)
    parser.add_argument("--seed", type=int, default=SCENARIO_SEED)
    parser.add_argument(
        "--model-source",
        choices=("local", "mlflow"),
        default=None,
        help="defaults to $VITALLOOP_MODEL_SOURCE, then 'local'",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="measure and report without writing drift_events rows",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip the Evidently HTML artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model = load_scoring_model(args.model_source)
    reference = load_reference(model)
    print(
        f"reference: {reference.rows} rows from {reference.source_path} "
        f"| model {model.version} ({model.source})"
    )

    session = None
    if not args.no_persist:
        from loop.monitor.database import open_session

        session = open_session()

    try:
        outcome = run_scenario(
            args.scenario,
            reference=reference,
            model=model,
            session=session,
            windows=args.windows,
            window_rows=args.window_rows,
            reports_dir=None if args.no_report else DRIFT_REPORTS_DIR,
            seed=args.seed,
        )
    finally:
        if session is not None:
            session.close()

    print(summarise(outcome))
    if outcome.persisted_event_ids:
        print(f"persisted {len(outcome.persisted_event_ids)} drift_events rows:")
        for event_id in outcome.persisted_event_ids:
            print(f"  {event_id}")
    print(f"scenario {args.scenario}: {'quiet' if outcome.quiet else 'drift detected'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
