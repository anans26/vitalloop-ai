"""Evaluating windows: the library call the worker makes, and the CLI a reviewer runs.

    python -m loop.engine.evaluate                 # every window with no card yet
    python -m loop.engine.evaluate --scenario S5   # one stream
    python -m loop.engine.evaluate --dry-run       # decide, print, write nothing

WORKFLOW.md step 13 puts the Decision Engine immediately after drift detection,
and ARCHITECTURE.md §5 puts both inside the one `monitor` service ("drift job +
decision engine"). So there is no second scheduler here: `evaluate_event` is
what `loop/monitor/worker.py` calls on each tick, and `evaluate_pending` is the
same thing over a backlog for anyone replaying Week 6's benchmark.
"""

import argparse
import sys
from datetime import datetime

from sqlalchemy.orm import Session

from db.models import DriftEvent
from loop.engine.card import DecisionCard
from loop.engine.engine import decide
from loop.engine.history import evidence_for
from loop.engine.persistence import record_decision
from loop.engine.policy import DEFAULT_POLICY_VERSION, Policy, load_policy


def candidate_data_version() -> str | None:
    """The DVC hash a retrain triggered by this decision would run on.

    ARCHITECTURE.md §3.3: "Every Decision Card references the exact dataset hash
    a retrain will run on". Read from `dvc.lock` -- the same helper Week 4 uses
    for MLflow lineage -- so the card pins bytes, not a branch name. Resolved
    here rather than inside `decide`, which has no filesystem access by design.
    """
    try:
        from ml.tracking import dvc_lineage

        return dvc_lineage().get("dvc_train_md5")
    except Exception:
        return None


def evaluate_event(
    session: Session,
    event: DriftEvent,
    policy: Policy,
    *,
    persist: bool = True,
    data_version: str | None = None,
    now: datetime | None = None,
) -> tuple[DecisionCard, bool]:
    """Decides one window. Returns `(card, written)`.

    `written` is False when the decision was not persisted -- either because
    `persist=False`, or because this policy version had already decided this
    window and the existing card stands.
    """
    evidence = evidence_for(session, event, policy, candidate_data_version=data_version)
    card = decide(evidence, policy, now=now)
    if not persist:
        return card, False
    _, created = record_decision(session, card)
    return card, created


def evaluate_pending(
    session: Session,
    policy: Policy,
    *,
    scenario: str | None = None,
    limit: int | None = None,
    persist: bool = True,
    data_version: str | None = None,
) -> list[tuple[DecisionCard, bool]]:
    """Decides every window this policy version has not decided yet.

    Oldest first, one stream at a time, because rule 3 reads the run of windows
    before the one being evaluated.
    """
    from loop.engine.history import events_without_cards

    pending = events_without_cards(session, policy.version, scenario=scenario, limit=limit)
    return [
        evaluate_event(session, event, policy, persist=persist, data_version=data_version)
        for event in pending
    ]


def summarise(card: DecisionCard) -> str:
    """One line per decision, for the CLI and the worker log."""
    features = ", ".join(breach.feature for breach in card.trigger.breaching_features) or "none"
    return (
        f"[{card.trigger.scenario}] {card.trigger.window_start.date()} "
        f"{card.card_id}: rule {card.rule_id} -> {card.action} / {card.disposition} "
        f"(confidence {card.confidence:.4f}, breaching: {features})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate drift windows against a versioned policy and emit Decision Cards."
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY_VERSION, help="policy version to apply")
    parser.add_argument("--scenario", default=None, help="restrict to one stream, e.g. S1")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N windows")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and print without writing decision_cards rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = load_policy(args.policy)

    from loop.monitor.database import open_session

    session = open_session()
    try:
        results = evaluate_pending(
            session,
            policy,
            scenario=args.scenario,
            limit=args.limit,
            persist=not args.dry_run,
            data_version=candidate_data_version(),
        )
    finally:
        session.close()

    print(f"policy {policy.version}: {len(results)} window(s) evaluated")
    for card, written in results:
        print(f"  {summarise(card)}{'' if written else '  [already on record]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
