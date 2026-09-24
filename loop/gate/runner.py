"""One card, end to end: retrain -> gate -> persist -> alias.

WORKFLOW.md steps 14-15 in code, and the roadmap's Week 8 deliverable in one
call: "drift -> card -> retrain -> gate PASS path fully automated; bad
challenger BLOCKED with reasons."

    python -m loop.gate.runner                      # every card awaiting a retrain
    python -m loop.gate.runner --card dc-...        # one card
    python -m loop.gate.runner --mode replay        # §3.11's cached-run replay
    python -m loop.gate.runner --dry-run            # gate, print, write nothing

Like `loop/monitor/runner.py`, every impure step arrives as an argument -- the
retrainer, the champion loader, the frame builder, the alias setter -- so the
function the CLI calls is the function the tests call, with synthetic pieces
and no MLflow server, no database service and no trained model.

**What authorises a retrain.** A card whose action is not a retrain action is
refused outright. A card whose disposition is `AUTO_PROCEED_SHADOW` runs on its
own authority: that is what §3.8 rule 4 means by "auto -> shadow". A card whose
disposition is `ESCALATE_HUMAN` runs only when a caller supplies
`authorized_by`, which is recorded on the row -- WORKFLOW.md §4: "nothing
retrains until an ops user acts". Week 8 supplied the field and the refusal;
Week 9 supplies the name: an ops user's `APPROVE` decision in `approvals`
(`loop/approval/retrain.py`) is read back here as the authorisation, and a
`REJECT` decision refuses the retrain even if a caller passes a name.

**Autonomy stops at shadow.** A PASS moves the `shadow` alias and nothing else.
`champion` is never touched here, by any code path, because §3.13 reserves it
for a logged human approval. A BLOCK moves nothing at all: §3.12's "nothing
changes in serving" is enforced by there being no line of code that could.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from db.models import DecisionCard as DecisionCardRow
from db.models import RetrainRun
from loop.engine.rules import AUTO_PROCEED_SHADOW, ESCALATE_HUMAN, RETRAIN_ACTIONS
from loop.gate.criteria import GateCriteria, from_card, load_criteria
from loop.gate.gate import GateResult, evaluate_gate
from loop.gate.metrics import metric_sets_for
from loop.gate.persistence import (
    build_retrain_row,
    cards_awaiting_retrain,
    latest_run_for,
    record_retrain_run,
    retrain_run_id,
)
from ml.retrain import MODE_LIVE, MODES, ChallengerRun

SHADOW_ALIAS = "shadow"


class GateRunnerError(RuntimeError):
    """The card could not be acted on as specified. Never swallowed."""


@dataclass(frozen=True)
class GateRun:
    """Everything one card produced, whether or not it was written."""

    card_id: str
    run_id: str
    mode: str
    gate_result: GateResult
    challenger: ChallengerRun | None = None
    champion_version: str | None = None
    challenger_version: str | None = None
    shadow_alias_moved: bool = False
    alias_audit: dict | None = None
    authorized_by: str | None = None
    row: RetrainRun | None = None
    created: bool = False
    resumed: bool = False
    lineage: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return self.gate_result.outcome

    @property
    def passed(self) -> bool:
        return self.gate_result.passed


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------
def check_authorised(card: dict, authorized_by: str | None) -> str | None:
    """Raises unless this card permits a retrain right now.

    Returns the authorisation to record: `None` on the automated path, the
    caller's name on the escalated one.
    """
    action = card.get("action")
    if action not in RETRAIN_ACTIONS:
        raise GateRunnerError(
            f"card {card.get('card_id')!r} decided {action!r}; only {RETRAIN_ACTIONS} "
            "authorise a retrain, and the engine's decision is not overridden here"
        )

    disposition = card.get("disposition")
    if disposition == AUTO_PROCEED_SHADOW:
        return authorized_by
    if disposition == ESCALATE_HUMAN:
        if not authorized_by:
            raise GateRunnerError(
                f"card {card.get('card_id')!r} escalated to a human "
                f"(confidence {card.get('confidence')}); it retrains only with an "
                "explicit authorisation to record on the run"
            )
        return authorized_by
    raise GateRunnerError(
        f"card {card.get('card_id')!r} has disposition {disposition!r}, which authorises nothing"
    )


# ---------------------------------------------------------------------------
# Default wiring (the impure half, injectable for tests)
# ---------------------------------------------------------------------------
def _default_retrainer(card: dict, *, mode: str, **kwargs) -> ChallengerRun:
    from ml.retrain import retrain_from_card

    return retrain_from_card(card, mode=mode, **kwargs)


def _default_champion_loader() -> tuple[Any, str | None]:
    from ml.retrain import load_champion

    return load_champion()


def _default_frames(scenario: str, window_start, sets: tuple[str, ...]):
    from loop.gate.datasets import evaluation_frames

    return evaluation_frames(scenario, window_start, sets=sets)


def _default_alias_setter(version: str, run_id: str | None, reason: str) -> dict:
    from mlflow.tracking import MlflowClient

    from ml.registry import set_alias
    from ml.tracking import resolve_tracking_uri
    from ml.tracking_config import REGISTERED_MODEL_NAME

    tracking_uri = resolve_tracking_uri()
    return set_alias(
        MlflowClient(tracking_uri),
        model_name=REGISTERED_MODEL_NAME,
        alias=SHADOW_ALIAS,
        version=version,
        run_id=run_id or "",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def gate_challenger(
    card: dict,
    challenger: ChallengerRun,
    champion_model,
    frames,
    *,
    criteria: GateCriteria | None = None,
) -> GateResult:
    """The comparison itself: both models, both sets, one verdict.

    The criteria default to the ones the *card* pinned, because §3.9 records
    `acceptance_criteria` at decision time precisely so a challenger is judged
    by the bar that was set before it existed.
    """
    criteria = criteria or from_card(card.get("acceptance_criteria") or {}, card_id=card["card_id"])
    champion_sets = metric_sets_for(
        champion_model,
        frames,
        subgroup_columns=criteria.subgroup_columns,
        top_decile_fraction=criteria.top_decile_fraction,
    )
    challenger_sets = metric_sets_for(
        challenger.calibrated_model,
        frames,
        subgroup_columns=criteria.subgroup_columns,
        top_decile_fraction=criteria.top_decile_fraction,
    )
    return evaluate_gate(champion_sets, challenger_sets, criteria)


def run_card(
    session: Session,
    card_row: DecisionCardRow,
    *,
    mode: str = MODE_LIVE,
    authorized_by: str | None = None,
    criteria: GateCriteria | None = None,
    retrainer: Callable[..., ChallengerRun] | None = None,
    champion_loader: Callable[[], tuple[Any, str | None]] | None = None,
    frames_builder: Callable[..., dict] | None = None,
    alias_setter: Callable[[str, str | None, str], dict] | None = None,
    persist: bool = True,
    move_shadow_alias: bool = True,
    reuse_existing: bool = True,
) -> GateRun:
    """Retrains, gates, persists and (on PASS) moves `shadow`, for one card."""
    if mode not in MODES:
        raise GateRunnerError(f"unknown retrain mode {mode!r}; expected one of {MODES}")

    card = dict(card_row.card_json)
    recorded_authorisation = check_authorised(
        card, _authorisation_on_record(session, card, authorized_by)
    )

    if reuse_existing:
        existing = latest_run_for(
            session,
            card_row.card_id,
            mode=mode,
            data_version=card.get("candidate_data_version"),
        )
        if existing is not None:
            return _resumed(existing, criteria)

    retrainer = retrainer or _default_retrainer
    champion_loader = champion_loader or _default_champion_loader
    frames_builder = frames_builder or _default_frames

    challenger = retrainer(card, mode=mode)
    champion_model, champion_version = champion_loader()

    effective = criteria or from_card(
        card.get("acceptance_criteria") or {}, card_id=card["card_id"]
    )
    frames = frames_builder(card_row.scenario, card_row.window_start, effective.evaluation_sets)
    result = gate_challenger(card, challenger, champion_model, frames, criteria=effective)

    alias_audit = None
    moved = False
    if result.passed and move_shadow_alias and challenger.registered_version:
        setter = alias_setter or _default_alias_setter
        alias_audit = setter(
            challenger.registered_version,
            challenger.mlflow_run,
            f"week-8 gate PASS on decision card {card_row.card_id} "
            f"({result.criteria_version}); autonomy stops at shadow",
        )
        moved = True

    run_id = retrain_run_id(card_row.card_id, mode, challenger.registered_version)
    row = build_retrain_row(
        run_id=run_id,
        card_id=card_row.card_id,
        mode=mode,
        gate_result=result,
        scenario=card_row.scenario,
        policy_version=card_row.policy_version,
        action=card_row.action,
        mlflow_run=challenger.mlflow_run,
        data_version=challenger.data_version or card.get("candidate_data_version"),
        champion_version=champion_version,
        challenger_version=challenger.registered_version,
        shadow_alias_moved=moved,
        authorized_by=recorded_authorisation,
    )

    written = None
    created = False
    if persist:
        written, created = record_retrain_run(session, row)

    return GateRun(
        card_id=card_row.card_id,
        run_id=run_id,
        mode=mode,
        gate_result=result,
        challenger=challenger,
        champion_version=champion_version,
        challenger_version=challenger.registered_version,
        shadow_alias_moved=moved,
        alias_audit=alias_audit,
        authorized_by=recorded_authorisation,
        row=written,
        created=created,
        lineage=dict(challenger.lineage),
    )


def _authorisation_on_record(session: Session, card: dict, authorized_by: str | None) -> str | None:
    """Week 9: the approval row speaks for an escalated card, and a rejection is final.

    A person who reviewed an escalated card and rejected the retrain has
    decided; a CLI flag must not quietly overrule them. A person who approved
    it supplies the name the run records, so the sweep can act on it.
    """
    if card.get("disposition") != ESCALATE_HUMAN:
        return authorized_by

    from loop.approval.persistence import DECISION_REJECT, KIND_RETRAIN, find_decision

    decision = find_decision(session, KIND_RETRAIN, card["card_id"])
    if decision is not None and decision.decision == DECISION_REJECT:
        raise GateRunnerError(
            f"card {card['card_id']!r}: retrain rejected by {decision.approver} "
            f"({decision.reason}); a rejected card is not retrained"
        )
    if authorized_by:
        return authorized_by
    return decision.approver if decision is not None else None


def _resumed(existing: RetrainRun, criteria: GateCriteria | None) -> GateRun:
    """A verdict already on record, rebuilt from the row rather than recomputed.

    The expensive half -- training and scoring -- is skipped entirely, which is
    what makes a worker that died mid-run safe to restart.
    """
    stored = dict(existing.gate_result or {})
    return GateRun(
        card_id=existing.card_id,
        run_id=existing.run_id,
        mode=existing.mode,
        gate_result=GateResult(
            outcome=existing.outcome,
            criteria_version=existing.criteria_version
            or (criteria.version if criteria else "unknown"),
            checks=(),
            reasons=tuple(stored.get("reasons") or ()),
            evaluation_sets=tuple(stored.get("evaluation_sets") or ()),
            criteria=stored.get("criteria") or {},
            champion_metrics=stored.get("champion_metrics") or {},
            challenger_metrics=stored.get("challenger_metrics") or {},
        ),
        champion_version=existing.champion_version,
        challenger_version=existing.challenger_version,
        shadow_alias_moved=bool(existing.shadow_alias_moved),
        authorized_by=existing.authorized_by,
        row=existing,
        created=False,
        resumed=True,
    )


def run_pending(
    session: Session,
    *,
    scenario: str | None = None,
    policy_version: str | None = None,
    limit: int | None = None,
    **kwargs,
) -> list[GateRun]:
    """Every card awaiting a retrain, oldest first.

    Automated cards, plus escalated cards an ops user has approved in
    `approvals` (Week 9). An escalated card nobody has approved is never swept
    up, because the authorisation it needs is a person's decision and not a
    default.
    """
    cards = cards_awaiting_retrain(
        session,
        scenario=scenario,
        policy_version=policy_version,
        include_approved=True,
        limit=limit,
    )
    return [run_card(session, card, **kwargs) for card in cards]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def summarise(run: GateRun) -> str:
    """One line per verdict, for the CLI and the demo log."""
    challenger = run.challenger_version or "unregistered"
    label = f"{run.mode}{' (replay)' if run.mode != MODE_LIVE else ''}"
    tail = ""
    if run.gate_result.blocked and run.gate_result.reasons:
        tail = f"  reasons: {len(run.gate_result.reasons)}"
    elif run.shadow_alias_moved:
        tail = "  shadow -> challenger"
    return (
        f"{run.card_id}: {label} challenger v{challenger} -> {run.outcome} "
        f"[{run.gate_result.criteria_version}]{tail}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain the challenger a Decision Card authorises and gate it."
    )
    parser.add_argument("--card", default=None, help="one card id; default is every pending card")
    parser.add_argument("--mode", default=MODE_LIVE, choices=list(MODES), help="live or replay")
    parser.add_argument("--scenario", default=None, help="restrict the backlog to one stream")
    parser.add_argument("--limit", type=int, default=None, help="gate at most N cards")
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
    criteria = load_criteria(args.criteria) if args.criteria else None

    # A live retrain starts an MLflow run, and MLflow prints a run URL
    # containing an emoji. On a cp1252 console that raises UnicodeEncodeError
    # from inside `end_run`, leaving the run stuck in RUNNING and skipping
    # registration -- a logging detail losing a challenger that trained fine.
    # `ml/tracking.py` met this in Week 4 and the guard is shared, not re-solved.
    from ml.tracking import make_console_encoding_safe

    make_console_encoding_safe()

    from loop.monitor.database import open_session

    session = open_session()
    options = {
        "mode": args.mode,
        "authorized_by": args.authorized_by,
        "criteria": criteria,
        "persist": not args.dry_run,
        "move_shadow_alias": not args.dry_run,
    }
    try:
        if args.card:
            card_row = session.get(DecisionCardRow, args.card)
            if card_row is None:
                print(f"no decision card {args.card!r}", file=sys.stderr)
                return 2
            runs = [run_card(session, card_row, **options)]
        else:
            runs = run_pending(session, scenario=args.scenario, limit=args.limit, **options)
    finally:
        session.close()

    print(f"{len(runs)} card(s) gated" + (" (dry run)" if args.dry_run else ""))
    for run in runs:
        print(f"  {summarise(run)}{'  [already on record]' if run.resumed else ''}")
        for reason in run.gate_result.reasons:
            print(f"      BLOCK: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
