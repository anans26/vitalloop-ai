"""Promotion: the one code path in the repository that moves `champion`.

ARCHITECTURE.md §3.13: "Promotion to `champion` requires a **human click in the
dashboard** (ops role), which writes an approval row (who, when, card
reference). Autonomy stops at shadow." §6 states the precondition in full:
"Alias change requires gate PASS + shadow window + logged human approval."

`decide_promotion` checks each of those before it moves anything, and one more
the documents imply but do not spell out:

1. **Gate PASS** -- the run's outcome is `PASS` and it actually moved `shadow`.
2. **The challenger is still the shadow** -- `shadow` points at the run's
   challenger. If a later PASS moved `shadow` on, this run's shadow window is
   about a model that is no longer being watched.
3. **The champion is still the one it beat** -- `champion` points at the
   version the gate compared against. The gate's verdict is "better than *that*
   model"; if the champion has changed since, the verdict no longer says
   anything about promotion and the challenger must be re-gated.
4. **The shadow window is complete** -- at least `min_shadow_requests`
   requests were dual-scored by exactly this `(champion, challenger)` pair.
5. **A logged human decision** -- the approver comes from a verified token
   with the `ops` role (enforced by the API), and gives a reason.

Conditions 2-4 apply to `APPROVE` only. A person may `REJECT` at any point
after the gate: stopping a model from reaching clinicians needs no evidence
threshold.

**What moves.** `APPROVE` moves `champion` to the challenger, then clears
`shadow` -- the promoted model no longer needs dual-scoring against itself.
`REJECT` moves `champion` nowhere and clears `shadow`, so the rejected model
stops scoring traffic. Every move goes through `ml.registry` and is audited with
the approver's name as the actor.

**Ordering, and the one compensation in the codebase.** The alias moves first
and the approval row is written second. If the row cannot be written, the
champion alias is moved *back* before the error propagates. The opposite order
would leave an approval row claiming a promotion that never happened, or --
worse, with this order and no compensation -- a champion swap with no approval
on record, which is precisely the "silent clinical model swap" RISK_ANALYSIS.md
§3 rates Critical.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Approval, RetrainRun
from db.models import DecisionCard as DecisionCardRow
from loop.approval.persistence import (
    DECISION_APPROVE,
    KIND_PROMOTION,
    AlreadyDecided,
    ApprovalConflict,
    ApprovalNotFound,
    approval_id,
    find_decision,
    record_decision,
    validate_decision,
)
from loop.approval.registry import AliasRegistry
from loop.approval.retrain import _card_summary
from loop.approval.rules import PromotionRules
from loop.gate.gate import PASS
from loop.shadow.persistence import window_stats

CHAMPION_ALIAS = "champion"
SHADOW_ALIAS = "shadow"

_HEADLINE_METRICS = ("roc_auc", "recall_at_top_decile", "brier_score", "expected_calibration_error")


@dataclass(frozen=True)
class Precondition:
    name: str
    satisfied: bool
    detail: str

    def to_record(self) -> dict:
        return {"name": self.name, "satisfied": self.satisfied, "detail": self.detail}


@dataclass(frozen=True)
class PromotionResult:
    approval: Approval
    champion_alias_moved: bool
    shadow_alias_cleared: bool
    alias_audit: tuple[dict, ...] = ()


# ---------------------------------------------------------------------------
# What is waiting, and what the person deciding is shown
# ---------------------------------------------------------------------------
def pending_promotions(session: Session) -> tuple[RetrainRun, ...]:
    """Gated challengers that reached shadow and nobody has decided yet, oldest first."""
    decided = select(Approval.subject).where(Approval.kind == KIND_PROMOTION)
    statement = (
        select(RetrainRun)
        .where(RetrainRun.outcome == PASS)
        .where(RetrainRun.shadow_alias_moved.is_(True))
        .where(RetrainRun.run_id.not_in(decided))
        .order_by(RetrainRun.created_at, RetrainRun.run_id)
    )
    return tuple(session.scalars(statement).all())


def _gate_summary(run: RetrainRun) -> dict:
    """The gate's verdict and headline numbers, champion beside challenger."""
    stored = dict(run.gate_result or {})
    champion = stored.get("champion_metrics") or {}
    challenger = stored.get("challenger_metrics") or {}

    headline = {}
    worst_subgroup_drop = {}
    for evaluation_set in stored.get("evaluation_sets") or sorted(champion):
        ours, theirs = champion.get(evaluation_set) or {}, challenger.get(evaluation_set) or {}
        headline[evaluation_set] = {
            metric: {
                "champion": ours.get(metric),
                "challenger": theirs.get(metric),
                "delta": (
                    round(theirs[metric] - ours[metric], 6)
                    if isinstance(ours.get(metric), int | float)
                    and isinstance(theirs.get(metric), int | float)
                    else None
                ),
            }
            for metric in _HEADLINE_METRICS
        }
        drops = []
        for column, groups in (ours.get("subgroup_roc_auc") or {}).items():
            for group, before in (groups or {}).items():
                after = ((theirs.get("subgroup_roc_auc") or {}).get(column) or {}).get(group)
                if isinstance(before, int | float) and isinstance(after, int | float):
                    drops.append((round(before - after, 6), f"{column}={group}"))
        if drops:
            drop, subgroup = max(drops)
            worst_subgroup_drop[evaluation_set] = {"subgroup": subgroup, "auroc_drop": drop}

    return {
        "run_id": run.run_id,
        "mode": run.mode,
        "outcome": run.outcome,
        "criteria_version": run.criteria_version,
        "failed_criteria_count": run.failed_criteria_count,
        "evaluation_sets": stored.get("evaluation_sets") or [],
        "headline": headline,
        "worst_subgroup_auroc_drop": worst_subgroup_drop,
        "authorized_by": run.authorized_by,
        "mlflow_run": run.mlflow_run,
        "data_version": run.data_version,
    }


def check_preconditions(
    session: Session, run: RetrainRun, registry: AliasRegistry, rules: PromotionRules
) -> tuple[tuple[Precondition, ...], dict]:
    """Every §6 precondition for `APPROVE`, evaluated now. Returns (checks, stats)."""
    champion_now = registry.version_of(CHAMPION_ALIAS)
    shadow_now = registry.version_of(SHADOW_ALIAS)
    challenger, beaten = run.challenger_version, run.champion_version

    stats = (
        window_stats(session, champion_now, challenger).to_record()
        if champion_now and challenger
        else None
    )
    seen = stats["requests"] if stats else 0

    checks = (
        Precondition(
            "gate_pass",
            run.outcome == PASS and bool(run.shadow_alias_moved),
            f"gate {run.outcome} under {run.criteria_version}; "
            f"shadow {'moved' if run.shadow_alias_moved else 'not moved'} by the run",
        ),
        Precondition(
            "challenger_in_shadow",
            challenger is not None and shadow_now == challenger,
            f"shadow alias -> {shadow_now}; this run's challenger is {challenger}",
        ),
        Precondition(
            "champion_unchanged_since_gate",
            beaten is not None and champion_now == beaten,
            f"champion alias -> {champion_now}; the gate compared against {beaten}",
        ),
        Precondition(
            "shadow_window_complete",
            seen >= rules.min_shadow_requests,
            f"{seen} of {rules.min_shadow_requests} requests dual-scored "
            f"(champion {champion_now}, shadow {challenger})",
        ),
    )
    return checks, stats or {}


def promotion_evidence(
    session: Session, run: RetrainRun, registry: AliasRegistry, rules: PromotionRules
) -> dict:
    """The full evidence chain RISK_ANALYSIS.md §3 asks the approval screen to show."""
    card_row = session.get(DecisionCardRow, run.card_id)
    checks, stats = check_preconditions(session, run, registry, rules)
    return {
        "card": _card_summary(card_row) if card_row is not None else {"card_id": run.card_id},
        "gate": _gate_summary(run),
        "shadow_window": {
            **stats,
            "required_requests": rules.min_shadow_requests,
        },
        "aliases": {
            CHAMPION_ALIAS: registry.version_of(CHAMPION_ALIAS),
            SHADOW_ALIAS: registry.version_of(SHADOW_ALIAS),
        },
        "preconditions": [check.to_record() for check in checks],
        "ready_to_approve": all(check.satisfied for check in checks),
        "rules": rules.to_record(),
    }


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def decide_promotion(
    session: Session,
    run_id: str,
    *,
    decision: str,
    approver: str,
    approver_role: str,
    reason: str,
    registry: AliasRegistry,
    rules: PromotionRules,
) -> PromotionResult:
    """Records a person's promotion decision and performs exactly its effects."""
    decision, reason = validate_decision(decision, reason, rules.min_reason_length)

    run = session.get(RetrainRun, run_id)
    if run is None:
        raise ApprovalNotFound(f"no retrain run {run_id!r}")
    if run.outcome != PASS or not run.shadow_alias_moved:
        raise ApprovalConflict(
            f"run {run_id} is {run.outcome} and "
            f"{'moved' if run.shadow_alias_moved else 'did not move'} shadow; only a gate "
            "PASS that reached shadow can be put to a person"
        )

    evidence = promotion_evidence(session, run, registry, rules)
    # A second decision is refused before any alias could move.
    if find_decision(session, KIND_PROMOTION, run_id) is not None:
        raise AlreadyDecided(f"promotion of {run_id} has already been decided")

    champion_before = registry.version_of(CHAMPION_ALIAS)
    approving = decision == DECISION_APPROVE
    if approving and not evidence["ready_to_approve"]:
        unmet = [c["detail"] for c in evidence["preconditions"] if not c["satisfied"]]
        raise ApprovalConflict("promotion preconditions not met: " + "; ".join(unmet))

    audit: list[dict] = []
    moved = False
    if approving:
        audit.append(
            registry.set(
                CHAMPION_ALIAS,
                run.challenger_version,
                run_id=run.mlflow_run or "",
                reason=(
                    f"week-9 promotion approved by {approver} for decision card {run.card_id} "
                    f"(retrain run {run_id}): {reason}"
                ),
                actor=approver,
            )
        )
        moved = True

    row = Approval(
        approval_id=approval_id(KIND_PROMOTION, run_id),
        card_id=run.card_id,
        kind=KIND_PROMOTION,
        subject=run_id,
        run_id=run_id,
        approver=approver,
        approver_role=approver_role,
        decision=decision,
        reason=reason,
        challenger_version=run.challenger_version,
        champion_version_before=champion_before,
        champion_version_after=run.challenger_version if moved else champion_before,
        champion_alias_moved=moved,
        shadow_alias_cleared=False,
        rules_version=rules.version,
        evidence=evidence,
    )

    # Clearing `shadow` is decided now and recorded on the row; it happens only
    # after the row is safely written, because an un-cleared shadow is harmless
    # (it keeps dual-scoring) while an unrecorded champion move is not.
    will_clear = registry.version_of(SHADOW_ALIAS) == run.challenger_version
    row.shadow_alias_cleared = will_clear

    try:
        written = record_decision(session, row)
    except Exception:
        if moved and champion_before is not None:
            registry.set(
                CHAMPION_ALIAS,
                champion_before,
                run_id=run.mlflow_run or "",
                reason=(
                    f"week-9 compensation: approval row for {run_id} could not be written, "
                    "so the promotion is reversed"
                ),
                actor=approver,
            )
        raise

    if will_clear:
        cleared = registry.clear(
            SHADOW_ALIAS,
            run_id=run.mlflow_run or "",
            reason=(
                f"week-9 promotion {'approved' if approving else 'rejected'} by {approver} "
                f"for retrain run {run_id}; shadow window closed"
            ),
            actor=approver,
        )
        if cleared is not None:
            audit.append(cleared)

    return PromotionResult(
        approval=written,
        champion_alias_moved=moved,
        shadow_alias_cleared=will_clear,
        alias_audit=tuple(audit),
    )
