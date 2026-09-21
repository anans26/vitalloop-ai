"""Reading `drift_events` back, and turning a window into evidence.

This is the only module in `loop/engine/` that talks to the database on the
read side, which keeps `engine.decide` a pure function of its arguments.

The interesting part is the persistence rule. §3.8 rule 3 fires when "the same
features breach for >= 2 consecutive windows", so the engine needs more than
the latest window: it needs the run of windows immediately before it, in the
same stream, and the intersection of their breaching feature sets. Week 6
persists *every* window including the quiet ones precisely so that run is
unbroken -- a missing quiet window would read here as two breaching windows
that were adjacent when they were not.

Streams are kept apart by `scenario`. Replaying S1 and then S5 must not make
S5's first window look like S1's fourth.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DecisionCard as DecisionCardRow
from db.models import DriftEvent
from loop.engine.evidence import DriftEvidence
from loop.engine.policy import Policy

# How far back the persistence rule ever needs to look. Rule 3 counts
# consecutive windows and `persistence` caps at `persistence_scale`, so a run
# longer than this changes no decision; the bound keeps a long-lived stream
# from loading its whole history to evaluate one window.
MAX_HISTORY_WINDOWS = 16


def breaching_feature_names(event: DriftEvent, psi_breach: float) -> set[str]:
    """Features whose *recorded* PSI reaches the policy's breach threshold.

    Read off `feature_stats` rather than trusting the row's own `breaching`
    flag: that flag was written under the thresholds in force at measurement
    time, and the policy evaluating the row now may use different ones. The
    card records both, so the difference is visible rather than silent.
    """
    return {
        str(stat["feature"])
        for stat in (event.feature_stats or [])
        if stat.get("feature") is not None
        and stat.get("psi") is not None
        and float(stat["psi"]) >= psi_breach
    }


def preceding_windows(
    session: Session, event: DriftEvent, *, limit: int = MAX_HISTORY_WINDOWS
) -> list[DriftEvent]:
    """The windows immediately before this one in the same stream, newest first."""
    statement = (
        select(DriftEvent)
        .where(DriftEvent.scenario == event.scenario)
        .where(DriftEvent.window_start < event.window_start)
        .order_by(DriftEvent.window_start.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def consecutive_breach_run(
    event: DriftEvent, history: Sequence[DriftEvent], psi_breach: float
) -> tuple[int, tuple[str, ...]]:
    """How many consecutive windows the same features have breached, and which.

    Walks backwards from the current window, intersecting breaching sets. The
    run ends at the first window that shares no breaching feature with the run
    so far -- "the *same* features breach", not "some feature breaches".

    Returns `(0, ())` when the current window breaches nothing: a run of
    breaches cannot include a window that did not breach.
    """
    current = breaching_feature_names(event, psi_breach)
    if not current:
        return 0, ()

    persistent = current
    count = 1
    for previous in history:
        overlap = persistent & breaching_feature_names(previous, psi_breach)
        if not overlap:
            break
        persistent = overlap
        count += 1
    return count, tuple(sorted(persistent))


def build_evidence(
    event: DriftEvent,
    history: Sequence[DriftEvent],
    policy: Policy,
    *,
    candidate_data_version: str | None = None,
) -> DriftEvidence:
    """One `drift_events` row plus its predecessors, as the engine's input.

    Nothing is recomputed: every statistic below was measured by Week 6 and is
    copied across verbatim.
    """
    consecutive, persistent = consecutive_breach_run(event, history, policy.psi_breach)
    thresholds = dict(event.policy_thresholds or {})
    monitored = int(thresholds.get("monitored_features") or len(event.feature_stats or []))

    return DriftEvidence(
        drift_event_id=event.event_id,
        scenario=event.scenario,
        window_start=event.window_start,
        window_end=event.window_end,
        feature_stats=tuple(event.feature_stats or []),
        prediction_drift=bool(event.prediction_drift),
        prediction_psi=event.prediction_psi,
        max_psi=event.max_psi,
        monitored_feature_count=monitored,
        measured_thresholds=thresholds,
        consecutive_breaching_windows=consecutive,
        persistent_features=persistent,
        model_version=event.model_version,
        data_version=event.data_version,
        candidate_data_version=candidate_data_version,
    )


def evidence_for(
    session: Session,
    event: DriftEvent,
    policy: Policy,
    *,
    candidate_data_version: str | None = None,
) -> DriftEvidence:
    """`build_evidence`, with the window's predecessors fetched for it."""
    return build_evidence(
        event,
        preceding_windows(session, event),
        policy,
        candidate_data_version=candidate_data_version,
    )


def with_retrain_history(
    evidence: DriftEvidence, *, days_since_last_retrain: float | None, retrains_in_cooldown: int
) -> DriftEvidence:
    """Attaches rule 6's inputs.

    Kept as an explicit step rather than a database read because the table it
    would read -- `retrain_runs` -- is Week 8. Rule 6 is therefore inert on
    every real Week 7 card, and this function is how the unit tests reach it.
    """
    return replace(
        evidence,
        days_since_last_retrain=days_since_last_retrain,
        retrains_in_cooldown=retrains_in_cooldown,
    )


def events_without_cards(
    session: Session,
    policy_version: str,
    *,
    scenario: str | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[DriftEvent]:
    """Windows this policy version has not decided on yet, oldest first.

    Oldest first matters: rule 3 reads the run of windows before the one being
    evaluated, so evaluating a backlog out of order would decide window 3
    against a history that did not yet contain window 2.
    """
    decided = select(DecisionCardRow.drift_event_id).where(
        DecisionCardRow.policy_version == policy_version
    )
    statement = (
        select(DriftEvent)
        .where(DriftEvent.event_id.not_in(decided))
        .order_by(DriftEvent.scenario, DriftEvent.window_start)
    )
    if scenario is not None:
        statement = statement.where(DriftEvent.scenario == scenario)
    if since is not None:
        statement = statement.where(DriftEvent.window_start >= since)
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))
