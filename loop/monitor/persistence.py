"""Turning a measured window into a `drift_events` row.

Append-only in application code, like `predictions`: nothing here issues UPDATE
or DELETE, and a re-run of the benchmark appends a fresh set of rows rather
than overwriting the previous one. Two runs of scenario S1 are two pieces of
evidence, not one piece of evidence edited twice.

The row is built field by field from `WindowDrift`, which carries only
statistics. There is no path from a patient row to this module.
"""

import uuid

from sqlalchemy.orm import Session

from db.models import DriftEvent
from loop.monitor.drift import WindowDrift


class DriftPersistenceError(RuntimeError):
    """The drift row could not be written. Never swallowed."""


def new_event_id(scenario: str, window_index: int) -> str:
    """A readable, unique id, shaped like `de-s1-w000-3f9c1a2b`.

    The scenario and window are in the id because an examiner reading a
    `decision_cards.drift_event_id` should be able to tell which window it came
    from without a join. The random suffix is what keeps a replay of the same
    scenario from colliding with the run before it.
    """
    return f"de-{scenario.lower()}-w{window_index:03d}-{uuid.uuid4().hex[:8]}"


def build_drift_event(result: WindowDrift, *, event_id: str | None = None) -> DriftEvent:
    """Assembles the row. Takes only values that are safe to persist."""
    return DriftEvent(
        event_id=event_id or new_event_id(result.scenario, result.window_index),
        window_start=result.window_start,
        window_end=result.window_end,
        scenario=result.scenario,
        feature_stats=[stat.to_record() for stat in result.feature_stats],
        prediction_drift=result.prediction_drift,
        prediction_psi=result.prediction_psi,
        max_psi=result.max_psi,
        breaching_feature_count=result.breaching_feature_count,
        reference_rows=result.reference_rows,
        current_rows=result.current_rows,
        report_uri=result.report_uri,
        model_version=result.model_version,
        data_version=result.data_version,
        policy_thresholds=result.policy_thresholds,
    )


def persist_drift_event(session: Session, event: DriftEvent) -> DriftEvent:
    """Writes the row, or raises DriftPersistenceError.

    Uses the ORM, so values are bound as parameters; no SQL is assembled from
    measured data anywhere in this module.
    """
    try:
        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    except Exception as error:
        session.rollback()
        raise DriftPersistenceError(str(error)) from error


def record_window(session: Session, result: WindowDrift) -> DriftEvent:
    """Build and write in one call -- what the runner and the worker use."""
    return persist_drift_event(session, build_drift_event(result))
