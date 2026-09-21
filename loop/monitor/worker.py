"""The scheduled monitor worker.

ARCHITECTURE.md §3.7: "A scheduled job (APScheduler inside a small worker
container) runs Evidently every monitoring window (demo: every N minutes;
design: daily)". This is that worker, and it is deliberately the smallest thing
that satisfies the sentence: one in-process `BlockingScheduler` with one
interval job. TECH_STACK.md names the alternative it is avoiding -- "Prefect /
Airflow (orchestrators for tens of DAGs, not one loop -- classic
over-engineering trap)".

    python -m loop.monitor.worker

Each tick measures the *next* window of the serving stream and writes its
`drift_events` row, quiet or not. When the stream runs out the worker stops
advancing and says so, rather than silently re-measuring the same window
forever: a repeated window would look, to Week 7's persistence rule, exactly
like drift that persisted.

Configuration is environment-only, the same way the API is configured:

    VITALLOOP_MONITOR_SCENARIO          S1-S5, or "live" (default: live)
    VITALLOOP_MONITOR_INTERVAL_SECONDS  seconds between windows (default: 300)
    VITALLOOP_MONITOR_MAX_WINDOWS       stop after N windows (default: unlimited)
    VITALLOOP_MODEL_SOURCE              local | mlflow (default: local)
    VITALLOOP_POLICY_VERSION            decision policy to apply (default: policy-v2)

Week 7 adds the second half of ARCHITECTURE.md §5's `monitor` service -- "drift
job + decision engine". Each tick measures a window, writes its `drift_events`
row, and then decides what that window means, emitting one Decision Card. There
is no second scheduler and no queue between the two halves: the worker owns
both, which is what §4.7 specifies.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.orm import Session

from api.logging_config import configure_logging, get_logger
from loop.engine.evaluate import candidate_data_version, evaluate_event
from loop.engine.policy import DEFAULT_POLICY_VERSION, Policy, PolicyError, load_policy
from loop.monitor.config import DRIFT_REPORTS_DIR, LIVE_SCENARIO, WINDOW_ROWS
from loop.monitor.database import open_session
from loop.monitor.persistence import record_window
from loop.monitor.reference import MonitoringReference, load_reference
from loop.monitor.runner import load_serving_stream, measure_window
from loop.monitor.scoring import ScoringModel, load_scoring_model
from loop.monitor.windows import iter_windows
from scenarios.injection import SCENARIO_SEED, apply_scenario

DEFAULT_INTERVAL_SECONDS = 300

logger = get_logger("vitalloop.monitor")


def _int_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


class MonitorCycle:
    """Walks the serving stream one window per tick, persisting each one."""

    def __init__(
        self,
        scenario: str,
        *,
        max_windows: int | None = None,
        model: ScoringModel | None = None,
        reference: MonitoringReference | None = None,
        session: Session | None = None,
        stream: pd.DataFrame | None = None,
        window_rows: int = WINDOW_ROWS,
        reports_dir: Path | None = DRIFT_REPORTS_DIR,
        policy: Policy | None = None,
        policy_version: str | None = DEFAULT_POLICY_VERSION,
    ):
        # Every collaborator is injectable so the cycle can be exercised without
        # a database service, a trained artifact or the dataset on disk.
        self.scenario = scenario
        self.max_windows = max_windows
        self.reports_dir = reports_dir
        self.model = model or load_scoring_model()
        self.reference = reference if reference is not None else load_reference(self.model)
        self.session = session if session is not None else open_session()

        stream = load_serving_stream() if stream is None else stream
        if scenario != LIVE_SCENARIO:
            stream = apply_scenario(stream, scenario, seed=SCENARIO_SEED)
        self.windows = iter_windows(stream, window_rows=window_rows, max_windows=max_windows)
        self.next_index = 0

        # Week 7. A policy that will not load is fatal here rather than on the
        # first tick: a monitor that silently stops deciding looks exactly like
        # a monitor that found nothing to decide.
        self.policy = policy
        if self.policy is None and policy_version is not None:
            self.policy = load_policy(policy_version)
        self.candidate_data_version = candidate_data_version()

        logger.info(
            "monitor_ready",
            scenario=scenario,
            windows=len(self.windows),
            reference_rows=self.reference.rows,
            model_version=self.model.version,
            model_source=self.model.source,
            policy_version=None if self.policy is None else self.policy.version,
        )

    @property
    def exhausted(self) -> bool:
        return self.next_index >= len(self.windows)

    def tick(self) -> None:
        """One monitoring window. Never raises out of the scheduler."""
        if self.exhausted:
            logger.info("monitor_stream_exhausted", scenario=self.scenario)
            return
        window = self.windows[self.next_index]
        self.next_index += 1
        try:
            result = measure_window(
                window,
                scenario=self.scenario,
                reference=self.reference,
                model=self.model,
                reports_dir=self.reports_dir,
            )
            event = record_window(self.session, result)
        except Exception as error:
            # A failed window must not take the worker down: the next window is
            # a fresh measurement, and the gap is visible in the row history.
            logger.error(
                "monitor_window_failed",
                scenario=self.scenario,
                window_index=window.index,
                error_category=type(error).__name__,
            )
            return
        logger.info(
            "drift_window_recorded",
            event_id=event.event_id,
            scenario=self.scenario,
            window_index=result.window_index,
            max_psi=round(result.max_psi, 6),
            breaching_feature_count=result.breaching_feature_count,
            prediction_drift=result.prediction_drift,
            report_uri=result.report_uri,
        )
        self._decide(event)

    def _decide(self, event) -> None:
        """Week 7: what the window means, decided in this same worker.

        ARCHITECTURE.md §5 gives the `monitor` service both jobs -- "drift job +
        decision engine" -- and §4.7 repeats it, so the decision runs on the
        same tick that produced the window rather than in a second scheduler.
        The measurement is already committed by the time this runs, so a policy
        failure loses the card, never the evidence.
        """
        if self.policy is None:
            return
        try:
            card, written = evaluate_event(
                self.session,
                event,
                self.policy,
                data_version=self.candidate_data_version,
            )
        except Exception as error:
            logger.error(
                "decision_failed",
                event_id=event.event_id,
                policy_version=self.policy.version,
                error_category=type(error).__name__,
            )
            return
        logger.info(
            "decision_card_emitted" if written else "decision_card_already_on_record",
            card_id=card.card_id,
            drift_event_id=card.trigger.drift_event_id,
            policy_version=card.policy_version,
            rule_id=card.rule_id,
            action=card.action,
            disposition=card.disposition,
            confidence=card.confidence,
            status=card.status,
        )

    def close(self) -> None:
        self.session.close()


def main() -> int:
    configure_logging()

    scenario = os.environ.get("VITALLOOP_MONITOR_SCENARIO", LIVE_SCENARIO)
    interval = _int_env("VITALLOOP_MONITOR_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS)
    max_windows = _int_env("VITALLOOP_MONITOR_MAX_WINDOWS", None)
    policy_version = os.environ.get("VITALLOOP_POLICY_VERSION", DEFAULT_POLICY_VERSION)

    try:
        cycle = MonitorCycle(scenario, max_windows=max_windows, policy_version=policy_version)
    except PolicyError as error:
        logger.error("policy_unavailable", policy_version=policy_version, error=str(error))
        return 1
    scheduler = BlockingScheduler(timezone="UTC")
    # The first window is measured immediately rather than one interval from
    # now, so `docker compose up` produces evidence without a wait.
    scheduler.add_job(cycle.tick, "interval", seconds=interval, id="drift")
    cycle.tick()

    logger.info("monitor_scheduled", scenario=scenario, interval_seconds=interval)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - signal path
        logger.info("monitor_stopped", scenario=scenario)
    finally:
        cycle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
