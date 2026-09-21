"""The scheduled worker's cycle: one window per tick, and what happens at the end.

The APScheduler loop itself is three lines of library call and is not worth a
test; what needs pinning is the cycle it drives -- that a tick advances exactly
one window, that it persists, and that an exhausted stream or a failed window
does not take the worker down.
"""

import pandas as pd
import pytest
from sqlalchemy import select

from db.models import DriftEvent
from loop.monitor.reference import build_reference
from loop.monitor.scoring import ScoringModel
from loop.monitor.worker import DEFAULT_INTERVAL_SECONDS, MonitorCycle, _int_env

STREAM_ROWS = 4000
WINDOW_ROWS = 1000


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    from tests.conftest import make_synthetic_clean_df

    return make_synthetic_clean_df(n_rows=STREAM_ROWS, seed=31)


@pytest.fixture(scope="module")
def model(stream):
    from ml.data.features import split_features_target
    from ml.train import build_base_model

    X, y = split_features_target(stream)
    base = build_base_model(n_estimators=5)
    base.fit(X, y)
    return ScoringModel(model=base, version="test-model", source="local")


@pytest.fixture(scope="module")
def reference(stream, model):
    half = STREAM_ROWS // 2
    return build_reference(stream.iloc[:half], model, rows=half)


@pytest.fixture
def cycle(stream, model, reference, drift_db):
    return MonitorCycle(
        "S5",
        model=model,
        reference=reference,
        session=drift_db,
        stream=stream.iloc[STREAM_ROWS // 2 :],
        window_rows=WINDOW_ROWS,
        reports_dir=None,
    )


def _rows(session) -> list[DriftEvent]:
    return list(session.scalars(select(DriftEvent)))


# ---------------------------------------------------------------------------
# Ticking
# ---------------------------------------------------------------------------
def test_a_cycle_plans_one_window_per_tick(cycle):
    assert len(cycle.windows) == 2
    assert cycle.next_index == 0
    assert not cycle.exhausted


def test_one_tick_writes_exactly_one_row(cycle, drift_db):
    cycle.tick()
    rows = _rows(drift_db)
    assert len(rows) == 1
    assert rows[0].scenario == "S5"
    assert cycle.next_index == 1


def test_consecutive_ticks_advance_through_the_stream(cycle, drift_db):
    cycle.tick()
    cycle.tick()
    rows = sorted(_rows(drift_db), key=lambda row: row.window_start)
    assert len(rows) == 2
    assert rows[0].window_end == rows[1].window_start
    assert rows[0].event_id != rows[1].event_id


def test_an_exhausted_stream_stops_advancing_instead_of_repeating_a_window(cycle, drift_db):
    """A re-measured window would read to Week 7 exactly like drift that persisted."""
    for _ in range(5):
        cycle.tick()
    assert cycle.exhausted
    assert len(_rows(drift_db)) == 2


def test_a_failed_window_is_logged_and_the_next_tick_still_runs(cycle, drift_db, monkeypatch):
    import loop.monitor.worker as worker

    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("evidently fell over")

    monkeypatch.setattr(worker, "measure_window", explode)
    cycle.tick()
    assert calls["n"] == 1
    assert _rows(drift_db) == []

    monkeypatch.undo()
    cycle.tick()
    assert len(_rows(drift_db)) == 1


def test_the_worker_persists_quiet_windows_too(cycle, drift_db):
    cycle.tick()
    assert _rows(drift_db)[0].breaching_feature_count == 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_the_interval_falls_back_to_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("VITALLOOP_MONITOR_INTERVAL_SECONDS", raising=False)
    assert _int_env("VITALLOOP_MONITOR_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS) == 300


def test_an_empty_environment_value_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("VITALLOOP_MONITOR_MAX_WINDOWS", "")
    assert _int_env("VITALLOOP_MONITOR_MAX_WINDOWS", None) is None


def test_the_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("VITALLOOP_MONITOR_INTERVAL_SECONDS", "45")
    assert _int_env("VITALLOOP_MONITOR_INTERVAL_SECONDS", 300) == 45


def test_max_windows_caps_the_run(stream, model, reference, drift_db):
    cycle = MonitorCycle(
        "S5",
        max_windows=1,
        model=model,
        reference=reference,
        session=drift_db,
        stream=stream.iloc[STREAM_ROWS // 2 :],
        window_rows=WINDOW_ROWS,
        reports_dir=None,
    )
    assert len(cycle.windows) == 1


# ---------------------------------------------------------------------------
# Week 7: the decision engine runs in this same worker
# ---------------------------------------------------------------------------
def test_a_tick_emits_one_decision_card_for_the_window_it_measured(cycle, drift_db):
    """ARCHITECTURE.md §5 gives the `monitor` service both jobs, on one tick."""
    from db.models import DecisionCard

    cycle.tick()

    events = _rows(drift_db)
    cards = list(drift_db.scalars(select(DecisionCard)))
    assert len(events) == len(cards) == 1
    assert cards[0].drift_event_id == events[0].event_id
    assert cards[0].policy_version == cycle.policy.version


def test_the_worker_loads_the_policy_at_startup(cycle):
    """A monitor that silently stopped deciding would look like a quiet stream."""
    assert cycle.policy is not None
    assert cycle.policy.version == "policy-v1"


def test_a_quiet_window_still_produces_a_card(cycle, drift_db):
    """Silence is a logged decision, not an absence -- WORKFLOW.md §4."""
    from db.models import DecisionCard

    cycle.tick()
    card = drift_db.scalars(select(DecisionCard)).one()
    assert card.action == "NO_OP"
    assert card.status == "CLOSED"


def test_a_failed_decision_does_not_lose_the_measurement(cycle, drift_db, monkeypatch):
    """The window is committed before the policy runs, so evidence survives."""
    import loop.monitor.worker as worker

    def explode(*args, **kwargs):
        raise RuntimeError("policy blew up")

    monkeypatch.setattr(worker, "evaluate_event", explode)
    cycle.tick()

    from db.models import DecisionCard

    assert len(_rows(drift_db)) == 1
    assert list(drift_db.scalars(select(DecisionCard))) == []


def test_the_worker_can_run_without_a_policy(stream, model, reference, drift_db):
    """Measurement never depends on the decision layer being configured."""
    from db.models import DecisionCard

    cycle = MonitorCycle(
        "S5",
        model=model,
        reference=reference,
        session=drift_db,
        stream=stream.iloc[STREAM_ROWS // 2 :],
        window_rows=WINDOW_ROWS,
        reports_dir=None,
        policy_version=None,
    )
    assert cycle.policy is None
    cycle.tick()
    assert len(_rows(drift_db)) == 1
    assert list(drift_db.scalars(select(DecisionCard))) == []
