"""The `drift_events` row: what is written, what is deliberately not, append-only.

Runs against a throwaway SQLite database so the suite needs no Postgres; the
real Postgres path is exercised by the documented manual verification (see
docs/RUNNING_THE_PROJECT.md).
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from db.models import DriftEvent
from loop.monitor.drift import FeatureDrift, WindowDrift, current_thresholds
from loop.monitor.persistence import (
    DriftPersistenceError,
    build_drift_event,
    new_event_id,
    persist_drift_event,
    record_window,
)

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)


def _result(**overrides) -> WindowDrift:
    stats = (
        FeatureDrift("num_lab_procedures", "numerical", 0.27, 0.001, True, True),
        FeatureDrift("num_medications", "numerical", 0.19, 0.004, True, False),
        FeatureDrift("race", "categorical", 0.01, None, False, False),
    )
    defaults = dict(
        scenario="S1",
        window_index=2,
        window_start=WINDOW_START,
        window_end=WINDOW_START + timedelta(days=1),
        feature_stats=stats,
        prediction_psi=0.14,
        prediction_drift=True,
        max_psi=0.27,
        breaching_feature_count=2,
        reference_rows=10_000,
        current_rows=2_000,
        policy_thresholds={**current_thresholds(), "monitored_features": 3},
        report_uri="reports/drift/s1/window_002.html",
        model_version="local-artifact",
        data_version="abc123",
    )
    return WindowDrift(**{**defaults, **overrides})


@pytest.fixture
def session(drift_db):
    yield drift_db
    drift_db.close()


# ---------------------------------------------------------------------------
# Event ids
# ---------------------------------------------------------------------------
def test_the_event_id_names_the_scenario_and_window():
    assert new_event_id("S1", 2).startswith("de-s1-w002-")


def test_event_ids_do_not_collide_when_a_scenario_is_replayed():
    """A re-run appends fresh evidence; it must not fail on the primary key."""
    assert new_event_id("S1", 0) != new_event_id("S1", 0)


# ---------------------------------------------------------------------------
# What the row carries
# ---------------------------------------------------------------------------
def test_every_measured_field_reaches_the_row():
    event = build_drift_event(_result())
    assert event.scenario == "S1"
    assert event.window_start == WINDOW_START
    assert event.prediction_drift is True
    assert event.prediction_psi == pytest.approx(0.14)
    assert event.max_psi == pytest.approx(0.27)
    assert event.breaching_feature_count == 2
    assert event.reference_rows == 10_000
    assert event.current_rows == 2_000
    assert event.report_uri == "reports/drift/s1/window_002.html"
    assert event.model_version == "local-artifact"
    assert event.data_version == "abc123"
    assert event.policy_thresholds["psi_breach"] == 0.10


def test_feature_stats_are_the_aggregate_records_and_nothing_more():
    event = build_drift_event(_result())
    assert [s["feature"] for s in event.feature_stats] == [
        "num_lab_procedures",
        "num_medications",
        "race",
    ]
    for record in event.feature_stats:
        assert set(record) == {"feature", "kind", "psi", "ks_p_value", "breaching", "severe"}


def test_a_supplied_event_id_is_used_verbatim():
    assert build_drift_event(_result(), event_id="de-0042").event_id == "de-0042"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_a_row_round_trips_through_the_database(session):
    written = record_window(session, _result())

    stored = session.scalars(
        select(DriftEvent).where(DriftEvent.event_id == written.event_id)
    ).one()
    assert stored.scenario == "S1"
    assert stored.max_psi == pytest.approx(0.27)
    assert stored.breaching_feature_count == 2
    assert stored.prediction_drift is True
    assert stored.feature_stats[0]["feature"] == "num_lab_procedures"
    assert stored.policy_thresholds["psi_severe"] == 0.25
    assert stored.created_at is not None


def test_a_quiet_window_is_persisted_as_a_row_not_as_an_absence(session):
    """PROJECT_DESIGN.md §6: "every window persisted". Silence is evidence."""
    record_window(
        session,
        _result(
            scenario="S5",
            feature_stats=(FeatureDrift("race", "categorical", 0.01, None, False, False),),
            prediction_psi=0.01,
            prediction_drift=False,
            max_psi=0.01,
            breaching_feature_count=0,
        ),
    )
    stored = session.scalars(select(DriftEvent).where(DriftEvent.scenario == "S5")).one()
    assert stored.breaching_feature_count == 0
    assert stored.prediction_drift is False


def test_replaying_a_scenario_appends_rather_than_overwrites(session):
    for _ in range(3):
        record_window(session, _result())
    assert len(list(session.scalars(select(DriftEvent)))) == 3


def test_consecutive_windows_are_stored_in_window_order(session):
    for index in range(3):
        record_window(session, _result(window_index=index, window_start=WINDOW_START))
    stored = list(session.scalars(select(DriftEvent).order_by(DriftEvent.event_id)))
    assert {row.event_id.split("-")[2] for row in stored} == {"w000", "w001", "w002"}


def test_a_duplicate_event_id_is_refused_rather_than_silently_replacing_evidence(session):
    persist_drift_event(session, build_drift_event(_result(), event_id="de-fixed"))
    with pytest.raises(DriftPersistenceError):
        persist_drift_event(session, build_drift_event(_result(), event_id="de-fixed"))


def test_a_failed_write_rolls_back_and_leaves_the_session_usable(session):
    persist_drift_event(session, build_drift_event(_result(), event_id="de-fixed"))
    with pytest.raises(DriftPersistenceError):
        persist_drift_event(session, build_drift_event(_result(), event_id="de-fixed"))
    record_window(session, _result())
    assert len(list(session.scalars(select(DriftEvent)))) == 2


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
def test_the_table_has_no_column_that_could_hold_a_patient_row():
    """The schema itself is the control: there is nowhere to put one."""
    columns = set(DriftEvent.__table__.columns.keys())
    assert columns == {
        "event_id",
        "created_at",
        "window_start",
        "window_end",
        "scenario",
        "feature_stats",
        "prediction_drift",
        "prediction_psi",
        "max_psi",
        "breaching_feature_count",
        "reference_rows",
        "current_rows",
        "report_uri",
        "model_version",
        "data_version",
        "policy_thresholds",
    }
    for forbidden in ("patient_nbr", "encounter_id", "readmitted_30d", "payload", "rows"):
        assert forbidden not in columns


def test_the_stored_report_reference_is_a_path_not_the_report(session):
    stored = record_window(session, _result())
    assert stored.report_uri.endswith(".html")
    assert "<html" not in stored.report_uri
