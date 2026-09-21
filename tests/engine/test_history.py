"""Reading `drift_events` back: the persistence rule's arithmetic.

Rule 3 is the only rule that needs more than one window, so this is where the
Week 6 / Week 7 seam is tested -- that the engine consumes what the monitor
wrote, in the right order, per stream, without recomputing any of it.
"""

from datetime import timedelta

from sqlalchemy import select

from db.models import DriftEvent
from loop.engine.engine import decide
from loop.engine.history import (
    breaching_feature_names,
    build_evidence,
    consecutive_breach_run,
    events_without_cards,
    evidence_for,
    preceding_windows,
    with_retrain_history,
)
from loop.engine.persistence import record_decision
from tests.engine.conftest import WINDOW_START, feature_stat

WINDOW = timedelta(days=1)


def make_event(session, *, index: int, scenario: str = "S5", stats=(), prediction_drift=False):
    """A `drift_events` row in exactly the shape Week 6 writes."""
    start = WINDOW_START + index * WINDOW
    event = DriftEvent(
        event_id=f"de-{scenario.lower()}-w{index:03d}-{index:08x}",
        window_start=start,
        window_end=start + WINDOW,
        scenario=scenario,
        feature_stats=list(stats),
        prediction_drift=prediction_drift,
        prediction_psi=0.02,
        max_psi=max((stat["psi"] for stat in stats), default=0.0),
        breaching_feature_count=sum(1 for stat in stats if stat["psi"] >= 0.10),
        reference_rows=10_000,
        current_rows=2_000,
        report_uri=f"reports/drift/{scenario.lower()}/window_{index:03d}.html",
        model_version="1",
        data_version="dvc-reference-hash",
        policy_thresholds={
            "psi_breach": 0.10,
            "psi_severe": 0.25,
            "ks_p_value": 0.05,
            "prediction_psi": 0.10,
            "monitored_features": 25,
        },
    )
    session.add(event)
    session.commit()
    return event


# ---------------------------------------------------------------------------
# Breaching sets
# ---------------------------------------------------------------------------
def test_breaching_names_are_read_against_the_policy_threshold():
    event = DriftEvent(feature_stats=[feature_stat("a", 0.12), feature_stat("b", 0.04)])
    assert breaching_feature_names(event, 0.10) == {"a"}
    assert breaching_feature_names(event, 0.02) == {"a", "b"}
    assert breaching_feature_names(event, 0.50) == set()


def test_a_row_with_no_stats_breaches_nothing():
    assert breaching_feature_names(DriftEvent(feature_stats=None), 0.10) == set()


def test_the_policy_threshold_wins_over_the_flag_recorded_on_the_row():
    """The row's `breaching` flag was written under the monitor's thresholds;
    the policy's may differ, and the card records both."""
    stat = feature_stat("a", 0.12)
    stat["breaching"] = False
    assert breaching_feature_names(DriftEvent(feature_stats=[stat]), 0.10) == {"a"}


# ---------------------------------------------------------------------------
# Consecutive runs
# ---------------------------------------------------------------------------
def test_a_window_that_breaches_nothing_has_no_run():
    event = DriftEvent(feature_stats=[feature_stat("a", 0.01)])
    assert consecutive_breach_run(event, [], 0.10) == (0, ())


def test_a_first_breaching_window_is_a_run_of_one():
    event = DriftEvent(feature_stats=[feature_stat("a", 0.12)])
    assert consecutive_breach_run(event, [], 0.10) == (1, ("a",))


def test_the_run_counts_only_features_common_to_every_window():
    """ "Same features breach", not "some feature breaches"."""
    current = DriftEvent(feature_stats=[feature_stat("a", 0.12), feature_stat("b", 0.13)])
    history = [
        DriftEvent(feature_stats=[feature_stat("a", 0.11), feature_stat("c", 0.14)]),
        DriftEvent(feature_stats=[feature_stat("a", 0.15)]),
    ]
    assert consecutive_breach_run(current, history, 0.10) == (3, ("a",))


def test_the_run_stops_at_a_window_with_nothing_in_common():
    current = DriftEvent(feature_stats=[feature_stat("a", 0.12)])
    history = [
        DriftEvent(feature_stats=[feature_stat("a", 0.11)]),
        DriftEvent(feature_stats=[feature_stat("z", 0.30)]),
        DriftEvent(feature_stats=[feature_stat("a", 0.40)]),
    ]
    assert consecutive_breach_run(current, history, 0.10) == (2, ("a",))


def test_a_quiet_window_in_the_history_breaks_the_run():
    """Which is why Week 6 persists quiet windows at all."""
    current = DriftEvent(feature_stats=[feature_stat("a", 0.12)])
    history = [
        DriftEvent(feature_stats=[feature_stat("a", 0.01)]),
        DriftEvent(feature_stats=[feature_stat("a", 0.30)]),
    ]
    assert consecutive_breach_run(current, history, 0.10) == (1, ("a",))


def test_persistent_features_are_returned_sorted():
    current = DriftEvent(feature_stats=[feature_stat("b", 0.12), feature_stat("a", 0.13)])
    history = [DriftEvent(feature_stats=[feature_stat("a", 0.11), feature_stat("b", 0.11)])]
    assert consecutive_breach_run(current, history, 0.10)[1] == ("a", "b")


# ---------------------------------------------------------------------------
# Reading from the database
# ---------------------------------------------------------------------------
def test_preceding_windows_are_returned_newest_first(drift_db):
    for index in range(3):
        make_event(drift_db, index=index, stats=(feature_stat("a", 0.12),))
    latest = drift_db.scalars(select(DriftEvent).order_by(DriftEvent.window_start.desc())).first()

    history = preceding_windows(drift_db, latest)
    assert [event.window_start for event in history] == sorted(
        (event.window_start for event in history), reverse=True
    )
    assert len(history) == 2


def test_streams_do_not_see_each_others_history(drift_db):
    """Replaying S1 then S5 must not make S5's first window look like S1's fourth."""
    for index in range(3):
        make_event(drift_db, index=index, scenario="S1", stats=(feature_stat("a", 0.30),))
    s5 = make_event(drift_db, index=3, scenario="S5", stats=(feature_stat("a", 0.12),))

    assert preceding_windows(drift_db, s5) == []


def test_evidence_is_built_from_the_row_without_recomputing_anything(drift_db, policy):
    event = make_event(
        drift_db, index=0, stats=(feature_stat("a", 0.30, ks_p=0.001), feature_stat("b", 0.04))
    )
    evidence = evidence_for(drift_db, event, policy, candidate_data_version="dvc-train")

    assert evidence.drift_event_id == event.event_id
    assert evidence.max_psi == event.max_psi
    assert evidence.monitored_feature_count == 25
    assert evidence.model_version == "1"
    assert evidence.candidate_data_version == "dvc-train"
    assert [breach.feature for breach in evidence.breaches(policy.psi_breach)] == ["a"]
    assert evidence.breaches(policy.psi_breach)[0].ks_p == 0.001


def test_the_monitored_count_falls_back_to_the_stats_when_the_row_omits_it(policy):
    event = DriftEvent(
        event_id="de-x",
        scenario="T",
        window_start=WINDOW_START,
        window_end=WINDOW_START + WINDOW,
        feature_stats=[feature_stat("a", 0.12), feature_stat("b", 0.01)],
        policy_thresholds=None,
    )
    assert build_evidence(event, [], policy).monitored_feature_count == 2


def test_retrain_history_is_attached_explicitly(policy):
    event = DriftEvent(
        event_id="de-x",
        scenario="T",
        window_start=WINDOW_START,
        window_end=WINDOW_START + WINDOW,
        feature_stats=[feature_stat("a", 0.40)],
    )
    evidence = build_evidence(event, [], policy)
    assert evidence.days_since_last_retrain is None

    with_history = with_retrain_history(
        evidence, days_since_last_retrain=2.0, retrains_in_cooldown=1
    )
    assert with_history.days_since_last_retrain == 2.0
    assert with_history.retrains_in_cooldown == 1
    assert evidence.days_since_last_retrain is None


# ---------------------------------------------------------------------------
# Finding undecided windows
# ---------------------------------------------------------------------------
def test_every_window_is_pending_before_anything_is_decided(drift_db, policy):
    for index in range(3):
        make_event(drift_db, index=index)
    assert len(events_without_cards(drift_db, policy.version)) == 3


def test_a_decided_window_stops_being_pending(drift_db, policy):
    event = make_event(drift_db, index=0, stats=(feature_stat("a", 0.30),))
    record_decision(drift_db, decide(evidence_for(drift_db, event, policy), policy))

    assert events_without_cards(drift_db, policy.version) == []


def test_a_different_policy_version_sees_the_window_as_pending_again(drift_db, policy):
    """Replaying history under a new policy is the governance story, not a duplicate."""
    event = make_event(drift_db, index=0, stats=(feature_stat("a", 0.30),))
    record_decision(drift_db, decide(evidence_for(drift_db, event, policy), policy))

    other = "policy-v1" if policy.version != "policy-v1" else "policy-v2"
    assert [e.event_id for e in events_without_cards(drift_db, other)] == [event.event_id]


def test_pending_windows_come_back_oldest_first(drift_db, policy):
    for index in reversed(range(4)):
        make_event(drift_db, index=index)
    pending = events_without_cards(drift_db, policy.version)
    assert [event.window_start for event in pending] == sorted(
        event.window_start for event in pending
    )


def test_pending_windows_can_be_restricted_to_one_stream(drift_db, policy):
    make_event(drift_db, index=0, scenario="S1")
    make_event(drift_db, index=1, scenario="S5")
    pending = events_without_cards(drift_db, policy.version, scenario="S5")
    assert [event.scenario for event in pending] == ["S5"]


def test_pending_windows_can_be_limited_and_filtered_by_time(drift_db, policy):
    for index in range(5):
        make_event(drift_db, index=index)
    assert len(events_without_cards(drift_db, policy.version, limit=2)) == 2

    # SQLite hands timezone-aware columns back naive, so the windows are
    # identified by id rather than re-compared in Python; the filter itself
    # runs in SQL either way.
    cutoff = WINDOW_START + 3 * WINDOW
    later = events_without_cards(drift_db, policy.version, since=cutoff)
    assert [event.event_id for event in later] == [
        "de-s5-w003-00000003",
        "de-s5-w004-00000004",
    ]
