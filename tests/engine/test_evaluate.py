"""Week 6 rows in, `decision_cards` rows out -- the whole Week 7 seam.

Including the case the Week 6 report flagged: the untouched S5 control stream
produces genuine PSI breaches in two administrative features. Both readings are
asserted here -- what the uncalibrated policy-v1 makes of them (the finding that
motivated the calibration) and what the calibrated policy-v2 makes of them (the
Week 7 deliverable). The Week 6 statistics are used exactly as measured;
nothing is special-cased to make the control look quieter than it is.
"""

from datetime import timedelta

import pytest
from sqlalchemy import select

from db.models import DecisionCard as DecisionCardRow
from db.models import DriftEvent
from loop.engine.evaluate import evaluate_event, evaluate_pending, summarise
from loop.engine.persistence import (
    DecisionPersistenceError,
    build_card_row,
    find_existing,
    persist_card,
    record_decision,
)
from loop.engine.rules import ALERT_ONLY, ESCALATE_HUMAN, FULL_RETRAIN, INCREMENTAL_RETRAIN, NO_OP
from tests.engine.conftest import WINDOW_START, feature_stat, mild_psi

WINDOW = timedelta(days=1)

# The S5 control, as Week 6 actually measured it (docs/RUNNING_THE_PROJECT.md
# §14.4): window 0 fully quiet, then payer_code and medical_specialty breaching
# mildly and persistently, with prediction drift never firing.
CONTROL_WINDOWS = [
    (),
    (("payer_code", 0.1278), ("medical_specialty", 0.1250)),
    (("medical_specialty", 0.1663), ("payer_code", 0.1131)),
]

# S1, as measured: the injected feature is severe from the first window.
COVARIATE_WINDOWS = [
    (("num_lab_procedures", 0.2777), ("num_medications", 0.1421)),
    (("num_lab_procedures", 0.2713), ("num_medications", 0.1318)),
]


def make_event(session, *, index, scenario, breaches=(), prediction_drift=False, commit=True):
    """A `drift_events` row in the shape Week 6 writes, with 25 monitored features."""
    start = WINDOW_START + index * WINDOW
    stats = [feature_stat(name, psi, ks_p=0.001) for name, psi in breaches]
    stats += [feature_stat(f"quiet_{i}", 0.01) for i in range(25 - len(stats))]
    event = DriftEvent(
        event_id=f"de-{scenario.lower()}-w{index:03d}-{index:08x}",
        window_start=start,
        window_end=start + WINDOW,
        scenario=scenario,
        feature_stats=stats,
        prediction_drift=prediction_drift,
        prediction_psi=0.20 if prediction_drift else 0.02,
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
    if commit:
        session.commit()
    return event


def seed_stream(session, scenario, windows, *, prediction_drift=False):
    return [
        make_event(
            session,
            index=index,
            scenario=scenario,
            breaches=breaches,
            prediction_drift=prediction_drift,
        )
        for index, breaches in enumerate(windows)
    ]


def cards(session, scenario=None):
    statement = select(DecisionCardRow).order_by(DecisionCardRow.window_start)
    if scenario is not None:
        statement = statement.where(DecisionCardRow.scenario == scenario)
    return list(session.scalars(statement))


# ---------------------------------------------------------------------------
# The Week 6 -> Week 7 seam
# ---------------------------------------------------------------------------
def test_a_quiet_window_produces_a_no_op_card(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S5")
    card, written = evaluate_event(drift_db, event, policy)

    assert written
    assert card.action == NO_OP
    assert card.rule_id == 1
    assert card.trigger.drift_event_id == event.event_id


def test_a_severe_window_produces_a_full_retrain_card(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("num_lab_procedures", 0.2777),))
    card, _ = evaluate_event(drift_db, event, policy)

    assert card.action == FULL_RETRAIN
    assert card.rule_id == 4
    assert card.disposition == ESCALATE_HUMAN


def test_prediction_drift_alone_produces_a_full_retrain_card(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S3", prediction_drift=True)
    card, _ = evaluate_event(drift_db, event, policy)

    assert card.action == FULL_RETRAIN
    assert card.trigger.prediction_drift is True


def test_a_window_is_decided_from_the_measurements_not_re_measured(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("num_lab_procedures", 0.2777),))
    card, _ = evaluate_event(drift_db, event, policy)

    assert card.trigger.max_psi == event.max_psi
    assert card.trigger.breaching_features[0].psi == 0.2777
    assert card.trigger.measured_thresholds == event.policy_thresholds


# ---------------------------------------------------------------------------
# The Week 6 natural control floor, under the documented policy
# ---------------------------------------------------------------------------
def test_the_uncalibrated_policy_escalates_on_the_control(drift_db, policy_v1):
    """What policy-v1 makes of Week 6's measured S5 control -- the finding.

    Window 0 is quiet -> rule 1 -> NO_OP.
    Window 1 has two mild breaches in their first window -> rule 2 -> ALERT_ONLY.
    Window 2 has the same two features again -> rule 3 -> INCREMENTAL_RETRAIN.

    The third line is why policy-v2 exists: at §3.8's uncalibrated 0.10 the
    untouched control reaches a retrain recommendation, because the drift in
    `payer_code` and `medical_specialty` is real and persistent rather than
    noisy. The rule table is not at fault -- rule 3 is doing exactly what it
    says -- so the remedy the roadmap prescribes is the threshold calibration,
    not a rule change. This test keeps the original behaviour on the record.
    """
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, policy_v1)

    assert [card.action for card in cards(drift_db, "S5")] == [
        NO_OP,
        ALERT_ONLY,
        INCREMENTAL_RETRAIN,
    ]
    assert [card.rule_id for card in cards(drift_db, "S5")] == ["1", "2", "3"]


def test_the_calibrated_policy_is_silent_on_the_control(drift_db, active_policy):
    """The Week 7 deliverable: "`NO_OP` on control", on the same measured rows.

    Nothing about the evidence changed -- these are Week 6's numbers, and the
    monitor still records `payer_code` and `medical_specialty` as breaching at
    its own 0.10. What changed is the threshold the *policy* deems actionable,
    calibrated on this very stream exactly as the roadmap's Week 6 risk line
    and RISK_ANALYSIS.md §2 require.
    """
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, active_policy)

    assert [card.action for card in cards(drift_db, "S5")] == [NO_OP, NO_OP, NO_OP]
    assert [card.rule_id for card in cards(drift_db, "S5")] == ["1", "1", "1"]
    assert all(card.breaching_feature_count == 0 for card in cards(drift_db, "S5"))


def test_the_control_evidence_is_untouched_by_the_calibration(drift_db, active_policy):
    """Calibrating the decision must not quietly edit the measurement.

    The drift rows still carry `payer_code` and `medical_specialty` above 0.10,
    and the card still records the monitor's threshold beside the policy's, so
    the gap between "observed" and "actionable" is on the audit trail.
    """
    events = seed_stream(drift_db, "S5", CONTROL_WINDOWS)
    breaching = [stat["feature"] for stat in events[2].feature_stats if stat["psi"] >= 0.10]
    assert sorted(breaching) == ["medical_specialty", "payer_code"]
    assert events[2].breaching_feature_count == 2

    card, _ = evaluate_event(drift_db, events[2], active_policy)
    assert card.trigger.measured_thresholds["psi_breach"] == 0.10
    assert card.policy_thresholds["psi_breach"] == active_policy.psi_breach
    assert card.trigger.max_psi == events[2].max_psi


def test_the_control_never_reaches_a_full_retrain(drift_db, policy):
    """Rule 4 needs PSI 0.25 or prediction drift; the control has neither."""
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, policy)

    assert all(card.action != FULL_RETRAIN for card in cards(drift_db, "S5"))
    assert all(card.prediction_drift is False for card in cards(drift_db, "S5"))


def test_any_control_escalation_is_shadow_bound_not_a_silent_promotion(drift_db, policy):
    """Autonomy stops at shadow under every policy: at worst a queued challenger."""
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, policy)

    for card in cards(drift_db, "S5"):
        assert card.disposition in {"NONE", "AUTO_PROCEED_SHADOW"}
        assert card.action != FULL_RETRAIN


def test_the_control_and_an_injected_scenario_are_distinguishable(drift_db, policy):
    """The benchmark is only useful if S1 and S5 do not decide the same way."""
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, policy)
    for event in seed_stream(drift_db, "S1", COVARIATE_WINDOWS):
        evaluate_event(drift_db, event, policy)

    control = {card.action for card in cards(drift_db, "S5")}
    injected = {card.action for card in cards(drift_db, "S1")}
    assert injected == {FULL_RETRAIN}
    assert FULL_RETRAIN not in control


def test_the_control_confidence_stays_well_below_the_auto_proceed_line(drift_db, policy):
    for event in seed_stream(drift_db, "S5", CONTROL_WINDOWS):
        evaluate_event(drift_db, event, policy)

    assert all(card.confidence < policy.auto_proceed_confidence for card in cards(drift_db, "S5"))


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_evaluating_the_same_window_twice_writes_one_card(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))

    first, written_first = evaluate_event(drift_db, event, policy)
    second, written_second = evaluate_event(drift_db, event, policy)

    assert written_first and not written_second
    assert first.card_id == second.card_id
    assert len(cards(drift_db)) == 1


def test_a_re_evaluation_does_not_change_the_recorded_decision(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    evaluate_event(drift_db, event, policy)
    before = cards(drift_db)[0].created_at

    evaluate_event(drift_db, event, policy)
    assert cards(drift_db)[0].created_at == before


def test_re_running_the_evaluator_over_a_backlog_is_a_no_op(drift_db, policy):
    seed_stream(drift_db, "S5", CONTROL_WINDOWS)

    first = evaluate_pending(drift_db, policy)
    second = evaluate_pending(drift_db, policy)

    assert [written for _, written in first] == [True, True, True]
    assert second == []
    assert len(cards(drift_db)) == 3


def test_a_duplicate_card_id_is_refused_rather_than_replacing_evidence(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy)

    with pytest.raises(DecisionPersistenceError):
        persist_card(drift_db, build_card_row(card))


def test_a_failed_write_rolls_back_and_leaves_the_session_usable(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy)
    with pytest.raises(DecisionPersistenceError):
        persist_card(drift_db, build_card_row(card))

    other = make_event(drift_db, index=1, scenario="S1", breaches=(("a", 0.40),))
    evaluate_event(drift_db, other, policy)
    assert len(cards(drift_db)) == 2


def test_a_dry_run_decides_without_writing(drift_db, policy):
    seed_stream(drift_db, "S5", CONTROL_WINDOWS)
    results = evaluate_pending(drift_db, policy, persist=False)

    assert len(results) == 3
    assert all(not written for _, written in results)
    assert cards(drift_db) == []


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def test_a_backlog_is_evaluated_oldest_first_so_persistence_is_correct(drift_db, policy):
    """Deciding window 2 before window 1 would give it a history that lacks window 1."""
    mild = mild_psi(policy)
    windows = [(("payer_code", mild),), (("payer_code", mild),), (("payer_code", mild),)]
    events = seed_stream(drift_db, "S5", windows)
    assert [event.window_start for event in events] == sorted(
        event.window_start for event in events
    )

    evaluate_pending(drift_db, policy)
    persisted = cards(drift_db, "S5")
    assert [card.card_json["trigger"]["consecutive_breaching_windows"] for card in persisted] == [
        1,
        2,
        3,
    ]


def test_one_stream_can_be_evaluated_without_touching_another(drift_db, policy):
    seed_stream(drift_db, "S5", CONTROL_WINDOWS)
    seed_stream(drift_db, "S1", COVARIATE_WINDOWS)

    evaluate_pending(drift_db, policy, scenario="S1")
    assert {card.scenario for card in cards(drift_db)} == {"S1"}


# ---------------------------------------------------------------------------
# What lands in the table
# ---------------------------------------------------------------------------
def test_the_row_projects_the_card_without_disagreeing_with_it(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy)
    row = cards(drift_db)[0]

    assert row.card_id == card.card_id
    assert row.action == card.action == row.card_json["action"]
    assert row.disposition == card.disposition
    assert row.confidence == card.confidence
    assert row.policy_version == card.policy_version
    assert row.rule_id == str(card.rule_id)
    assert row.status == card.status
    assert row.breaching_feature_count == len(card.trigger.breaching_features)


def test_the_row_carries_the_lineage_back_to_the_window_and_the_model(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    evaluate_event(drift_db, event, policy, data_version="dvc-train-hash")
    row = cards(drift_db)[0]

    assert row.drift_event_id == event.event_id
    assert row.scenario == event.scenario
    # SQLite drops the offset on read-back, so the bounds are compared naively;
    # in Postgres both sides are timestamptz. Either way they must be the window's.
    assert row.window_start.replace(tzinfo=None) == event.window_start.replace(tzinfo=None)
    assert row.window_end.replace(tzinfo=None) == event.window_end.replace(tzinfo=None)
    assert row.model_version == "1"
    assert row.data_version == "dvc-reference-hash"
    assert row.candidate_data_version == "dvc-train-hash"


def test_the_stored_card_is_a_complete_valid_card(drift_db, policy):
    from loop.engine.card import DecisionCard

    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy)

    restored = DecisionCard.model_validate(cards(drift_db)[0].card_json)
    assert restored.model_dump() == card.model_dump()


def test_find_existing_locates_the_card_for_a_window_and_policy(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    evaluate_event(drift_db, event, policy)

    assert find_existing(drift_db, event.event_id, policy.version) is not None
    assert find_existing(drift_db, event.event_id, "policy-v99") is None
    assert find_existing(drift_db, "de-nope", policy.version) is None


def test_record_decision_reports_whether_it_wrote(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("a", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy, persist=False)

    _, created = record_decision(drift_db, card)
    assert created
    _, created_again = record_decision(drift_db, card)
    assert not created_again


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
def test_the_table_has_no_column_that_could_hold_a_patient_row():
    columns = set(DecisionCardRow.__table__.columns.keys())
    for forbidden in ("patient_nbr", "encounter_id", "readmitted_30d", "payload", "input_hash"):
        assert forbidden not in columns


def test_a_persisted_card_carries_only_names_statistics_and_metadata(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("num_lab_procedures", 0.40),))
    evaluate_event(drift_db, event, policy)
    blob = cards(drift_db)[0].card_json

    for breach in blob["trigger"]["breaching_features"]:
        assert set(breach) == {"feature", "psi", "ks_p"}
    for forbidden in ("patient_nbr", "encounter_id", "readmitted", "caller", "input_hash"):
        assert forbidden not in str(blob)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_the_summary_names_the_rule_the_action_and_the_confidence(drift_db, policy):
    event = make_event(drift_db, index=0, scenario="S1", breaches=(("num_lab_procedures", 0.40),))
    card, _ = evaluate_event(drift_db, event, policy)
    line = summarise(card)

    assert "S1" in line
    assert card.card_id in line
    assert "FULL_RETRAIN" in line
    assert "num_lab_procedures" in line
