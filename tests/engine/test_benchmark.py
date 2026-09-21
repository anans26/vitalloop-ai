"""The S1-S5 benchmark, decided: the Week 7 deliverable, asserted end to end.

IMPLEMENTATION_ROADMAP.md Week 7 deliverable:

    "all five drift scenarios S1-S5 produce the correct card
     (incl. `NO_OP` on control)"

PROJECT_DESIGN.md §13 says what "correct" means in numbers:

    "drift detection within one monitoring window on S1-S3;
     zero false triggers on the control"

The PSI values below are **the ones Week 6 actually measured** against the
frozen-evaluation reference, three 2000-row windows per scenario, copied from
the live `drift_events` rows. Nothing is rounded toward a nicer answer and
nothing is omitted: if a feature breached the monitor's 0.10, it is here.
Regenerate them with `python -m scenarios.run_scenario <S> --windows 3`.

S4 shares S5's numbers by construction -- it degrades the feature/label
relationship without touching a feature -- so input drift must decide the two
identically. That is asserted rather than assumed.
"""

import pytest
from sqlalchemy import select

from db.models import DecisionCard as DecisionCardRow
from loop.engine.evaluate import evaluate_pending
from loop.engine.rules import FULL_RETRAIN, NO_OP
from tests.engine.test_evaluate import make_event

# Measured Week 6 output. Each entry is one window: the features that reached
# the monitor's 0.10 breach line, with their PSI, plus whether the champion's
# score distribution moved.
BENCHMARK = {
    "S1": [  # covariate shift: num_lab_procedures / num_medications +20%
        ((("num_lab_procedures", 0.2777), ("num_medications", 0.1423)), False),
        (
            (
                ("num_lab_procedures", 0.2713),
                ("num_medications", 0.1317),
                ("payer_code", 0.1278),
                ("medical_specialty", 0.1249),
            ),
            False,
        ),
        (
            (
                ("num_lab_procedures", 0.2998),
                ("medical_specialty", 0.1663),
                ("num_medications", 0.1224),
                ("payer_code", 0.1125),
            ),
            False,
        ),
    ],
    "S2": [  # coding change: HbA1c recorded for 40% of previously blank encounters
        ((("A1Cresult", 0.4086),), False),
        ((("A1Cresult", 0.4231), ("payer_code", 0.1278), ("medical_specialty", 0.1249)), False),
        ((("A1Cresult", 0.3931), ("medical_specialty", 0.1663), ("payer_code", 0.1125)), False),
    ],
    "S3": [  # prevalence shift: resampled toward elderly, high-utilisation patients
        ((("age", 0.3717), ("payer_code", 0.1438)), True),
        (
            (
                ("age", 0.3984),
                ("medical_specialty", 0.2016),
                ("payer_code", 0.1855),
                ("number_inpatient", 0.1600),
            ),
            True,
        ),
        (
            (
                ("age", 0.4519),
                ("number_inpatient", 0.2818),
                ("payer_code", 0.2449),
                ("medical_specialty", 0.2313),
                ("number_diagnoses", 0.1289),
            ),
            True,
        ),
    ],
    "S4": [  # label drift: features untouched, so identical to the control
        ((), False),
        ((("payer_code", 0.1278), ("medical_specialty", 0.1249)), False),
        ((("medical_specialty", 0.1663), ("payer_code", 0.1125)), False),
    ],
    "S5": [  # the no-drift control: the untouched stream
        ((), False),
        ((("payer_code", 0.1278), ("medical_specialty", 0.1249)), False),
        ((("medical_specialty", 0.1663), ("payer_code", 0.1125)), False),
    ],
}

# The control's worst measured PSI. The calibrated threshold must clear it.
CONTROL_FLOOR = 0.1663

# What the injected scenarios move, and by how much, in their first window.
INJECTED_SIGNALS = {
    "S1": ("num_lab_procedures", 0.2777),
    "S2": ("A1Cresult", 0.4086),
    "S3": ("age", 0.3717),
}


def seed(session, scenario):
    return [
        make_event(
            session,
            index=index,
            scenario=scenario,
            breaches=breaches,
            prediction_drift=prediction_drift,
        )
        for index, (breaches, prediction_drift) in enumerate(BENCHMARK[scenario])
    ]


def decide_scenario(session, scenario, policy):
    seed(session, scenario)
    evaluate_pending(session, policy, scenario=scenario)
    return list(
        session.scalars(
            select(DecisionCardRow)
            .where(DecisionCardRow.scenario == scenario)
            .order_by(DecisionCardRow.window_start)
        )
    )


# ---------------------------------------------------------------------------
# The deliverable, scenario by scenario
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["S1", "S2", "S3"])
def test_an_injected_scenario_is_caught_in_its_first_window(drift_db, active_policy, scenario):
    """§13: "drift detection within one monitoring window on S1-S3"."""
    cards = decide_scenario(drift_db, scenario, active_policy)

    assert [card.action for card in cards] == [FULL_RETRAIN] * 3
    assert [card.rule_id for card in cards] == ["4"] * 3
    assert cards[0].window_start == min(card.window_start for card in cards)


@pytest.mark.parametrize("scenario", ["S4", "S5"])
def test_an_uninjected_scenario_produces_no_op_on_every_window(drift_db, active_policy, scenario):
    """§13: "zero false triggers on the control", and the Week 7 deliverable."""
    cards = decide_scenario(drift_db, scenario, active_policy)

    assert [card.action for card in cards] == [NO_OP] * 3
    assert [card.rule_id for card in cards] == ["1"] * 3
    assert all(card.breaching_feature_count == 0 for card in cards)
    assert all(card.disposition == "NONE" for card in cards)
    assert all(card.status == "CLOSED" for card in cards)


def test_label_drift_decides_identically_to_the_control(drift_db, active_policy):
    """S4 moves labels, not features, so input drift must not react to it."""
    control = decide_scenario(drift_db, "S5", active_policy)
    labels = decide_scenario(drift_db, "S4", active_policy)

    assert [card.action for card in labels] == [card.action for card in control]
    assert [card.confidence for card in labels] == [card.confidence for card in control]


def test_each_injected_scenario_is_flagged_by_the_feature_it_moved(drift_db, active_policy):
    """The benchmark is only evidence if the card names the right cause."""
    for scenario, (feature, _) in INJECTED_SIGNALS.items():
        cards = decide_scenario(drift_db, scenario, active_policy)
        first = cards[0].card_json["trigger"]["breaching_features"]
        assert first[0]["feature"] == feature


def test_the_prevalence_shift_is_the_only_scenario_that_moves_the_scores(drift_db, active_policy):
    """S3 changes who is being scored, so prediction drift is expected there alone."""
    moved = {
        scenario
        for scenario in BENCHMARK
        if any(card.prediction_drift for card in decide_scenario(drift_db, scenario, active_policy))
    }
    assert moved == {"S3"}


def test_no_benchmark_card_auto_proceeds_on_leading_indicators(drift_db, active_policy):
    """Confidence tops out at 0.875 without matured labels, so a human sees every retrain."""
    for scenario in BENCHMARK:
        for card in decide_scenario(drift_db, scenario, active_policy):
            assert card.disposition != "AUTO_PROCEED_SHADOW"
            if card.action == FULL_RETRAIN:
                assert card.disposition == "ESCALATE_HUMAN"
                assert card.confidence < active_policy.auto_proceed_confidence


# ---------------------------------------------------------------------------
# Why the calibration separates them
# ---------------------------------------------------------------------------
def test_the_calibrated_threshold_sits_between_the_control_and_every_injected_signal(
    active_policy,
):
    """The separation the calibration relies on, stated as an inequality.

    Every uninjected feature stays below the threshold; every injected signal
    clears it. If a future policy change broke either side, this fails before
    the benchmark's dispositions silently change.
    """
    control_values = [
        psi
        for scenario in ("S4", "S5")
        for breaches, _ in BENCHMARK[scenario]
        for _, psi in breaches
    ]
    assert max(control_values) == CONTROL_FLOOR
    assert CONTROL_FLOOR < active_policy.psi_breach

    for feature, psi in INJECTED_SIGNALS.values():
        assert psi >= active_policy.psi_severe, feature


def test_the_control_rows_still_record_the_drift_the_monitor_measured(drift_db, active_policy):
    """The calibration changed what is actionable, never what was observed."""
    events = seed(drift_db, "S5")
    assert events[2].breaching_feature_count == 2
    assert events[2].max_psi == CONTROL_FLOOR
