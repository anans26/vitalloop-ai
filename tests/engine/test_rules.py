"""Every rule branch and every disposition in the §3.8 table.

IMPLEMENTATION_ROADMAP.md Week 7 asks for "exhaustive unit tests -- every rule
branch, every disposition". This file is that: each of the six rules is tested
for firing, for *not* firing on each of its conditions, and for losing to the
rules above it in precedence.
"""

import pytest

from loop.engine.confidence import compute_confidence
from loop.engine.engine import decide
from loop.engine.evidence import MATURED_LABELS
from loop.engine.rules import (
    ALERT_ONLY,
    AUTO_PROCEED_SHADOW,
    DISPOSITION_NONE,
    ESCALATE_HUMAN,
    FULL_RETRAIN,
    INCREMENTAL_RETRAIN,
    NO_OP,
    RULE_UNCOVERED,
    apply_policy,
)
from tests.engine.conftest import FIXED_NOW, feature_stat, make_evidence


def outcome(evidence, policy):
    """The rule table's verdict, without building a whole card."""
    return apply_policy(evidence, policy, compute_confidence(evidence, policy))


# ---------------------------------------------------------------------------
# Rule 1 -- no drift
# ---------------------------------------------------------------------------
def test_rule_1_fires_when_nothing_breached_and_scores_held(policy):
    result = outcome(make_evidence(stats=(feature_stat("race", 0.01),)), policy)
    assert result.rule_id == 1
    assert result.action == NO_OP
    assert result.disposition == DISPOSITION_NONE


def test_rule_1_fires_on_a_window_with_no_features_at_all(policy):
    assert outcome(make_evidence(), policy).action == NO_OP


def test_rule_1_does_not_fire_when_a_feature_breached(policy):
    assert outcome(make_evidence(stats=(feature_stat("race", 0.11),)), policy).rule_id != 1


def test_rule_1_does_not_fire_on_prediction_drift_alone(policy):
    """Silence is only a decision when the scores held too."""
    evidence = make_evidence(
        stats=(feature_stat("race", 0.01),), prediction_drift=True, prediction_psi=0.18
    )
    assert outcome(evidence, policy).rule_id == 4


def test_a_feature_exactly_at_the_breach_threshold_counts_as_breaching(policy):
    assert outcome(make_evidence(stats=(feature_stat("race", 0.10),)), policy).rule_id == 2


# ---------------------------------------------------------------------------
# Rule 2 -- first-window mild breach
# ---------------------------------------------------------------------------
def test_rule_2_fires_for_one_mild_feature_in_its_first_window(policy):
    result = outcome(make_evidence(stats=(feature_stat("payer_code", 0.12),)), policy)
    assert result.rule_id == 2
    assert result.action == ALERT_ONLY
    assert result.disposition == DISPOSITION_NONE


def test_rule_2_fires_at_the_two_feature_limit(policy):
    evidence = make_evidence(
        stats=(feature_stat("payer_code", 0.12), feature_stat("medical_specialty", 0.13))
    )
    assert outcome(evidence, policy).rule_id == 2


def test_rule_2_does_not_fire_above_the_feature_limit(policy):
    """Three mild features in a first window match no rule in the table."""
    evidence = make_evidence(
        stats=(
            feature_stat("a", 0.12),
            feature_stat("b", 0.13),
            feature_stat("c", 0.14),
        )
    )
    assert outcome(evidence, policy).rule_id == RULE_UNCOVERED


def test_rule_2_does_not_fire_when_a_feature_is_severe(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.12), feature_stat("b", 0.30)))
    assert outcome(evidence, policy).rule_id == 4


def test_rule_2_does_not_fire_when_the_scores_moved(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.12),), prediction_drift=True, prediction_psi=0.15
    )
    assert outcome(evidence, policy).rule_id == 4


def test_rule_2_does_not_fire_once_the_breach_persists(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.12),), consecutive=2, persistent=("a",))
    assert outcome(evidence, policy).rule_id == 3


# ---------------------------------------------------------------------------
# Rule 3 -- persistence
# ---------------------------------------------------------------------------
def test_rule_3_fires_on_the_second_consecutive_window(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.12),), consecutive=2, persistent=("a",))
    result = outcome(evidence, policy)
    assert result.rule_id == 3
    assert result.action == INCREMENTAL_RETRAIN
    assert result.disposition == AUTO_PROCEED_SHADOW


def test_rule_3_names_the_features_that_persisted(policy):
    evidence = make_evidence(
        stats=(feature_stat("payer_code", 0.12), feature_stat("medical_specialty", 0.13)),
        consecutive=3,
        persistent=("medical_specialty", "payer_code"),
    )
    result = outcome(evidence, policy)
    assert "payer_code" in result.rationale
    assert "medical_specialty" in result.rationale


def test_rule_3_does_not_fire_without_a_persisting_feature(policy):
    """A run of breaching windows with nothing in common is not persistence."""
    evidence = make_evidence(stats=(feature_stat("a", 0.12),), consecutive=3, persistent=())
    assert outcome(evidence, policy).rule_id != 3


def test_rule_3_loses_to_rule_4_when_a_feature_is_severe(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.30),), consecutive=3, persistent=("a",))
    assert outcome(evidence, policy).rule_id == 4


# ---------------------------------------------------------------------------
# Rule 4 -- severe drift or prediction drift
# ---------------------------------------------------------------------------
def test_rule_4_fires_at_the_severe_threshold(policy):
    result = outcome(make_evidence(stats=(feature_stat("a", 0.25),)), policy)
    assert result.rule_id == 4
    assert result.action == FULL_RETRAIN


def test_rule_4_fires_on_prediction_drift_with_no_feature_breach(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.02),), prediction_drift=True, prediction_psi=0.20
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 4
    assert "prediction drift" in result.rationale


def test_rule_4_escalates_when_confidence_is_below_the_threshold(policy):
    """Leading indicators alone rarely clear 0.75, so escalation is the norm."""
    evidence = make_evidence(stats=(feature_stat("a", 0.30),))
    result = outcome(evidence, policy)
    assert result.disposition == ESCALATE_HUMAN
    assert compute_confidence(evidence, policy).confidence < policy.auto_proceed_confidence


def test_rule_4_auto_proceeds_when_confidence_reaches_the_threshold(policy):
    """The other disposition branch: maximal evidence on every term."""
    evidence = make_evidence(
        stats=tuple(feature_stat(f"f{i}", 0.60) for i in range(25)),
        consecutive=3,
        persistent=tuple(f"f{i}" for i in range(25)),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.0,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 4
    assert result.disposition == AUTO_PROCEED_SHADOW
    assert compute_confidence(evidence, policy).confidence >= policy.auto_proceed_confidence


# ---------------------------------------------------------------------------
# Rule 5 -- matured-label regression
# ---------------------------------------------------------------------------
def test_rule_5_fires_when_matured_labels_confirm_a_drop(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.01),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.05,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 5
    assert result.action == FULL_RETRAIN


def test_rule_5_always_escalates_even_at_maximum_confidence(policy):
    """WORKFLOW.md §4: performance drops are never auto-handled silently."""
    evidence = make_evidence(
        stats=tuple(feature_stat(f"f{i}", 0.90) for i in range(25)),
        consecutive=3,
        persistent=tuple(f"f{i}" for i in range(25)),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.20,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 5
    assert result.disposition == ESCALATE_HUMAN
    assert compute_confidence(evidence, policy).confidence >= policy.auto_proceed_confidence


def test_rule_5_does_not_fire_at_or_below_the_drop_threshold(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.01),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=policy.matured_auroc_drop,
    )
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_does_not_fire_while_labels_are_immature(policy):
    """Every real Week 7 window is in this state: no matured labels exist yet."""
    evidence = make_evidence(stats=(feature_stat("a", 0.01),), matured_auroc_drop=0.20)
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_does_not_fire_when_matured_labels_carry_no_measurement(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.01),), label_maturity=MATURED_LABELS)
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_outranks_rule_4(policy):
    """Confirmed regression beats leading indicators -- RESEARCH_NOVELTY.md C3."""
    evidence = make_evidence(
        stats=(feature_stat("a", 0.40),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.10,
    )
    assert outcome(evidence, policy).rule_id == 5


# ---------------------------------------------------------------------------
# Rule 6 -- the downgrade
# ---------------------------------------------------------------------------
def test_rule_6_downgrades_a_full_retrain_inside_the_cooldown(policy):
    evidence = make_evidence(stats=(feature_stat("a", 0.40),), days_since_last_retrain=1.0)
    result = outcome(evidence, policy)
    assert result.rule_id == 6
    assert result.action == ALERT_ONLY
    assert result.disposition == ESCALATE_HUMAN
    assert result.downgraded_from == FULL_RETRAIN


def test_rule_6_downgrades_an_incremental_retrain_too(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.12),),
        consecutive=2,
        persistent=("a",),
        days_since_last_retrain=0.5,
    )
    result = outcome(evidence, policy)
    assert result.downgraded_from == INCREMENTAL_RETRAIN
    assert result.action == ALERT_ONLY


def test_rule_6_fires_on_an_exhausted_budget_alone(policy):
    """§3.8 joins the two conditions with "or", so either is sufficient."""
    evidence = make_evidence(
        stats=(feature_stat("a", 0.40),), retrains_in_cooldown=policy.retrain_budget
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 6
    assert "budget" in result.rationale


def test_rule_6_does_not_fire_outside_the_cooldown_with_budget_left(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.40),),
        days_since_last_retrain=policy.retrain_cooldown_days + 1,
        retrains_in_cooldown=0,
    )
    assert outcome(evidence, policy).rule_id == 4


def test_rule_6_never_downgrades_a_no_op(policy):
    """There is nothing to downgrade about silence."""
    evidence = make_evidence(
        stats=(feature_stat("a", 0.01),), days_since_last_retrain=0.0, retrains_in_cooldown=99
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 1
    assert result.action == NO_OP


def test_rule_6_never_downgrades_an_alert_only(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.12),), days_since_last_retrain=0.0, retrains_in_cooldown=99
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 2
    assert result.action == ALERT_ONLY


def test_rule_6_downgrades_rule_5_as_well(policy):
    """The table says "downgrade", without excepting the escalating rule."""
    evidence = make_evidence(
        stats=(feature_stat("a", 0.01),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.10,
        days_since_last_retrain=0.0,
    )
    result = outcome(evidence, policy)
    assert result.downgraded_from == FULL_RETRAIN
    assert result.disposition == ESCALATE_HUMAN


# ---------------------------------------------------------------------------
# The uncovered region
# ---------------------------------------------------------------------------
def test_uncovered_breaches_take_the_weakest_non_silent_action(policy):
    evidence = make_evidence(
        stats=(feature_stat("a", 0.12), feature_stat("b", 0.13), feature_stat("c", 0.14))
    )
    result = outcome(evidence, policy)
    assert result.rule_id == RULE_UNCOVERED
    assert result.action == policy.uncovered_breach_action == ALERT_ONLY


def test_the_uncovered_region_never_retrains(policy):
    evidence = make_evidence(stats=tuple(feature_stat(f"f{i}", 0.12) for i in range(10)))
    assert not outcome(evidence, policy).requires_retrain


# ---------------------------------------------------------------------------
# Every action and disposition is reachable
# ---------------------------------------------------------------------------
def test_every_documented_action_is_reachable(policy):
    reached = {
        outcome(evidence, policy).action
        for evidence in (
            make_evidence(stats=(feature_stat("a", 0.01),)),
            make_evidence(stats=(feature_stat("a", 0.12),)),
            make_evidence(stats=(feature_stat("a", 0.12),), consecutive=2, persistent=("a",)),
            make_evidence(stats=(feature_stat("a", 0.40),)),
        )
    }
    assert reached == {NO_OP, ALERT_ONLY, INCREMENTAL_RETRAIN, FULL_RETRAIN}


def test_every_documented_disposition_is_reachable(policy):
    reached = {
        outcome(evidence, policy).disposition
        for evidence in (
            make_evidence(stats=(feature_stat("a", 0.01),)),
            make_evidence(stats=(feature_stat("a", 0.12),), consecutive=2, persistent=("a",)),
            make_evidence(stats=(feature_stat("a", 0.40),)),
        )
    }
    assert reached == {DISPOSITION_NONE, AUTO_PROCEED_SHADOW, ESCALATE_HUMAN}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("psi", [0.01, 0.12, 0.30, 0.60])
def test_the_same_evidence_always_decides_the_same_way(policy, psi):
    evidence = make_evidence(stats=(feature_stat("a", psi),))
    first = decide(evidence, policy, now=FIXED_NOW)
    second = decide(evidence, policy, now=FIXED_NOW)
    assert first.model_dump() == second.model_dump()


def test_breaching_features_are_listed_worst_first(policy):
    """A card that reorders its own evidence is not re-derivable."""
    evidence = make_evidence(
        stats=(feature_stat("b", 0.12), feature_stat("a", 0.40), feature_stat("c", 0.20))
    )
    card = decide(evidence, policy, now=FIXED_NOW)
    assert [breach.feature for breach in card.trigger.breaching_features] == ["a", "c", "b"]
