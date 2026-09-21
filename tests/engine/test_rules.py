"""Every rule branch and every disposition in the §3.8 table.

IMPLEMENTATION_ROADMAP.md Week 7 asks for "exhaustive unit tests -- every rule
branch, every disposition". This file is that: each of the six rules is tested
for firing, for *not* firing on each of its conditions, and for losing to the
rules above it in precedence.

The `policy` fixture is parameterised over every shipped policy version, and
every PSI here is expressed relative to that policy's own thresholds, so the
whole table is re-verified whenever a policy is added or a threshold moves.
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
from tests.engine.conftest import (
    FIXED_NOW,
    evidence_for,
    feature_stat,
    mild_psi,
    quiet_psi,
    severe_psi,
)


def outcome(evidence, policy):
    """The rule table's verdict, without building a whole card."""
    return apply_policy(evidence, policy, compute_confidence(evidence, policy))


# ---------------------------------------------------------------------------
# Rule 1 -- no drift
# ---------------------------------------------------------------------------
def test_rule_1_fires_when_nothing_breached_and_scores_held(policy):
    evidence = evidence_for(policy, stats=(feature_stat("race", quiet_psi(policy)),))
    result = outcome(evidence, policy)
    assert result.rule_id == 1
    assert result.action == NO_OP
    assert result.disposition == DISPOSITION_NONE


def test_rule_1_fires_on_a_window_with_no_features_at_all(policy):
    assert outcome(evidence_for(policy), policy).action == NO_OP


def test_rule_1_does_not_fire_when_a_feature_breached(policy):
    evidence = evidence_for(policy, stats=(feature_stat("race", mild_psi(policy)),))
    assert outcome(evidence, policy).rule_id != 1


def test_rule_1_does_not_fire_on_prediction_drift_alone(policy):
    """Silence is only a decision when the scores held too."""
    evidence = evidence_for(
        policy,
        stats=(feature_stat("race", quiet_psi(policy)),),
        prediction_drift=True,
        prediction_psi=0.18,
    )
    assert outcome(evidence, policy).rule_id == 4


def test_a_feature_exactly_at_the_breach_threshold_counts_as_breaching(policy):
    """The threshold is inclusive: `psi >= psi_breach`."""
    evidence = evidence_for(policy, stats=(feature_stat("race", policy.psi_breach),))
    assert outcome(evidence, policy).rule_id == 2


# ---------------------------------------------------------------------------
# Rule 2 -- first-window mild breach
# ---------------------------------------------------------------------------
def test_rule_2_fires_for_one_mild_feature_in_its_first_window(policy):
    evidence = evidence_for(policy, stats=(feature_stat("payer_code", mild_psi(policy)),))
    result = outcome(evidence, policy)
    assert result.rule_id == 2
    assert result.action == ALERT_ONLY
    assert result.disposition == DISPOSITION_NONE


def test_rule_2_fires_at_the_two_feature_limit(policy):
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("payer_code", mild_psi(policy)),
            feature_stat("medical_specialty", mild_psi(policy, 0.001)),
        ),
    )
    assert outcome(evidence, policy).rule_id == 2


def test_rule_2_does_not_fire_above_the_feature_limit(policy):
    """Three mild features in a first window match no rule in the table."""
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("a", mild_psi(policy)),
            feature_stat("b", mild_psi(policy, 0.001)),
            feature_stat("c", mild_psi(policy, 0.002)),
        ),
    )
    assert outcome(evidence, policy).rule_id == RULE_UNCOVERED


def test_rule_2_does_not_fire_when_a_feature_is_severe(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", mild_psi(policy)), feature_stat("b", severe_psi(policy))),
    )
    assert outcome(evidence, policy).rule_id == 4


def test_rule_2_does_not_fire_when_the_scores_moved(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", mild_psi(policy)),),
        prediction_drift=True,
        prediction_psi=0.15,
    )
    assert outcome(evidence, policy).rule_id == 4


def test_rule_2_does_not_fire_once_the_breach_persists(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", mild_psi(policy)),), consecutive=2, persistent=("a",)
    )
    assert outcome(evidence, policy).rule_id == 3


# ---------------------------------------------------------------------------
# Rule 3 -- persistence
# ---------------------------------------------------------------------------
def test_rule_3_fires_on_the_second_consecutive_window(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", mild_psi(policy)),), consecutive=2, persistent=("a",)
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 3
    assert result.action == INCREMENTAL_RETRAIN
    assert result.disposition == AUTO_PROCEED_SHADOW


def test_rule_3_names_the_features_that_persisted(policy):
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("payer_code", mild_psi(policy)),
            feature_stat("medical_specialty", mild_psi(policy, 0.001)),
        ),
        consecutive=3,
        persistent=("medical_specialty", "payer_code"),
    )
    result = outcome(evidence, policy)
    assert "payer_code" in result.rationale
    assert "medical_specialty" in result.rationale


def test_rule_3_does_not_fire_without_a_persisting_feature(policy):
    """A run of breaching windows with nothing in common is not persistence."""
    evidence = evidence_for(
        policy, stats=(feature_stat("a", mild_psi(policy)),), consecutive=3, persistent=()
    )
    assert outcome(evidence, policy).rule_id != 3


def test_rule_3_loses_to_rule_4_when_a_feature_is_severe(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", severe_psi(policy)),), consecutive=3, persistent=("a",)
    )
    assert outcome(evidence, policy).rule_id == 4


# ---------------------------------------------------------------------------
# Rule 4 -- severe drift or prediction drift
# ---------------------------------------------------------------------------
def test_rule_4_fires_at_the_severe_threshold(policy):
    """Inclusive again: `psi >= psi_severe`."""
    evidence = evidence_for(policy, stats=(feature_stat("a", policy.psi_severe),))
    result = outcome(evidence, policy)
    assert result.rule_id == 4
    assert result.action == FULL_RETRAIN


def test_rule_4_fires_on_prediction_drift_with_no_feature_breach(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", quiet_psi(policy)),),
        prediction_drift=True,
        prediction_psi=0.20,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 4
    assert "prediction drift" in result.rationale


def test_rule_4_escalates_when_confidence_is_below_the_threshold(policy):
    """Leading indicators alone rarely clear 0.75, so escalation is the norm."""
    evidence = evidence_for(policy, stats=(feature_stat("a", severe_psi(policy)),))
    result = outcome(evidence, policy)
    assert result.disposition == ESCALATE_HUMAN
    assert compute_confidence(evidence, policy).confidence < policy.auto_proceed_confidence


def test_rule_4_auto_proceeds_when_confidence_reaches_the_threshold(policy):
    """The other disposition branch: maximal evidence on every term."""
    evidence = evidence_for(
        policy,
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
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", quiet_psi(policy)),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.05,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 5
    assert result.action == FULL_RETRAIN


def test_rule_5_always_escalates_even_at_maximum_confidence(policy):
    """WORKFLOW.md §4: performance drops are never auto-handled silently."""
    evidence = evidence_for(
        policy,
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
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", quiet_psi(policy)),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=policy.matured_auroc_drop,
    )
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_does_not_fire_while_labels_are_immature(policy):
    """Every real Week 7 window is in this state: no matured labels exist yet."""
    evidence = evidence_for(
        policy, stats=(feature_stat("a", quiet_psi(policy)),), matured_auroc_drop=0.20
    )
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_does_not_fire_when_matured_labels_carry_no_measurement(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", quiet_psi(policy)),), label_maturity=MATURED_LABELS
    )
    assert outcome(evidence, policy).rule_id == 1


def test_rule_5_outranks_rule_4(policy):
    """Confirmed regression beats leading indicators -- RESEARCH_NOVELTY.md C3."""
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", severe_psi(policy)),),
        label_maturity=MATURED_LABELS,
        matured_auroc_drop=0.10,
    )
    assert outcome(evidence, policy).rule_id == 5


# ---------------------------------------------------------------------------
# Rule 6 -- the downgrade
# ---------------------------------------------------------------------------
def test_rule_6_downgrades_a_full_retrain_inside_the_cooldown(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", severe_psi(policy)),), days_since_last_retrain=1.0
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 6
    assert result.action == ALERT_ONLY
    assert result.disposition == ESCALATE_HUMAN
    assert result.downgraded_from == FULL_RETRAIN


def test_rule_6_downgrades_an_incremental_retrain_too(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", mild_psi(policy)),),
        consecutive=2,
        persistent=("a",),
        days_since_last_retrain=0.5,
    )
    result = outcome(evidence, policy)
    assert result.downgraded_from == INCREMENTAL_RETRAIN
    assert result.action == ALERT_ONLY


def test_rule_6_fires_on_an_exhausted_budget_alone(policy):
    """§3.8 joins the two conditions with "or", so either is sufficient."""
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", severe_psi(policy)),),
        retrains_in_cooldown=policy.retrain_budget,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 6
    assert "budget" in result.rationale


def test_rule_6_does_not_fire_outside_the_cooldown_with_budget_left(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", severe_psi(policy)),),
        days_since_last_retrain=policy.retrain_cooldown_days + 1,
        retrains_in_cooldown=0,
    )
    assert outcome(evidence, policy).rule_id == 4


def test_rule_6_never_downgrades_a_no_op(policy):
    """There is nothing to downgrade about silence."""
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", quiet_psi(policy)),),
        days_since_last_retrain=0.0,
        retrains_in_cooldown=99,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 1
    assert result.action == NO_OP


def test_rule_6_never_downgrades_an_alert_only(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", mild_psi(policy)),),
        days_since_last_retrain=0.0,
        retrains_in_cooldown=99,
    )
    result = outcome(evidence, policy)
    assert result.rule_id == 2
    assert result.action == ALERT_ONLY


def test_rule_6_downgrades_rule_5_as_well(policy):
    """The table says "downgrade", without excepting the escalating rule."""
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", quiet_psi(policy)),),
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
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("a", mild_psi(policy)),
            feature_stat("b", mild_psi(policy, 0.001)),
            feature_stat("c", mild_psi(policy, 0.002)),
        ),
    )
    result = outcome(evidence, policy)
    assert result.rule_id == RULE_UNCOVERED
    assert result.action == policy.uncovered_breach_action == ALERT_ONLY


def test_the_uncovered_region_never_retrains(policy):
    evidence = evidence_for(
        policy,
        stats=tuple(feature_stat(f"f{i}", mild_psi(policy, 0.001 * i)) for i in range(10)),
    )
    assert not outcome(evidence, policy).requires_retrain


# ---------------------------------------------------------------------------
# Every action and disposition is reachable
# ---------------------------------------------------------------------------
def test_every_documented_action_is_reachable(policy):
    reached = {
        outcome(evidence, policy).action
        for evidence in (
            evidence_for(policy, stats=(feature_stat("a", quiet_psi(policy)),)),
            evidence_for(policy, stats=(feature_stat("a", mild_psi(policy)),)),
            evidence_for(
                policy,
                stats=(feature_stat("a", mild_psi(policy)),),
                consecutive=2,
                persistent=("a",),
            ),
            evidence_for(policy, stats=(feature_stat("a", severe_psi(policy)),)),
        )
    }
    assert reached == {NO_OP, ALERT_ONLY, INCREMENTAL_RETRAIN, FULL_RETRAIN}


def test_every_documented_disposition_is_reachable(policy):
    reached = {
        outcome(evidence, policy).disposition
        for evidence in (
            evidence_for(policy, stats=(feature_stat("a", quiet_psi(policy)),)),
            evidence_for(
                policy,
                stats=(feature_stat("a", mild_psi(policy)),),
                consecutive=2,
                persistent=("a",),
            ),
            evidence_for(policy, stats=(feature_stat("a", severe_psi(policy)),)),
        )
    }
    assert reached == {DISPOSITION_NONE, AUTO_PROCEED_SHADOW, ESCALATE_HUMAN}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("band", ["quiet", "mild", "severe"])
def test_the_same_evidence_always_decides_the_same_way(policy, band):
    psi = {"quiet": quiet_psi, "mild": mild_psi, "severe": severe_psi}[band](policy)
    evidence = evidence_for(policy, stats=(feature_stat("a", psi),))
    first = decide(evidence, policy, now=FIXED_NOW)
    second = decide(evidence, policy, now=FIXED_NOW)
    assert first.model_dump() == second.model_dump()


def test_breaching_features_are_listed_worst_first(policy):
    """A card that reorders its own evidence is not re-derivable."""
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("b", mild_psi(policy)),
            feature_stat("a", severe_psi(policy, 0.20)),
            feature_stat("c", severe_psi(policy)),
        ),
    )
    card = decide(evidence, policy, now=FIXED_NOW)
    assert [breach.feature for breach in card.trigger.breaching_features] == ["a", "c", "b"]
