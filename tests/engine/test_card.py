"""The Decision Card contract, and what it is structurally unable to carry.

RISK_ANALYSIS.md §1 freezes this schema in week 7 so Weeks 8-9 cannot develop
hidden coupling. These tests are the freeze: they pin the field set, pin that
Week 9's narration slots already exist, and pin that nothing on the card can
hold a patient row.
"""

import json

import pytest
from pydantic import ValidationError

from loop.engine.card import (
    CARD_STATUS_CLOSED,
    CARD_STATUS_OPEN,
    CARD_STATUSES,
    Action,
    CardStatus,
    DecisionCard,
    Disposition,
)
from loop.engine.engine import card_id_for, decide
from loop.engine.rules import ACTIONS, DISPOSITIONS
from tests.engine.conftest import (
    FIXED_NOW,
    WINDOW_START,
    evidence_for,
    feature_stat,
    mild_psi,
    quiet_psi,
    severe_psi,
)


@pytest.fixture
def card(policy) -> DecisionCard:
    evidence = evidence_for(
        policy,
        stats=(
            feature_stat("num_lab_procedures", severe_psi(policy), ks_p=0.001),
            feature_stat("race", quiet_psi(policy)),
        ),
        consecutive=2,
        persistent=("num_lab_procedures",),
    )
    return decide(evidence, policy, now=FIXED_NOW)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
def test_the_card_carries_every_field_the_documents_require(card):
    """RESEARCH_NOVELTY.md C1's six, plus §3.9's identifiers and status."""
    fields = set(DecisionCard.model_fields)
    assert {
        "card_id",
        "created_at",
        "policy_version",
        "trigger",
        "action",
        "disposition",
        "confidence",
        "confidence_breakdown",
        "candidate_data_version",
        "acceptance_criteria",
        "status",
        "narrative",
        "narrative_source",
    } <= fields


def test_week_nine_narration_slots_exist_and_are_empty(card):
    """The contract must not move when the narration layer arrives."""
    assert card.narrative is None
    assert card.narrative_source is None


def test_the_card_is_immutable(card):
    with pytest.raises(ValidationError):
        card.action = "NO_OP"


def test_the_card_rejects_unknown_fields(card):
    payload = card.to_json_dict() | {"surprise": 1}
    with pytest.raises(ValidationError):
        DecisionCard.model_validate(payload)


def test_the_card_round_trips_through_json(card):
    restored = DecisionCard.model_validate(json.loads(json.dumps(card.to_json_dict())))
    assert restored.model_dump() == card.model_dump()


def test_the_schema_and_the_rule_table_agree_on_the_vocabulary():
    """A card the engine can emit but not validate is the one failure a frozen
    contract must not have."""
    assert set(Action.__args__) == set(ACTIONS)
    assert set(Disposition.__args__) == set(DISPOSITIONS)
    assert set(CardStatus.__args__) == set(CARD_STATUSES)


@pytest.mark.parametrize("value", ["RETRAIN", "no_op", ""])
def test_an_undocumented_action_is_refused(card, value):
    payload = card.to_json_dict() | {"action": value}
    with pytest.raises(ValidationError):
        DecisionCard.model_validate(payload)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_confidence_outside_the_unit_interval_is_refused(card, value):
    payload = card.to_json_dict() | {"confidence": value}
    with pytest.raises(ValidationError):
        DecisionCard.model_validate(payload)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_the_card_id_follows_the_documented_shape(card):
    """§3.9's `dc-2026-07-13-001`, with a deterministic suffix."""
    parts = card.card_id.split("-")
    assert parts[0] == "dc"
    assert "-".join(parts[1:4]) == WINDOW_START.date().isoformat()
    assert len(parts[4]) == 8


def test_the_card_id_is_derived_from_the_event_and_the_policy():
    first = card_id_for("de-1", "policy-v1", WINDOW_START)
    assert first == card_id_for("de-1", "policy-v1", WINDOW_START)
    assert first != card_id_for("de-2", "policy-v1", WINDOW_START)
    assert first != card_id_for("de-1", "policy-v2", WINDOW_START)


def test_the_card_id_does_not_depend_on_when_the_decision_was_taken(policy):
    """Re-evaluating a window tomorrow must produce the same card, not a second one."""
    evidence = evidence_for(policy, stats=(feature_stat("a", severe_psi(policy)),))
    early = decide(evidence, policy, now=FIXED_NOW)
    late = decide(evidence, policy, now=FIXED_NOW.replace(year=2027))
    assert early.card_id == late.card_id
    assert early.created_at != late.created_at


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------
def test_the_trigger_points_back_at_the_drift_event_and_its_window(card):
    assert card.trigger.drift_event_id == "de-test-w000-0000abcd"
    assert card.trigger.scenario == "T1"
    assert card.trigger.window_start < card.trigger.window_end


def test_the_card_records_the_policy_that_decided_it(card, policy):
    assert card.policy_version == policy.version
    assert card.policy_thresholds == policy.summary()


def test_the_card_pins_the_data_a_retrain_would_use(card):
    """§3.3: "the exact dataset hash a retrain will run on"."""
    assert card.candidate_data_version == "dvc-train-hash"
    assert card.data_version == "dvc-reference-hash"
    assert card.model_version == "1"


def test_the_card_records_the_acceptance_criteria_before_any_training(card, policy):
    assert card.acceptance_criteria == policy.acceptance_criteria
    assert "auroc_non_inferiority_margin" in card.acceptance_criteria


def test_the_card_records_both_the_measured_and_the_policy_thresholds(card, policy):
    """What the monitor compared against, and what the policy compares against.

    These are separate facts and both are on the record. The monitor measures at
    0.10 so the evidence stays at full sensitivity; the policy decides at its
    own calibrated threshold. A reader can see the gap without opening a config.
    """
    assert card.trigger.measured_thresholds["psi_breach"] == 0.10
    assert card.policy_thresholds["psi_breach"] == policy.psi_breach


def test_the_card_names_the_rule_that_fired_and_says_why(card):
    assert card.rule_id in {1, 2, 3, 4, 5, 6, "uncovered"}
    assert len(card.rationale) > 20


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "band, consecutive, expected",
    [
        ("quiet", 0, CARD_STATUS_CLOSED),
        ("mild", 1, CARD_STATUS_CLOSED),
        ("mild", 2, CARD_STATUS_OPEN),
        ("severe", 1, CARD_STATUS_OPEN),
    ],
)
def test_status_records_whether_downstream_work_remains(policy, band, consecutive, expected):
    psi = {"quiet": quiet_psi, "mild": mild_psi, "severe": severe_psi}[band](policy)
    evidence = evidence_for(
        policy, stats=(feature_stat("a", psi),), consecutive=consecutive, persistent=("a",)
    )
    assert decide(evidence, policy, now=FIXED_NOW).status == expected


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
def test_a_breaching_feature_carries_a_name_and_two_statistics_and_nothing_else(card):
    for breach in card.trigger.breaching_features:
        assert set(breach.model_dump()) == {"feature", "psi", "ks_p"}


def test_only_breaching_features_reach_the_card(card):
    """A card is evidence for a decision, not a copy of the whole measurement."""
    assert [breach.feature for breach in card.trigger.breaching_features] == ["num_lab_procedures"]


def test_the_card_has_no_field_that_could_hold_a_patient_row(card):
    """§3.10's claim that the narration layer is "structurally incapable of
    seeing PHI" is a claim about this schema."""
    blob = json.dumps(card.to_json_dict())
    for forbidden in ("patient_nbr", "encounter_id", "readmitted", "input_hash", "caller"):
        assert forbidden not in blob


def test_every_string_on_the_card_is_a_name_a_version_or_prose(card):
    """A structural check: no field is free to carry a measurement's rows."""
    trigger = card.trigger
    assert all(isinstance(name, str) for name in trigger.persistent_features)
    assert set(trigger.model_dump()) == {
        "drift_event_id",
        "scenario",
        "window_start",
        "window_end",
        "breaching_features",
        "prediction_drift",
        "prediction_psi",
        "max_psi",
        "monitored_feature_count",
        "consecutive_breaching_windows",
        "persistent_features",
        "label_maturity",
        "matured_auroc_drop",
        "measured_thresholds",
    }
