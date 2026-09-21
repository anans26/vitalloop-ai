"""The policy artifact: what ships, what loads, and what is refused.

ARCHITECTURE.md §3.8 makes the policy file the governance surface -- "changing
a threshold is a reviewed pull request". These tests pin the values that are
under review, and pin that a policy the engine could not defend is rejected at
load time rather than silently half-applied.
"""

import pytest
import yaml

from loop.engine.policy import (
    DEFAULT_POLICY_VERSION,
    POLICY_DIR,
    PolicyError,
    load_policy,
    parse_policy,
    policy_path,
)


@pytest.fixture
def document() -> dict:
    """The shipped policy, as a mutable document tests can break on purpose."""
    return yaml.safe_load(policy_path().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# What ships
# ---------------------------------------------------------------------------
def test_policy_v1_is_the_default_and_exists():
    assert DEFAULT_POLICY_VERSION == "policy-v1"
    assert policy_path().exists()
    assert policy_path().parent == POLICY_DIR


def test_the_shipped_thresholds_are_the_documented_ones(policy):
    """§3.8's rule table, as numbers."""
    assert policy.psi_breach == 0.10
    assert policy.psi_severe == 0.25
    assert policy.consecutive_windows == 2
    assert policy.alert_only_max_features == 2
    assert policy.auto_proceed_confidence == 0.75
    assert policy.matured_auroc_drop == 0.03


def test_the_shipped_confidence_weights_are_the_documented_ones(policy):
    """policy-v1: w1=0.35, w2=0.20, w3=0.20, w4=0.25."""
    assert policy.weights.severity == 0.35
    assert policy.weights.breadth == 0.20
    assert policy.weights.persistence == 0.20
    assert policy.weights.evidence == 0.25
    assert policy.weights.severity_scale == 0.5
    assert policy.weights.persistence_scale == 3
    assert policy.weights.leading_indicators == 0.5
    assert policy.weights.matured_labels == 1.0


def test_the_shipped_acceptance_criteria_match_the_validation_gate(policy):
    """§3.12 / §4.5. Recorded on the card before any training starts."""
    assert policy.acceptance_criteria == {
        "auroc_non_inferiority_margin": 0.005,
        "recall_top_decile_min_ratio": 1.0,
        "max_brier_increase": 0.005,
        "max_ece": 0.05,
        "max_subgroup_auroc_drop": 0.01,
    }


def test_the_policy_summary_is_what_a_card_records(policy):
    summary = policy.summary()
    assert set(summary) == {
        "psi_breach",
        "psi_severe",
        "consecutive_windows",
        "auto_proceed_confidence",
    }


def test_the_policy_is_immutable(policy):
    with pytest.raises(Exception):
        policy.psi_breach = 0.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_a_missing_policy_is_reported_not_guessed(tmp_path):
    with pytest.raises(PolicyError, match="No policy file"):
        load_policy("policy-v9", directory=tmp_path)


def test_invalid_yaml_is_reported_as_a_policy_error(tmp_path):
    (tmp_path / "policy-vx.yaml").write_text("version: [unclosed", encoding="utf-8")
    with pytest.raises(PolicyError, match="not valid YAML"):
        load_policy("policy-vx", directory=tmp_path)


def test_a_file_whose_version_disagrees_with_its_name_is_refused(tmp_path, document):
    """A card's policy_version must name a file a reader can open."""
    document["version"] = "policy-v7"
    (tmp_path / "policy-v1.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(PolicyError, match="does not match its filename"):
        load_policy("policy-v1", directory=tmp_path)


def test_a_policy_file_round_trips_from_disk(tmp_path, document, policy):
    (tmp_path / "policy-v1.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_policy("policy-v1", directory=tmp_path) == policy


def test_a_non_mapping_document_is_refused():
    with pytest.raises(PolicyError, match="mapping"):
        parse_policy(["not", "a", "policy"])


# ---------------------------------------------------------------------------
# Validation -- a policy the engine could not defend must not load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "section, key",
    [
        (None, "version"),
        (None, "precedence"),
        (None, "uncovered_breach_action"),
        (None, "acceptance_criteria"),
        ("thresholds", "psi_breach"),
        ("thresholds", "psi_severe"),
        ("thresholds", "consecutive_windows"),
        ("thresholds", "auto_proceed_confidence"),
        ("confidence", "weights"),
        ("confidence", "severity_scale"),
    ],
)
def test_a_missing_key_fails_the_load_rather_than_defaulting(document, section, key):
    """There is no default threshold anywhere in the engine, by design."""
    target = document if section is None else document[section]
    target.pop(key)
    with pytest.raises(PolicyError):
        parse_policy(document)


def test_weights_that_do_not_sum_to_one_are_refused(document):
    document["confidence"]["weights"]["severity"] = 0.50
    with pytest.raises(PolicyError, match="must sum to 1.0"):
        parse_policy(document)


def test_a_breach_threshold_above_the_severe_threshold_is_refused(document):
    document["thresholds"]["psi_breach"] = 0.40
    with pytest.raises(PolicyError, match="below psi_severe"):
        parse_policy(document)


def test_a_non_positive_breach_threshold_is_refused(document):
    document["thresholds"]["psi_breach"] = 0.0
    with pytest.raises(PolicyError):
        parse_policy(document)


def test_a_single_window_persistence_rule_is_refused(document):
    document["thresholds"]["consecutive_windows"] = 1
    with pytest.raises(PolicyError, match="not a persistence rule"):
        parse_policy(document)


def test_a_confidence_threshold_outside_zero_to_one_is_refused(document):
    document["thresholds"]["auto_proceed_confidence"] = 1.5
    with pytest.raises(PolicyError, match="within"):
        parse_policy(document)


def test_a_non_positive_scale_is_refused(document):
    document["confidence"]["severity_scale"] = 0
    with pytest.raises(PolicyError, match="scales must be positive"):
        parse_policy(document)


@pytest.mark.parametrize("precedence", [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 1, 2, 3, 4]])
def test_a_precedence_that_does_not_order_rules_one_to_five_is_refused(document, precedence):
    """Rule 6 is a downgrade applied after the table, never a row within it."""
    document["precedence"] = precedence
    with pytest.raises(PolicyError, match="precedence"):
        parse_policy(document)


def test_empty_acceptance_criteria_are_refused(document):
    document["acceptance_criteria"] = {}
    with pytest.raises(PolicyError, match="acceptance_criteria"):
        parse_policy(document)
