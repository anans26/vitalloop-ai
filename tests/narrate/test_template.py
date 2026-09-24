"""The Jinja2 narrative: deterministic, and grounded on every card the engine emits.

The template is the fallback the grounding check falls back *to*, so if its own
output ever failed the check, the fallback path would produce a narrative the
system's own guardrail rejects. `test_the_template_is_grounded_on_every_branch`
runs over every rule of the §3.8 table under every shipped policy.
"""

import pytest

from loop.narrate.grounding import check_grounding
from loop.narrate.template import (
    TEMPLATE_SOURCE,
    TEMPLATE_VERSION,
    format_number,
    headline,
    render_template,
)


def test_the_template_is_grounded_on_every_branch(branch_card):
    text = render_template(branch_card)
    result = check_grounding(text, branch_card)
    assert result.grounded, result.problems


def test_the_template_is_deterministic(branch_card):
    assert render_template(branch_card) == render_template(branch_card)


def test_the_template_names_the_decision(branch_card):
    text = render_template(branch_card)
    assert branch_card["card_id"] in text
    assert branch_card["policy_version"] in text
    assert f"rule {branch_card['rule_id']}" in text


def test_the_source_label_carries_the_template_version():
    assert TEMPLATE_SOURCE == f"template/{TEMPLATE_VERSION}"


def test_an_escalated_retrain_says_nothing_retrains_without_a_person(escalated_card):
    text = render_template(escalated_card)
    assert "nothing retrains until an ops user authorises it" in text
    assert "shadow" in text
    assert escalated_card["candidate_data_version"] in text


def test_a_retrain_states_the_acceptance_criteria(escalated_card):
    text = render_template(escalated_card)
    criteria = escalated_card["acceptance_criteria"]
    for key in ("auroc_non_inferiority_margin", "max_ece", "max_subgroup_auroc_drop"):
        assert format_number(criteria[key]) in text


def test_a_quiet_window_does_not_claim_the_inputs_moved():
    card = dict(dict(_cards())["rule1_quiet"])
    text = render_template(card)
    assert "Nothing happens" in text
    assert "the inputs moved" not in text


def test_a_downgraded_card_says_what_it_was_downgraded_from():
    card = dict(_cards())["rule6_cooldown"]
    text = render_template(card)
    assert card["downgraded_from"] in text
    assert "held back to an alert" in text


def test_matured_labels_are_described_as_confirmed_evidence():
    text = render_template(dict(_cards())["rule5_matured"])
    assert "confirmed performance evidence" in text


def test_every_action_disposition_pair_has_a_headline(branch_card):
    assert "/" not in headline(branch_card)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.417667, "0.4177"),
        (0.1, "0.1"),
        (0.25, "0.25"),
        (2, "2"),
        (1.0, "1"),
        (0.0, "0"),
        (None, "n/a"),
    ],
)
def test_format_number_only_rounds(value, text):
    assert format_number(value) == text


def _cards():
    from loop.engine.policy import load_policy
    from tests.narrate.conftest import BRANCH_CARDS

    prefix = f"{load_policy().version}:"
    return [
        (name.removeprefix(prefix), card) for name, card in BRANCH_CARDS if name.startswith(prefix)
    ]
