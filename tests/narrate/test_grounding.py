"""The grounding check (ARCHITECTURE.md §3.10): every quoted number must be in the card.

Most of these tests are *refusals*. The check exists to catch a narrative that
invents a figure, so the cases that matter are the ones where it says no.
"""

import pytest

from loop.narrate.grounding import (
    card_dates,
    card_numbers,
    check_grounding,
    quoted_dates,
    quoted_numbers,
)


# ---------------------------------------------------------------------------
# Reading numbers out of text
# ---------------------------------------------------------------------------
def test_reads_integers_decimals_and_percentages():
    found = quoted_numbers("PSI 0.27 on 2 features, 41.8% confidence, 10,498 rows")
    assert [q.text for q in found] == ["0.27", "2", "41.8%", "10,498"]
    assert found[2].percent and found[2].decimals == 1
    assert found[3].value == 10498


def test_reads_a_sentence_final_number_without_the_full_stop():
    assert [q.text for q in quoted_numbers("confidence was 0.4177.")] == ["0.4177"]


def test_reads_negative_numbers():
    (quoted,) = quoted_numbers("a shift of -0.01 in mean score")
    assert quoted.value == -0.01


@pytest.mark.parametrize(
    "text",
    [
        "policy-v2",
        "scenario S1",
        "card dc-2026-01-01-11d1d16c",
        "hash ebbfae60a1",
        "the 30-day label",
    ],
)
def test_digits_inside_identifiers_are_not_quantities(text):
    assert quoted_numbers(text) == ()


def test_reads_iso_dates_separately():
    assert quoted_dates("from 2026-01-01 to 2026-01-02.") == ("2026-01-01", "2026-01-02")
    assert quoted_numbers("from 2026-01-01 to 2026-01-02.") == ()


# ---------------------------------------------------------------------------
# What counts as "in the card"
# ---------------------------------------------------------------------------
def test_card_numbers_include_leaves_list_lengths_and_the_rationale(escalated_card):
    numbers = card_numbers(escalated_card)
    assert escalated_card["confidence"] in numbers
    assert float(len(escalated_card["trigger"]["breaching_features"])) in numbers
    assert escalated_card["trigger"]["monitored_feature_count"] in numbers


def test_booleans_are_not_numbers():
    assert card_numbers({"prediction_drift": True, "x": 0.5}) == (0.5,)


def test_narrative_fields_are_never_evidence_for_themselves():
    card = {"confidence": 0.5, "narrative": "confidence 0.9", "narrative_source": "x"}
    assert 0.9 not in card_numbers(card)
    assert not check_grounding("confidence 0.9", card).grounded


def test_card_dates_are_read_from_timestamps(escalated_card):
    assert escalated_card["trigger"]["window_start"][:10] in card_dates(escalated_card)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def test_a_rounded_quotation_is_grounded():
    card = {"confidence": 0.417667}
    for text in ("0.417667", "0.4177", "0.418", "0.42"):
        assert check_grounding(f"confidence {text}", card).grounded, text


def test_a_percentage_of_a_card_value_is_grounded():
    card = {"confidence": 0.417667}
    assert check_grounding("confidence 42%", card).grounded
    assert check_grounding("confidence 41.8%", card).grounded


def test_a_mis_rounded_quotation_is_refused():
    """0.417667 rounds to 0.42 at two places, never to 0.41."""
    result = check_grounding("confidence 0.41", {"confidence": 0.417667})
    assert not result.grounded
    assert result.ungrounded_numbers == ("0.41",)


def test_an_invented_number_is_refused_and_named(escalated_card):
    result = check_grounding(
        "Confidence was 0.9 and 17 features breached, so retraining is urgent.", escalated_card
    )
    assert not result.grounded
    assert "0.9" in result.ungrounded_numbers
    assert "17" in result.ungrounded_numbers
    assert "numbers not in the card" in result.problems[0]


def test_an_invented_date_is_refused(escalated_card):
    result = check_grounding("The window closed on 2031-05-05.", escalated_card)
    assert not result.grounded
    assert result.ungrounded_dates == ("2031-05-05",)


def test_a_narrative_with_no_numbers_is_grounded(escalated_card):
    assert check_grounding("The inputs moved and the policy escalated.", escalated_card).grounded


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_empty_narrative_is_never_grounded(empty, escalated_card):
    assert not check_grounding(empty, escalated_card).grounded


def test_numbers_quoted_from_the_rationale_are_grounded():
    """Rule 6's rationale names days since the last retrain, which no other field holds."""
    card = {"rationale": "the last retrain was 1.50 days ago", "confidence": 0.5}
    assert check_grounding("It retrained 1.5 days ago.", card).grounded


def test_the_result_serialises_to_a_record(escalated_card):
    record = check_grounding("confidence 0.9", escalated_card).to_record()
    assert record["grounded"] is False
    assert record["ungrounded_numbers"] == ["0.9"]
