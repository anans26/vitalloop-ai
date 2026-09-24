"""§3.13's shadow agreement statistics, tested on values.

A pure function over score pairs, so every figure a person will approve a
promotion on can be checked here by hand.
"""

import pytest

from loop.shadow.stats import shadow_stats


def test_identical_models_agree_perfectly():
    pairs = [(0.1, 0.1), (0.4, 0.4), (0.7, 0.7), (0.9, 0.9)]
    stats = shadow_stats(pairs, threshold=0.5)

    assert stats.requests == stats.scored == 4
    assert stats.failed == 0
    assert stats.decision_agreement == 1.0
    assert stats.flips_to_positive == stats.flips_to_negative == 0
    assert stats.mean_abs_diff == stats.max_abs_diff == 0.0
    assert stats.spearman == 1.0
    assert stats.mean_shift == 0.0


def test_flips_are_counted_in_each_direction():
    pairs = [
        (0.40, 0.60),  # would now be flagged
        (0.45, 0.55),  # would now be flagged
        (0.70, 0.30),  # would no longer be flagged
        (0.10, 0.20),  # agrees
    ]
    stats = shadow_stats(pairs, threshold=0.5)
    assert stats.flips_to_positive == 2
    assert stats.flips_to_negative == 1
    assert stats.decision_agreement == 0.25


def test_the_threshold_itself_counts_as_positive():
    """Same rule as the API: predicted_class = score >= threshold."""
    stats = shadow_stats([(0.5, 0.49)], threshold=0.5)
    assert stats.flips_to_negative == 1


def test_score_distance_statistics():
    pairs = [(0.1, 0.2), (0.2, 0.2), (0.3, 0.6)]
    stats = shadow_stats(pairs, threshold=0.5)
    assert stats.mean_abs_diff == pytest.approx((0.1 + 0.0 + 0.3) / 3, abs=1e-6)
    assert stats.max_abs_diff == pytest.approx(0.3)
    assert stats.champion_mean_score == pytest.approx(0.2)
    assert stats.shadow_mean_score == pytest.approx(1.0 / 3, abs=1e-6)
    assert stats.mean_shift == pytest.approx(1.0 / 3 - 0.2, abs=1e-6)


def test_a_systematically_higher_shadow_shows_as_a_mean_shift_even_when_ranks_agree():
    pairs = [(s, s + 0.05) for s in (0.1, 0.2, 0.3, 0.4)]
    stats = shadow_stats(pairs, threshold=0.9)
    assert stats.spearman == 1.0
    assert stats.decision_agreement == 1.0
    assert stats.mean_shift == pytest.approx(0.05)


def test_a_reversed_ranking_has_negative_rank_correlation():
    pairs = [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)]
    assert shadow_stats(pairs, threshold=0.5).spearman == -1.0


def test_failures_are_counted_not_dropped():
    pairs = [(0.2, 0.3), (0.4, None), (0.6, 0.5)]
    stats = shadow_stats(pairs, threshold=0.5)
    assert stats.requests == 3
    assert stats.scored == 2
    assert stats.failed == 1


def test_an_all_failed_window_has_no_agreement_rather_than_perfect_agreement():
    stats = shadow_stats([(0.2, None), (0.4, None)], threshold=0.5)
    assert stats.scored == 0
    assert stats.failed == 2
    assert stats.decision_agreement is None
    assert stats.mean_abs_diff is None


def test_an_empty_window():
    stats = shadow_stats([], threshold=0.5)
    assert stats.requests == 0
    assert stats.decision_agreement is None


def test_rank_correlation_is_undefined_for_a_constant_model():
    stats = shadow_stats([(0.1, 0.5), (0.2, 0.5), (0.3, 0.5)], threshold=0.9)
    assert stats.spearman is None


def test_the_record_carries_the_versions_and_is_json_shaped():
    import json

    stats = shadow_stats(
        [(0.1, 0.2), (0.3, 0.4)], threshold=0.5, champion_version="1", shadow_version="2"
    )
    record = stats.to_record()
    json.dumps(record)
    assert record["champion_version"] == "1"
    assert record["shadow_version"] == "2"
