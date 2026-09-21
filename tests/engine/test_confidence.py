"""The confidence formula, term by term.

PROJECT_DESIGN.md §4.2 replaced v1's LLM self-consistency score with this
"decomposable, re-derivable by hand" number. These tests are the hand
derivation.
"""

import pytest

from loop.engine.confidence import (
    breadth_term,
    compute_confidence,
    evidence_term,
    max_attainable_confidence,
    persistence_term,
    severity_term,
)
from loop.engine.evidence import MATURED_LABELS
from tests.engine.conftest import evidence_for, feature_stat, quiet_psi


# ---------------------------------------------------------------------------
# The individual terms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "max_psi, expected",
    [(0.0, 0.0), (0.10, 0.2), (0.25, 0.5), (0.27, 0.54), (0.5, 1.0), (0.9, 1.0)],
)
def test_severity_is_max_psi_over_the_scale_capped_at_one(max_psi, expected):
    assert severity_term(max_psi, 0.5) == pytest.approx(expected)


def test_severity_of_a_negative_psi_is_zero():
    assert severity_term(-1.0, 0.5) == 0.0


@pytest.mark.parametrize(
    "breaching, monitored, expected",
    [(0, 25, 0.0), (2, 25, 0.08), (2, 11, pytest.approx(0.1818, abs=1e-4)), (25, 25, 1.0)],
)
def test_breadth_is_the_breaching_fraction(breaching, monitored, expected):
    assert breadth_term(breaching, monitored) == expected


def test_quiet_psi_is_below_every_shipped_breach_threshold(policy):
    """The helper the rule tests rely on must actually mean 'quiet'."""
    assert quiet_psi(policy) < policy.psi_breach


def test_breadth_is_zero_when_nothing_is_monitored():
    """A window that measured no features is not evidence of breadth."""
    assert breadth_term(3, 0) == 0.0


@pytest.mark.parametrize(
    "windows, expected",
    [(0, 0.0), (1, pytest.approx(1 / 3)), (2, pytest.approx(2 / 3)), (3, 1.0), (9, 1.0)],
)
def test_persistence_is_consecutive_windows_over_the_scale_capped_at_one(windows, expected):
    assert persistence_term(windows, 3) == expected


def test_evidence_is_halved_while_labels_are_immature(policy):
    """RESEARCH_NOVELTY.md C3's explicit evidence-completeness term."""
    assert evidence_term(False, policy) == 0.5
    assert evidence_term(True, policy) == 1.0


# ---------------------------------------------------------------------------
# The whole formula
# ---------------------------------------------------------------------------
def test_confidence_is_the_weighted_sum_of_its_own_breakdown(policy):
    evidence = evidence_for(
        policy,
        stats=(feature_stat("a", 0.27), feature_stat("b", 0.19)),
        consecutive=2,
        persistent=("a", "b"),
        monitored=11,
    )
    breakdown = compute_confidence(evidence, policy)

    recomputed = (
        0.35 * breakdown.severity
        + 0.20 * breakdown.breadth
        + 0.20 * breakdown.persistence
        + 0.25 * breakdown.evidence
    )
    assert breakdown.confidence == pytest.approx(recomputed, abs=1e-6)


def test_the_architecture_example_breakdown_is_reproduced(policy_v1):
    """§3.9's worked example: severity 0.54, breadth ~0.18, persistence 0.67, evidence 0.5.

    Pinned to policy-v1, the literal transcription of §3.8, because the
    example's breadth of 2/11 depends on both quoted features (PSI 0.27 and
    0.19) counting as breaching -- which they do at §3.8's 0.10, and not at
    policy-v2's calibrated 0.20.

    The document's headline `"confidence": 0.86` is *not* reproducible from
    those four terms -- under policy-v1's weights they sum to 0.484. The formula
    is normative (§3.8, §4.3) and the example is illustrative, so the engine
    implements the formula. This test pins that reading so the discrepancy
    cannot be quietly resolved in the wrong direction later.
    """
    evidence = evidence_for(
        policy_v1,
        stats=(feature_stat("num_lab_procedures", 0.27), feature_stat("num_medications", 0.19)),
        consecutive=2,
        persistent=("num_lab_procedures", "num_medications"),
        monitored=11,
    )
    breakdown = compute_confidence(evidence, policy_v1)

    assert breakdown.severity == pytest.approx(0.54, abs=1e-6)
    assert breakdown.breadth == pytest.approx(0.1818, abs=1e-4)
    assert breakdown.persistence == pytest.approx(0.6667, abs=1e-4)
    assert breakdown.evidence == 0.5
    assert breakdown.confidence == pytest.approx(0.4840, abs=1e-3)
    assert breakdown.confidence != pytest.approx(0.86, abs=0.01)


def test_a_quiet_window_still_carries_the_evidence_floor(policy):
    """0.25 * 0.5: a window with no drift is not zero-confidence, it is low-evidence."""
    breakdown = compute_confidence(evidence_for(policy, stats=(feature_stat("a", 0.0),)), policy)
    assert breakdown.severity == 0.0
    assert breakdown.breadth == 0.0
    assert breakdown.persistence == 0.0
    assert breakdown.confidence == pytest.approx(0.125)


def test_confidence_never_leaves_the_unit_interval(policy):
    extreme = evidence_for(
        policy,
        stats=tuple(feature_stat(f"f{i}", 5.0) for i in range(25)),
        consecutive=99,
        persistent=tuple(f"f{i}" for i in range(25)),
        label_maturity=MATURED_LABELS,
    )
    assert compute_confidence(extreme, policy).confidence == pytest.approx(1.0)
    assert 0.0 <= compute_confidence(evidence_for(policy), policy).confidence <= 1.0


def test_leading_indicators_cap_attainable_confidence_below_certainty(policy):
    """Label latency is a ceiling on autonomy, not a rounding detail."""
    assert max_attainable_confidence(policy, labels_matured=False) == pytest.approx(0.875)
    assert max_attainable_confidence(policy, labels_matured=True) == pytest.approx(1.0)


def test_the_auto_proceed_threshold_is_reachable_on_leading_indicators(policy):
    """0.75 sits under the 0.875 ceiling: demanding, but not unreachable."""
    assert policy.auto_proceed_confidence < max_attainable_confidence(policy)


def test_confidence_is_deterministic(policy):
    evidence = evidence_for(
        policy, stats=(feature_stat("a", 0.31),), consecutive=2, persistent=("a",)
    )
    assert compute_confidence(evidence, policy) == compute_confidence(evidence, policy)


def test_the_breakdown_record_carries_the_four_terms_and_not_the_total(policy):
    record = compute_confidence(evidence_for(policy), policy).to_record()
    assert set(record) == {"severity", "breadth", "persistence", "evidence"}
