"""The deterministic confidence formula.

ARCHITECTURE.md §3.8 and PROJECT_DESIGN.md §4.3 give it in full:

    confidence = w1*severity + w2*breadth + w3*persistence + w4*evidence
      severity    = min(max_PSI / 0.5, 1.0)
      breadth     = fraction of monitored features breaching PSI >= 0.10
      persistence = min(consecutive breaching windows / 3, 1.0)
      evidence    = 1.0 if matured labels confirm degradation,
                    0.5 if only leading indicators
      weights (policy-v1): w1=0.35, w2=0.20, w3=0.20, w4=0.25

PROJECT_DESIGN.md §4.2 replaced v1's LLM self-consistency score with this
precisely so the number would be "decomposable, re-derivable by hand". The
breakdown is therefore returned alongside the total and stored on the card:
a confidence that cannot be checked by arithmetic is the thing this formula
exists to replace.

**A note on the documented example.** The card in ARCHITECTURE.md §3.9 shows
`"confidence": 0.86` beside `{"severity": 0.54, "breadth": 0.18,
"persistence": 0.67, "evidence": 0.5}`. Those four terms under policy-v1's
weights sum to 0.484, not 0.86, so the example's headline figure is not
reproducible from its own breakdown. The formula is normative and the example
is illustrative, so this module implements the formula.
"""

from dataclasses import dataclass

from loop.engine.evidence import DriftEvidence
from loop.engine.policy import Policy

# Confidence is rounded before it is stored and compared, so that re-deriving a
# card by hand from the rounded breakdown gives the number the card carries.
CONFIDENCE_PRECISION = 6


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """The four terms and the total they produce."""

    severity: float
    breadth: float
    persistence: float
    evidence: float
    confidence: float

    def to_record(self) -> dict:
        """The §3.9 `confidence_breakdown` shape."""
        return {
            "severity": self.severity,
            "breadth": self.breadth,
            "persistence": self.persistence,
            "evidence": self.evidence,
        }


def severity_term(max_psi: float, scale: float) -> float:
    """How far the worst feature moved, capped at 1.0."""
    return min(max(max_psi, 0.0) / scale, 1.0)


def breadth_term(breaching_count: int, monitored_count: int) -> float:
    """How much of the monitored surface moved.

    Zero when nothing is monitored: a window that measured no features is not
    evidence of breadth, and dividing by zero to say so would be worse.
    """
    if monitored_count <= 0:
        return 0.0
    return min(max(breaching_count, 0) / monitored_count, 1.0)


def persistence_term(consecutive_windows: int, scale: float) -> float:
    """How long the same features have been breaching, capped at 1.0."""
    return min(max(consecutive_windows, 0) / scale, 1.0)


def evidence_term(labels_matured: bool, policy: Policy) -> float:
    """Evidence completeness -- RESEARCH_NOVELTY.md C3's explicit policy term.

    Leading indicators alone are worth half. Because no producer of matured
    labels exists yet, every real Week 7 card carries the lower value, which
    caps attainable confidence at `w1 + w2 + w3 + w4*leading_indicators` and is
    exactly the restraint the label-latency argument asks for.
    """
    weights = policy.weights
    return weights.matured_labels if labels_matured else weights.leading_indicators


def compute_confidence(evidence: DriftEvidence, policy: Policy) -> ConfidenceBreakdown:
    """The formula, over one window's evidence. Pure, and free of clocks."""
    weights = policy.weights
    severity = severity_term(evidence.observed_max_psi(), weights.severity_scale)
    breadth = breadth_term(
        len(evidence.breaches(policy.psi_breach)), evidence.monitored_feature_count
    )
    persistence = persistence_term(
        evidence.consecutive_breaching_windows, weights.persistence_scale
    )
    completeness = evidence_term(evidence.labels_matured, policy)

    total = (
        weights.severity * severity
        + weights.breadth * breadth
        + weights.persistence * persistence
        + weights.evidence * completeness
    )
    return ConfidenceBreakdown(
        severity=round(severity, CONFIDENCE_PRECISION),
        breadth=round(breadth, CONFIDENCE_PRECISION),
        persistence=round(persistence, CONFIDENCE_PRECISION),
        evidence=round(completeness, CONFIDENCE_PRECISION),
        confidence=round(total, CONFIDENCE_PRECISION),
    )


def max_attainable_confidence(policy: Policy, *, labels_matured: bool = False) -> float:
    """The ceiling under a given evidence completeness.

    Useful in tests and in the docs: under both shipped policies' weights, with
    leading indicators only it is 0.875, so `auto_proceed_confidence = 0.75` is
    reachable but demanding -- which is the intended shape, not an accident.
    """
    weights = policy.weights
    completeness = weights.matured_labels if labels_matured else weights.leading_indicators
    return round(
        weights.severity + weights.breadth + weights.persistence + weights.evidence * completeness,
        CONFIDENCE_PRECISION,
    )
