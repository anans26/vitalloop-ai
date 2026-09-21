"""`decide(evidence, policy) -> DecisionCard`. The whole engine, in one call.

ARCHITECTURE.md §3.8: "**No LLM, no randomness, no network.**" This module has
no database handle, no filesystem access and no clock of its own -- `now` is an
argument. Everything it needs arrives in the `DriftEvidence`, which is what
makes RESEARCH_NOVELTY.md C1's claim ("every card in the S1-S5 benchmark can be
re-derived by hand from the policy table") a property a test can assert.

`created_at` is the only field that varies between two evaluations of the same
window. The decision itself -- `card_id`, `action`, `disposition`,
`confidence`, `rule_id` -- does not.
"""

import hashlib
from datetime import UTC, datetime

from loop.engine.card import (
    CARD_STATUS_CLOSED,
    CARD_STATUS_OPEN,
    BreachingFeature,
    CardTrigger,
    ConfidenceBreakdownModel,
    DecisionCard,
)
from loop.engine.confidence import compute_confidence
from loop.engine.evidence import DriftEvidence
from loop.engine.policy import Policy
from loop.engine.rules import RETRAIN_ACTIONS, apply_policy

CARD_ID_HASH_LENGTH = 8


def card_id_for(drift_event_id: str, policy_version: str, window_start: datetime) -> str:
    """`dc-2026-01-03-1a2b3c4d` -- the §3.9 shape, made deterministic.

    The date comes from the window's own start rather than the wall clock, and
    the suffix from the drift event and policy version, so re-evaluating a
    window tomorrow produces the *same* id. That is what makes the write
    idempotent: a second evaluation collides on the primary key instead of
    quietly appending a duplicate decision about the same evidence.

    A different policy version deliberately produces a different id. Replaying
    history under policy-v2 is a governance feature, not a duplicate.
    """
    digest = hashlib.sha256(f"{drift_event_id}|{policy_version}".encode()).hexdigest()
    return f"dc-{window_start.date().isoformat()}-{digest[:CARD_ID_HASH_LENGTH]}"


def _status_for(action: str) -> str:
    """`OPEN` when the action leaves work for a later week, `CLOSED` when it does not."""
    return CARD_STATUS_OPEN if action in RETRAIN_ACTIONS else CARD_STATUS_CLOSED


def decide(
    evidence: DriftEvidence,
    policy: Policy,
    *,
    now: datetime | None = None,
) -> DecisionCard:
    """Evaluates one window against one policy version.

    Confidence is computed first because rule 4's disposition reads it, which is
    the only place in §3.8 where the formula feeds the table rather than merely
    describing the result.
    """
    breakdown = compute_confidence(evidence, policy)
    outcome = apply_policy(evidence, policy, breakdown)

    trigger = CardTrigger(
        drift_event_id=evidence.drift_event_id,
        scenario=evidence.scenario,
        window_start=evidence.window_start,
        window_end=evidence.window_end,
        breaching_features=tuple(
            BreachingFeature(feature=breach.feature, psi=breach.psi, ks_p=breach.ks_p)
            for breach in evidence.breaches(policy.psi_breach)
        ),
        prediction_drift=evidence.prediction_drift,
        prediction_psi=evidence.prediction_psi,
        max_psi=evidence.observed_max_psi(),
        monitored_feature_count=evidence.monitored_feature_count,
        consecutive_breaching_windows=evidence.consecutive_breaching_windows,
        persistent_features=tuple(evidence.persistent_features),
        label_maturity=evidence.label_maturity,
        matured_auroc_drop=evidence.matured_auroc_drop,
        measured_thresholds=dict(evidence.measured_thresholds),
    )

    return DecisionCard(
        card_id=card_id_for(evidence.drift_event_id, policy.version, evidence.window_start),
        created_at=now or datetime.now(UTC),
        policy_version=policy.version,
        trigger=trigger,
        action=outcome.action,
        disposition=outcome.disposition,
        confidence=breakdown.confidence,
        confidence_breakdown=ConfidenceBreakdownModel(**breakdown.to_record()),
        rule_id=outcome.rule_id,
        rationale=outcome.rationale,
        downgraded_from=outcome.downgraded_from,
        candidate_data_version=evidence.candidate_data_version,
        model_version=evidence.model_version,
        data_version=evidence.data_version,
        acceptance_criteria=dict(policy.acceptance_criteria),
        policy_thresholds=policy.summary(),
        status=_status_for(outcome.action),
    )
