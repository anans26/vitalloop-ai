"""Week 9 narration fixtures: one real Decision Card per branch of the policy table.

The cards come from `decide` itself, not from hand-written JSON, so a test that
says "the template is grounded on every card" is a statement about the cards the
engine actually emits -- under every shipped policy.
"""

import pytest

from loop.engine.engine import decide
from loop.engine.evidence import MATURED_LABELS
from loop.engine.policy import available_policies, load_policy
from tests.engine.conftest import (
    FIXED_NOW,
    evidence_for,
    feature_stat,
    mild_psi,
    quiet_psi,
    severe_psi,
)


def branch_evidence(policy) -> dict:
    """Evidence that lands on each rule (and disposition) of the §3.8 table."""
    return {
        "rule1_quiet": evidence_for(policy, stats=(feature_stat("a", quiet_psi(policy), 0.4),)),
        "rule2_alert": evidence_for(policy, stats=(feature_stat("a", mild_psi(policy), 0.01),)),
        "rule3_incremental": evidence_for(
            policy,
            stats=(feature_stat("a", mild_psi(policy), 0.01),),
            consecutive=2,
            persistent=("a",),
        ),
        "rule4_escalate": evidence_for(
            policy,
            stats=(
                feature_stat("num_lab_procedures", severe_psi(policy), 0.001),
                feature_stat("num_medications", mild_psi(policy), 0.004),
            ),
            prediction_drift=True,
            prediction_psi=0.31,
        ),
        "rule4_prediction_only": evidence_for(
            policy, stats=(), prediction_drift=True, prediction_psi=0.27
        ),
        "rule4_auto": evidence_for(
            policy,
            stats=tuple(feature_stat(f"f{i}", 0.60) for i in range(25)),
            consecutive=3,
            persistent=tuple(f"f{i}" for i in range(25)),
            label_maturity=MATURED_LABELS,
            matured_auroc_drop=0.0,
        ),
        "rule5_matured": evidence_for(
            policy,
            stats=(feature_stat("a", quiet_psi(policy)),),
            label_maturity=MATURED_LABELS,
            matured_auroc_drop=0.05,
        ),
        "rule6_cooldown": evidence_for(
            policy,
            stats=(feature_stat("a", severe_psi(policy)),),
            days_since_last_retrain=1.5,
        ),
        "rule6_budget": evidence_for(
            policy,
            stats=(feature_stat("a", severe_psi(policy)),),
            retrains_in_cooldown=policy.retrain_budget,
        ),
        "uncovered": evidence_for(
            policy,
            stats=tuple(feature_stat(f"m{i}", mild_psi(policy, i / 1000)) for i in range(3)),
        ),
    }


def all_branch_cards() -> list[tuple[str, dict]]:
    cards = []
    for version in available_policies():
        policy = load_policy(version)
        for name, evidence in branch_evidence(policy).items():
            card = decide(evidence, policy, now=FIXED_NOW)
            cards.append((f"{version}:{name}", card.to_json_dict()))
    return cards


BRANCH_CARDS = all_branch_cards()


@pytest.fixture(params=BRANCH_CARDS, ids=[name for name, _ in BRANCH_CARDS])
def branch_card(request) -> dict:
    return request.param[1]


@pytest.fixture
def escalated_card() -> dict:
    """The §3.9-shaped headline card: FULL_RETRAIN, escalated, two features."""
    return dict(BRANCH_CARDS)[f"{load_policy().version}:rule4_escalate"]
