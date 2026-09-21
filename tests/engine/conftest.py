"""Shared Week 7 fixtures: policies, and evidence to feed them.

Everything here is synthetic. The engine's inputs are aggregate statistics, so
a test never needs a dataset, a model, or a database -- which is also the point
of keeping `decide` a pure function.

**PSI values are expressed relative to the policy under test**, never as
literals. Two policies ship -- `policy-v1`, ARCHITECTURE.md §3.8's literal
transcription, and `policy-v2`, the same six rules with the breach threshold
calibrated on the no-drift control -- and a rule test that hard-coded `0.12` as
"mild" would silently stop testing the branch it names the moment a threshold
moved. `mild_psi(policy)` is mild under whichever policy is in force, so the
rule table is verified under every shipped version.
"""

from datetime import UTC, datetime, timedelta

import pytest

from loop.engine.evidence import DriftEvidence
from loop.engine.policy import available_policies, load_policy

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DURATION = timedelta(days=1)
MONITORED_FEATURES = 25

# Pinned so `created_at` does not vary between two evaluations of the same
# window -- everything else about a decision is already deterministic.
FIXED_NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Policy-relative PSI values
# ---------------------------------------------------------------------------
def quiet_psi(policy) -> float:
    """Below the breach threshold: this feature has not moved."""
    return round(policy.psi_breach / 2, 6)


def mild_psi(policy, offset: float = 0.0) -> float:
    """Inside [psi_breach, psi_severe): breaching, but not severe on its own.

    `offset` distinguishes several mild features from one another without
    leaving the band -- the band is at least 0.05 wide in both shipped
    policies, and offsets are thousandths.
    """
    return round((policy.psi_breach + policy.psi_severe) / 2 + offset, 6)


def severe_psi(policy, offset: float = 0.05) -> float:
    """At or above psi_severe: actionable on its own under rule 4."""
    return round(policy.psi_severe + offset, 6)


def feature_stat(feature: str, psi: float, ks_p: float | None = None) -> dict:
    """A `drift_events.feature_stats` entry, in exactly the Week 6 shape."""
    return {
        "feature": feature,
        "kind": "numerical" if ks_p is not None else "categorical",
        "psi": psi,
        "ks_p_value": ks_p,
        "breaching": psi >= 0.10,
        "severe": psi >= 0.25,
    }


def make_evidence(
    *,
    stats: tuple[dict, ...] = (),
    event_id: str = "de-test-w000-0000abcd",
    scenario: str = "T1",
    window_index: int = 0,
    prediction_drift: bool = False,
    prediction_psi: float | None = 0.01,
    consecutive: int | None = None,
    persistent: tuple[str, ...] | None = None,
    monitored: int = MONITORED_FEATURES,
    breach_threshold: float = 0.10,
    **overrides,
) -> DriftEvidence:
    """Evidence with sensible defaults, so each test states only what it is about.

    `breach_threshold` is only used to derive the default breaching set for
    `consecutive_breaching_windows` and `persistent_features`; the engine reads
    the policy's own threshold off `feature_stats` regardless.
    """
    start = WINDOW_START + window_index * WINDOW_DURATION
    breaching = tuple(stat["feature"] for stat in stats if stat["psi"] >= breach_threshold)
    return DriftEvidence(
        drift_event_id=event_id,
        scenario=scenario,
        window_start=start,
        window_end=start + WINDOW_DURATION,
        feature_stats=stats,
        prediction_drift=prediction_drift,
        prediction_psi=prediction_psi,
        max_psi=max((stat["psi"] for stat in stats), default=0.0),
        monitored_feature_count=monitored,
        measured_thresholds={
            "psi_breach": 0.10,
            "psi_severe": 0.25,
            "monitored_features": monitored,
        },
        consecutive_breaching_windows=(
            consecutive if consecutive is not None else (1 if breaching else 0)
        ),
        persistent_features=persistent if persistent is not None else breaching,
        model_version="1",
        data_version="dvc-reference-hash",
        candidate_data_version="dvc-train-hash",
        **overrides,
    )


def evidence_for(policy, **kwargs) -> DriftEvidence:
    """`make_evidence`, with the breaching set derived from the policy's threshold."""
    kwargs.setdefault("breach_threshold", policy.psi_breach)
    return make_evidence(**kwargs)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
@pytest.fixture(params=available_policies())
def policy(request):
    """Every shipped policy in turn.

    The rule table, the confidence formula and the card contract must hold
    under all of them, so the tests covering those run once per version.
    """
    return load_policy(request.param)


@pytest.fixture(scope="session")
def active_policy():
    """The policy in force -- what the worker and the CLI decide with."""
    return load_policy()


@pytest.fixture(scope="session")
def policy_v1():
    """ARCHITECTURE.md §3.8's literal transcription, retained unedited.

    Cards were emitted under it, so it must stay loadable and unchanged; tests
    that assert against the document's own worked examples pin it explicitly.
    """
    return load_policy("policy-v1")
