"""Shared Week 7 fixtures: a policy, and evidence to feed it.

Everything here is synthetic. The engine's inputs are aggregate statistics, so
a test never needs a dataset, a model, or a database -- which is also the point
of keeping `decide` a pure function.
"""

from datetime import UTC, datetime, timedelta

import pytest

from loop.engine.evidence import DriftEvidence
from loop.engine.policy import load_policy

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DURATION = timedelta(days=1)
MONITORED_FEATURES = 25

# Pinned so `created_at` does not vary between two evaluations of the same
# window -- everything else about a decision is already deterministic.
FIXED_NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


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
    **overrides,
) -> DriftEvidence:
    """Evidence with sensible defaults, so each test states only what it is about."""
    start = WINDOW_START + window_index * WINDOW_DURATION
    breaching = tuple(stat["feature"] for stat in stats if stat["psi"] >= 0.10)
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


@pytest.fixture(scope="session")
def policy():
    """The real `configs/policy-v1.yaml`. Tests assert against the shipped policy."""
    return load_policy()
