"""Shared Week 8 fixtures: metric sets, challengers, and a card to act on.

Two kinds of fixture live here, and the split is deliberate.

**Metric sets** are hand-built, because §3.12 calls the gate "a pure function of
two metric sets" and a pure function should be tested on values, not on models.
Every criterion boundary in `test_gate.py` is expressed relative to the
criteria under test rather than as a literal, for the reason Week 7's fixtures
give: a test that hard-coded `0.005` would silently stop testing the branch it
names the moment a criterion moved.

**Models** are only used where the deliverable is about models: the
"deliberately-bad-challenger fixture proving BLOCK" the roadmap asks for, and
the end-to-end card -> retrain -> gate path. `InvertedChallenger` is bad by
construction rather than by luck -- it reverses the champion's ranking, so it
fails on discrimination, on recall, on calibration and on every subgroup at
once, deterministically, with no fitting and no random seed to drift.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from loop.gate.criteria import FROZEN_HOLDOUT, RECENT_LABELED_WINDOW, load_criteria
from loop.gate.metrics import MetricSet
from ml.retrain import MODE_LIVE, ChallengerRun

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DURATION = timedelta(days=1)

SUBGROUPS = {
    "age": {"[50-60)": 0.70, "[60-70)": 0.68, "[70-80)": 0.66},
    "gender": {"Female": 0.69, "Male": 0.67},
    "race": {"Caucasian": 0.70, "AfricanAmerican": 0.65},
}


# ---------------------------------------------------------------------------
# Metric sets
# ---------------------------------------------------------------------------
def make_metric_set(evaluation_set: str = FROZEN_HOLDOUT, **overrides) -> MetricSet:
    """A healthy champion-shaped metric set; each test states only its own change."""
    defaults = {
        "evaluation_set": evaluation_set,
        "rows": 2000,
        "positive_rate": 0.11,
        "roc_auc": 0.680000,
        "recall_at_top_decile": 0.240000,
        "brier_score": 0.090000,
        "expected_calibration_error": 0.020000,
        "subgroup_roc_auc": {column: dict(groups) for column, groups in SUBGROUPS.items()},
    }
    defaults.update(overrides)
    return MetricSet(**defaults)


def both_sets(**overrides) -> dict[str, MetricSet]:
    """The same numbers on both sets §3.12 gates on."""
    return {
        FROZEN_HOLDOUT: make_metric_set(FROZEN_HOLDOUT, **overrides),
        RECENT_LABELED_WINDOW: make_metric_set(RECENT_LABELED_WINDOW, **overrides),
    }


@pytest.fixture
def criteria():
    """The criteria version in force."""
    return load_criteria()


@pytest.fixture
def champion_sets():
    return both_sets()


@pytest.fixture
def equal_challenger_sets():
    """A challenger that matches the champion exactly -- the trivial PASS."""
    return both_sets()


# ---------------------------------------------------------------------------
# Models: the deliberately bad challenger
# ---------------------------------------------------------------------------
class InvertedChallenger:
    """A challenger that reverses the champion's ranking.

    The roadmap's "deliberately-bad-challenger fixture proving BLOCK", built so
    the BLOCK is a property of the construction rather than of a lucky seed:
    reversing a ranking maps AUROC `a` to `1 - a`, sends top-decile recall
    toward zero, and leaves probabilities pointing the wrong way, so
    discrimination, recall, calibration and subgroup non-regression all fail at
    once on every evaluation set.
    """

    def __init__(self, champion):
        self._champion = champion

    def predict_proba(self, frame):
        good = np.asarray(self._champion.predict_proba(frame), dtype=float)[:, 1]
        bad = 1.0 - good
        return np.column_stack([1.0 - bad, bad])


class ConstantChallenger:
    """A challenger with no discrimination at all: every patient scores alike.

    A second shape of badness -- AUROC is undefined-flat at 0.5 rather than
    inverted -- so the gate is shown to refuse "useless" as well as "backwards".
    """

    def __init__(self, score: float = 0.5):
        self._score = float(score)

    def predict_proba(self, frame):
        column = np.full(len(frame), self._score, dtype=float)
        return np.column_stack([1.0 - column, column])


# The shared 300-row `synthetic_clean_df` is too small to gate against: a model
# fitted on it scores ECE around 0.09 on any half, and §3.12's calibration
# ceiling is an *absolute* 0.05. A champion that cannot clear the real bar would
# force these tests to relax it, which is the one thing a promotion-safety test
# must never do. So the gate fixtures use their own, larger population, sized
# until a genuinely well-calibrated champion falls out of the same generator.
GATE_POPULATION_ROWS = 4000
GATE_POPULATION_SEED = 11
GATE_TREES = 3


@pytest.fixture(scope="session")
def gate_population():
    """A synthetic population large enough for calibration to mean something."""
    from tests.conftest import make_synthetic_clean_df

    return make_synthetic_clean_df(n_rows=GATE_POPULATION_ROWS, seed=GATE_POPULATION_SEED)


@pytest.fixture(scope="session")
def gate_frames(gate_population):
    """Two labeled frames standing in for the frozen holdout and a live window.

    Disjoint halves of the same generator, so both carry the same contract and
    neither is a copy of the other.
    """
    half = len(gate_population) // 2
    return {
        FROZEN_HOLDOUT: gate_population.iloc[:half].reset_index(drop=True),
        RECENT_LABELED_WINDOW: gate_population.iloc[half:].reset_index(drop=True),
    }


@pytest.fixture(scope="session")
def champion_model(gate_population):
    """A small calibrated model standing in for the registered champion.

    Deliberately weak (few trees): an overfitted champion is overconfident, and
    an overconfident champion fails §3.12's ECE ceiling on its own -- which
    would make every PASS test a test of the fixture rather than of the gate.
    """
    from ml.data.features import split_features_target
    from ml.train import build_base_model, build_calibrated_model

    X, y = split_features_target(gate_population)
    model = build_calibrated_model(build_base_model(n_estimators=GATE_TREES))
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def bad_challenger_model(champion_model):
    return InvertedChallenger(champion_model)


def challenger_run(model, *, mode: str = MODE_LIVE, version: str = "2", **overrides):
    """A `ChallengerRun` around an already-built model, with no MLflow involved."""
    defaults = {
        "mode": mode,
        "calibrated_model": model,
        "mlflow_run": "run-challenger",
        "registered_version": version,
        "data_version": "dvc-train-hash",
    }
    defaults.update(overrides)
    return ChallengerRun(**defaults)


# ---------------------------------------------------------------------------
# A database with a card in it
# ---------------------------------------------------------------------------
def make_card_json(
    *,
    card_id: str = "dc-2026-01-01-aaaabbbb",
    action: str = "FULL_RETRAIN",
    disposition: str = "AUTO_PROCEED_SHADOW",
    acceptance_criteria: dict | None = None,
    candidate_data_version: str | None = "dvc-train-hash",
    confidence: float = 0.86,
) -> dict:
    """A Decision Card payload in the frozen Week 7 shape, as JSON."""
    from loop.engine.card import DecisionCard
    from loop.engine.policy import load_policy

    policy = load_policy()
    card = DecisionCard(
        card_id=card_id,
        created_at=WINDOW_START,
        policy_version=policy.version,
        trigger={
            # Derived from the card so two seeded cards never collide on Week 7's
            # (drift_event_id, policy_version) uniqueness constraint.
            "drift_event_id": f"de-{card_id.removeprefix('dc-')}",
            "scenario": "S1",
            "window_start": WINDOW_START,
            "window_end": WINDOW_START + WINDOW_DURATION,
            "breaching_features": [{"feature": "num_lab_procedures", "psi": 0.28}],
            "prediction_drift": True,
            "max_psi": 0.28,
            "monitored_feature_count": 25,
        },
        action=action,
        disposition=disposition,
        confidence=confidence,
        confidence_breakdown={
            "severity": 0.56,
            "breadth": 0.04,
            "persistence": 0.33,
            "evidence": 0.5,
        },
        rule_id=4,
        rationale="test card",
        candidate_data_version=candidate_data_version,
        model_version="1",
        data_version="dvc-reference-hash",
        acceptance_criteria=acceptance_criteria or dict(policy.acceptance_criteria),
        policy_thresholds=policy.summary(),
        status="OPEN",
    )
    return card.to_json_dict()


@pytest.fixture
def gate_db(tmp_path):
    """A throwaway SQLite database brought up through the real `init_db` path."""
    from db.session import configure_engine, get_session, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{(tmp_path / 'gate.db').as_posix()}")
    init_db()
    session = get_session()
    try:
        yield session
    finally:
        session.close()
        reset_engine()


def seed_card(session, **overrides):
    """Writes a drift event and the card that decided it, and returns the card row."""
    from db.models import DecisionCard as DecisionCardRow
    from db.models import DriftEvent

    payload = make_card_json(**overrides)
    trigger = payload["trigger"]

    if session.get(DriftEvent, trigger["drift_event_id"]) is None:
        session.add(
            DriftEvent(
                event_id=trigger["drift_event_id"],
                window_start=WINDOW_START,
                window_end=WINDOW_START + WINDOW_DURATION,
                scenario=trigger["scenario"],
                prediction_drift=trigger["prediction_drift"],
                max_psi=trigger["max_psi"],
                breaching_feature_count=len(trigger["breaching_features"]),
            )
        )

    row = DecisionCardRow(
        card_id=payload["card_id"],
        created_at=WINDOW_START,
        drift_event_id=trigger["drift_event_id"],
        scenario=trigger["scenario"],
        window_start=WINDOW_START,
        window_end=WINDOW_START + WINDOW_DURATION,
        policy_version=payload["policy_version"],
        action=payload["action"],
        disposition=payload["disposition"],
        confidence=payload["confidence"],
        rule_id=str(payload["rule_id"]),
        status=payload["status"],
        breaching_feature_count=len(trigger["breaching_features"]),
        max_psi=trigger["max_psi"],
        prediction_drift=trigger["prediction_drift"],
        model_version=payload["model_version"],
        data_version=payload["data_version"],
        candidate_data_version=payload["candidate_data_version"],
        card_json=payload,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@pytest.fixture
def card_row(gate_db):
    """A card that authorises an automated retrain."""
    return seed_card(gate_db)
