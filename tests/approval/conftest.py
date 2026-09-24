"""Week 9 approval fixtures: a database with a gated challenger in it, and a fake registry.

`FakeRegistry` stands in for MLflow the way Week 8's `AliasRecorder` did: it
holds aliases in a dict and remembers every move, with who made it. That is
enough to assert the thing that matters -- which aliases a decision moved, and
in whose name -- without a tracking server.
"""

from datetime import timedelta

import pytest

from db.models import RetrainRun, ShadowPrediction
from loop.approval.rules import load_rules
from loop.gate.gate import BLOCK, PASS
from tests.gate.conftest import WINDOW_START, seed_card

CHAMPION, CHALLENGER = "1", "2"


class FakeRegistry:
    """An in-memory `AliasRegistry` that records every move."""

    def __init__(self, **aliases):
        self.aliases = dict(aliases)
        self.moves: list[dict] = []
        self.fail_on_set: str | None = None

    def version_of(self, alias):
        return self.aliases.get(alias)

    def set(self, alias, version, *, run_id, reason, actor):
        if self.fail_on_set == alias:
            raise RuntimeError("registry unavailable")
        move = {
            "action": "set_alias",
            "alias": alias,
            "from_version": self.aliases.get(alias),
            "to_version": version,
            "actor": actor,
            "reason": reason,
        }
        self.aliases[alias] = version
        self.moves.append(move)
        return move

    def clear(self, alias, *, run_id, reason, actor):
        if alias not in self.aliases:
            return None
        move = {
            "action": "delete_alias",
            "alias": alias,
            "from_version": self.aliases.pop(alias),
            "to_version": None,
            "actor": actor,
            "reason": reason,
        }
        self.moves.append(move)
        return move


@pytest.fixture
def registry():
    """The state right after a Week 8 PASS: champion 1 serving, challenger 2 in shadow."""
    return FakeRegistry(champion=CHAMPION, shadow=CHALLENGER, challenger=CHALLENGER)


@pytest.fixture
def rules():
    return load_rules()


@pytest.fixture
def approval_db(tmp_path):
    from db.session import configure_engine, get_session, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{(tmp_path / 'approval.db').as_posix()}")
    init_db()
    session = get_session()
    try:
        yield session
    finally:
        session.close()
        reset_engine()


def metric_set(evaluation_set, auc, subgroups):
    return {
        "evaluation_set": evaluation_set,
        "rows": 2000,
        "positive_rate": 0.11,
        "roc_auc": auc,
        "recall_at_top_decile": 0.24,
        "brier_score": 0.09,
        "expected_calibration_error": 0.02,
        "subgroup_roc_auc": {"age": subgroups},
    }


def seed_run(
    session,
    card_id,
    *,
    outcome=PASS,
    challenger=CHALLENGER,
    champion=CHAMPION,
    shadow_moved=True,
    run_id=None,
    mode="replay",
):
    run = RetrainRun(
        run_id=run_id or f"rr-{card_id.removeprefix('dc-')}-{challenger}",
        card_id=card_id,
        scenario="S1",
        policy_version="policy-v2",
        action="FULL_RETRAIN",
        mode=mode,
        mlflow_run=f"mlflow-run-{challenger}",
        data_version="dvc-train-hash",
        champion_version=champion,
        challenger_version=challenger,
        outcome=outcome,
        criteria_version="card:" + card_id,
        failed_criteria_count=0 if outcome == PASS else 3,
        shadow_alias_moved=shadow_moved,
        authorized_by="ops-alice",
        gate_result={
            "outcome": outcome,
            "evaluation_sets": ["frozen_holdout"],
            "champion_metrics": {
                "frozen_holdout": metric_set(
                    "frozen_holdout", 0.60, {"[50-60)": 0.62, "[60-70)": 0.58}
                )
            },
            "challenger_metrics": {
                "frozen_holdout": metric_set(
                    "frozen_holdout", 0.61, {"[50-60)": 0.615, "[60-70)": 0.59}
                )
            },
            "reasons": [],
        },
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def seed_shadow_rows(session, n, *, champion=CHAMPION, shadow=CHALLENGER, start=0):
    for i in range(start, start + n):
        score = (i % 10) / 10 + 0.05
        session.add(
            ShadowPrediction(
                request_id=f"req-{champion}-{shadow}-{i:05d}",
                ts=WINDOW_START + timedelta(seconds=i),
                model_name="test-readmission",
                champion_version=champion,
                shadow_version=shadow,
                champion_score=score,
                shadow_score=min(score + 0.01, 1.0),
                decision_threshold=0.5,
                status="scored",
            )
        )
    session.commit()


@pytest.fixture
def gated(approval_db):
    """A card, and the PASS run that put its challenger in shadow."""
    card = seed_card(approval_db, disposition="ESCALATE_HUMAN", confidence=0.42)
    run = seed_run(approval_db, card.card_id)
    return card, run


@pytest.fixture
def blocked(approval_db):
    card = seed_card(approval_db, card_id="dc-2026-01-02-ccccdddd")
    return card, seed_run(approval_db, card.card_id, outcome=BLOCK, shadow_moved=False)
