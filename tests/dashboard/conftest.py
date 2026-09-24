"""Week 10 fixtures: a database with the whole lifecycle in it, and a fake API.

The rows are seeded through the same helpers Weeks 8 and 9 test with, so the
dashboard is tested against the shapes those weeks actually write -- not a
second, dashboard-only idea of what a retrain run looks like.
"""

from datetime import UTC, datetime, timedelta

import pytest

from db.models import Approval, Prediction
from tests.approval.conftest import CHALLENGER, CHAMPION, seed_run, seed_shadow_rows
from tests.gate.conftest import seed_card

FIXED_AT = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def dash_db(tmp_path):
    from db.session import configure_engine, get_session, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{(tmp_path / 'dashboard.db').as_posix()}")
    init_db()
    session = get_session()
    try:
        yield session
    finally:
        session.close()
        reset_engine()


def seed_approval(session, card_id, *, kind, decision, run_id=None, before=None, after=None):
    subject = run_id if kind == "PROMOTION" else card_id
    row = Approval(
        approval_id=f"ap-{kind.lower()}-{subject}",
        card_id=card_id,
        kind=kind,
        subject=subject,
        run_id=run_id,
        approver="ops-alice",
        approver_role="ops",
        decision=decision,
        reason=f"{decision.lower()} after reviewing the evidence",
        champion_version_before=before,
        champion_version_after=after,
        champion_alias_moved=decision == "APPROVE" and kind == "PROMOTION",
        rules_version="promotion-v1",
        evidence={"ready_to_approve": True},
    )
    session.add(row)
    session.commit()
    return row


def seed_prediction(session, request_id, *, version="1", caller="dr-a", ts=None):
    row = Prediction(
        request_id=request_id,
        ts=ts or FIXED_AT,
        caller=caller,
        caller_role="clinician",
        model_name="test-readmission",
        model_version=version,
        model_source="mlflow",
        data_version="dvc-train-hash",
        input_hash=f"hash-{request_id}",
        risk_score=0.2,
        predicted_class=0,
        decision_threshold=0.5,
        top_shap=[{"feature": "num__number_inpatient", "contribution": 0.1}],
        status="success",
        latency_ms=120.0,
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def lifecycle(dash_db):
    """One card per lifecycle state the dashboard derives, plus traffic and shadow rows."""
    s = dash_db
    cards = {
        "awaiting": seed_card(
            s, card_id="dc-2026-01-01-00000001", disposition="ESCALATE_HUMAN", confidence=0.4
        ),
        "promoted": seed_card(
            s, card_id="dc-2026-01-01-00000002", disposition="ESCALATE_HUMAN", confidence=0.5
        ),
        "blocked": seed_card(s, card_id="dc-2026-01-01-00000003"),
        "shadow": seed_card(s, card_id="dc-2026-01-01-00000004"),
        "noop": seed_card(s, card_id="dc-2026-01-01-00000005", action="NO_OP", disposition="NONE"),
        "rejected": seed_card(
            s, card_id="dc-2026-01-01-00000006", disposition="ESCALATE_HUMAN", confidence=0.4
        ),
    }
    seed_approval(s, cards["promoted"].card_id, kind="RETRAIN", decision="APPROVE")
    promoted_run = seed_run(s, cards["promoted"].card_id)
    seed_approval(
        s,
        cards["promoted"].card_id,
        kind="PROMOTION",
        decision="APPROVE",
        run_id=promoted_run.run_id,
        before=CHAMPION,
        after=CHALLENGER,
    )
    seed_run(
        s,
        cards["blocked"].card_id,
        outcome="BLOCK",
        shadow_moved=False,
        challenger="inverted-v1",
        mode="demo-bad",
    )
    seed_run(s, cards["shadow"].card_id, challenger="3")
    seed_approval(s, cards["rejected"].card_id, kind="RETRAIN", decision="REJECT")

    for i in range(4):
        seed_prediction(
            s, f"req-{i}", version="1" if i < 3 else "2", ts=FIXED_AT + timedelta(minutes=i)
        )
    seed_shadow_rows(s, 5)
    return cards


class FakeApi:
    """Stands in for `OpsApi`, recording every call; decisions return Week 9 shapes."""

    def __init__(self, *, identity=None, refuse=None):
        self.identity = identity or {"subject": "ops-alice", "role": "ops"}
        self.refuse = refuse or {}
        self.calls: list[tuple] = []
        self.promotions: list[dict] = []
        self.retrains: list[dict] = []

    def _maybe_refuse(self, name):
        if name in self.refuse:
            from dashboard.api_client import ApiError

            status, detail = self.refuse[name]
            raise ApiError(status, detail)

    def whoami(self):
        self.calls.append(("whoami",))
        self._maybe_refuse("whoami")
        return self.identity

    def ready(self):
        return {"status": "ready", "model_version": "2"}

    def shadow(self):
        return {
            "champion_version": "2",
            "shadow_version": None,
            "shadow_active": False,
            "stats": None,
        }

    def reload(self):
        self.calls.append(("reload",))
        return {"champion_version": "2", "shadow_version": "3"}

    def predict(self, payload):
        self.calls.append(("predict",))
        self._maybe_refuse("predict")
        return {"model_version": "2"}

    def pending_retrains(self):
        return self.retrains

    def pending_promotions(self):
        return self.promotions

    def decide_retrain(self, card_id, decision, reason):
        self.calls.append(("decide_retrain", card_id, decision, reason))
        self._maybe_refuse("decide_retrain")
        return {"approval_id": f"ap-retrain-{card_id}", "decision": decision}

    def decide_promotion(self, run_id, decision, reason):
        self.calls.append(("decide_promotion", run_id, decision, reason))
        self._maybe_refuse("decide_promotion")
        moved = decision == "APPROVE"
        return {
            "approval_id": f"ap-promotion-{run_id}",
            "decision": decision,
            "champion_alias_moved": moved,
            "champion_version_before": "2",
            "champion_version_after": "3" if moved else "2",
            "serving": {"champion_version": "3" if moved else "2"},
        }
