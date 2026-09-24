"""The `/ops` endpoints: role-gated, identity from the token, decisions persisted.

The promotion logic itself is tested in `tests/approval/`; these tests pin the
HTTP contract around it -- who may call, what each refusal maps to, and that a
decision refreshes the models this instance serves.
"""

import pytest
from sqlalchemy import select

from db.models import Approval
from tests.approval.conftest import CHALLENGER, CHAMPION, FakeRegistry, seed_run, seed_shadow_rows
from tests.gate.conftest import seed_card

OPS_ENDPOINTS = [
    ("get", "/ops/shadow"),
    ("post", "/ops/models/reload"),
    ("get", "/ops/cards/dc-x/narrative"),
    ("get", "/ops/retrains/pending"),
    ("post", "/ops/retrains/dc-x/decision"),
    ("get", "/ops/promotions/pending"),
    ("get", "/ops/promotions/rr-x"),
    ("post", "/ops/promotions/rr-x/decision"),
]
DECISION = {"decision": "APPROVE", "reason": "gate passed and shadow agreement is high"}


@pytest.fixture
def ops_headers(api_env):
    from api.auth import create_access_token

    token = create_access_token("ops-alice", role="ops", settings=api_env)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def session(api_client):
    from db.session import get_session

    session = get_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def registry(api_client):
    fake = FakeRegistry(champion=CHAMPION, shadow=CHALLENGER)
    api_client.app.state.alias_registry = fake
    return fake


@pytest.fixture
def refreshes(api_client, monkeypatch):
    """Records model refreshes instead of loading models from disk."""
    calls = []

    def refresh(state, settings):
        calls.append(True)
        return {"champion_version": "2", "shadow_version": None, "champion_reload_failed": False}

    monkeypatch.setattr("api.routers.ops.refresh_serving_models", refresh)
    return calls


def _call(client, method, path, headers):
    if method == "post":
        return client.post(path, headers=headers, json=DECISION)
    return client.get(path, headers=headers)


# ---------------------------------------------------------------------------
# Who may call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("method", "path"), OPS_ENDPOINTS)
def test_a_clinician_cannot_reach_any_ops_endpoint(method, path, api_client, auth_headers):
    """TECH_STACK.md: whoever sees a score is not whoever promotes a model."""
    assert _call(api_client, method, path, auth_headers).status_code == 403


@pytest.mark.parametrize(("method", "path"), OPS_ENDPOINTS)
def test_an_anonymous_caller_cannot_reach_any_ops_endpoint(method, path, api_client):
    assert _call(api_client, method, path, {}).status_code == 401


def test_the_body_cannot_name_the_approver(api_client, ops_headers):
    body = {**DECISION, "approver": "someone-else"}
    response = api_client.post("/ops/retrains/dc-x/decision", headers=ops_headers, json=body)
    assert response.status_code == 422


def test_openapi_documents_the_ops_endpoints_as_authenticated(api_client):
    paths = api_client.get("/openapi.json").json()["paths"]
    for _, path in OPS_ENDPOINTS:
        templated = path.replace("dc-x", "{card_id}").replace("rr-x", "{run_id}")
        assert templated in paths, templated
        for operation in paths[templated].values():
            assert operation.get("security")


# ---------------------------------------------------------------------------
# Shadow
# ---------------------------------------------------------------------------
def test_shadow_status_without_a_shadow(api_client, ops_headers):
    api_client.app.state.shadow_bundle = None
    body = api_client.get("/ops/shadow", headers=ops_headers).json()
    assert body == {
        "champion_version": "1",
        "shadow_version": None,
        "shadow_active": False,
        "stats": None,
    }


def test_shadow_status_reports_the_windows_statistics(api_client, ops_headers, session):
    from api.shadow import ShadowBundle

    api_client.app.state.shadow_bundle = ShadowBundle(object(), "test-readmission", CHALLENGER)
    seed_shadow_rows(session, 12)

    body = api_client.get("/ops/shadow", headers=ops_headers).json()
    assert body["shadow_active"] is True
    assert body["stats"]["requests"] == 12
    assert body["stats"]["shadow_version"] == CHALLENGER


def test_reload_reports_what_is_served(api_client, ops_headers, refreshes):
    body = api_client.post("/ops/models/reload", headers=ops_headers).json()
    assert body["champion_version"] == "2"
    assert refreshes == [True]


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------
def test_a_card_without_a_stored_narrative_is_rendered_by_the_template(
    api_client, ops_headers, session
):
    card = seed_card(session)
    body = api_client.get(f"/ops/cards/{card.card_id}/narrative", headers=ops_headers).json()
    assert body["stored"] is False
    assert body["narrative_source"].startswith("template/")
    assert body["grounding"]["grounded"] is True
    assert card.card_id in body["narrative"]


def test_an_unknown_card_has_no_narrative(api_client, ops_headers):
    assert api_client.get("/ops/cards/dc-nope/narrative", headers=ops_headers).status_code == 404


# ---------------------------------------------------------------------------
# Retrain authorisation
# ---------------------------------------------------------------------------
def test_an_escalated_card_is_listed_then_authorised(api_client, ops_headers, session):
    card = seed_card(session, disposition="ESCALATE_HUMAN", confidence=0.42)
    pending = api_client.get("/ops/retrains/pending", headers=ops_headers).json()
    assert [entry["card"]["card_id"] for entry in pending] == [card.card_id]

    response = api_client.post(
        f"/ops/retrains/{card.card_id}/decision", headers=ops_headers, json=DECISION
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "RETRAIN"
    assert body["approver"] == "ops-alice"
    assert api_client.get("/ops/retrains/pending", headers=ops_headers).json() == []


def test_a_second_retrain_decision_is_a_conflict(api_client, ops_headers, session):
    card = seed_card(session, disposition="ESCALATE_HUMAN", confidence=0.42)
    path = f"/ops/retrains/{card.card_id}/decision"
    api_client.post(path, headers=ops_headers, json=DECISION)
    assert api_client.post(path, headers=ops_headers, json=DECISION).status_code == 409


def test_a_bare_reason_is_refused(api_client, ops_headers, session):
    card = seed_card(session, disposition="ESCALATE_HUMAN", confidence=0.42)
    response = api_client.post(
        f"/ops/retrains/{card.card_id}/decision",
        headers=ops_headers,
        json={"decision": "APPROVE", "reason": "ok"},
    )
    assert response.status_code == 422
    assert "reason" in response.json()["detail"]


def test_an_unknown_card_is_not_found(api_client, ops_headers):
    response = api_client.post("/ops/retrains/dc-nope/decision", headers=ops_headers, json=DECISION)
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
@pytest.fixture
def gated_run(session):
    card = seed_card(session, disposition="ESCALATE_HUMAN", confidence=0.42)
    return seed_run(session, card.card_id)


def test_pending_promotions_carry_the_evidence(api_client, ops_headers, registry, gated_run):
    (entry,) = api_client.get("/ops/promotions/pending", headers=ops_headers).json()
    assert entry["run_id"] == gated_run.run_id
    assert entry["evidence"]["ready_to_approve"] is False
    assert entry["evidence"]["aliases"] == {"champion": CHAMPION, "shadow": CHALLENGER}


def test_an_approval_promotes_and_refreshes_serving(
    api_client, ops_headers, registry, refreshes, session, gated_run
):
    from loop.approval.rules import load_rules

    seed_shadow_rows(session, load_rules().min_shadow_requests)
    response = api_client.post(
        f"/ops/promotions/{gated_run.run_id}/decision", headers=ops_headers, json=DECISION
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["champion_alias_moved"] is True
    assert body["champion_version_after"] == CHALLENGER
    assert body["approver"] == "ops-alice"
    assert body["serving"]["champion_version"] == "2"
    assert registry.aliases["champion"] == CHALLENGER
    assert refreshes == [True]
    (row,) = session.scalars(select(Approval)).all()
    assert row.approver == "ops-alice"


def test_an_incomplete_window_is_a_conflict_and_moves_nothing(
    api_client, ops_headers, registry, refreshes, gated_run
):
    response = api_client.post(
        f"/ops/promotions/{gated_run.run_id}/decision", headers=ops_headers, json=DECISION
    )
    assert response.status_code == 409
    assert "preconditions not met" in response.json()["detail"]
    assert registry.aliases["champion"] == CHAMPION
    assert refreshes == []


def test_a_rejection_is_recorded_and_clears_shadow(
    api_client, ops_headers, registry, refreshes, gated_run
):
    response = api_client.post(
        f"/ops/promotions/{gated_run.run_id}/decision",
        headers=ops_headers,
        json={"decision": "REJECT", "reason": "shadow scores drift upward at night"},
    )
    assert response.status_code == 200
    assert response.json()["champion_alias_moved"] is False
    assert "shadow" not in registry.aliases
    assert refreshes == [True]


def test_an_unknown_run_is_not_found(api_client, ops_headers, registry):
    assert api_client.get("/ops/promotions/rr-nope", headers=ops_headers).status_code == 404
    response = api_client.post(
        "/ops/promotions/rr-nope/decision", headers=ops_headers, json=DECISION
    )
    assert response.status_code == 404


def test_promotion_needs_a_registry(api_client, ops_headers, gated_run):
    """An API serving the local artifact has no aliases to move."""
    api_client.app.state.alias_registry = None
    response = api_client.post(
        f"/ops/promotions/{gated_run.run_id}/decision", headers=ops_headers, json=DECISION
    )
    assert response.status_code == 503
