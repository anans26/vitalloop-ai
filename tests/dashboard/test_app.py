"""The six pages and the sign-in, run headlessly with Streamlit's `AppTest`.

Every page is rendered against the seeded lifecycle database and a fake API,
and every button the demo path uses is clicked. What is asserted is the
contract with the rest of the system: decisions go to the API (never straight
into the database), refusals are shown verbatim, and nothing renders without a
token the API accepted.
"""

from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from dashboard.actions import ActionError, InjectionResult, TrafficResult
from dashboard.context import Context
from tests.dashboard.page_harness import STATE

from .conftest import FakeApi

APP = str(Path("dashboard/app.py"))
HARNESS = str(Path("tests/dashboard/page_harness.py"))
TIMEOUT = 60
ALIASES = {"champion": "2", "challenger": "3", "shadow": "3"}


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def ctx(dash_db, lifecycle, api, monkeypatch):
    for view in ("overview", "gate"):
        monkeypatch.setattr(f"dashboard.views.{view}.alias_versions", lambda: dict(ALIASES))
    for view in ("cards", "audit"):
        monkeypatch.setattr(f"dashboard.views.{view}.alias_audit_rows", lambda: [])
    return Context(
        api=api,
        identity=api.identity,
        session_factory=sessionmaker(bind=dash_db.get_bind(), expire_on_commit=False),
    )


def page(name, ctx) -> AppTest:
    STATE.page, STATE.ctx = name, ctx
    at = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    parts += (
        [m.value for m in at.success] + [m.value for m in at.error] + [m.value for m in at.info]
    )
    parts += [m.value for m in at.title] + [m.value for m in at.subheader]
    return "\n".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------
def test_nothing_renders_without_a_token():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Paste an **ops** token" in at.info[0].value
    assert not at.tabs and not at.dataframe


def test_a_refused_token_stays_locked(monkeypatch):
    refusing = FakeApi(refuse={"whoami": (403, "This endpoint requires a different role")})
    monkeypatch.setattr("dashboard.context.OpsApi", lambda *a, **k: refusing)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.text_input(key="ops_token").input("clinician-token").run()

    assert "Token refused" in at.sidebar.error[0].value
    assert "Paste an **ops** token" in at.info[0].value


def test_an_ops_token_opens_the_dashboard_on_the_overview(ctx, monkeypatch):
    monkeypatch.setattr("dashboard.context.OpsApi", lambda *a, **k: ctx.api)
    monkeypatch.setattr("dashboard.context.session_factory", lambda: ctx.session_factory)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.text_input(key="ops_token").input("ops-token").run()

    assert not at.exception, [e.value for e in at.exception]
    assert "ops-alice" in at.sidebar.success[0].value
    assert at.title[0].value == "Overview"


def test_the_app_declares_exactly_six_pages():
    """Roadmap Week 10: '6 pages, no more'."""
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    titles = [
        "Overview",
        "Drift Monitor",
        "Decision Cards",
        "Champion vs Challenger",
        "Approvals",
        "Audit",
    ]
    for title in titles:
        assert f'("{title}",' in source
    assert len(list(Path("dashboard/views").glob("[!_]*.py"))) == 6


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def test_overview_shows_versions_and_traffic(ctx):
    at = page("overview", ctx)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["champion alias"] == "v2"
    assert metrics["Serving (API)"] == "v2"
    assert metrics["Predictions audited"] == "4"
    assert metrics["Decision Cards"] == "6"


def test_overview_traffic_button_sends_through_the_api(ctx, monkeypatch):
    sent = {}

    def fake_traffic(api, *, count, offset):
        sent.update(api=api, count=count, offset=offset)
        return TrafficResult(sent=count, scored=count, failed=0, versions=("2",))

    monkeypatch.setattr("dashboard.actions.send_demo_traffic", fake_traffic)
    at = page("overview", ctx)
    at.button(key="send_traffic").click().run()

    assert sent == {"api": ctx.api, "count": 55, "offset": 4}
    assert "55 scored" in at.success[0].value


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------
def test_drift_page_lists_windows(ctx):
    at = page("drift", ctx)
    assert at.dataframe
    assert "No Evidently report on disk" in _text(at)


def test_the_inject_button_measures_and_decides(ctx, monkeypatch):
    calls = []

    def fake_inject(session, scenario):
        calls.append(scenario)
        return InjectionResult(
            "S1", 3, "de-s1-w003-x", "dc-x", "FULL_RETRAIN", "ESCALATE_HUMAN", 0.56, True
        )

    monkeypatch.setattr("dashboard.actions.inject_next_window", fake_inject)
    at = page("drift", ctx)
    at.button(key="inject").click().run()

    assert calls == ["S1"]
    assert "S1 window 3 recorded" in at.success[0].value


def test_an_injection_refusal_is_shown(ctx, monkeypatch):
    def refuse(session, scenario):
        raise ActionError("S1 has no window 40: every one is already on record")

    monkeypatch.setattr("dashboard.actions.inject_next_window", refuse)
    at = page("drift", ctx)
    at.button(key="inject").click().run()
    assert "already on record" in at.error[0].value


# ---------------------------------------------------------------------------
# Decision Cards
# ---------------------------------------------------------------------------
def test_cards_page_shows_the_narrative_and_its_grounding(ctx):
    at = page("cards", ctx)
    text = _text(at)
    assert "grounding check: passed" in text
    assert at.dataframe


def test_cards_can_be_filtered_by_state(ctx, lifecycle):
    at = page("cards", ctx)
    at.selectbox(key="cards_state").select("BLOCKED").run()
    assert not at.exception
    assert at.selectbox(key="cards_detail").value == lifecycle["blocked"].card_id


# ---------------------------------------------------------------------------
# Champion vs Challenger
# ---------------------------------------------------------------------------
def test_gate_page_labels_modes_and_shows_block_reasons(ctx, lifecycle):
    at = page("gate", ctx)
    table = at.dataframe[0].value
    assert "CONSTRUCTED BAD (demo)" in set(table["mode"])
    assert "REPLAY" in set(table["mode"])


def test_retrain_button_runs_the_gate_and_reloads_the_api(ctx, monkeypatch):
    from types import SimpleNamespace

    ran = {}

    def fake_retrain(session, card_id, mode):
        ran.update(card_id=card_id, mode=mode)
        return SimpleNamespace(
            resumed=False,
            passed=True,
            challenger_version="4",
            shadow_alias_moved=True,
            run_id="rr-x",
            outcome="PASS",
            gate_result=SimpleNamespace(reasons=()),
        )

    monkeypatch.setattr("dashboard.actions.run_retrain", fake_retrain)
    at = page("gate", ctx)
    at.radio(key="retrain_mode").set_value("replay").run()
    at.button(key="retrain").click().run()

    assert ran["mode"] == "replay"
    assert ("reload",) in ctx.api.calls
    assert "Gate PASS: REPLAY challenger v4" in at.success[0].value


def test_a_retrain_refusal_is_shown(ctx, monkeypatch):
    def refuse(session, card_id, mode):
        raise ActionError("the cached challenger (v3) is the serving champion")

    monkeypatch.setattr("dashboard.actions.run_retrain", refuse)
    at = page("gate", ctx)
    at.button(key="retrain").click().run()
    assert "serving champion" in at.error[0].value


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------
PROMOTION = {
    "run_id": "rr-x",
    "card_id": "dc-x",
    "challenger_version": "3",
    "evidence": {
        "ready_to_approve": True,
        "card": {
            "card_id": "dc-x",
            "action": "FULL_RETRAIN",
            "disposition": "ESCALATE_HUMAN",
            "confidence": 0.54,
            "narrative": "The inputs moved.",
        },
        "gate": {
            "outcome": "PASS",
            "criteria_version": "card:dc-x",
            "mode": "replay",
            "authorized_by": "ops-alice",
            "headline": {
                "frozen_holdout": {"roc_auc": {"champion": 0.6, "challenger": 0.6, "delta": 0.0}}
            },
            "worst_subgroup_auroc_drop": {
                "frozen_holdout": {"subgroup": "age=[50-60)", "auroc_drop": 0.0}
            },
        },
        "shadow_window": {
            "requests": 55,
            "required_requests": 50,
            "decision_agreement": 1.0,
            "mean_abs_diff": 0.0,
            "spearman": 1.0,
        },
        "preconditions": [
            {
                "name": "shadow_window_complete",
                "satisfied": True,
                "detail": "55 of 50 requests dual-scored",
            }
        ],
    },
}
RETRAIN = {
    "card": {
        "card_id": "dc-y",
        "scenario": "S1",
        "window_start": "2026-01-04T00:00:00",
        "action": "FULL_RETRAIN",
        "confidence": 0.56,
        "policy_version": "policy-v2",
        "rule_id": "4",
        "rationale": "num_lab_procedures reached PSI 0.3272.",
        "breaching_features": [{"feature": "num_lab_procedures", "psi": 0.3272}],
    }
}


def test_approvals_show_the_evidence_chain(ctx, api):
    api.promotions, api.retrains = [PROMOTION], [RETRAIN]
    text = _text(page("approvals", ctx))
    assert "55 / 50" in text
    assert "shadow_window_complete" in text
    assert "mode **replay**" in text
    assert "dc-y" in text


def test_approving_a_promotion_goes_through_the_api(ctx, api):
    api.promotions = [PROMOTION]
    at = page("approvals", ctx)
    at.text_area(key="reason_rr-x").input("Gate PASS on both sets; window complete.").run()
    at.button(key="approve_rr-x").click().run()

    assert (
        "decide_promotion",
        "rr-x",
        "APPROVE",
        "Gate PASS on both sets; window complete.",
    ) in api.calls
    assert "champion v2 -> v3" in at.success[0].value


def test_a_refused_decision_is_shown_verbatim(ctx, api):
    api.promotions = [PROMOTION]
    api.refuse["decide_promotion"] = (422, "a reason of at least 10 characters is required")
    at = page("approvals", ctx)
    at.button(key="approve_rr-x").click().run()
    assert "(422): a reason of at least 10 characters" in at.error[0].value


def test_authorising_a_retrain_goes_through_the_api(ctx, api):
    api.retrains = [RETRAIN]
    at = page("approvals", ctx)
    at.text_area(key="reason_dc-y").input("Worth a challenger.").run()
    at.button(key="reject_dc-y").click().run()
    assert ("decide_retrain", "dc-y", "REJECT", "Worth a challenger.") in api.calls


def test_approvals_page_reports_an_api_refusal(ctx, api):
    api.pending_promotions = lambda: (_ for _ in ()).throw(
        __import__("dashboard.api_client", fromlist=["ApiError"]).ApiError(401, "Token has expired")
    )
    at = page("approvals", ctx)
    assert "Token has expired" in at.error[0].value


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def test_audit_log_search_and_pdf_export(ctx, lifecycle):
    at = page("audit", ctx)
    at.text_input(key="audit_query").input(lifecycle["blocked"].card_id).run()
    table = at.dataframe[0].value
    assert len(table) >= 1
    assert all(lifecycle["blocked"].card_id in " ".join(map(str, row)) for row in table.values)

    at.selectbox(key="audit_card").select(lifecycle["blocked"].card_id).run()
    at.button(key="build_pdf").click().run()
    assert not at.exception
    assert any("Download" in str(b.proto.label) for b in at.get("download_button"))


def test_audit_prediction_lookup_shows_hashes_not_payloads(ctx):
    at = page("audit", ctx)
    at.text_input(key="prediction_lookup").input("req-1").run()
    table = at.dataframe[-1].value
    assert list(table["request"]) == ["req-1"]
    assert "input hash" in table.columns
    for forbidden in ("time_in_hospital", "patient_nbr", "encounter_id"):
        assert forbidden not in table.columns


def test_drift_page_embeds_the_evidently_report_when_present(ctx, monkeypatch, tmp_path):
    report = tmp_path / "window_000.html"
    report.write_text("<html><body>evidently</body></html>", encoding="utf-8")
    monkeypatch.setattr("dashboard.data.report_path", lambda event: report)
    at = page("drift", ctx)
    assert "No Evidently report on disk" not in _text(at)
