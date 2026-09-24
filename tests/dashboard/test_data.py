"""The dashboard's read side: every number a page shows, tested without a browser."""

from pathlib import Path

import pytest

from dashboard import data
from db.models import DriftEvent

from .conftest import FIXED_AT


# ---------------------------------------------------------------------------
# Lifecycle state -- derived from appended rows, never stored
# ---------------------------------------------------------------------------
def test_every_seeded_card_lands_in_its_state(dash_db, lifecycle):
    states = {card.card_id: state for card, state in data.cards_with_state(dash_db)}
    assert states[lifecycle["awaiting"].card_id] == data.STATE_AWAITING_AUTHORISATION
    assert states[lifecycle["promoted"].card_id] == data.STATE_PROMOTED
    assert states[lifecycle["blocked"].card_id] == data.STATE_BLOCKED
    assert states[lifecycle["shadow"].card_id] == data.STATE_IN_SHADOW
    assert states[lifecycle["noop"].card_id] == data.STATE_NO_OP
    assert states[lifecycle["rejected"].card_id] == data.STATE_RETRAIN_REJECTED


class _Card:
    def __init__(self, action, disposition="NONE"):
        self.action, self.disposition = action, disposition


class _Run:
    def __init__(self, run_id, outcome, moved=True, created_at=FIXED_AT):
        self.run_id, self.outcome, self.shadow_alias_moved, self.created_at = (
            run_id,
            outcome,
            moved,
            created_at,
        )


class _Approval:
    def __init__(self, kind, decision, run_id=None):
        self.kind, self.decision, self.run_id = kind, decision, run_id


@pytest.mark.parametrize(
    ("card", "runs", "approvals", "state"),
    [
        (_Card("ALERT_ONLY"), [], [], data.STATE_ALERT),
        (_Card("ALERT_ONLY", "ESCALATE_HUMAN"), [], [], data.STATE_ALERT_ESCALATED),
        (_Card("FULL_RETRAIN", "AUTO_PROCEED_SHADOW"), [], [], data.STATE_AWAITING_RETRAIN),
        (
            _Card("FULL_RETRAIN", "ESCALATE_HUMAN"),
            [],
            [_Approval("RETRAIN", "APPROVE")],
            data.STATE_AWAITING_RETRAIN,
        ),
        (_Card("FULL_RETRAIN"), [_Run("r1", "PASS", moved=False)], [], data.STATE_PASSED),
        (
            _Card("FULL_RETRAIN"),
            [_Run("r1", "PASS")],
            [_Approval("PROMOTION", "REJECT", "r1")],
            data.STATE_PROMOTION_REJECTED,
        ),
        (_Card("SOMETHING_ELSE"), [], [], "SOMETHING_ELSE"),
    ],
)
def test_lifecycle_state_branches(card, runs, approvals, state):
    assert data.lifecycle_state(card, runs, approvals) == state


def test_the_latest_run_decides_a_promoted_then_blocked_card():
    """The demo's step 6: a promoted card re-run with the bad challenger is BLOCKED now."""
    from datetime import timedelta

    runs = [
        _Run("r1", "PASS"),
        _Run("r2", "BLOCK", moved=False, created_at=FIXED_AT + timedelta(hours=1)),
    ]
    approvals = [_Approval("PROMOTION", "APPROVE", "r1")]
    assert data.lifecycle_state(_Card("FULL_RETRAIN"), runs, approvals) == data.STATE_BLOCKED


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def test_overview_counts(dash_db, lifecycle):
    view = data.overview(dash_db)
    assert view.predictions == view.successful_predictions == 4
    assert view.predictions_by_version == {"1": 3, "2": 1}
    assert view.median_latency_ms == 120.0
    assert view.cards == 6
    assert view.shadow_scores == 5
    assert sum(view.cards_by_state.values()) == 6


def test_overview_of_an_empty_database(dash_db):
    view = data.overview(dash_db)
    assert view.predictions == 0
    assert view.median_latency_ms is None
    assert view.cards_by_state == {}


# ---------------------------------------------------------------------------
# Cards, history, gate
# ---------------------------------------------------------------------------
def test_card_filters(dash_db, lifecycle):
    assert [c.card_id for c, _ in data.cards_with_state(dash_db, action="NO_OP")] == [
        lifecycle["noop"].card_id
    ]
    assert data.cards_with_state(dash_db, scenario="S9") == []


def test_card_history_carries_the_lineage(dash_db, lifecycle):
    card_id = lifecycle["promoted"].card_id
    alias_rows = [
        {"reason": f"week-9 promotion approved for decision card {card_id}", "alias": "champion"},
        {"reason": "unrelated move", "alias": "shadow"},
    ]
    history = data.card_history(dash_db, card_id, alias_rows)

    assert history.state == data.STATE_PROMOTED
    assert history.event.event_id == history.card.drift_event_id
    assert len(history.runs) == 1
    assert [a.kind for a in history.approvals] == ["RETRAIN", "PROMOTION"]
    assert history.alias_moves == [alias_rows[0]]


def test_card_history_of_an_unknown_card(dash_db):
    assert data.card_history(dash_db, "dc-nope") is None


def test_alias_moves_match_on_run_ids_too():
    rows = [{"reason": "gate PASS for rr-x-1"}, {"reason": "other"}]
    assert data.alias_moves_for("dc-x", ["rr-x-1"], rows) == [rows[0]]


def test_headline_metrics_put_champion_beside_challenger(dash_db, lifecycle):
    (run,) = data.card_history(dash_db, lifecycle["shadow"].card_id).runs
    auc = next(m for m in data.headline_metrics(run) if m["metric"] == "roc_auc")
    assert auc == {
        "evaluation_set": "frozen_holdout",
        "metric": "roc_auc",
        "champion": 0.60,
        "challenger": 0.61,
        "delta": 0.01,
    }


def test_gate_checks_filter_failures():
    class Run:
        gate_result = {"checks": [{"passed": True}, {"passed": False, "criterion": "auroc"}]}

    assert len(data.gate_checks(Run())) == 2
    assert data.gate_checks(Run(), failed_only=True) == [{"passed": False, "criterion": "auroc"}]


def test_retrain_runs_are_newest_first(dash_db, lifecycle):
    runs = data.retrain_runs(dash_db)
    assert len(runs) == 3
    stamps = [data._aware(r.created_at) for r in runs]
    assert stamps == sorted(stamps, reverse=True)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def test_drift_events_by_scenario(dash_db, lifecycle):
    assert data.scenarios(dash_db) == ["S1"]
    assert len(data.drift_events(dash_db, "S1")) == 6
    assert data.drift_events(dash_db, "S2") == []


def test_report_path_refuses_to_leave_the_project(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "w.html").write_text("<html/>", encoding="utf-8")
    inside = DriftEvent(report_uri="reports/w.html")
    outside = DriftEvent(report_uri="../secrets.html")
    missing = DriftEvent(report_uri="reports/none.html")

    assert data.report_path(inside, root=tmp_path) == (tmp_path / "reports" / "w.html").resolve()
    assert data.report_path(outside, root=tmp_path) is None
    assert data.report_path(missing, root=tmp_path) is None
    assert data.report_path(DriftEvent(report_uri=None), root=Path(tmp_path)) is None


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def test_the_audit_log_spans_every_table(dash_db, lifecycle):
    alias_rows = [
        {
            "timestamp_utc": "2026-09-24T13:00:00+00:00",
            "action": "set_alias",
            "alias": "champion",
            "from_version": "1",
            "to_version": "2",
            "actor": "ops-alice",
            "reason": "promotion",
        }
    ]
    events = data.audit_events(dash_db, alias_rows)
    assert {e["kind"] for e in events} == set(data.EVENT_KINDS)
    stamps = [e["ts"] for e in events if e["ts"] is not None]
    assert stamps == sorted(stamps, key=data._aware, reverse=True)


def test_the_audit_log_is_searchable_and_filterable(dash_db, lifecycle):
    card_id = lifecycle["promoted"].card_id
    hits = data.audit_events(dash_db, query=card_id)
    assert hits and all(card_id in " ".join(str(v) for v in e.values()) for e in hits)
    assert {e["kind"] for e in data.audit_events(dash_db, kinds=("approval",))} == {"approval"}
    assert data.audit_events(dash_db, query="no-such-thing-anywhere") == []


def test_the_audit_log_names_the_bad_challenger_mode(dash_db, lifecycle):
    (event,) = data.audit_events(dash_db, query="demo-bad", kinds=("retrain",))
    assert "BLOCK" in event["summary"]


def test_prediction_lookup_by_request_hash_or_caller(dash_db, lifecycle):
    assert [p.request_id for p in data.find_predictions(dash_db, "req-1")] == ["req-1"]
    assert [p.request_id for p in data.find_predictions(dash_db, "hash-req-2")] == ["req-2"]
    assert len(data.find_predictions(dash_db, "dr-a")) == 4
    assert data.find_predictions(dash_db, "  ") == []
