"""The dashboard's demo buttons: injection, retrain, traffic.

Injection runs the worker's own chain on a synthetic stream, model and
reference (the same pieces `tests/loop/test_worker.py` uses), so it needs no
dataset, trained artifact or database service.
"""

import pandas as pd
import pytest
from sqlalchemy import func, select

from dashboard import actions
from db.models import DecisionCard, DriftEvent
from loop.monitor.reference import build_reference
from loop.monitor.scoring import ScoringModel

from .conftest import FakeApi

STREAM_ROWS = 4000
WINDOW_ROWS = 1000


@pytest.fixture(scope="module")
def pieces():
    from ml.data.features import split_features_target
    from ml.train import build_base_model
    from tests.conftest import make_synthetic_clean_df

    stream = make_synthetic_clean_df(n_rows=STREAM_ROWS, seed=31)
    X, y = split_features_target(stream)
    base = build_base_model(n_estimators=5)
    base.fit(X, y)
    model = ScoringModel(model=base, version="test-model", source="local")
    half = STREAM_ROWS // 2
    return {
        "model": model,
        "reference": build_reference(stream.iloc[:half], model, rows=half),
        "stream": stream.iloc[half:].reset_index(drop=True),
    }


def _inject(session, pieces, scenario="S1"):
    return actions.inject_next_window(
        session,
        scenario,
        **pieces,
        window_rows=WINDOW_ROWS,
        reports_dir=None,
        data_version="dvc-train-hash",
    )


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


# ---------------------------------------------------------------------------
# Inject drift
# ---------------------------------------------------------------------------
def test_injection_measures_and_decides_one_window(dash_db, pieces):
    result = _inject(dash_db, pieces)

    assert result.window_index == 0
    assert result.card_written
    assert _count(dash_db, DriftEvent) == 1
    assert _count(dash_db, DecisionCard) == 1
    card = dash_db.get(DecisionCard, result.card_id)
    assert card.drift_event_id == result.event_id
    assert card.card_json["narrative_source"].startswith("template/")


def test_pressing_again_measures_the_next_window_never_the_same_one(dash_db, pieces):
    first, second = _inject(dash_db, pieces), _inject(dash_db, pieces)

    assert (first.window_index, second.window_index) == (0, 1)
    starts = list(
        dash_db.scalars(select(DriftEvent.window_start).order_by(DriftEvent.window_start))
    )
    assert len(starts) == len(set(starts)) == 2


def test_scenarios_are_counted_independently(dash_db, pieces):
    _inject(dash_db, pieces, "S1")
    assert _inject(dash_db, pieces, "S2").window_index == 0
    assert actions.next_window_index(dash_db, "S1") == 1


def test_an_exhausted_stream_is_refused(dash_db, pieces):
    for _ in range(2):  # 2000 rows of stream / 1000 per window
        _inject(dash_db, pieces)
    with pytest.raises(actions.ActionError, match="already on record"):
        _inject(dash_db, pieces)
    assert _count(dash_db, DriftEvent) == 2


def test_only_the_roadmaps_scenarios_are_wired(dash_db, pieces):
    with pytest.raises(actions.ActionError, match="wired to"):
        _inject(dash_db, pieces, "S5")


def test_next_window_index_reads_naive_sqlite_timestamps(dash_db, lifecycle):
    """The Week 7 seed cards sit on 2026-01-01, window 0 of the benchmark epoch."""
    assert actions.next_window_index(dash_db, "S1") == 1
    assert actions.next_window_index(dash_db, "S2") == 0


# ---------------------------------------------------------------------------
# Retrain + gate
# ---------------------------------------------------------------------------
class _Registry:
    def __init__(self, **aliases):
        self.aliases = aliases

    def version_of(self, alias):
        return self.aliases.get(alias)


def test_replay_of_the_serving_champion_is_refused_up_front(dash_db, lifecycle):
    card_id = lifecycle["awaiting"].card_id
    with pytest.raises(actions.ActionError, match="is the serving champion"):
        actions.run_retrain(
            dash_db, card_id, "replay", registry=_Registry(champion="3", challenger="3")
        )


def test_replay_without_a_cached_challenger_is_refused(dash_db, lifecycle):
    with pytest.raises(actions.ActionError, match="no cached challenger"):
        actions.run_retrain(
            dash_db, lifecycle["awaiting"].card_id, "replay", registry=_Registry(champion="1")
        )


def test_unknown_card_and_mode_are_refused(dash_db, lifecycle):
    with pytest.raises(actions.ActionError, match="no decision card"):
        actions.run_retrain(dash_db, "dc-nope", "live")
    with pytest.raises(actions.ActionError, match="unknown mode"):
        actions.run_retrain(dash_db, lifecycle["awaiting"].card_id, "turbo")


def test_runner_refusals_surface_as_action_errors(dash_db, lifecycle):
    """An escalated card nobody approved: Week 8's refusal, shown verbatim."""
    with pytest.raises(actions.ActionError, match="escalated"):
        actions.run_retrain(dash_db, lifecycle["awaiting"].card_id, "demo-bad")


def test_retrain_delegates_to_the_week_8_runner(dash_db, lifecycle, monkeypatch):
    seen = {}

    def fake_run_card(session, card, *, mode, **kwargs):
        seen.update(card_id=card.card_id, mode=mode, kwargs=kwargs)
        return "run"

    monkeypatch.setattr("loop.gate.runner.run_card", fake_run_card)
    monkeypatch.setattr("ml.tracking.make_console_encoding_safe", lambda: None)
    card_id = lifecycle["shadow"].card_id
    assert actions.run_retrain(dash_db, card_id, "live", persist=True) == "run"
    assert seen == {"card_id": card_id, "mode": "live", "kwargs": {"persist": True}}


# ---------------------------------------------------------------------------
# Demo traffic
# ---------------------------------------------------------------------------
def _rows(synthetic_clean_df, n=4):
    return synthetic_clean_df.head(n)


def test_traffic_goes_through_the_api(synthetic_clean_df):
    api = FakeApi()
    result = actions.send_demo_traffic(api, count=4, rows=_rows(synthetic_clean_df))
    assert (result.sent, result.scored, result.failed) == (4, 4, 0)
    assert result.versions == ("2",)
    assert api.calls.count(("predict",)) == 4


def test_traffic_failures_are_counted_and_reported(synthetic_clean_df):
    api = FakeApi(refuse={"predict": (503, "Model is not loaded")})
    result = actions.send_demo_traffic(api, count=3, rows=_rows(synthetic_clean_df, 3))
    assert (result.scored, result.failed) == (0, 3)
    assert result.errors == ("HTTP 503: Model is not loaded",)


def test_traffic_reads_the_stream_when_no_rows_are_given(synthetic_clean_df, monkeypatch):
    calls = {}

    def load_rows(*, offset, count):
        calls.update(offset=offset, count=count)
        return pd.DataFrame(_rows(synthetic_clean_df, count))

    monkeypatch.setattr("scripts.send_traffic.load_rows", load_rows)
    result = actions.send_demo_traffic(FakeApi(), count=2, offset=7)
    assert calls == {"offset": 7, "count": 2}
    assert result.scored == 2
