"""Week 11: the Overview's *Demo state* panel.

Replay is only a fast path while the cached challenger is not the serving
champion. The panel tells the person running the demo which of the three
states the stack is in, and what to run when it is not ready.
"""

import pytest

from dashboard.views.overview import DEMO_PATH, replay_status
from tests.dashboard.test_app import ALIASES, api, ctx, page  # noqa: F401  (fixtures)


@pytest.mark.parametrize(
    ("aliases", "ready", "expected"),
    [
        ({"champion": "1", "challenger": "2", "shadow": None}, True, "Replay ready"),
        ({"champion": "2", "challenger": "2", "shadow": None}, False, "v2 is already the champion"),
        ({"champion": "1", "challenger": None, "shadow": None}, False, "none is cached"),
        ({"champion": None, "challenger": None, "shadow": None}, False, "not been seeded"),
    ],
)
def test_replay_status_names_the_state_and_the_way_out(aliases, ready, expected):
    is_ready, message = replay_status(aliases)
    assert is_ready is ready
    assert expected in message
    if not ready:
        assert "scripts.seed_demo" in message


def test_replay_ready_says_the_run_is_recorded_as_replay():
    _, message = replay_status({"champion": "1", "challenger": "2"})
    assert "REPLAY" in message and "never as a retrain" in message


def test_unavailable_replay_points_at_the_archiving_reset_not_a_bare_wipe():
    _, message = replay_status({"champion": "2", "challenger": "2"})
    assert "scripts.reset_demo --yes" in message
    assert "archives first" in message


def test_the_demo_path_starts_from_the_seeded_state():
    assert "scripts.reset_demo --yes" in DEMO_PATH
    assert "scripts.seed_demo" in DEMO_PATH
    assert "Replay ready" in DEMO_PATH


def test_overview_renders_the_panel(ctx):  # noqa: F811
    at = page("overview", ctx)
    assert "Demo state" in [h.value for h in at.subheader]
    # tests/dashboard/test_app.py's aliases: champion v2, cached challenger v3.
    assert ALIASES["challenger"] != ALIASES["champion"]
    assert any("Replay ready" in s.value for s in at.success)


def test_overview_warns_when_replay_is_unavailable(ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        "dashboard.views.overview.alias_versions",
        lambda: {"champion": "3", "challenger": "3", "shadow": None},
    )
    at = page("overview", ctx)
    assert any("Replay unavailable" in w.value for w in at.warning)
    assert not any("Replay ready" in s.value for s in at.success)
