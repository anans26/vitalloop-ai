"""The gate runner's command line, the replay script, and the default wiring.

`test_runner.py` covers the orchestration with every impure step injected. This
file covers the parts around it: argument parsing, the CLI's own reporting, and
the four `_default_*` helpers that decide what those injected arguments are in
production. Those helpers are thin by design -- each one resolves a collaborator
and delegates -- so what is asserted is the delegation, not the collaborator.
"""

import pytest

from db.models import RetrainRun
from loop.gate import runner as runner_module
from loop.gate.criteria import FROZEN_HOLDOUT, RECENT_LABELED_WINDOW
from loop.gate.runner import SHADOW_ALIAS, build_parser, main
from ml.retrain import MODE_LIVE, MODE_REPLAY
from scripts import replay_retrain

from .conftest import challenger_run, seed_card


@pytest.fixture
def cli(gate_db, monkeypatch, gate_frames, champion_model, card_row):
    """Runs the CLI against the throwaway SQLite session instead of Postgres."""
    monkeypatch.setattr("loop.monitor.database.open_session", lambda: gate_db)
    monkeypatch.setattr(gate_db, "close", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "_default_champion_loader",
        lambda: (champion_model, "1"),
    )
    monkeypatch.setattr(
        runner_module,
        "_default_frames",
        lambda scenario, window_start, sets: {name: gate_frames[name] for name in sets},
    )
    monkeypatch.setattr(
        runner_module,
        "_default_alias_setter",
        lambda version, run_id, reason: {"alias": SHADOW_ALIAS, "version": version},
    )
    monkeypatch.setattr(
        runner_module,
        "_default_retrainer",
        lambda card, *, mode, **_: challenger_run(champion_model, mode=mode),
    )
    return gate_db


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def test_the_parser_defaults_to_a_live_run_over_the_backlog():
    args = build_parser().parse_args([])
    assert args.mode == MODE_LIVE
    assert args.card is None
    assert args.dry_run is False
    assert args.authorized_by is None


def test_the_parser_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "teleport"])


def test_the_cli_gates_the_backlog_and_reports_it(cli, capsys):
    assert main([]) == 0
    output = capsys.readouterr().out

    assert "1 card(s) gated" in output
    assert "PASS" in output
    assert cli.query(RetrainRun).count() == 1


def test_the_cli_is_idempotent_across_runs(cli, capsys):
    main([])
    capsys.readouterr()

    assert main([]) == 0
    assert "0 card(s) gated" in capsys.readouterr().out
    assert cli.query(RetrainRun).count() == 1


def test_the_cli_can_gate_one_named_card(cli, card_row, capsys):
    assert main(["--card", card_row.card_id]) == 0
    assert card_row.card_id in capsys.readouterr().out


def test_an_unknown_card_is_reported_rather_than_crashed(cli, capsys):
    assert main(["--card", "dc-nope"]) == 2
    assert "no decision card" in capsys.readouterr().err


def test_the_cli_can_restrict_itself_to_one_stream(cli, capsys):
    assert main(["--scenario", "S9"]) == 0
    assert "0 card(s) gated" in capsys.readouterr().out


def test_the_cli_can_limit_how_many_cards_it_gates(cli):
    seed_card(cli, card_id="dc-2026-01-02-ccccdddd")
    main(["--limit", "1"])
    assert cli.query(RetrainRun).count() == 1


def test_a_dry_run_writes_nothing(cli, capsys):
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "(dry run)" in output
    assert cli.query(RetrainRun).count() == 0


def test_the_cli_can_replay(cli, capsys):
    assert main(["--mode", MODE_REPLAY]) == 0
    assert "replay" in capsys.readouterr().out


def test_the_cli_prints_every_block_reason(cli, card_row, champion_model, monkeypatch, capsys):
    """ "bad challenger BLOCKED with reasons" has to reach the operator's screen."""
    from .conftest import InvertedChallenger

    monkeypatch.setattr(
        runner_module,
        "_default_retrainer",
        lambda card, *, mode, **_: challenger_run(InvertedChallenger(champion_model), mode=mode),
    )
    assert main([]) == 0
    output = capsys.readouterr().out

    assert "BLOCK" in output
    stored = cli.query(RetrainRun).one()
    assert output.count("      BLOCK:") == stored.failed_criteria_count


def test_an_unknown_criteria_version_fails_loudly(cli):
    from loop.gate.criteria import CriteriaError

    with pytest.raises(CriteriaError):
        main(["--criteria", "gate-v99"])


def test_the_cli_forwards_an_authorisation(cli, capsys):
    escalated = seed_card(
        cli, card_id="dc-2026-01-02-ccccdddd", disposition="ESCALATE_HUMAN", confidence=0.46
    )
    assert main(["--card", escalated.card_id, "--authorized-by", "ops-alice"]) == 0
    assert cli.get(RetrainRun, cli.query(RetrainRun).one().run_id).authorized_by == "ops-alice"


# ---------------------------------------------------------------------------
# scripts/replay_retrain.py (ARCHITECTURE.md §10 names this file)
# ---------------------------------------------------------------------------
def test_the_replay_script_requires_a_card():
    with pytest.raises(SystemExit):
        replay_retrain.build_parser().parse_args([])


def test_the_replay_script_gates_a_cached_challenger(
    cli, card_row, champion_model, monkeypatch, capsys
):
    """The script resolves `retrain_from_card` through its own module globals."""
    seen = {}

    def fake(card, *, mode, replay_version=None, **_):
        seen.update({"mode": mode, "replay_version": replay_version})
        return challenger_run(champion_model, mode=mode)

    monkeypatch.setattr(replay_retrain, "retrain_from_card", fake)

    assert replay_retrain.main(["--card", card_row.card_id, "--version", "4"]) == 0
    output = capsys.readouterr().out

    assert output.startswith("REPLAY")
    assert seen == {"mode": MODE_REPLAY, "replay_version": "4"}
    assert cli.query(RetrainRun).one().mode == MODE_REPLAY


def test_the_replay_script_exits_non_zero_on_a_block(
    cli, card_row, champion_model, monkeypatch, capsys
):
    """A blocked replay must not look like a successful demo to a shell."""
    from .conftest import InvertedChallenger

    monkeypatch.setattr(
        replay_retrain,
        "retrain_from_card",
        lambda card, *, mode, **_: challenger_run(InvertedChallenger(champion_model), mode=mode),
    )
    assert replay_retrain.main(["--card", card_row.card_id]) == 1
    assert "BLOCK:" in capsys.readouterr().out


def test_the_replay_script_can_dry_run(cli, card_row, champion_model, monkeypatch, capsys):
    monkeypatch.setattr(
        replay_retrain,
        "retrain_from_card",
        lambda card, *, mode, **_: challenger_run(champion_model, mode=mode),
    )
    assert replay_retrain.main(["--card", card_row.card_id, "--dry-run"]) == 0
    assert "(dry run)" in capsys.readouterr().out
    assert cli.query(RetrainRun).count() == 0


def test_the_replay_script_reports_an_unknown_card(cli, capsys):
    assert replay_retrain.main(["--card", "dc-nope"]) == 2
    assert "no decision card" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The default wiring
# ---------------------------------------------------------------------------
def test_the_default_retrainer_delegates_to_the_retrain_pipeline(monkeypatch):
    seen = {}

    def fake(card, *, mode, **kwargs):
        seen.update({"card": card, "mode": mode})
        return "challenger"

    monkeypatch.setattr("ml.retrain.retrain_from_card", fake)
    assert runner_module._default_retrainer({"card_id": "dc-x"}, mode=MODE_LIVE) == "challenger"
    assert seen == {"card": {"card_id": "dc-x"}, "mode": MODE_LIVE}


def test_the_default_champion_loader_delegates_to_the_registry(monkeypatch):
    monkeypatch.setattr("ml.retrain.load_champion", lambda: ("model", "7"))
    assert runner_module._default_champion_loader() == ("model", "7")


def test_the_default_frames_builder_delegates_to_the_dataset_module(monkeypatch):
    seen = {}

    def fake(scenario, window_start, *, sets):
        seen.update({"scenario": scenario, "sets": sets})
        return {"frozen_holdout": "frame"}

    monkeypatch.setattr("loop.gate.datasets.evaluation_frames", fake)
    sets = (FROZEN_HOLDOUT, RECENT_LABELED_WINDOW)
    assert runner_module._default_frames("S1", "2026-01-01", sets) == {"frozen_holdout": "frame"}
    assert seen == {"scenario": "S1", "sets": sets}


def test_the_default_alias_setter_moves_shadow_and_only_shadow(monkeypatch, tmp_path):
    """§3.5: the registry helper is the only way an alias moves, and it audits."""
    seen = {}

    def fake_set_alias(client, **kwargs):
        seen.update(kwargs)
        return kwargs

    monkeypatch.setattr("ml.registry.set_alias", fake_set_alias)
    monkeypatch.setattr(
        "ml.tracking.resolve_tracking_uri",
        lambda: f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
    )

    runner_module._default_alias_setter("3", "run-1", "because")
    assert seen["alias"] == SHADOW_ALIAS
    assert seen["version"] == "3"
    assert seen["reason"] == "because"
