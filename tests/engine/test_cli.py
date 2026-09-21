"""The evaluator's command line, and the odd corners of the engine's plumbing.

The roadmap's Week 7 deliverable asks for ">= 95% branch coverage on the
engine"; the policy modules reach that from the rule tests alone, and this file
covers the parts around them -- argument parsing, the DVC lookup, and the
guards that only a malformed input reaches.
"""

import pytest

from loop.engine import evaluate as evaluate_module
from loop.engine.evaluate import build_parser, candidate_data_version, main
from loop.engine.evidence import DriftEvidence, FeatureBreach
from loop.engine.policy import DEFAULT_POLICY_VERSION, parse_policy
from loop.engine.rules import rule_2_first_window_mild
from tests.engine.conftest import WINDOW_START, feature_stat, make_evidence
from tests.engine.test_evaluate import CONTROL_WINDOWS, cards, seed_stream


@pytest.fixture
def cli(drift_db, monkeypatch):
    """Runs the CLI against the throwaway SQLite session instead of Postgres."""
    monkeypatch.setattr("loop.monitor.database.open_session", lambda: drift_db)
    monkeypatch.setattr(drift_db, "close", lambda: None)
    return drift_db


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def test_the_parser_defaults_to_the_shipped_policy():
    args = build_parser().parse_args([])
    assert args.policy == DEFAULT_POLICY_VERSION
    assert args.scenario is None
    assert args.dry_run is False


def test_the_cli_evaluates_pending_windows_and_reports_them(cli, capsys):
    seed_stream(cli, "S5", CONTROL_WINDOWS)

    assert main([]) == 0
    output = capsys.readouterr().out
    assert f"policy {DEFAULT_POLICY_VERSION}: 3 window(s) evaluated" in output
    assert "NO_OP" in output
    assert len(cards(cli)) == 3


def test_the_cli_is_idempotent_across_runs(cli, capsys):
    seed_stream(cli, "S5", CONTROL_WINDOWS)
    main([])
    capsys.readouterr()

    assert main([]) == 0
    assert "0 window(s) evaluated" in capsys.readouterr().out
    assert len(cards(cli)) == 3


def test_the_cli_can_restrict_itself_to_one_stream(cli, capsys):
    seed_stream(cli, "S5", CONTROL_WINDOWS)
    seed_stream(cli, "S1", [(("num_lab_procedures", 0.30),)])

    main(["--scenario", "S1"])
    assert {card.scenario for card in cards(cli)} == {"S1"}


def test_the_cli_can_limit_how_many_windows_it_decides(cli):
    seed_stream(cli, "S5", CONTROL_WINDOWS)
    main(["--limit", "1"])
    assert len(cards(cli)) == 1


def test_a_dry_run_writes_nothing(cli, capsys):
    seed_stream(cli, "S5", CONTROL_WINDOWS)

    assert main(["--dry-run"]) == 0
    assert "3 window(s) evaluated" in capsys.readouterr().out
    assert cards(cli) == []


def test_an_unknown_policy_version_fails_loudly(cli):
    from loop.engine.policy import PolicyError

    with pytest.raises(PolicyError):
        main(["--policy", "policy-v99"])


# ---------------------------------------------------------------------------
# The pinned candidate data version
# ---------------------------------------------------------------------------
def test_the_candidate_data_version_comes_from_dvc_lock():
    """§3.3: the card pins the exact dataset hash a retrain will run on."""
    resolved = candidate_data_version()
    assert resolved is None or (isinstance(resolved, str) and len(resolved) >= 8)


def test_an_unreadable_dvc_lock_yields_no_candidate_version(monkeypatch):
    """A missing hash must not stop a decision being recorded."""

    def explode(*args, **kwargs):
        raise OSError("dvc.lock is gone")

    monkeypatch.setattr("ml.tracking.dvc_lineage", explode)
    assert evaluate_module.candidate_data_version() is None


# ---------------------------------------------------------------------------
# Guards only a malformed input reaches
# ---------------------------------------------------------------------------
def test_a_policy_with_an_empty_version_is_refused():
    import yaml

    from loop.engine.policy import PolicyError, policy_path

    document = yaml.safe_load(policy_path().read_text(encoding="utf-8"))
    document["version"] = ""
    with pytest.raises(PolicyError, match="empty version"):
        parse_policy(document)


def test_a_breach_record_is_a_name_and_two_statistics():
    record = FeatureBreach("num_medications", 0.19, 0.004).to_record()
    assert record == {"feature": "num_medications", "psi": 0.19, "ks_p": 0.004}


def test_the_max_psi_falls_back_to_the_stats_when_the_row_omits_it():
    evidence = DriftEvidence(
        drift_event_id="de-x",
        scenario="T",
        window_start=WINDOW_START,
        window_end=WINDOW_START,
        feature_stats=(feature_stat("a", 0.31), feature_stat("b", 0.04)),
        max_psi=None,
    )
    assert evidence.observed_max_psi() == 0.31


def test_the_max_psi_of_a_window_with_no_stats_is_zero():
    evidence = DriftEvidence(
        drift_event_id="de-x",
        scenario="T",
        window_start=WINDOW_START,
        window_end=WINDOW_START,
        max_psi=None,
    )
    assert evidence.observed_max_psi() == 0.0


def test_rule_2_declines_a_severe_breach_when_asked_directly(policy):
    """Unreachable through the precedence -- rule 4 claims it first -- but the
    rule must still be correct on its own terms."""
    from loop.engine.confidence import compute_confidence

    evidence = make_evidence(stats=(feature_stat("a", 0.30),))
    assert rule_2_first_window_mild(evidence, policy, compute_confidence(evidence, policy)) is None
