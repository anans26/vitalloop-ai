"""Week 11: the CI training smoke run and gate check (`scripts/ci_smoke.py`)."""

import dataclasses
import json

import pytest

from ml.data.clean import TARGET_COLUMN
from ml.data.features import MEDICATION_COLUMNS
from scripts import ci_smoke
from scripts.ci_smoke import SmokeError, run_smoke, synthetic_clean_frame
from tests.conftest import make_synthetic_clean_df

# For the paths where the gate is replaced and the sample's quality is moot.
SMALL = 1_000


@pytest.fixture(scope="module")
def summary():
    # The size CI runs. Smaller samples are genuinely too noisy for the gate's
    # absolute 0.05 ECE ceiling on a 15% window -- the gate is right to refuse.
    return run_smoke(ci_smoke.DEFAULT_ROWS)


def test_the_sample_is_deterministic():
    assert synthetic_clean_frame(300, seed=4).equals(synthetic_clean_frame(300, seed=4))
    assert not synthetic_clean_frame(300, seed=4).equals(synthetic_clean_frame(300, seed=5))


def test_the_sample_has_the_cleaned_contract():
    frame = synthetic_clean_frame(500)
    # Every column the shared synthetic fixture (the cleaned-split contract) has.
    assert set(make_synthetic_clean_df(10).columns) <= set(frame.columns)
    assert set(MEDICATION_COLUMNS) <= set(frame.columns)
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}
    # One encounter per patient and increasing encounters, as the split requires.
    assert frame["patient_nbr"].is_unique
    assert frame["encounter_id"].is_monotonic_increasing


def test_smoke_trains_and_the_gate_passes_the_retrain_and_blocks_the_bad_challenger(summary):
    assert summary["retrained_challenger"]["outcome"] == "PASS"
    assert summary["retrained_challenger"]["reasons"] == []
    assert summary["bad_challenger"]["outcome"] == "BLOCK"
    assert summary["bad_challenger"]["failed_checks"] > 0
    assert summary["criteria_version"] == "gate-v1"
    assert sum(summary["rows"].values()) == ci_smoke.DEFAULT_ROWS
    for ece in summary["retrained_challenger"]["ece_by_set"].values():
        assert ece <= 0.05
    assert 0.5 < summary["calibrated_roc_auc"] <= 1.0


def test_a_gate_that_blocks_the_retrain_fails_the_smoke(monkeypatch):
    from loop.gate import gate

    real = gate.evaluate_gate

    def strict(champion, challenger, criteria):
        return real(champion, challenger, dataclasses.replace(criteria, max_ece=0.0))

    monkeypatch.setattr("loop.gate.gate.evaluate_gate", strict)
    with pytest.raises(SmokeError, match="retrained challenger was BLOCKED"):
        run_smoke(SMALL)


def test_a_gate_that_passes_the_bad_challenger_fails_the_smoke(monkeypatch):
    from loop.gate.gate import PASS, GateResult

    def lenient(champion, challenger, criteria):
        return GateResult(PASS, criteria.version, (), (), (), {}, {}, {})

    monkeypatch.setattr("loop.gate.gate.evaluate_gate", lenient)
    with pytest.raises(SmokeError, match="bad challenger PASSED"):
        run_smoke(SMALL)


def test_cli_writes_the_summary_and_exits_zero(tmp_path, capsys, monkeypatch, summary):
    monkeypatch.setattr(ci_smoke, "run_smoke", lambda rows, seed: summary)
    out = tmp_path / "smoke.json"
    assert ci_smoke.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == summary
    assert (
        "gate PASS for the retrained challenger, BLOCK for the bad one" in capsys.readouterr().out
    )


def test_cli_exits_nonzero_on_a_smoke_failure(monkeypatch, capsys):
    def failing(rows, seed):
        raise SmokeError("boom")

    monkeypatch.setattr(ci_smoke, "run_smoke", failing)
    assert ci_smoke.main([]) == 1
    assert "SMOKE FAILED: boom" in capsys.readouterr().err
