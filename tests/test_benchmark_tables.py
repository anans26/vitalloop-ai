"""Week 12: `scenarios/benchmark.py` -- the S1-S5 results tables.

The metrics are pure functions over decisions and are tested as such. The
decision loop is then run end to end on injected measurements (no dataset, no
Evidently) through the real Decision Engine under both shipped policies, in the
module's own throwaway database -- asserting the properties the tables claim.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from loop.monitor.drift import FeatureDrift, WindowDrift, current_thresholds
from scenarios import benchmark
from scenarios.benchmark import (
    WindowDecision,
    breach_rate,
    detection_latency,
    render_markdown,
    run_benchmark,
    summarise,
    trigger_rate,
)

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def decision(scenario, index, action="NO_OP", breaches=0, **extra) -> WindowDecision:
    return WindowDecision(
        scenario=scenario,
        window_index=index,
        breaching_features=breaches,
        max_psi=0.3 if breaches else 0.02,
        prediction_drift=False,
        action=action,
        disposition="ESCALATE_HUMAN" if action != "NO_OP" else "NONE",
        confidence=0.4 if action != "NO_OP" else 0.19,
        rule_id="4" if action != "NO_OP" else "1",
        card_id=f"dc-{scenario}-{index}",
        **extra,
    )


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------
def test_latency_is_the_first_acting_window_counted_from_onset():
    rows = [decision("S1", 2, "FULL_RETRAIN"), decision("S1", 0), decision("S1", 1)]
    assert detection_latency(rows) == 2
    assert detection_latency([decision("S1", 0, "FULL_RETRAIN")]) == 0
    assert detection_latency([decision("S5", 0), decision("S5", 1)]) is None


def test_trigger_and_breach_rates_count_windows():
    rows = [decision("S5", 0, breaches=1), decision("S5", 1), decision("S5", 2, "ALERT_ONLY")]
    assert trigger_rate(rows) == pytest.approx(1 / 3)
    assert breach_rate(rows) == pytest.approx(1 / 3)
    assert trigger_rate([]) == 0.0 and breach_rate([]) == 0.0


def test_summary_reports_the_control_false_trigger_rate_and_gate_counts():
    rows = [
        decision("S1", 0, "FULL_RETRAIN", 2, gate_retrained="PASS", gate_bad="BLOCK"),
        decision("S1", 1, "FULL_RETRAIN", 2, gate_retrained="BLOCK", gate_bad="BLOCK"),
        decision("S5", 0),
        decision("S5", 1, "ALERT_ONLY", 1),
    ]
    summary = summarise({"policy-x": rows})["policy-x"]
    assert summary["control_false_trigger_rate"] == 0.5
    s1 = summary["scenarios"]["S1"]
    assert (s1["retrained_pass"], s1["retrained_block"], s1["bad_challenger_block"]) == (1, 1, 2)
    assert s1["detection_latency_windows"] == 0


def test_the_markdown_carries_every_scenario_and_the_gate_table():
    rows = [
        decision(
            "S1",
            0,
            "FULL_RETRAIN",
            2,
            gate_retrained="PASS",
            gate_bad="BLOCK",
            gate_bad_failed_checks=37,
        ),
        decision("S5", 0),
    ]
    text = render_markdown(
        {
            "lineage": {"git_commit": "abc", "git_dirty": False, "dvc_train_md5": "d1"},
            "gated_policy": "policy-v2",
            "summary": summarise({"policy-v2": rows}),
            "windows": {"policy-v2": [benchmark.asdict(r) for r in rows]},
        }
    )
    assert "| S1 | 1 |" in text and "| S5 | 1 |" in text
    assert "False-trigger rate on the S5 control: 0%" in text
    assert "PASS" in text and "BLOCK (37 failed checks)" in text
    assert "not detected" in text


# ---------------------------------------------------------------------------
# The loop, on injected measurements, through the real engine
# ---------------------------------------------------------------------------
def window(scenario: str, index: int, psi: float) -> WindowDrift:
    breaching = psi >= 0.10
    stats = (
        FeatureDrift("num_lab_procedures", "numerical", psi, 0.001, breaching, psi >= 0.25),
        *(FeatureDrift(f"quiet_{i}", "numerical", 0.01, 0.5, False, False) for i in range(24)),
    )
    start = EPOCH + timedelta(days=index)
    return WindowDrift(
        scenario=scenario,
        window_index=index,
        window_start=start,
        window_end=start + timedelta(days=1),
        feature_stats=stats,
        prediction_psi=0.01,
        prediction_drift=False,
        max_psi=psi,
        breaching_feature_count=int(breaching),
        reference_rows=10_000,
        current_rows=2_000,
        policy_thresholds={**current_thresholds(), "monitored_features": 25},
    )


def fake_measure(scenarios, *, max_windows=None):
    psi = {"S1": 0.30, "S2": 0.40, "S3": 0.30, "S4": 0.02, "S5": 0.02}
    return [window(s, i, psi[s]) for s in scenarios for i in range(max_windows or 2)]


def test_the_benchmark_decides_injected_windows_with_the_real_engine():
    result = run_benchmark(max_windows=2, gate=False, measure=fake_measure)

    for policy in benchmark.POLICIES:
        block = result["summary"][policy]
        assert block["control_false_trigger_rate"] == 0.0
        for scenario in ("S1", "S2", "S3"):
            assert block["scenarios"][scenario]["detection_latency_windows"] == 0
        assert block["scenarios"]["S4"]["detection_latency_windows"] is None
    assert result["gated_policy"] is None
    json.dumps(result, default=str)  # the artifact is serialisable


def test_the_gate_runs_only_on_retrain_cards_and_labels_both_challengers(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_gate(card, challenger, champion, frames):
        calls.append((card["action"], challenger.mode))
        blocked = challenger.mode == "demo-bad"
        return SimpleNamespace(
            outcome="BLOCK" if blocked else "PASS",
            reasons=("bad",) if blocked else (),
            failed_checks=lambda: ("x",) * (3 if blocked else 0),
        )

    monkeypatch.setattr("loop.gate.runner.gate_challenger", fake_gate)
    monkeypatch.setattr("loop.gate.datasets.evaluation_frames", lambda *a, **k: {})
    monkeypatch.setattr("loop.monitor.runner.load_serving_stream", lambda: None)
    monkeypatch.setattr("ml.train.load_models", lambda: (None, object()))

    result = run_benchmark(max_windows=1, gate=True, measure=fake_measure, challenger=object())

    assert calls and all(action == "FULL_RETRAIN" for action, _ in calls)
    assert {mode for _, mode in calls} == {"live", "demo-bad"}
    v2 = result["summary"]["policy-v2"]["scenarios"]
    assert v2["S1"]["retrained_pass"] == 1 and v2["S1"]["bad_challenger_block"] == 1
    assert v2["S5"]["gated_cards"] == 0
    # Only the policy in force is gated.
    assert all(r["gate_retrained"] is None for r in result["windows"]["policy-v1"])


def test_cli_writes_both_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda **kwargs: run_benchmark(max_windows=1, gate=False, measure=fake_measure),
    )
    assert benchmark.main(["--no-gate", "--output-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert set(data["summary"]) == set(benchmark.POLICIES)
    assert (tmp_path / "benchmark.md").read_text(encoding="utf-8").startswith("# S1-S5")
    assert "Detection under policy-v2" in capsys.readouterr().out
