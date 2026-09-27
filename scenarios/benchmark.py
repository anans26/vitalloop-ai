"""The S1-S5 benchmark results tables (IMPLEMENTATION_ROADMAP.md Week 12).

Week 12: *"results tables from the S1-S5 benchmark (detection latency,
false-trigger rate, gate outcomes)"*. RESEARCH_NOVELTY.md C6 makes the same
benchmark the project's evaluation harness. This module regenerates those
tables from nothing but the committed code and the DVC-tracked data:

    python -m scenarios.benchmark                  # all complete windows, gate included
    python -m scenarios.benchmark --no-gate        # decisions only (no retrain, ~5 min)

It writes `reports/benchmark.json` (every number, per window) and
`reports/benchmark.md` (the tables the final report quotes).

**Nothing here is a second implementation.** Each window is measured by the
monitor's `measure_window`, recorded by `record_window`, and decided by the
Decision Engine's `evaluate_event` under each shipped policy -- the worker's
own chain. The gate is `loop.gate.runner.gate_challenger` with each card's
pinned criteria and `loop.gate.datasets.evaluation_frames`, exactly as a
card-driven retrain is gated.

**Nothing touches the live system.** Rows go to a throwaway SQLite database in
a temporary directory; no Evidently report is written; no model is registered
and no alias moves.

**One retrain, gated per card.** Every card pins the same training-data hash
(`candidate_data_version`), and `ml.retrain` trains on exactly that data
whatever the card, so the challenger a live retrain would produce is the same
model for every card. It is therefore trained once, with the production entry
point (`ml.retrain.train_challenger`), and then gated against each retrain
card's own evaluation sets (the frozen holdout plus that card's own labelled
window). Beside it, every card is also gated against the deliberately bad
challenger (`loop.gate.demo.InvertedModel`).

Definitions used in the tables:

* **Onset.** Every scenario transforms the whole serving stream, so drift is
  present from window 0; S5 is the untouched stream.
* **Detection latency** -- the index of the first window whose card is not
  `NO_OP`, counted from onset (0 = caught in the first drifted window). "not
  detected" when no window's card acts.
* **False-trigger rate** -- on the S5 control, the fraction of windows whose
  card is not `NO_OP`. S4 is *not* a control: its labels drift, so a quiet S4
  is a miss by design (rule 5 needs matured labels), reported as such.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ml.config import REPORTS_DIR

POLICIES = ("policy-v1", "policy-v2")
GATED_POLICY = "policy-v2"  # the version in force; its cards are the ones gated
CONTROL = "S5"
LABEL_DRIFT = "S4"
JSON_PATH = REPORTS_DIR / "benchmark.json"
MARKDOWN_PATH = REPORTS_DIR / "benchmark.md"
NO_OP = "NO_OP"


@dataclass
class WindowDecision:
    scenario: str
    window_index: int
    breaching_features: int
    max_psi: float
    prediction_drift: bool
    action: str
    disposition: str
    confidence: float
    rule_id: str
    card_id: str
    gate_retrained: str | None = None
    gate_retrained_reasons: list[str] = field(default_factory=list)
    gate_bad: str | None = None
    gate_bad_failed_checks: int | None = None


# ---------------------------------------------------------------------------
# The pure half: the metrics, from decisions
# ---------------------------------------------------------------------------
def detection_latency(decisions: Iterable[WindowDecision]) -> int | None:
    """Windows from onset to the first acting card; None if none acts."""
    for decision in sorted(decisions, key=lambda d: d.window_index):
        if decision.action != NO_OP:
            return decision.window_index
    return None


def trigger_rate(decisions: Iterable[WindowDecision]) -> float:
    """Fraction of windows whose card acts. On the control, the false-trigger rate."""
    decisions = list(decisions)
    if not decisions:
        return 0.0
    return sum(d.action != NO_OP for d in decisions) / len(decisions)


def breach_rate(decisions: Iterable[WindowDecision]) -> float:
    """Fraction of windows where the *monitor* saw a breach (before any policy)."""
    decisions = list(decisions)
    if not decisions:
        return 0.0
    return sum(d.breaching_features > 0 or d.prediction_drift for d in decisions) / len(decisions)


def summarise(results: dict[str, list[WindowDecision]]) -> dict:
    """`results` maps a policy version to its decisions over every scenario."""
    summary: dict = {}
    for policy, decisions in results.items():
        per_scenario = {}
        for scenario in sorted({d.scenario for d in decisions}):
            rows = [d for d in decisions if d.scenario == scenario]
            gated = [d for d in rows if d.gate_retrained is not None]
            per_scenario[scenario] = {
                "windows": len(rows),
                "monitor_breach_rate": round(breach_rate(rows), 4),
                "acting_card_rate": round(trigger_rate(rows), 4),
                "detection_latency_windows": detection_latency(rows),
                "actions": sorted({d.action for d in rows}),
                "dispositions": sorted({d.disposition for d in rows}),
                "confidence_range": [
                    round(min(d.confidence for d in rows), 4),
                    round(max(d.confidence for d in rows), 4),
                ],
                "gated_cards": len(gated),
                "retrained_pass": sum(d.gate_retrained == "PASS" for d in gated),
                "retrained_block": sum(d.gate_retrained == "BLOCK" for d in gated),
                "bad_challenger_block": sum(d.gate_bad == "BLOCK" for d in gated),
            }
        control = [d for d in decisions if d.scenario == CONTROL]
        summary[policy] = {
            "scenarios": per_scenario,
            "control_false_trigger_rate": round(trigger_rate(control), 4) if control else None,
        }
    return summary


# ---------------------------------------------------------------------------
# The impure half: measure, decide, gate
# ---------------------------------------------------------------------------
def _private_session(directory: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from db.models import Base

    engine = create_engine(f"sqlite:///{(directory / 'benchmark.db').as_posix()}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)(), engine


def measure_all(scenarios: Iterable[str], *, max_windows: int | None = None) -> list:
    """Every complete window of every scenario, measured once. No report files."""
    from loop.monitor.config import WINDOW_ROWS
    from loop.monitor.reference import load_reference
    from loop.monitor.runner import load_serving_stream, measure_window
    from loop.monitor.scoring import load_scoring_model
    from loop.monitor.windows import iter_windows
    from scenarios.injection import SCENARIO_SEED, apply_scenario

    model = load_scoring_model("local")
    reference = load_reference(model)
    stream = load_serving_stream()
    measured = []
    for scenario in scenarios:
        injected = apply_scenario(stream, scenario, seed=SCENARIO_SEED)
        for window in iter_windows(injected, window_rows=WINDOW_ROWS, max_windows=max_windows):
            measured.append(
                measure_window(
                    window, scenario=scenario, reference=reference, model=model, reports_dir=None
                )
            )
    return measured


def decide_all(measured: list, policy_version: str, directory: Path) -> list[tuple]:
    """Records and decides every window under one policy, in its own database.

    Order matters and is kept: the persistence rule reads the scenario's prior
    windows, exactly as the worker's history would present them.
    """
    from loop.engine.evaluate import candidate_data_version, evaluate_event
    from loop.engine.policy import load_policy
    from loop.monitor.persistence import record_window

    policy = load_policy(policy_version)
    session, engine = _private_session(directory)
    decided = []
    try:
        for result in measured:
            event = record_window(session, result)
            card, _ = evaluate_event(
                session,
                event,
                policy,
                data_version=candidate_data_version(),
                narrator=lambda card: card,  # decisions only; narration is not measured here
            )
            decided.append((result, card))
    finally:
        session.close()
        engine.dispose()
    return decided


def to_decision(result, card) -> WindowDecision:
    return WindowDecision(
        scenario=result.scenario,
        window_index=result.window_index,
        breaching_features=result.breaching_feature_count,
        max_psi=round(float(result.max_psi), 4),
        prediction_drift=bool(result.prediction_drift),
        action=str(card.action),
        disposition=str(card.disposition),
        confidence=round(float(card.confidence), 4),
        rule_id=str(card.rule_id),
        card_id=card.card_id,
    )


def gate_all(decided: list[tuple], decisions: list[WindowDecision], *, challenger=None) -> None:
    """Gates the retrained and the bad challenger against every retrain card."""
    from loop.engine.rules import RETRAIN_ACTIONS
    from loop.gate.datasets import evaluation_frames
    from loop.gate.demo import InvertedModel
    from loop.gate.runner import gate_challenger
    from loop.monitor.runner import load_serving_stream
    from ml.retrain import MODE_DEMO_BAD, MODE_LIVE, ChallengerRun
    from ml.train import load_models

    _, champion = load_models()
    if challenger is None:
        from ml.config import EVAL_PATH, TRAIN_PATH
        from ml.retrain import train_challenger
        from ml.train import load_split

        _, challenger, _, _ = train_challenger(load_split(TRAIN_PATH), load_split(EVAL_PATH))
    retrained = ChallengerRun(mode=MODE_LIVE, calibrated_model=challenger)
    bad = ChallengerRun(mode=MODE_DEMO_BAD, calibrated_model=InvertedModel(champion))
    stream = load_serving_stream()

    for (result, card), decision in zip(decided, decisions, strict=True):
        if card.action not in RETRAIN_ACTIONS:
            continue
        card_json = card.model_dump(mode="json")
        frames = evaluation_frames(result.scenario, result.window_start, stream=stream)
        good = gate_challenger(card_json, retrained, champion, frames)
        worse = gate_challenger(card_json, bad, champion, frames)
        decision.gate_retrained = good.outcome
        decision.gate_retrained_reasons = list(good.reasons)
        decision.gate_bad = worse.outcome
        decision.gate_bad_failed_checks = len(worse.failed_checks())


def run_benchmark(
    *,
    max_windows: int | None = None,
    gate: bool = True,
    measure: Callable[..., list] = measure_all,
    challenger=None,
) -> dict:
    from scenarios.injection import SCENARIOS

    measured = measure(list(SCENARIOS), max_windows=max_windows)
    results: dict[str, list[WindowDecision]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for policy in POLICIES:
            directory = Path(tmp) / policy
            directory.mkdir()
            decided = decide_all(measured, policy, directory)
            decisions = [to_decision(result, card) for result, card in decided]
            if gate and policy == GATED_POLICY:
                gate_all(decided, decisions, challenger=challenger)
            results[policy] = decisions

    from ml.tracking import dvc_lineage, git_lineage

    return {
        "lineage": {**git_lineage(), **dvc_lineage()},
        "gated_policy": GATED_POLICY if gate else None,
        "summary": summarise(results),
        "windows": {policy: [asdict(d) for d in rows] for policy, rows in results.items()},
    }


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------
def _latency(value) -> str:
    return "not detected" if value is None else str(value)


def render_markdown(benchmark: dict) -> str:
    lineage = benchmark["lineage"]
    lines = [
        "# S1-S5 benchmark results",
        "",
        "Generated by `python -m scenarios.benchmark` from git commit "
        f"`{lineage.get('git_commit')}` (dirty: {lineage.get('git_dirty')}), training data "
        f"`{lineage.get('dvc_train_md5')}`. Every complete 2,000-row window of each seeded "
        "scenario; drift is present from window 0. Definitions: `scenarios/benchmark.py`.",
        "",
    ]
    for policy, block in benchmark["summary"].items():
        lines += [
            f"## Detection under {policy}",
            "",
            "| Scenario | Windows | Monitor breach rate | Acting-card rate | "
            "Detection latency (windows) | Actions | Dispositions | Confidence |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scenario, row in block["scenarios"].items():
            low, high = row["confidence_range"]
            lines.append(
                f"| {scenario} | {row['windows']} | {row['monitor_breach_rate']:.0%} | "
                f"{row['acting_card_rate']:.0%} | {_latency(row['detection_latency_windows'])} | "
                f"{', '.join(row['actions'])} | {', '.join(row['dispositions'])} | "
                f"{low:.4f}-{high:.4f} |"
            )
        rate = block["control_false_trigger_rate"]
        lines += ["", f"**False-trigger rate on the S5 control: {rate:.0%}.**", ""]

    gated = benchmark.get("gated_policy")
    if gated:
        lines += [
            f"## Gate outcomes ({gated} retrain cards)",
            "",
            "| Card | Scenario | Window | Retrained challenger | Deliberately bad challenger |",
            "|---|---|---|---|---|",
        ]
        for row in benchmark["windows"][gated]:
            if row["gate_retrained"] is None:
                continue
            reason = "; ".join(row["gate_retrained_reasons"]) or "-"
            retrained = row["gate_retrained"] + ("" if reason == "-" else f" ({reason})")
            lines.append(
                f"| `{row['card_id']}` | {row['scenario']} | {row['window_index']} | "
                f"{retrained} | {row['gate_bad']} ({row['gate_bad_failed_checks']} failed checks) |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate the S1-S5 benchmark tables.")
    parser.add_argument("--windows", type=int, default=None, help="windows per scenario")
    parser.add_argument("--no-gate", action="store_true", help="skip the retrain and the gate")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ml.tracking import make_console_encoding_safe

    make_console_encoding_safe()
    benchmark = run_benchmark(max_windows=args.windows, gate=not args.no_gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / JSON_PATH.name).write_text(
        json.dumps(benchmark, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = render_markdown(benchmark)
    (args.output_dir / MARKDOWN_PATH.name).write_text(
        markdown + "\n", encoding="utf-8", newline="\n"
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
