"""The CI training smoke run and gate check (IMPLEMENTATION_ROADMAP.md Week 11).

ARCHITECTURE.md §5 draws the pipeline as "lint (ruff) -> unit tests -> training
smoke run on a data sample -> gate check -> image build". This script is the
middle two stages:

    python -m scripts.ci_smoke                 # 5,000 rows, the roadmap's figure
    python -m scripts.ci_smoke --rows 2000 --output smoke.json

**Why the sample is synthetic.** The real dataset is DVC-tracked against a
*local* remote (`dvc-storage/`, gitignored), so a CI runner has no copy of it,
and fetching it from UCI would make every pipeline run depend on a third-party
server. The sample is instead generated here, deterministically, in the shape
`ml.data.clean` produces -- every column the feature pipeline consumes,
identifiers, the binary target -- with a learnable signal. No patient record is
involved at any point. It proves the *code path* trains, evaluates and gates;
it says nothing about the real model's quality, which is `dvc repro`'s job.

**What runs is the production code, unchanged:** the patient-level
chronological split (`ml.data.split`), the training entry point that fitted the
champion (`ml.train.train_models`), the evaluation report (`ml.evaluate`), and
the validation gate with the default criteria artifact (`configs/gate-v1.yaml`,
whose header names this stage as one of its users).

**The gate check asserts both outcomes.** A retrained challenger -- the same
entry point on the same data -- must PASS, and the deliberately bad challenger
(`loop.gate.demo.InvertedModel`, the champion's ranking reversed) must BLOCK.
A gate that passed everything, or blocked everything, fails the pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.clean import TARGET_COLUMN
from ml.data.features import MEDICATION_COLUMNS

DEFAULT_ROWS = 5_000
DEFAULT_SEED = 11

RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Other"]
AGES = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
SPECIALTIES = ["InternalMedicine", "Cardiology", "Surgery", "Family/GeneralPractice", "missing"]
DIAGNOSES = ["250.83", "428", "486", "V27", "789", "401", "414", "786"]
PAYERS = ["MC", "HM", "BC", "missing"]


class SmokeError(RuntimeError):
    """The smoke run did not produce the outcome the pipeline requires."""


def synthetic_clean_frame(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """A deterministic stand-in for `ml.data.clean.clean()` output. No real rows.

    One encounter per synthetic patient and increasing encounter ids, so the
    patient-level chronological split accepts it exactly as it accepts the
    cleaned dataset.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(1, n_rows + 1),
            "patient_nbr": np.arange(500_000, 500_000 + n_rows),
            "time_in_hospital": rng.integers(1, 14, n_rows),
            "num_lab_procedures": rng.integers(1, 90, n_rows),
            "num_procedures": rng.integers(0, 6, n_rows),
            "num_medications": rng.integers(1, 40, n_rows),
            "number_outpatient": rng.poisson(0.4, n_rows),
            "number_emergency": rng.poisson(0.2, n_rows),
            "number_inpatient": rng.poisson(0.6, n_rows),
            "number_diagnoses": rng.integers(1, 16, n_rows),
            "race": rng.choice(RACES, n_rows),
            "gender": rng.choice(["Female", "Male"], n_rows),
            "age": rng.choice(AGES, n_rows),
            "payer_code": rng.choice(PAYERS, n_rows),
            "medical_specialty": rng.choice(SPECIALTIES, n_rows),
            "max_glu_serum": rng.choice([None, "Norm", ">200", ">300"], n_rows),
            "A1Cresult": rng.choice([None, "Norm", ">7", ">8"], n_rows),
            "change": rng.choice(["Ch", "No"], n_rows),
            "diabetesMed": rng.choice(["Yes", "No"], n_rows),
            "diag_1": rng.choice(DIAGNOSES, n_rows),
            "diag_2": rng.choice(DIAGNOSES, n_rows),
            "diag_3": rng.choice(DIAGNOSES, n_rows),
            "admission_source_id": rng.choice([1, 4, 7, 8], n_rows),
        }
    )
    for column in MEDICATION_COLUMNS:
        frame[column] = rng.choice(["No", "Steady", "Up", "Down"], n_rows)

    # Prior utilisation drives readmission, as it does in the real data; the
    # noise keeps the problem honest rather than separable.
    logit = (
        -2.0
        + 0.55 * frame["number_inpatient"]
        + 0.30 * frame["number_emergency"]
        + 0.04 * frame["time_in_hospital"]
        + 0.02 * frame["number_diagnoses"]
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame[TARGET_COLUMN] = (rng.random(n_rows) < probability).astype(int)
    return frame


def run_smoke(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> dict:
    """Split, train, evaluate, then gate a retrained and a bad challenger."""
    from loop.gate.criteria import load_criteria
    from loop.gate.demo import InvertedModel
    from loop.gate.gate import evaluate_gate
    from loop.gate.metrics import metric_sets_for
    from ml.config import SUBGROUP_COLUMNS
    from ml.data.features import split_features_target
    from ml.data.split import time_sliced_patient_split
    from ml.evaluate import build_report
    from ml.train import train_models

    train_df, eval_df, stream_df = time_sliced_patient_split(synthetic_clean_frame(n_rows, seed))

    base_model, champion, n_estimators = train_models(train_df)
    X_eval, y_eval = split_features_target(eval_df)
    report = build_report(
        eval_df,
        y_eval,
        base_model.predict_proba(X_eval)[:, 1],
        champion.predict_proba(X_eval)[:, 1],
        evaluation_slice="synthetic-smoke-sample",
        subgroup_columns=SUBGROUP_COLUMNS,
    )

    # The retrain: the same entry point on the same data, as ml/retrain.py does.
    _, retrained, _ = train_models(train_df)

    criteria = load_criteria()
    frames = {"frozen_holdout": eval_df, "recent_labeled_window": stream_df}

    def sets(model):
        return metric_sets_for(
            model,
            frames,
            subgroup_columns=criteria.subgroup_columns,
            top_decile_fraction=criteria.top_decile_fraction,
        )

    champion_sets = sets(champion)
    retrained_sets = sets(retrained)
    retrained_gate = evaluate_gate(champion_sets, retrained_sets, criteria)
    bad_gate = evaluate_gate(champion_sets, sets(InvertedModel(champion)), criteria)

    summary = {
        "rows": {"train": len(train_df), "eval": len(eval_df), "stream": len(stream_df)},
        "seed": seed,
        "n_estimators": n_estimators,
        "calibrated_roc_auc": report["calibrated"]["roc_auc"],
        "calibrated_ece": report["calibrated"]["expected_calibration_error"],
        "criteria_version": criteria.version,
        "retrained_challenger": {
            "outcome": retrained_gate.outcome,
            "checks": len(retrained_gate.checks),
            "reasons": list(retrained_gate.reasons),
            # The one absolute criterion, so its headroom is visible in the log:
            # a small sample's calibration error is noisy, and the ceiling is 0.05.
            "ece_by_set": {
                name: metric.expected_calibration_error for name, metric in retrained_sets.items()
            },
        },
        "bad_challenger": {
            "outcome": bad_gate.outcome,
            "checks": len(bad_gate.checks),
            "failed_checks": len(bad_gate.failed_checks()),
        },
    }
    if not retrained_gate.passed:
        raise SmokeError(
            "the retrained challenger was BLOCKED: " + "; ".join(retrained_gate.reasons)
        )
    if not bad_gate.blocked:
        raise SmokeError("the deliberately bad challenger PASSED the gate")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CI training smoke run and gate check.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="synthetic sample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sample seed")
    parser.add_argument("--output", type=Path, default=None, help="also write the summary here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_smoke(args.rows, args.seed)
    except SmokeError as error:
        print(f"SMOKE FAILED: {error}", file=sys.stderr)
        return 1

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(
        f"smoke: trained on {summary['rows']['train']} synthetic rows; gate "
        f"{summary['retrained_challenger']['outcome']} for the retrained challenger, "
        f"{summary['bad_challenger']['outcome']} for the bad one"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
