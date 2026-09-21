"""Week 4: MLflow experiment tracking and lineage for the Week 3 workflow.

This module adds *observation* around the Week 3 model. It does not retrain
differently, re-tune, or alter the evaluation protocol: it calls the same
`ml.train`, `ml.evaluate`, and `ml.explain` entry points and records what they
produced, so the tracked numbers are the Week 3 numbers by construction.

Two tracking destinations are supported, both local:

* `MLFLOW_TRACKING_URI` set (e.g. ``http://localhost:5000``) -- the Compose
  service. This is the documented path for the UI and the registry.
* unset -- a SQLite store under ``mlflow/``. Registry operations need a database
  backend, which a bare file store cannot provide, so SQLite rather than
  ``./mlruns``.

If a configured server is unreachable, the run fails loudly. Silently falling
back to a local store would let a "tracked" run quietly go nowhere, which is the
one failure mode experiment tracking exists to prevent.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

import mlflow
from ml.config import (
    CALIBRATION_FIGURE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    PROJECT_ROOT,
    RELIABILITY_PATH,
    SHAP_EXAMPLE_PATH,
    SHAP_FIGURE_PATH,
    SHAP_IMPORTANCE_PATH,
)
from ml.registry import ensure_initial_champion, register_model
from ml.tracking_config import (
    BASE_MODEL_ARTIFACT,
    CALIBRATED_MODEL_ARTIFACT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_LOCAL_ARTIFACTS,
    MLFLOW_LOCAL_DB,
    MLFLOW_LOCAL_DIR,
    REGISTERED_MODEL_NAME,
)

# Aggregates, configuration and figures only. The raw and processed patient
# CSVs are never logged to MLflow -- they are DVC's responsibility, and copying
# them into a tracking server would duplicate clinical data for no benefit.
SAFE_ARTIFACTS = (
    METRICS_PATH,
    RELIABILITY_PATH,
    SHAP_IMPORTANCE_PATH,
    METADATA_PATH,
    CALIBRATION_FIGURE_PATH,
    SHAP_FIGURE_PATH,
)


# ---------------------------------------------------------------------------
# Tracking destination
# ---------------------------------------------------------------------------
def resolve_tracking_uri() -> str:
    """The configured server if there is one, else the local SQLite store."""
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured

    MLFLOW_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{MLFLOW_LOCAL_DB.as_posix()}"


def verify_tracking_reachable(tracking_uri: str) -> None:
    """Fails loudly rather than letting a supposedly tracked run vanish."""
    try:
        MlflowClient(tracking_uri).search_experiments(max_results=1)
    except Exception as error:
        raise RuntimeError(
            f"MLflow tracking backend at {tracking_uri!r} is not reachable: {error}\n"
            "Start it with `docker compose -f docker/docker-compose.yml up -d mlflow`, "
            "or unset MLFLOW_TRACKING_URI to use the local SQLite store."
        ) from error


# ---------------------------------------------------------------------------
# Lineage: which code and which data produced this run
# ---------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def git_lineage() -> dict:
    """Commit, branch, and whether the tree was dirty when the run happened.

    `git_dirty` matters: a run produced from an uncommitted tree is not
    reproducible from its commit alone, and the run should say so.
    """
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


DVC_TRACKED_PATHS = {
    "datasets/raw/diabetic_data.csv": "dvc_raw_md5",
    "datasets/processed/train.csv": "dvc_train_md5",
    "datasets/processed/eval_frozen.csv": "dvc_eval_frozen_md5",
    "models/readmission_model.joblib": "dvc_model_md5",
}


def dvc_lineage(lock_path: Path | None = None) -> dict:
    """Dataset and model hashes, read from dvc.lock.

    These are the same hashes the Decision Card will pin in Week 7, so a run and
    a future retrain decision can be tied to identical bytes.
    """
    lock_path = lock_path or (PROJECT_ROOT / "dvc.lock")
    if not lock_path.exists():
        return {}

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    found: dict = {}
    for stage in (lock.get("stages") or {}).values():
        entries = list(stage.get("deps") or []) + list(stage.get("outs") or [])
        for entry in entries:
            path = str(entry.get("path", "")).replace("\\", "/")
            key = DVC_TRACKED_PATHS.get(path)
            if key and entry.get("md5"):
                found[key] = entry["md5"]
    return found


# ---------------------------------------------------------------------------
# Parameters and metrics
# ---------------------------------------------------------------------------
def build_params(metadata: dict) -> dict:
    """Flat parameters describing how the model was produced.

    Sourced entirely from `models/model_metadata.json`, so the tracked
    configuration cannot drift from what training actually recorded.
    """
    params = {
        "model_type": metadata["model_type"],
        "inference_model": metadata["inference_model"],
        "target": metadata["target"],
        "feature_pipeline": metadata["feature_pipeline"],
        "excluded_from_features": ",".join(metadata["excluded_from_features"]),
        "random_seed": metadata["random_seed"],
        "calibration_method": metadata["calibration"]["method"],
        "calibration_cv_folds": metadata["calibration"]["cv_folds"],
        "train_path": metadata["data"]["train_path"],
        "eval_path": metadata["data"]["eval_path"],
        "train_rows": metadata["data"]["train_rows"],
        "eval_rows": metadata["data"]["eval_rows"],
        "train_positive_rate": metadata["data"]["train_positive_rate"],
        "eval_positive_rate": metadata["data"]["eval_positive_rate"],
        "future_stream_used": metadata["data"]["future_stream_used"],
    }
    for key, value in metadata["lightgbm_params"].items():
        params[f"lgbm_{key}"] = value
    for key, value in metadata["n_estimators_selection"].items():
        params[f"n_estimators_{key}"] = value
    for library, version in metadata["versions"].items():
        params[f"version_{library}"] = version
    return params


def _flatten_metric_block(prefix: str, block: dict) -> dict:
    """One level of nesting is enough: the threshold-count sub-dicts."""
    flat: dict = {}
    for key, value in block.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, (int, float)) and not isinstance(inner_value, bool):
                    flat[f"{prefix}_{key}_{inner_key}"] = float(inner_value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}_{key}"] = float(value)
    return flat


def _safe_metric_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(value))


def build_metrics(report: dict) -> dict:
    """Every number the Week 3 evaluation already produced. Nothing invented.

    The raw/calibrated distinction is preserved in the metric names so the
    effect of calibration stays visible in the MLflow UI.
    """
    metrics: dict = {
        "eval_rows": float(report["eval_rows"]),
        "positive_rate": float(report["positive_rate"]),
    }
    metrics.update(_flatten_metric_block("raw", report["raw"]))
    metrics.update(_flatten_metric_block("calibrated", report["calibrated"]))

    for key, value in report["calibration_effect"].items():
        metrics[f"calibration_effect_{key}"] = float(value)

    for column, groups in (report.get("subgroups_calibrated") or {}).items():
        for group_value, stats in groups.items():
            if stats.get("roc_auc") is None:
                continue
            metrics[f"subgroup_{column}_{_safe_metric_name(group_value)}_roc_auc"] = float(
                stats["roc_auc"]
            )
    return metrics


def redacted_local_explanation(example: dict) -> dict:
    """The local SHAP example with one encounter's feature values removed.

    `reports/shap_local_example.json` carries the transformed feature values of a
    single encounter. It holds no identifiers, but the project's posture is that
    no individual's record reaches a log or a tracking server, so the tracked
    copy keeps the attributions and drops the values.
    """
    contributions = [
        {"feature": item["feature"], "shap_value": item["shap_value"]}
        for item in example.get("top_contributions", [])
    ]
    redacted = {key: value for key, value in example.items() if key != "top_contributions"}
    redacted["note"] = "feature_value removed before logging (no per-encounter values tracked)"
    redacted["top_contributions"] = contributions
    return redacted


# ---------------------------------------------------------------------------
# The tracked run
# ---------------------------------------------------------------------------
def log_run(
    base_model,
    calibrated_model,
    metadata: dict,
    report: dict,
    local_explanation: dict,
    register: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = REGISTERED_MODEL_NAME,
    artifacts=SAFE_ARTIFACTS,
    pip_requirements=None,
) -> dict:
    """Records one tracked run and returns a summary of what was written."""
    tracking_uri = tracking_uri or resolve_tracking_uri()
    verify_tracking_reachable(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = build_params(metadata)
    metrics = build_metrics(report)
    lineage = {**git_lineage(), **dvc_lineage()}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_params({k: v for k, v in lineage.items() if v is not None})
        mlflow.set_tags(
            {
                "week": "4",
                "pipeline": "week3-readmission",
                "inference_model": "calibrated",
                "future_stream_used": "false",
            }
        )
        mlflow.log_metrics(metrics)

        for artifact in artifacts:
            if Path(artifact).exists():
                mlflow.log_artifact(str(artifact), artifact_path="reports")

        with tempfile.TemporaryDirectory() as tmp:
            redacted_path = Path(tmp) / "shap_local_example_redacted.json"
            redacted_path.write_text(
                json.dumps(redacted_local_explanation(local_explanation), indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            mlflow.log_artifact(str(redacted_path), artifact_path="reports")

        # No input_example/signature: inferring one would embed a real encounter's
        # feature row in the MLmodel file.
        #
        # cloudpickle rather than MLflow 3's skops default: the pipeline contains
        # a LightGBM booster and this project's own FeatureEngineer, which skops
        # refuses as untrusted types. Trusting them back one by one would be a
        # list to keep in sync with no security gain here -- the registry is
        # local, self-hosted, and only ever loads models this repo produced.
        calibrated_info = mlflow.sklearn.log_model(
            calibrated_model,
            name=CALIBRATED_MODEL_ARTIFACT,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=pip_requirements,
        )
        base_info = mlflow.sklearn.log_model(
            base_model,
            name=BASE_MODEL_ARTIFACT,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=pip_requirements,
        )

    summary = {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment": experiment_name,
        "params": params,
        "metrics": metrics,
        "lineage": lineage,
        "registered_version": None,
        "alias_audit": None,
        "calibrated_model_uri": calibrated_info.model_uri,
        "base_model_uri": base_info.model_uri,
    }

    if register:
        client = MlflowClient(tracking_uri)
        version = register_model(client, run_id, calibrated_info.model_uri, model_name)
        summary["registered_version"] = version
        summary["alias_audit"] = ensure_initial_champion(
            client,
            model_name=model_name,
            version=version,
            run_id=run_id,
            git_commit=lineage.get("git_commit"),
        )

    return summary


def make_console_encoding_safe() -> None:
    """Stops a Windows console from killing a run that already succeeded.

    MLflow prints a run URL containing an emoji. On a cp1252 console that raises
    UnicodeEncodeError from inside `end_run`, which leaves the run stuck in
    RUNNING and skips registration -- a logging detail breaking the actual work.
    Replacing unencodable characters keeps the console readable and harmless.

    Public because every entry point that starts an MLflow run needs it, not
    only this one: Week 8's retrain and gate CLIs hit the identical failure.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def main() -> None:
    """The one documented Week 4 command: train, evaluate, explain, then track."""
    from ml import evaluate, explain, train

    make_console_encoding_safe()

    train.main()
    evaluate.main()
    explain.main()

    base_model, calibrated_model = train.load_models()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    local_explanation = json.loads(SHAP_EXAMPLE_PATH.read_text(encoding="utf-8"))

    summary = log_run(
        base_model=base_model,
        calibrated_model=calibrated_model,
        metadata=metadata,
        report=report,
        local_explanation=local_explanation,
    )

    print(f"tracking uri : {summary['tracking_uri']}")
    print(f"experiment   : {summary['experiment']}")
    print(f"run id       : {summary['run_id']}")
    print(f"params       : {len(summary['params'])}")
    print(f"metrics      : {len(summary['metrics'])}")
    lineage = summary["lineage"]
    print(f"git commit   : {lineage.get('git_commit')} (dirty={lineage.get('git_dirty')})")
    print(f"dvc train md5: {lineage.get('dvc_train_md5')}")
    print(f"registered   : {REGISTERED_MODEL_NAME} v{summary['registered_version']}")
    if summary["alias_audit"]:
        print(f"alias        : champion -> v{summary['alias_audit']['to_version']} (audited)")
    else:
        print("alias        : champion already set; left unchanged (promotion is Week 8/9)")


if __name__ == "__main__":
    main()
