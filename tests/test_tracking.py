"""Week 4 tracking contract.

Everything here runs against a temporary SQLite tracking store, so the suite
never needs the Compose MLflow service. That matters twice over: CI has no
server, and a test that silently depended on one would be exactly the kind of
hidden external dependency Week 4 is supposed to remove.
"""

import ast
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from ml import evaluate
from ml.data.features import split_features_target
from ml.registry import read_audit_rows
from ml.tracking import (
    SAFE_ARTIFACTS,
    build_metrics,
    build_params,
    dvc_lineage,
    git_lineage,
    log_run,
    redacted_local_explanation,
    resolve_tracking_uri,
    verify_tracking_reachable,
)
from ml.tracking_config import MODEL_ALIASES
from ml.train import build_base_model, build_calibrated_model, build_metadata

TRACKING_MODULES = ("ml/tracking.py", "ml/registry.py")

# Pinned so MLflow skips dependency inference, which dominates test runtime.
TEST_PIP_REQUIREMENTS = ["scikit-learn", "lightgbm", "cloudpickle"]


# ---------------------------------------------------------------------------
# Destination resolution
# ---------------------------------------------------------------------------
def test_resolve_tracking_uri_prefers_the_configured_server(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid:5000")
    assert resolve_tracking_uri() == "http://example.invalid:5000"


def test_resolve_tracking_uri_falls_back_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = resolve_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_unreachable_backend_raises_instead_of_degrading_silently(monkeypatch):
    """A 'tracked' run that quietly goes nowhere is the failure to prevent.

    Retries are disabled here only for speed: MLflow's default backoff turns
    this single assertion into minutes of waiting.
    """
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")

    with pytest.raises(RuntimeError, match="not reachable"):
        verify_tracking_reachable("http://127.0.0.1:1/")


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------
def test_git_lineage_captures_commit_branch_and_dirtiness():
    lineage = git_lineage()
    assert lineage["git_commit"] is not None
    assert len(lineage["git_commit"]) == 40
    assert lineage["git_branch"]
    assert isinstance(lineage["git_dirty"], bool)


def test_dvc_lineage_reads_dataset_hashes_from_the_lock():
    lineage = dvc_lineage()
    assert lineage["dvc_train_md5"]
    assert lineage["dvc_eval_frozen_md5"]
    assert all(len(value) == 32 for value in lineage.values())


def test_dvc_lineage_is_empty_when_the_lock_is_absent(tmp_path):
    assert dvc_lineage(tmp_path / "missing.lock") == {}


# ---------------------------------------------------------------------------
# Params / metrics shaping
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metadata(synthetic_clean_df) -> dict:
    return build_metadata(synthetic_clean_df, synthetic_clean_df, n_estimators=5)


@pytest.fixture(scope="module")
def fitted_models(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    base = build_base_model(n_estimators=5)
    base.fit(X, y)
    calibrated = build_calibrated_model(build_base_model(n_estimators=5))
    calibrated.fit(X, y)
    return base, calibrated


@pytest.fixture(scope="module")
def report(synthetic_clean_df, fitted_models) -> dict:
    """A report with the same shape ml.evaluate.main() writes."""
    base, calibrated = fitted_models
    X, y = split_features_target(synthetic_clean_df)
    raw_prob = base.predict_proba(X)[:, 1]
    calibrated_prob = calibrated.predict_proba(X)[:, 1]

    raw = evaluate.score_predictions(y, raw_prob)
    cal = evaluate.score_predictions(y, calibrated_prob)
    return {
        "evaluation_slice": "synthetic",
        "eval_rows": len(X),
        "target": "readmitted_30d",
        "positive_rate": float(y.mean()),
        "inference_model": "calibrated",
        "raw": raw,
        "calibrated": cal,
        "calibration_effect": {
            "brier_delta": cal["brier_score"] - raw["brier_score"],
            "log_loss_delta": cal["log_loss"] - raw["log_loss"],
            "ece_delta": 0.0,
            "roc_auc_delta": cal["roc_auc"] - raw["roc_auc"],
        },
        "subgroups_calibrated": evaluate.subgroup_metrics(
            synthetic_clean_df, y, calibrated_prob, ["gender"]
        ),
    }


def test_build_params_records_the_training_contract(metadata):
    params = build_params(metadata)
    for key in (
        "model_type",
        "target",
        "random_seed",
        "calibration_method",
        "calibration_cv_folds",
        "train_path",
        "eval_path",
        "feature_pipeline",
        "future_stream_used",
    ):
        assert key in params

    assert params["calibration_method"] == "isotonic"
    assert params["future_stream_used"] is False
    assert params["lgbm_n_estimators"] == 5
    assert "readmitted" in params["excluded_from_features"]
    assert params["version_lightgbm"]


def test_build_metrics_preserves_the_raw_versus_calibrated_distinction(report):
    metrics = build_metrics(report)
    for stem in (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "recall_at_top_decile",
    ):
        assert f"raw_{stem}" in metrics
        assert f"calibrated_{stem}" in metrics

    assert "calibrated_counts_at_top_decile_threshold_true_positives" in metrics
    assert any(key.startswith("subgroup_gender_") for key in metrics)
    assert all(isinstance(value, float) for value in metrics.values())


def test_build_metrics_invents_nothing(report):
    """Every metric must trace back to a key the Week 3 evaluation produced."""
    produced = set(report["raw"]) | set(report["calibrated"])
    for name in build_metrics(report):
        if name in ("eval_rows", "positive_rate") or name.startswith(
            ("calibration_effect_", "subgroup_")
        ):
            continue
        stem = name.split("_", 1)[1]
        assert any(stem == key or stem.startswith(f"{key}_") for key in produced), name


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
def test_redaction_drops_per_encounter_feature_values():
    example = {
        "row_position": 0,
        "predicted_probability": 0.09,
        "top_contributions": [
            {"feature": "num__time_in_hospital", "shap_value": -0.09, "feature_value": 2.0}
        ],
    }
    redacted = redacted_local_explanation(example)

    assert redacted["top_contributions"] == [
        {"feature": "num__time_in_hospital", "shap_value": -0.09}
    ]
    assert "feature_value" not in json.dumps(redacted["top_contributions"])
    assert redacted["predicted_probability"] == 0.09


def test_safe_artifact_list_contains_no_patient_datasets():
    names = [Path(path).name for path in SAFE_ARTIFACTS]
    assert "diabetic_data.csv" not in names
    assert not any(name in names for name in ("train.csv", "eval_frozen.csv", "future_stream.csv"))


def _executable_source(path: str) -> str:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)
    return ast.unparse(tree)


def test_tracking_modules_never_touch_the_future_stream():
    for module in TRACKING_MODULES:
        assert "future_stream.csv" not in _executable_source(module)


# ---------------------------------------------------------------------------
# End-to-end against a temporary SQLite store
# ---------------------------------------------------------------------------
@pytest.fixture
def local_tracking(tmp_path):
    """A throwaway tracking store with artifacts inside tmp_path."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    MlflowClient(uri).create_experiment("test-experiment", artifact_location=artifacts.as_uri())
    return uri


def test_log_run_records_a_complete_tracked_run(
    local_tracking, tmp_path, fitted_models, metadata, report
):
    base, calibrated = fitted_models
    artifact = tmp_path / "metrics.json"
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = log_run(
        base_model=base,
        calibrated_model=calibrated,
        metadata=metadata,
        report=report,
        local_explanation={"row_position": 0, "top_contributions": []},
        tracking_uri=local_tracking,
        experiment_name="test-experiment",
        model_name="test-model",
        artifacts=(artifact,),
        pip_requirements=TEST_PIP_REQUIREMENTS,
    )

    client = MlflowClient(local_tracking)
    run = client.get_run(summary["run_id"])

    assert run.info.status == "FINISHED"
    assert run.data.params["model_type"] == "lightgbm.LGBMClassifier"
    assert run.data.params["calibration_method"] == "isotonic"
    assert run.data.params["git_commit"] == git_lineage()["git_commit"]
    assert run.data.params["dvc_train_md5"] == dvc_lineage()["dvc_train_md5"]
    assert run.data.tags["future_stream_used"] == "false"
    assert run.data.metrics["calibrated_roc_auc"] == pytest.approx(
        report["calibrated"]["roc_auc"], abs=1e-6
    )

    logged = [item.path for item in client.list_artifacts(summary["run_id"], "reports")]
    assert any(path.endswith("metrics.json") for path in logged)
    assert any(path.endswith("shap_local_example_redacted.json") for path in logged)


def test_logged_model_reloads_and_scores_identically(
    local_tracking, tmp_path, fitted_models, metadata, report, synthetic_clean_df
):
    base, calibrated = fitted_models
    summary = log_run(
        base_model=base,
        calibrated_model=calibrated,
        metadata=metadata,
        report=report,
        local_explanation={"row_position": 0, "top_contributions": []},
        tracking_uri=local_tracking,
        experiment_name="test-experiment",
        model_name="test-model",
        artifacts=(),
        pip_requirements=TEST_PIP_REQUIREMENTS,
    )

    mlflow.set_tracking_uri(local_tracking)
    reloaded = mlflow.sklearn.load_model(summary["calibrated_model_uri"])

    X, _ = split_features_target(synthetic_clean_df)
    expected = calibrated.predict_proba(X)[:, 1]
    actual = reloaded.predict_proba(X)[:, 1]

    assert actual.shape == expected.shape
    assert np.all((actual >= 0.0) & (actual <= 1.0))
    assert np.allclose(actual, expected)


def test_registration_sets_champion_once_and_audits_it(
    local_tracking, tmp_path, monkeypatch, fitted_models, metadata, report
):
    audit_path = tmp_path / "registry_audit.jsonl"
    monkeypatch.setattr("ml.registry.REGISTRY_AUDIT_PATH", audit_path)

    def run_once():
        return log_run(
            base_model=fitted_models[0],
            calibrated_model=fitted_models[1],
            metadata=metadata,
            report=report,
            local_explanation={"row_position": 0, "top_contributions": []},
            tracking_uri=local_tracking,
            experiment_name="test-experiment",
            model_name="test-model",
            artifacts=(),
            pip_requirements=TEST_PIP_REQUIREMENTS,
        )

    first = run_once()
    assert first["registered_version"] == "1"
    assert first["alias_audit"]["alias"] == "champion"
    assert first["alias_audit"]["to_version"] == "1"

    client = MlflowClient(local_tracking)
    assert str(client.get_model_version_by_alias("test-model", "champion").version) == "1"

    second = run_once()
    assert second["registered_version"] == "2"
    # Week 4 registers; it does not promote. Champion stays on v1.
    assert second["alias_audit"] is None
    assert str(client.get_model_version_by_alias("test-model", "champion").version) == "1"
    assert len(read_audit_rows(audit_path)) == 1


def test_champion_is_one_of_the_documented_aliases():
    assert "champion" in MODEL_ALIASES
    assert set(MODEL_ALIASES) == {"champion", "challenger", "shadow"}
