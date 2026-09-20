"""Guards on the Week 3 artifacts actually produced by a training run.

These assert properties of `models/` and `reports/`, which are DVC outputs rather
than git contents, so they skip when the workflow has not been run -- the same
convention as tests/data/test_processed_datasets.py.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.config import (
    EVAL_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RELIABILITY_PATH,
    SHAP_EXAMPLE_PATH,
    SHAP_IMPORTANCE_PATH,
    TRAIN_PATH,
)
from ml.data.clean import RAW_TARGET_COLUMN, TARGET_COLUMN

REQUIRED_PATHS = (
    MODEL_PATH,
    METADATA_PATH,
    METRICS_PATH,
    RELIABILITY_PATH,
    SHAP_IMPORTANCE_PATH,
    SHAP_EXAMPLE_PATH,
)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_PATHS),
    reason="Week 3 artifacts not built (run `dvc repro`)",
)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_saved_model_bundle_loads():
    from ml.train import load_models

    base_model, calibrated_model = load_models()
    assert hasattr(base_model, "predict_proba")
    assert hasattr(calibrated_model, "predict_proba")


def test_metadata_records_the_training_contract(metadata):
    assert metadata["target"] == TARGET_COLUMN
    assert metadata["data"]["future_stream_used"] is False
    assert metadata["data"]["train_path"] == "datasets/processed/train.csv"
    assert metadata["data"]["eval_path"] == "datasets/processed/eval_frozen.csv"
    assert RAW_TARGET_COLUMN in metadata["excluded_from_features"]
    assert metadata["calibration"]["method"] == "isotonic"


def test_reported_metrics_are_present_and_in_range(metrics):
    for variant in ("raw", "calibrated"):
        block = metrics[variant]
        assert 0.0 <= block["roc_auc"] <= 1.0
        assert 0.0 <= block["average_precision"] <= 1.0
        assert 0.0 <= block["brier_score"] <= 1.0
        assert block["log_loss"] > 0.0
        assert 0.0 <= block["expected_calibration_error"] <= 1.0
        assert 0.0 <= block["recall_at_top_decile"] <= 1.0
        for key in ("counts_at_default_threshold", "counts_at_top_decile_threshold"):
            counts = block[key]
            total = sum(
                counts[field]
                for field in (
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                    "true_positives",
                )
            )
            assert total == metrics["eval_rows"]


def test_metrics_describe_the_frozen_evaluation_slice(metrics):
    assert metrics["evaluation_slice"] == "datasets/processed/eval_frozen.csv"
    assert metrics["eval_rows"] == len(pd.read_csv(EVAL_PATH, low_memory=False))
    assert metrics["inference_model"] == "calibrated"


def test_calibrated_probabilities_on_the_eval_slice_are_valid():
    from ml.data.features import split_features_target
    from ml.train import load_models, load_split

    eval_df = load_split(EVAL_PATH).head(500)
    X_eval, _ = split_features_target(eval_df)
    _, calibrated_model = load_models()

    probabilities = calibrated_model.predict_proba(X_eval)[:, 1]
    assert probabilities.shape == (len(X_eval),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert not np.isnan(probabilities).any()


def test_shap_importance_covers_the_models_full_feature_set():
    from ml.explain import transformed_feature_names
    from ml.train import load_models

    base_model, _ = load_models()
    importance = pd.read_csv(SHAP_IMPORTANCE_PATH)

    assert len(importance) == len(transformed_feature_names(base_model))
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_local_shap_example_is_well_formed():
    example = json.loads(SHAP_EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert 0.0 <= example["predicted_probability"] <= 1.0
    assert len(example["top_contributions"]) > 0


def test_reliability_table_reports_both_model_variants():
    table = pd.read_csv(RELIABILITY_PATH)
    assert set(table["model"].unique()) == {"raw", "calibrated"}


def test_train_and_eval_slices_remain_disjoint_by_patient():
    """The Week 2 separation the Week 3 model depends on."""
    train_patients = set(pd.read_csv(TRAIN_PATH, usecols=["patient_nbr"])["patient_nbr"])
    eval_patients = set(pd.read_csv(EVAL_PATH, usecols=["patient_nbr"])["patient_nbr"])
    assert train_patients.isdisjoint(eval_patients)
