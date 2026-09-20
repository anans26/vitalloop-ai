"""Week 3 training contract: leakage safety, probability validity, persistence."""

import ast
from pathlib import Path

import numpy as np
import pytest

from ml.config import LIGHTGBM_PARAMS, MIN_ESTIMATORS
from ml.data.clean import RAW_TARGET_COLUMN, TARGET_COLUMN
from ml.data.features import NON_FEATURE_COLUMNS, split_features_target
from ml.train import (
    build_base_model,
    build_calibrated_model,
    build_metadata,
    load_models,
    save_models,
    select_n_estimators,
    train_models,
)

WEEK3_MODULES = ("ml/train.py", "ml/evaluate.py", "ml/explain.py")


@pytest.fixture(scope="module")
def fitted_base_model(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    model = build_base_model(n_estimators=20)
    model.fit(X, y)
    return model


def test_target_is_absent_from_the_model_feature_matrix(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN


def test_known_leakage_columns_cannot_enter_model_features(synthetic_clean_df):
    frame = synthetic_clean_df.copy()
    # Even if a raw target column reappears upstream, it must not reach X.
    frame[RAW_TARGET_COLUMN] = "<30"

    X, _ = split_features_target(frame)
    for column in NON_FEATURE_COLUMNS:
        assert column not in X.columns


def _executable_source(path: str) -> str:
    """Module source with comments and docstrings stripped.

    Week 3 modules legitimately *discuss* future_stream in their docstrings; what
    must never appear is a line of code that reads it.
    """
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


def test_week3_modules_never_read_the_future_stream():
    """future_stream.csv is reserved for the Week 6 drift scenarios.

    Recording `future_stream_used: False` in the run metadata is fine; naming the
    file in code, or exposing a path constant pointing at it, is not.
    """
    for module in WEEK3_MODULES:
        assert "future_stream.csv" not in _executable_source(module), (
            f"{module} contains code referencing future_stream.csv"
        )


def test_config_exposes_no_path_to_the_future_stream():
    import ml.config as config

    future_paths = [
        name
        for name, value in vars(config).items()
        if isinstance(value, Path) and "future_stream" in value.name
    ]
    assert future_paths == [], f"ml.config exposes future-stream paths: {future_paths}"


def test_base_model_produces_probabilities_with_expected_shape_and_range(
    fitted_base_model, synthetic_clean_df
):
    X, _ = split_features_target(synthetic_clean_df)
    probabilities = fitted_base_model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibrated_model_produces_valid_probabilities(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    calibrated = build_calibrated_model(build_base_model(n_estimators=10))
    calibrated.fit(X, y)

    probabilities = calibrated.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert not np.isnan(probabilities).any()


def test_select_n_estimators_returns_a_bounded_tree_count(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    n_estimators = select_n_estimators(X, y)

    assert isinstance(n_estimators, int)
    assert MIN_ESTIMATORS <= n_estimators <= LIGHTGBM_PARAMS["n_estimators"]


def test_saved_model_bundle_reloads_and_scores_identically(
    synthetic_clean_df, tmp_path, monkeypatch
):
    X, y = split_features_target(synthetic_clean_df)
    base = build_base_model(n_estimators=10)
    base.fit(X, y)
    calibrated = build_calibrated_model(build_base_model(n_estimators=10))
    calibrated.fit(X, y)

    monkeypatch.setattr("ml.train.MODELS_DIR", tmp_path)
    monkeypatch.setattr("ml.train.MODEL_PATH", tmp_path / "model.joblib")
    save_models(base, calibrated)
    reloaded_base, reloaded_calibrated = load_models()

    assert np.allclose(base.predict_proba(X)[:, 1], reloaded_base.predict_proba(X)[:, 1])
    assert np.allclose(
        calibrated.predict_proba(X)[:, 1], reloaded_calibrated.predict_proba(X)[:, 1]
    )


def test_metadata_records_the_data_contract_and_excludes_future_stream(synthetic_clean_df):
    metadata = build_metadata(synthetic_clean_df, synthetic_clean_df, n_estimators=42)

    assert metadata["target"] == TARGET_COLUMN
    assert RAW_TARGET_COLUMN in metadata["excluded_from_features"]
    assert metadata["data"]["future_stream_used"] is False
    assert metadata["n_estimators_selection"]["selected_n_estimators"] == 42
    assert metadata["calibration"]["method"] == "isotonic"


def test_train_models_returns_both_models_and_the_selected_tree_count(synthetic_clean_df):
    base, calibrated, n_estimators = train_models(synthetic_clean_df)

    assert hasattr(base, "predict_proba")
    assert hasattr(calibrated, "predict_proba")
    assert n_estimators >= MIN_ESTIMATORS
