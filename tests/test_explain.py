"""SHAP explanation contract: explanations must describe the matrix the model sees."""

import numpy as np
import pytest

from ml.data.features import split_features_target
from ml.explain import (
    build_explainer,
    explain_instance,
    global_importance,
    transform_features,
    transformed_feature_names,
)
from ml.train import build_base_model


@pytest.fixture(scope="module")
def fitted_model(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    model = build_base_model(n_estimators=20)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def eval_features(synthetic_clean_df):
    X, _ = split_features_target(synthetic_clean_df)
    return X


def test_feature_names_align_with_the_transformed_matrix(fitted_model, eval_features):
    """The core contract: one name per column the model actually receives."""
    matrix = transform_features(fitted_model, eval_features.head(10))
    names = transformed_feature_names(fitted_model)

    assert matrix.shape[1] == len(names)
    assert matrix.shape[0] == 10
    assert len(set(names)) == len(names)


def test_transformed_matrix_width_matches_the_models_expected_input(fitted_model, eval_features):
    matrix = transform_features(fitted_model, eval_features.head(5))
    assert matrix.shape[1] == fitted_model.named_steps["model"].n_features_in_


def test_global_importance_covers_every_transformed_feature(fitted_model, eval_features):
    importance = global_importance(fitted_model, eval_features, sample_size=100)

    assert len(importance) == len(transformed_feature_names(fitted_model))
    assert set(importance.columns) == {"feature", "mean_abs_shap", "mean_shap"}
    assert (importance["mean_abs_shap"] >= 0).all()
    # sorted descending by magnitude
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_local_explanation_returns_valid_named_contributions(fitted_model, eval_features):
    explanation = explain_instance(fitted_model, eval_features, row_position=0, top_n=5)
    names = set(transformed_feature_names(fitted_model))

    assert 0.0 <= explanation["predicted_probability"] <= 1.0
    assert len(explanation["top_contributions"]) == 5
    for contribution in explanation["top_contributions"]:
        assert contribution["feature"] in names


def test_local_contributions_are_ranked_by_absolute_impact(fitted_model, eval_features):
    explanation = explain_instance(fitted_model, eval_features, row_position=1, top_n=6)
    magnitudes = [abs(c["shap_value"]) for c in explanation["top_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_values_are_additive_to_the_model_margin(fitted_model, eval_features):
    """base_value + sum(shap) must equal the model's raw log-odds output.

    This is what proves the explanation describes *this* model on *this* encoding
    rather than some other preprocessing representation.
    """
    row = eval_features.iloc[[3]]
    matrix = transform_features(fitted_model, row)
    explainer = build_explainer(fitted_model)

    values = np.asarray(explainer.shap_values(matrix, check_additivity=False))
    if values.ndim == 3:
        values = values[:, :, -1]
    base_value = np.asarray(explainer.expected_value).ravel()[-1]

    predicted_probability = fitted_model.predict_proba(row)[0, 1]
    reconstructed = 1.0 / (1.0 + np.exp(-(base_value + values[0].sum())))

    assert reconstructed == pytest.approx(predicted_probability, abs=1e-6)


def test_explanations_are_deterministic(fitted_model, eval_features):
    first = explain_instance(fitted_model, eval_features, row_position=0)
    second = explain_instance(fitted_model, eval_features, row_position=0)
    assert first == second
