"""SHAP explanations for the Week 3 model.

The contract this module exists to enforce: explanations are computed on *the
exact matrix the model consumes*. The feature pipeline engineers columns and then
one-hot encodes them, so the model never sees `race` -- it sees `cat__race_Other`
and friends. Explaining the pre-transform frame would produce confident, wrong
attributions, so every function here goes through `transform_features` and labels
its output with the ColumnTransformer's own feature names.

SHAP is applied to the **base** (uncalibrated) LightGBM pipeline. Isotonic
calibration is a monotonic remap of the score, so it changes the probability a
patient receives but not the ranking or the relative contribution of features;
TreeExplainer also needs the tree ensemble itself, which the calibrated wrapper
hides behind five per-fold clones.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap
from sklearn.pipeline import Pipeline

from ml.config import SHAP_SAMPLE_SIZE, SHAP_TOP_FEATURES


def transformed_feature_names(base_model: Pipeline) -> list[str]:
    """Names of the columns the LightGBM step actually receives."""
    encoder = base_model.named_steps["features"].named_steps["encode"]
    return [str(name) for name in encoder.get_feature_names_out()]


def transform_features(base_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Runs X through the fitted feature pipeline and returns a dense matrix.

    Dense because TreeExplainer's output handling is unreliable on sparse input;
    callers are expected to pass a sample, not the whole dataset.
    """
    transformed = base_model.named_steps["features"].transform(X)
    if sp.issparse(transformed):
        transformed = transformed.toarray()
    return np.asarray(transformed)


def build_explainer(base_model: Pipeline) -> shap.TreeExplainer:
    """TreeExplainer over the fitted LightGBM estimator."""
    return shap.TreeExplainer(base_model.named_steps["model"])


def _positive_class_shap(explainer: shap.TreeExplainer, matrix: np.ndarray) -> np.ndarray:
    """SHAP values for the positive class, as (n_rows, n_features).

    Binary LightGBM returns a 2-D array, but some SHAP/model combinations return
    a 3-D array or a per-class list; this normalises all of them.
    """
    values = explainer.shap_values(matrix, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def sample_matrix(
    base_model: Pipeline, X: pd.DataFrame, sample_size: int = SHAP_SAMPLE_SIZE, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic row sample of the transformed matrix, with its row indices."""
    n_rows = len(X)
    if n_rows <= sample_size:
        positions = np.arange(n_rows)
    else:
        positions = np.sort(np.random.default_rng(seed).choice(n_rows, sample_size, replace=False))
    return transform_features(base_model, X.iloc[positions]), positions


def global_importance(
    base_model: Pipeline, X: pd.DataFrame, sample_size: int = SHAP_SAMPLE_SIZE
) -> pd.DataFrame:
    """Mean |SHAP| per transformed feature, ranked -- the global summary."""
    matrix, _ = sample_matrix(base_model, X, sample_size)
    values = _positive_class_shap(build_explainer(base_model), matrix)

    frame = pd.DataFrame(
        {
            "feature": transformed_feature_names(base_model),
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    )
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_instance(
    base_model: Pipeline, X: pd.DataFrame, row_position: int = 0, top_n: int = SHAP_TOP_FEATURES
) -> dict:
    """Per-prediction explanation: the top contributing features for one patient.

    `row_position` is positional (iloc), not a DataFrame index label.
    """
    row = X.iloc[[row_position]]
    matrix = transform_features(base_model, row)
    explainer = build_explainer(base_model)
    values = _positive_class_shap(explainer, matrix)[0]

    names = transformed_feature_names(base_model)
    contributions = (
        pd.DataFrame(
            {"feature": names, "shap_value": values, "feature_value": matrix[0]},
        )
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
    )

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).ravel()[-1]

    return {
        "row_position": int(row_position),
        "predicted_probability": round(float(base_model.predict_proba(row)[0, 1]), 6),
        "shap_base_value": round(float(base_value), 6),
        "shap_sum": round(float(values.sum()), 6),
        "top_contributions": [
            {
                "feature": str(r.feature),
                "shap_value": round(float(r.shap_value), 6),
                "feature_value": round(float(r.feature_value), 6),
            }
            for r in contributions.itertuples()
        ],
    }


def _write_shap_figure(importance: pd.DataFrame, path, top_n: int = SHAP_TOP_FEATURES) -> None:
    """Horizontal bar chart of the strongest global contributors.

    A plain bar chart of the exported table rather than `shap.summary_plot`, so
    the figure is deterministic and matches `shap_global_importance.csv` exactly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).iloc[::-1]

    figure, axes = plt.subplots(figsize=(8, max(4, 0.32 * len(top))))
    axes.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    axes.set_xlabel("mean |SHAP value|")
    axes.set_title(f"Global feature importance (top {len(top)})")
    figure.tight_layout()
    figure.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(figure)


def main() -> None:
    import json

    from ml.config import (
        EVAL_PATH,
        REPORTS_DIR,
        SHAP_EXAMPLE_PATH,
        SHAP_FIGURE_PATH,
        SHAP_IMPORTANCE_PATH,
    )
    from ml.data.features import split_features_target
    from ml.train import load_models, load_split

    eval_df = load_split(EVAL_PATH)
    X_eval, _ = split_features_target(eval_df)
    base_model, _ = load_models()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    importance = global_importance(base_model, X_eval)
    importance.to_csv(SHAP_IMPORTANCE_PATH, index=False, lineterminator="\n")

    example = explain_instance(base_model, X_eval, row_position=0)
    SHAP_EXAMPLE_PATH.write_text(
        json.dumps(example, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    _write_shap_figure(importance, SHAP_FIGURE_PATH)

    print(
        f"global importance over {len(importance)} transformed features -> {SHAP_IMPORTANCE_PATH}"
    )
    print(f"top feature: {importance.iloc[0]['feature']}")
    print(f"local example -> {SHAP_EXAMPLE_PATH}")


if __name__ == "__main__":
    main()
