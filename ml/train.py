"""Week 3 training entry point: LightGBM + isotonic calibration on the Week 2 data.

Two fitted objects come out of a run, and the distinction matters downstream:

* the **base model** -- a plain sklearn Pipeline of (feature pipeline -> LightGBM).
  It is what SHAP explains, because TreeExplainer needs the tree ensemble itself.
* the **calibrated model** -- the base pipeline wrapped in CalibratedClassifierCV
  (isotonic, 5-fold). This is the model used for inference, because a readmission
  score is only actionable if it is a trustworthy probability.

Both are fitted on `datasets/processed/train.csv` only. The frozen evaluation
slice is read solely to score an already-fitted model, and `future_stream.csv` is
never opened here at all.
"""

import json
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ml.config import (
    CALIBRATION_CV_FOLDS,
    CALIBRATION_METHOD,
    EARLY_STOPPING_ROUNDS,
    EARLY_STOPPING_VALIDATION_FRACTION,
    EVAL_PATH,
    LIGHTGBM_PARAMS,
    METADATA_PATH,
    MIN_ESTIMATORS,
    MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_PATH,
)
from ml.data.clean import TARGET_COLUMN
from ml.data.features import NON_FEATURE_COLUMNS, build_feature_pipeline, split_features_target


def load_split(path: Path) -> pd.DataFrame:
    """Reads one processed split, failing loudly if Week 2 has not been run."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `dvc repro` to build the Week 2 datasets."
        )
    return pd.read_csv(path, low_memory=False)


def select_n_estimators(X_train: pd.DataFrame, y_train: pd.Series) -> int:
    """Lets early stopping choose the tree count, using training data only.

    The validation tail is the chronologically last slice of TRAIN (train.csv is
    already ordered by encounter_id), which mirrors how the model will be used --
    fitted on the past, scored on the future -- without touching the frozen
    evaluation slice or the reserved future stream.
    """
    cut = int(len(X_train) * (1.0 - EARLY_STOPPING_VALIDATION_FRACTION))
    X_fit, X_validation = X_train.iloc[:cut], X_train.iloc[cut:]
    y_fit, y_validation = y_train.iloc[:cut], y_train.iloc[cut:]

    feature_pipeline = build_feature_pipeline()
    X_fit_t = feature_pipeline.fit_transform(X_fit)
    X_validation_t = feature_pipeline.transform(X_validation)

    probe = lightgbm.LGBMClassifier(**LIGHTGBM_PARAMS)
    probe.fit(
        X_fit_t,
        y_fit,
        eval_X=X_validation_t,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return max(int(probe.best_iteration_ or LIGHTGBM_PARAMS["n_estimators"]), MIN_ESTIMATORS)


def build_base_model(n_estimators: int | None = None) -> Pipeline:
    """Feature pipeline + LightGBM, unfitted.

    The feature pipeline is the *same* ml.data.features pipeline used everywhere
    else; Week 3 does not introduce a second preprocessing path.
    """
    params = dict(LIGHTGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("model", lightgbm.LGBMClassifier(**params)),
        ]
    )


def build_calibrated_model(base_model: Pipeline) -> CalibratedClassifierCV:
    """Wraps an *unfitted* clone of the base model in cross-validated calibration.

    Cloning matters: CalibratedClassifierCV refits the whole pipeline inside each
    fold, so preprocessing is fitted on fold-train only and the calibration map
    never sees data the underlying model was fitted on.
    """
    return CalibratedClassifierCV(
        estimator=clone(base_model),
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV_FOLDS,
    )


def train_models(train_df: pd.DataFrame) -> tuple[Pipeline, CalibratedClassifierCV, int]:
    """Fits the base and calibrated models on the training slice only.

    Returns the tree count chosen by early stopping alongside the models so it can
    be recorded in the run metadata.
    """
    X_train, y_train = split_features_target(train_df)
    n_estimators = select_n_estimators(X_train, y_train)

    base_model = build_base_model(n_estimators)
    base_model.fit(X_train, y_train)

    calibrated_model = build_calibrated_model(build_base_model(n_estimators))
    calibrated_model.fit(X_train, y_train)

    return base_model, calibrated_model, n_estimators


def build_metadata(train_df: pd.DataFrame, eval_df: pd.DataFrame, n_estimators: int) -> dict:
    """Everything needed to identify what was trained, on what, and how."""
    return {
        "model_type": "lightgbm.LGBMClassifier",
        "inference_model": (
            f"CalibratedClassifierCV({CALIBRATION_METHOD}, cv={CALIBRATION_CV_FOLDS}) "
            "over the base pipeline"
        ),
        "target": TARGET_COLUMN,
        "excluded_from_features": list(NON_FEATURE_COLUMNS),
        "feature_pipeline": "ml.data.features.build_feature_pipeline",
        "random_seed": RANDOM_SEED,
        "lightgbm_params": {**LIGHTGBM_PARAMS, "n_estimators": n_estimators},
        "n_estimators_selection": {
            "method": "early stopping on a chronological validation tail of the training slice",
            "validation_fraction": EARLY_STOPPING_VALIDATION_FRACTION,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "max_n_estimators": LIGHTGBM_PARAMS["n_estimators"],
            "selected_n_estimators": int(n_estimators),
        },
        "calibration": {"method": CALIBRATION_METHOD, "cv_folds": CALIBRATION_CV_FOLDS},
        "data": {
            "train_path": TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "eval_path": EVAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "train_positive_rate": round(float(train_df[TARGET_COLUMN].mean()), 6),
            "eval_positive_rate": round(float(eval_df[TARGET_COLUMN].mean()), 6),
            "future_stream_used": False,
        },
        "versions": {
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }


def save_models(base_model: Pipeline, calibrated_model: CalibratedClassifierCV) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"base_model": base_model, "calibrated_model": calibrated_model},
        MODEL_PATH,
    )


def load_models() -> tuple[Pipeline, CalibratedClassifierCV]:
    """Reloads what `save_models` wrote."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["base_model"], bundle["calibrated_model"]


def main() -> None:
    train_df = load_split(TRAIN_PATH)
    eval_df = load_split(EVAL_PATH)

    base_model, calibrated_model, n_estimators = train_models(train_df)
    save_models(base_model, calibrated_model)

    # Deliberately no wall-clock timestamp: model_metadata.json is a DVC stage
    # output, and a timestamp would change its hash on every run. Run timing is
    # MLflow's job in Week 4.
    metadata = build_metadata(train_df, eval_df, n_estimators)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"trained on {len(train_df)} rows; early stopping selected {n_estimators} trees")
    print(f"saved model to {MODEL_PATH}")
    print(f"metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    main()
