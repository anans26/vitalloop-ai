"""Week 3 training configuration: paths, seed, and model hyperparameters.

Deliberately plain module-level constants. The versioned policy/gate YAML files
described in project_docs/ARCHITECTURE.md belong to Week 7; this module is only
what the training entry point needs in order to be reproducible.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Week 2 produced these; Week 3 only reads them.
TRAIN_PATH = PROCESSED_DIR / "train.csv"
EVAL_PATH = PROCESSED_DIR / "eval_frozen.csv"
# future_stream.csv is deliberately absent: it is reserved for the Week 6 drift
# scenarios and must never be read during training, calibration, or evaluation.

MODEL_PATH = MODELS_DIR / "readmission_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
RELIABILITY_PATH = REPORTS_DIR / "reliability_curve.csv"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_global_importance.csv"
SHAP_EXAMPLE_PATH = REPORTS_DIR / "shap_local_example.json"
CALIBRATION_FIGURE_PATH = FIGURES_DIR / "calibration_curve.png"
SHAP_FIGURE_PATH = FIGURES_DIR / "shap_summary.png"

RANDOM_SEED = 42

# Defensible defaults, not a tuned configuration. The roadmap caps Week 3 tuning
# explicitly ("freeze eval set, cap tuning time to 1 day") because over-tuning to
# a leaderboard AUROC is a named risk in project_docs/RISK_ANALYSIS.md.
# n_jobs=1 + deterministic + force_row_wise make training bit-reproducible, which
# matters because the model is a DVC stage output.
LIGHTGBM_PARAMS = {
    "objective": "binary",
    # Upper bound only -- the actual tree count is chosen by early stopping on a
    # chronological validation tail carved out of the TRAINING slice. At a fixed
    # 300 trees this model reached train AUROC 0.82 against eval 0.59, which is
    # memorisation, not signal.
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

# Isotonic over Platt: project_docs/PROJECT_DESIGN.md §6 -- "fine, isotonic is
# better with 100k rows". Cross-validated so calibration never sees the frozen
# evaluation slice.
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV_FOLDS = 5

# Early stopping, not a hyperparameter search: one principled mechanism that lets
# the data choose the tree count. The validation tail is the chronologically last
# slice of TRAIN, so it never touches eval_frozen.csv or future_stream.csv.
EARLY_STOPPING_VALIDATION_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 50
MIN_ESTIMATORS = 25

# Two reported operating points (see docs): the nominal 0.5 cut, and the
# top-decile cut that matches how a readmission risk list is actually worked.
DECISION_THRESHOLD = 0.5
TOP_DECILE_FRACTION = 0.10

RELIABILITY_BINS = 10
ECE_BINS = 10

# Capped so SHAP stays fast enough for CI and for a 3-minute demo.
SHAP_SAMPLE_SIZE = 2000
SHAP_TOP_FEATURES = 20

SUBGROUP_COLUMNS = ("age", "gender", "race")
