"""Week 4 MLflow settings, deliberately kept out of ml/config.py.

`ml/config.py` is a declared dependency of the DVC model stages, so editing it
invalidates `dvc.lock` and forces a retrain. Tracking settings have no effect
on the trained model, so coupling them to the training pipeline would mean a
spurious rebuild every time an experiment-tracking detail changes.
"""

from ml.config import PROJECT_ROOT

# --- Week 4: MLflow tracking + registry -------------------------------------
# project_docs/ARCHITECTURE.md §3.5 specifies MLflow Tracking + Model Registry
# with three aliases; IMPLEMENTATION_ROADMAP.md Week 4 adds "MLflow in Compose"
# and "MLflow UI reachable at :5000".

MLFLOW_EXPERIMENT_NAME = "vitalloop-readmission"
REGISTERED_MODEL_NAME = "vitalloop-readmission"

# Local (no-Docker) store, used by tests and by offline development. SQLite
# rather than a file store because the Model Registry requires a database
# backend. The Compose service keeps its own store; see docs.
MLFLOW_LOCAL_DIR = PROJECT_ROOT / "mlflow"
MLFLOW_LOCAL_DB = MLFLOW_LOCAL_DIR / "mlflow.db"
MLFLOW_LOCAL_ARTIFACTS = MLFLOW_LOCAL_DIR / "artifacts"

# Append-only alias-move audit trail. The roadmap's Week 4 task calls for a
# promotion helper that "writes an audit row on every alias move"; the Postgres
# schema that will hold those rows is Week 5, so Week 4 writes JSONL and Week 5
# migrates it.
REGISTRY_AUDIT_PATH = MLFLOW_LOCAL_DIR / "registry_audit.jsonl"

MODEL_ALIASES = ("champion", "challenger", "shadow")
INITIAL_ALIAS = "champion"

# Artifact names inside an MLflow run.
CALIBRATED_MODEL_ARTIFACT = "calibrated_model"
BASE_MODEL_ARTIFACT = "base_model"
