"""Loads the already-trained inference model for serving.

The API never trains. It loads what Weeks 3-4 produced, once, at startup:

* `model_source="mlflow"` -- the registered model by **alias**, which is what
  the roadmap means by "champion loaded via alias". An alias is deliberate;
  resolving "latest" would let the served model change without anyone deciding
  that it should.
* `model_source="local"` -- the DVC-tracked joblib bundle, for offline
  development and for tests, which must not need a tracking server.

Two models come back, as in Week 3: the calibrated model produces the score a
clinician sees, and the uncalibrated base model backs the SHAP explainer.
`ARCHITECTURE.md` §3.6 requires top-3 SHAP factors in the response, so a bundle
without a usable explainer is a failure, not a degraded mode.
"""

from dataclasses import dataclass
from typing import Any

import joblib

from api.config import Settings
from ml.config import MODEL_PATH
from ml.tracking_config import BASE_MODEL_ARTIFACT, CALIBRATED_MODEL_ARTIFACT


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded. Never swallowed."""


@dataclass
class ModelBundle:
    """Everything a request needs, resolved once at startup."""

    calibrated_model: Any
    base_model: Any
    explainer: Any
    feature_names: list[str]
    model_name: str
    model_version: str
    model_source: str
    data_version: str | None = None

    def version_label(self) -> str:
        return f"{self.model_name}:{self.model_version}"


def _load_local() -> tuple[Any, Any, str]:
    """The DVC-tracked bundle written by `ml.train`.

    Read directly with joblib rather than importing `ml.train`: the API has no
    business importing a module whose `main()` trains a model.
    """
    if not MODEL_PATH.exists():
        raise ModelUnavailableError(
            f"Model artifact not found at {MODEL_PATH.name}. Run `dvc repro` to build it."
        )
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle["calibrated_model"], bundle["base_model"], "local-artifact"
    except Exception as error:
        raise ModelUnavailableError(f"Could not load the local model artifact: {error}") from error


def _load_mlflow(settings: Settings) -> tuple[Any, Any, str]:
    """The registered champion, plus the base model logged in the same run."""
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(settings.mlflow_tracking_uri)
    name, alias = settings.registered_model_name, settings.model_alias

    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as error:
        raise ModelUnavailableError(
            f"No model registered as {name!r} with alias {alias!r} at "
            f"{settings.mlflow_tracking_uri}: {error}"
        ) from error

    try:
        calibrated = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    except Exception as error:
        raise ModelUnavailableError(f"Could not load {name}@{alias}: {error}") from error

    # The explainer needs the tree ensemble, which the calibrated wrapper hides.
    base_uri = _find_base_model_uri(client, version.run_id)
    if base_uri is None:
        raise ModelUnavailableError(
            f"Run {version.run_id} has no {BASE_MODEL_ARTIFACT!r} logged model, so SHAP "
            "explanations cannot be produced."
        )
    try:
        base = mlflow.sklearn.load_model(base_uri)
    except Exception as error:
        raise ModelUnavailableError(f"Could not load the base model: {error}") from error

    return calibrated, base, str(version.version)


def _find_base_model_uri(client, run_id: str) -> str | None:
    """Locates the base model logged beside the champion in the same run.

    `search_logged_models` requires the experiment, so it is resolved from the
    run first. Failures here raise rather than returning None: a swallowed
    exception reports "no base model logged" for what may be an entirely
    different problem.
    """
    try:
        experiment_id = client.get_run(run_id).info.experiment_id
        logged = client.search_logged_models(experiment_ids=[experiment_id])
    except Exception as error:
        raise ModelUnavailableError(
            f"Could not search logged models for run {run_id}: {error}"
        ) from error

    for model in logged:
        if model.name == BASE_MODEL_ARTIFACT and model.source_run_id == run_id:
            return f"models:/{model.model_id}"
    return None


def _data_version() -> str | None:
    """The DVC hash of the training data, recorded alongside each prediction."""
    try:
        from ml.tracking import dvc_lineage

        return dvc_lineage().get("dvc_train_md5")
    except Exception:
        return None


def load_model_bundle(settings: Settings) -> ModelBundle:
    """Resolves the configured model. Raises ModelUnavailableError on any failure."""
    if settings.model_source == "mlflow":
        calibrated, base, version = _load_mlflow(settings)
        model_name = settings.registered_model_name
    else:
        calibrated, base, version = _load_local()
        model_name = f"{settings.registered_model_name} ({CALIBRATED_MODEL_ARTIFACT})"

    from ml.explain import build_explainer, transformed_feature_names

    try:
        explainer = build_explainer(base)
        feature_names = transformed_feature_names(base)
    except Exception as error:
        raise ModelUnavailableError(f"Could not build the SHAP explainer: {error}") from error

    return ModelBundle(
        calibrated_model=calibrated,
        base_model=base,
        explainer=explainer,
        feature_names=feature_names,
        model_name=model_name,
        model_version=str(version),
        model_source=settings.model_source,
        data_version=_data_version(),
    )
