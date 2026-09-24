"""Shadow scoring: the `shadow` model scores every request, after the response.

ARCHITECTURE.md §3.6: "when a `shadow` alias exists, middleware also scores the
request with the shadow model and logs it. The response *only ever* contains
the champion score." RISK_ANALYSIS.md §3 adds the constraint that shapes the
implementation: "Shadow scoring is post-response/async in middleware; champion
path never waits on it."

So the shadow runs as a FastAPI background task, which Starlette executes only
after the response has been sent. The clinician's answer, its latency and its
audit row are already settled by then; nothing the shadow does -- slow,
failing, or loaded with a broken model -- can change any of them. Every error
here is caught and recorded as a row with a failure status, because a shadow
that fails on some inputs is a finding about that shadow, not an outage.

Only the calibrated model is loaded: a shadow score is never explained to
anyone, so it needs no SHAP explainer. It is loaded by **version**, not by
alias, so the model scoring and the version written on the row cannot come
apart if the alias moves while the API is running.
"""

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from api.config import Settings
from api.logging_config import get_logger

logger = get_logger()

SHADOW_ALIAS = "shadow"


@dataclass(frozen=True)
class ShadowBundle:
    calibrated_model: Any
    model_name: str
    model_version: str


def load_shadow_bundle(settings: Settings, champion_version: str | None) -> ShadowBundle | None:
    """The registered `shadow` model, or None when there is nothing to shadow.

    None -- never an exception -- when shadowing is switched off, when the API
    serves the local artifact (there is no registry to hold a shadow alias),
    when no shadow alias is set, when it points at the champion itself, or when
    it cannot be loaded. The API must start and serve regardless.
    """
    if not settings.shadow_enabled or settings.model_source != "mlflow":
        return None

    try:
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        client = MlflowClient(settings.mlflow_tracking_uri)
        version = client.get_model_version_by_alias(settings.registered_model_name, SHADOW_ALIAS)
    except Exception:
        return None

    version_label = str(version.version)
    if champion_version is not None and version_label == str(champion_version):
        return None

    try:
        model = mlflow.sklearn.load_model(
            f"models:/{settings.registered_model_name}/{version_label}"
        )
    except Exception as error:
        logger.warning(
            "shadow_model_unavailable",
            model_version=version_label,
            error_category=type(error).__name__,
        )
        return None

    return ShadowBundle(
        calibrated_model=model,
        model_name=settings.registered_model_name,
        model_version=version_label,
    )


def score_in_shadow(
    shadow: ShadowBundle,
    features: dict,
    *,
    request_id: str,
    champion_version: str,
    champion_score: float,
    threshold: float,
    session_factory=None,
) -> None:
    """Scores one request with the shadow model and writes the row. Never raises.

    `features` is held in memory for the length of this call only; the row
    stores scores and versions, never the payload.
    """
    from db.models import ShadowPrediction
    from loop.shadow.persistence import STATUS_FAILED, STATUS_SCORED, persist_shadow_row

    started = time.perf_counter()
    score, status, error_category = None, STATUS_SCORED, None
    try:
        frame = pd.DataFrame([features])
        score = float(shadow.calibrated_model.predict_proba(frame)[0, 1])
    except Exception as error:
        status, error_category = STATUS_FAILED, type(error).__name__

    row = ShadowPrediction(
        request_id=request_id,
        model_name=shadow.model_name,
        champion_version=str(champion_version),
        shadow_version=shadow.model_version,
        champion_score=champion_score,
        shadow_score=score,
        decision_threshold=threshold,
        status=status,
        error_category=error_category,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

    try:
        if session_factory is None:
            from db.session import get_session as session_factory
        session = session_factory()
        try:
            persist_shadow_row(session, row)
        finally:
            session.close()
    except Exception as error:
        # Post-response: there is no one to return an error to. Logged with the
        # request id so the gap in the shadow window can be traced.
        logger.error(
            "shadow_write_failed",
            request_id=request_id,
            shadow_version=shadow.model_version,
            error_category=type(error).__name__,
        )
        return

    logger.info(
        "shadow_scored",
        request_id=request_id,
        shadow_version=shadow.model_version,
        status=status,
    )
