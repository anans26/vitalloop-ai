"""The prediction endpoint.

Flow, per `ARCHITECTURE.md` §3.6: validated payload -> feature pipeline ->
champion model -> calibrated risk + top-3 SHAP factors, with one audit row per
request. The audit write is fail-closed: see `api.audit`.
"""

import time
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.audit import (
    STATUS_PREDICTION_FAILED,
    STATUS_SUCCESS,
    AuditWriteError,
    build_audit_row,
    hash_payload,
    persist_audit_row,
)
from api.auth import CLINICIAN_ROLE, OPS_ROLE, Principal, require_role
from api.config import Settings, get_settings
from api.logging_config import get_logger, request_log_fields
from api.model_loader import ModelBundle
from api.schemas import ErrorResponse, PredictionRequest, PredictionResponse, ShapContribution
from db.session import session_scope

router = APIRouter(tags=["prediction"])
logger = get_logger()


def get_model_bundle(request: Request) -> ModelBundle:
    """The bundle loaded once during startup; never loaded per request."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded; the service is not ready.",
        )
    return bundle


def _top_contributions(bundle: ModelBundle, frame: pd.DataFrame, top_n: int) -> list[dict]:
    """Top-N SHAP contributors for this row.

    Runs on the transformed matrix, so the names line up with what the model
    actually consumes. Only names and contributions leave this function.
    """
    from ml.explain import _positive_class_shap, transform_features

    matrix = transform_features(bundle.base_model, frame)
    values = _positive_class_shap(bundle.explainer, matrix)[0]
    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {"feature": str(bundle.feature_names[i]), "contribution": round(float(values[i]), 6)}
        for i in order
    ]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score one encounter for 30-day readmission risk",
    description=(
        "Returns a calibrated readmission probability with the top contributing "
        "factors, and writes one audit row. Requires a JWT bearer token with the "
        "`clinician` or `ops` role. The request body is hashed for the audit "
        "trail and is never stored."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Missing, malformed, or expired token"},
        403: {"model": ErrorResponse, "description": "Token lacks a permitted role"},
        422: {"model": ErrorResponse, "description": "Request failed validation"},
        503: {"model": ErrorResponse, "description": "Model unavailable or audit write failed"},
    },
)
def predict(
    payload: PredictionRequest,
    principal: Principal = Depends(require_role(CLINICIAN_ROLE, OPS_ROLE)),
    bundle: ModelBundle = Depends(get_model_bundle),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(session_scope),
) -> PredictionResponse:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    # by_alias so hyphenated medication columns match the training contract.
    features = payload.model_dump(by_alias=True)
    input_hash = hash_payload(features)

    risk_score: float | None = None
    predicted_class: int | None = None
    top_shap: list[dict] | None = None
    audit_status, error_category = STATUS_SUCCESS, None

    try:
        frame = pd.DataFrame([features])
        risk_score = float(bundle.calibrated_model.predict_proba(frame)[0, 1])
        predicted_class = int(risk_score >= settings.decision_threshold)
        top_shap = _top_contributions(bundle, frame, settings.top_shap_features)
    except Exception as error:
        audit_status, error_category = STATUS_PREDICTION_FAILED, type(error).__name__
        logger.error(
            "prediction_failed",
            **request_log_fields(
                request_id=request_id,
                caller=principal.subject,
                input_hash=input_hash,
                model_version=bundle.model_version,
                status=audit_status,
                error_category=error_category,
            ),
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    row = build_audit_row(
        request_id=request_id,
        caller=principal.subject,
        caller_role=principal.role,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        model_source=bundle.model_source,
        data_version=bundle.data_version,
        input_hash=input_hash,
        risk_score=risk_score,
        predicted_class=predicted_class,
        decision_threshold=settings.decision_threshold if risk_score is not None else None,
        top_shap=top_shap,
        status=audit_status,
        error_category=error_category,
        latency_ms=latency_ms,
    )

    try:
        persist_audit_row(session, row)
    except AuditWriteError as error:
        # Fail closed: no audit row, no score. Returning the prediction anyway
        # would make the audit trail best-effort.
        logger.error(
            "audit_write_failed",
            **request_log_fields(
                request_id=request_id,
                caller=principal.subject,
                input_hash=input_hash,
                model_version=bundle.model_version,
                status="audit_write_failed",
                latency_ms=latency_ms,
                error_category=type(error).__name__,
            ),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction could not be audited and was therefore not returned.",
        ) from error

    if audit_status != STATUS_SUCCESS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction failed. The attempt has been recorded.",
        )

    logger.info(
        "prediction_served",
        **request_log_fields(
            request_id=request_id,
            caller=principal.subject,
            input_hash=input_hash,
            model_version=bundle.model_version,
            status=audit_status,
            latency_ms=latency_ms,
        ),
    )

    return PredictionResponse(
        request_id=request_id,
        readmission_probability=round(risk_score, 6),
        predicted_class=predicted_class,
        decision_threshold=settings.decision_threshold,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        data_version=bundle.data_version,
        top_factors=[ShapContribution(**item) for item in top_shap],
    )
