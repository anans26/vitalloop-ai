"""Liveness and readiness.

Two different questions, deliberately kept apart. `/health` answers "is this
process up", so an orchestrator does not restart a container merely because a
database blipped. `/ready` answers "can this instance serve a prediction right
now", which means the model is loaded *and* the audit database is reachable --
serving without the latter would be refused anyway, since the audit write is
fail-closed.

Neither response carries configuration, credentials, or exception detail.
"""

from fastapi import APIRouter, Request, Response, status

from api.schemas import HealthResponse, ReadinessResponse
from db.session import check_connection

router = APIRouter(tags=["operations"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Reports that the process is running. Requires no authentication.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="vitalloop-api")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Reports whether this instance can serve a prediction: the model is "
        "loaded and the audit database is reachable. Returns 503 when not ready. "
        "Requires no authentication."
    ),
)
def ready(request: Request, response: Response) -> ReadinessResponse:
    bundle = getattr(request.app.state, "model_bundle", None)
    model_loaded = bundle is not None
    database_available = check_connection()

    is_ready = model_loaded and database_available
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        model_loaded=model_loaded,
        database_available=database_available,
        # A version string, never a path or a URI.
        model_version=bundle.model_version if bundle else None,
    )
