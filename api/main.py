"""FastAPI application for the VitalLoop serving layer.

Run it with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Startup loads the already-trained model once (see `api.model_loader`) and
creates the audit table if it is missing. Nothing here trains: the API depends
on `ml.data.features` for the feature contract and `ml.explain` for SHAP, but
never imports `ml.train`.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.logging_config import configure_logging, get_logger
from api.model_loader import ModelUnavailableError, load_model_bundle
from api.routers import health, ops, predict
from api.shadow import load_shadow_bundle
from db.session import configure_engine, init_db

logger = get_logger()

API_DESCRIPTION = """
Authenticated 30-day hospital readmission risk scoring.

Every prediction is written to an append-only audit row in PostgreSQL before it
is returned. The request body itself is never stored -- only a SHA-256 hash of
it, which is enough to verify later which input produced a score without the
audit trail becoming a second copy of the clinical record.

All prediction endpoints require a JWT bearer token carrying a `clinician` or
`ops` role claim. The `/ops` endpoints -- shadow statistics, retrain
authorisation and promotion -- require the `ops` role: whoever *sees* a risk
score is not automatically whoever *promotes* a model.

When a `shadow` model is registered, every prediction is also scored by it
after the response is sent. The response only ever carries the champion's
score.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve configuration, database and model once, at startup."""
    configure_logging()
    settings = get_settings()

    configure_engine(settings.resolved_database_url())
    try:
        init_db()
    except Exception as error:
        # Not fatal: /ready will report the database as unavailable, and the
        # fail-closed audit path refuses predictions until it recovers.
        logger.warning("database_init_failed", error_category=type(error).__name__)
        _ = error

    try:
        app.state.model_bundle = load_model_bundle(settings)
        logger.info(
            "model_loaded",
            model_name=app.state.model_bundle.model_name,
            model_version=app.state.model_bundle.model_version,
            model_source=app.state.model_bundle.model_source,
        )
    except ModelUnavailableError as error:
        # Start anyway so /health and /ready can report the problem; /predict
        # returns 503 rather than the process crash-looping.
        app.state.model_bundle = None
        logger.error("model_unavailable", error_category=type(error).__name__)

    # Week 9: the shadow model, if one is registered. Never fatal -- the
    # champion serves whether or not anything is shadowing it.
    champion = app.state.model_bundle
    app.state.shadow_bundle = load_shadow_bundle(
        settings, champion.model_version if champion else None
    )
    if app.state.shadow_bundle is not None:
        logger.info("shadow_loaded", model_version=app.state.shadow_bundle.model_version)

    yield

    app.state.model_bundle = None
    app.state.shadow_bundle = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="VitalLoop Readmission API",
        description=API_DESCRIPTION,
        version="0.9.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        """Field-level detail, but never the submitted values.

        FastAPI's default body echoes the offending input back to the caller,
        which for this service would put clinical values into an error response
        and any log that captures it.
        """
        fields = [
            {"field": ".".join(str(p) for p in err.get("loc", ())[1:]), "error": err.get("msg")}
            for err in exc.errors()
        ]
        return JSONResponse(
            # Literal 422: starlette renamed the constant and deprecated the old name.
            status_code=422,
            content={"detail": "Request validation failed.", "errors": fields},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        """Last resort: log the category, return nothing revealing."""
        logger.error("unhandled_error", path=request.url.path, error_category=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(ops.router)
    return app


app = create_app()
