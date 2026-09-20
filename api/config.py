"""Serving configuration, read from the environment.

Nothing here carries a usable default for a secret. `VITALLOOP_JWT_SECRET` has
no fallback at all, so a misconfigured deployment fails at startup rather than
signing tokens with a value that happens to be in the repository.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_NAME = "vitalloop-readmission"
DEFAULT_MODEL_ALIAS = "champion"


class Settings(BaseSettings):
    """Every knob the API needs. Prefixed `VITALLOOP_` except the POSTGRES_* vars,
    which already exist in docker/.env for the Postgres container itself."""

    model_config = SettingsConfigDict(env_prefix="VITALLOOP_", extra="ignore")

    # --- Authentication ---------------------------------------------------
    # No default: an unset secret must break startup, not silently weaken auth.
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    jwt_issuer: str = "vitalloop"
    jwt_audience: str = "vitalloop-api"

    # --- Model ------------------------------------------------------------
    # "mlflow" is the documented path (roadmap Week 5: "champion loaded via
    # alias"). "local" loads the DVC-tracked artifact, for offline development
    # and tests. Either way the choice is explicit -- never an implicit "latest".
    model_source: Literal["mlflow", "local"] = "mlflow"
    mlflow_tracking_uri: str = "http://localhost:5000"
    registered_model_name: str = DEFAULT_MODEL_NAME
    model_alias: str = DEFAULT_MODEL_ALIAS

    # --- Serving ----------------------------------------------------------
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_shap_features: int = Field(default=3, ge=1, le=20)

    # --- Database ---------------------------------------------------------
    # Set VITALLOOP_DATABASE_URL directly, or leave it unset and let the
    # POSTGRES_* variables compose it.
    database_url: str | None = None
    postgres_user: str = Field(default="vitalloop", alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="vitalloop", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    def resolved_database_url(self) -> str:
        """The SQLAlchemy URL, composed from POSTGRES_* when not given directly."""
        if self.database_url:
            return self.database_url
        if not self.postgres_password:
            raise RuntimeError(
                "No database configuration: set VITALLOOP_DATABASE_URL, or "
                "POSTGRES_PASSWORD (plus the other POSTGRES_* variables)."
            )
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings. Raises a message that names the missing variable."""
    try:
        return Settings()
    except ValidationError as error:
        missing = ", ".join(
            f"VITALLOOP_{'.'.join(str(p) for p in err['loc']).upper()}"
            for err in error.errors()
            if err["type"] == "missing"
        )
        raise RuntimeError(
            f"API configuration is incomplete. Missing: {missing or 'see details'}. "
            "See docs/RUNNING_THE_PROJECT.md for the required environment."
        ) from error
