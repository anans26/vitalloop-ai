"""The monitor's database configuration.

Deliberately not `api.config.Settings`. That class requires
`VITALLOOP_JWT_SECRET` and fails at startup without it -- correct for a service
that signs tokens, wrong for a worker that never issues one. Importing it here
would mean handing the monitor container a signing secret it has no use for,
which is exactly the kind of quiet credential spread ARCHITECTURE.md §6 exists
to prevent.

So the monitor reads the database variables and nothing else. They are the same
variables, resolved the same way, against the same database -- there is one
`drift_events` table and one audit store.
"""

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.orm import Session

from db.session import configure_engine, get_session, init_db


class MonitorDatabaseSettings(BaseSettings):
    """Set `VITALLOOP_DATABASE_URL`, or let the POSTGRES_* variables compose it."""

    model_config = SettingsConfigDict(env_prefix="VITALLOOP_", extra="ignore")

    database_url: str | None = None
    postgres_user: str = Field(default="vitalloop", alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="vitalloop", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    def resolved_database_url(self) -> str:
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


def monitor_database_settings() -> MonitorDatabaseSettings:
    try:
        return MonitorDatabaseSettings()
    except ValidationError as error:
        raise RuntimeError(
            "Monitor database configuration is incomplete. "
            "See docs/RUNNING_THE_PROJECT.md for the required environment."
        ) from error


def open_session() -> Session:
    """Configures the engine, creates any missing tables, returns a session.

    `init_db` is the documented initialisation path for every table in the
    schema, `drift_events` included -- a monitor started against a database that
    predates Week 6 brings itself up to the current schema rather than failing
    on the first write.
    """
    configure_engine(monitor_database_settings().resolved_database_url())
    init_db()
    return get_session()
