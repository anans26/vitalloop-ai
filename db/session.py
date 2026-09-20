"""Engine and session handling, plus reproducible schema initialisation.

`init_db` is deliberately `create_all` rather than Alembic. The project has one
table in Week 5 and the roadmap adds the rest week by week; a migration system
would be infrastructure nobody is using yet. What matters for now is that a
fresh database can be brought to the current schema in one documented call,
which this provides.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def configure_engine(database_url: str, **kwargs) -> Engine:
    """Creates (or replaces) the process-wide engine."""
    global _engine, _session_factory

    options = {"pool_pre_ping": True, "future": True}
    options.update(kwargs)
    _engine = create_engine(database_url, **options)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("database engine is not configured; call configure_engine() first")
    return _engine


def get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("database engine is not configured; call configure_engine() first")
    return _session_factory()


def session_scope() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Creates any missing tables. Safe to run repeatedly."""
    Base.metadata.create_all(bind=engine or get_engine())


def check_connection(engine: Engine | None = None) -> bool:
    """A cheap liveness probe for the readiness endpoint."""
    try:
        with (engine or get_engine()).connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_engine() -> None:
    """Drops the cached engine. Used by tests between configurations."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
