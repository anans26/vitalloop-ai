"""What every page is handed: a database session, the API under the user's token, the registry.

**Who is looking.** The dashboard has no user store of its own and must not
hold the JWT signing secret (that would let it mint any identity -- the quiet
credential spread `loop/monitor/database.py` warns about). So an ops user
pastes a token issued the documented way (`python -m scripts.issue_dev_token
--role ops`), the dashboard asks the API who it belongs to (`/ops/whoami`), and
every page stays locked until the API says `ops`. Decisions are then made
*through* the API with that same token, so the name on an approval row is the
API's verified subject, never something the dashboard typed.

**The database** is reached with the monitor's settings (the POSTGRES_* or
VITALLOOP_DATABASE_URL variables -- no JWT secret), through an engine private
to the dashboard and keyed by URL, so it never swaps the process-wide engine
another component configured.
"""

import os
from dataclasses import dataclass
from typing import Any

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from dashboard.api_client import ApiError, OpsApi

DEFAULT_API_URL = "http://localhost:8000"
ALIASES = ("champion", "challenger", "shadow")


@dataclass
class Context:
    api: Any
    identity: dict
    session_factory: Any

    def session(self) -> Session:
        return self.session_factory()


def api_url() -> str:
    return os.environ.get("VITALLOOP_API_URL", DEFAULT_API_URL)


def database_url() -> str:
    from loop.monitor.database import monitor_database_settings

    return monitor_database_settings().resolved_database_url()


@st.cache_resource(show_spinner=False)
def _session_factory(url: str):
    from db.session import init_db

    engine = create_engine(url, pool_pre_ping=True, future=True)
    # The documented schema initialisation: a dashboard pointed at an older
    # database brings it to the current schema rather than failing on a read.
    init_db(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def session_factory():
    return _session_factory(database_url())


def alias_versions() -> dict[str, str | None]:
    """Where each registry alias points right now; None if unset or unreachable."""
    try:
        from loop.approval.registry import MlflowAliasRegistry
        from ml.tracking import resolve_tracking_uri
        from ml.tracking_config import REGISTERED_MODEL_NAME

        registry = MlflowAliasRegistry(resolve_tracking_uri(), REGISTERED_MODEL_NAME)
        return {alias: registry.version_of(alias) for alias in ALIASES}
    except Exception:
        return dict.fromkeys(ALIASES)


def alias_audit_rows() -> list[dict]:
    """The registry's append-only alias audit trail (mlflow/registry_audit.jsonl)."""
    from ml.registry import read_audit_rows

    try:
        return read_audit_rows()
    except Exception:
        return []


def authenticate() -> Context | None:
    """The sidebar sign-in. Returns a Context only for a token the API verifies as `ops`."""
    st.sidebar.markdown("### VitalLoop ops")
    token = st.sidebar.text_input(
        "Ops token",
        type="password",
        key="ops_token",
        help="Issue one with: python -m scripts.issue_dev_token --subject <you> --role ops",
    )
    if not token:
        return None

    cache = st.session_state.setdefault("_whoami", {})
    if token not in cache:
        try:
            cache[token] = OpsApi(api_url(), token).whoami()
        except ApiError as error:
            cache.pop(token, None)
            st.sidebar.error(f"Token refused: {error.detail}")
            return None

    identity = cache[token]
    st.sidebar.success(f"Signed in as **{identity['subject']}** ({identity['role']})")
    st.sidebar.caption(f"API: {api_url()}")
    return Context(
        api=OpsApi(api_url(), token), identity=identity, session_factory=session_factory()
    )
