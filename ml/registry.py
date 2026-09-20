"""MLflow Model Registry helper with an append-only alias audit trail.

project_docs/ARCHITECTURE.md §3.5: "Alias moves are the *only* way a model
changes state, and every move writes an audit row." This module is the single
place an alias is allowed to move, so that rule is enforceable rather than
aspirational.

Week 4 scope is registration plus the initial `champion` alias on version 1,
which is the roadmap's stated deliverable. It deliberately does **not** promote
later versions: moving a champion is a governed act that Week 8's validation
gate and Week 9's human approval exist to authorise, and inventing that policy
here would pre-empt them.

The audit rows land in JSONL because the Postgres schema that will hold them
arrives in Week 5; the record shape is chosen to migrate cleanly into a table.
"""

import getpass
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from mlflow.tracking import MlflowClient

from ml.tracking_config import (
    INITIAL_ALIAS,
    MODEL_ALIASES,
    REGISTRY_AUDIT_PATH,
)


def _actor() -> str:
    """Who performed the move. Best-effort; never fails a registration."""
    for key in ("VITALLOOP_ACTOR", "USERNAME", "USER"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def write_audit_row(record: dict, audit_path=None) -> dict:
    """Appends one immutable record. Never rewrites existing lines."""
    audit_path = Path(audit_path) if audit_path else REGISTRY_AUDIT_PATH
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": datetime.now(UTC).isoformat(), **record}
    with open(audit_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def read_audit_rows(audit_path=None) -> list[dict]:
    audit_path = Path(audit_path) if audit_path else REGISTRY_AUDIT_PATH
    if not audit_path.exists():
        return []
    with open(audit_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def register_model(client: MlflowClient, run_id: str, source_uri: str, model_name: str) -> str:
    """Registers an already-logged model and returns its version number.

    `source_uri` must be the URI returned by `log_model` (``models:/m-...`` in
    MLflow 3), not a ``runs:/<id>/<name>`` artifact path. In MLflow 3 a logged
    model is a first-class object stored outside the run's artifact tree, so a
    run-artifact source registers a version that points at nothing and fails
    only later, at load time.
    """
    try:
        client.create_registered_model(model_name)
    except Exception:
        # Already exists; registering a new version is the normal path.
        pass

    version = client.create_model_version(name=model_name, source=source_uri, run_id=run_id)
    # Normalised to str: the SQLite store returns an int while the HTTP server
    # returns a string, and callers compare these values.
    return str(version.version)


def current_alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    """The version an alias currently points at, or None if unset."""
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except Exception:
        return None


def set_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
    run_id: str,
    reason: str,
    git_commit: str | None = None,
    audit_path=None,
) -> dict:
    """Moves an alias and records the move. The only sanctioned way to do either."""
    if alias not in MODEL_ALIASES:
        raise ValueError(f"unknown alias {alias!r}; expected one of {MODEL_ALIASES}")

    previous_version = current_alias_version(client, model_name, alias)
    client.set_registered_model_alias(model_name, alias, version)

    return write_audit_row(
        {
            "action": "set_alias",
            "model_name": model_name,
            "alias": alias,
            "from_version": previous_version,
            "to_version": str(version),
            "run_id": run_id,
            "git_commit": git_commit,
            "reason": reason,
            "actor": _actor(),
        },
        audit_path=audit_path,
    )


def ensure_initial_champion(
    client: MlflowClient,
    model_name: str,
    version: str,
    run_id: str,
    git_commit: str | None = None,
    audit_path=None,
) -> dict | None:
    """Sets `champion` only when nothing holds it yet.

    This delivers the roadmap's "registered champion v1" without inventing a
    promotion policy: once a champion exists, replacing it requires the Week 8
    gate and the Week 9 approval flow, so later runs register a version and stop.
    Returns the audit row, or None when a champion already exists.
    """
    if current_alias_version(client, model_name, INITIAL_ALIAS) is not None:
        return None

    return set_alias(
        client,
        model_name=model_name,
        alias=INITIAL_ALIAS,
        version=version,
        run_id=run_id,
        reason="initial registration of the Week 3 baseline model (roadmap Week 4 deliverable)",
        git_commit=git_commit,
        audit_path=audit_path,
    )
