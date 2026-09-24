"""Registry alias handling and the append-only audit trail.

These use MLflow's SQLite backend in a temp directory: the Model Registry needs
a database store, but it does not need a running server, so the tests stay
CI-safe.
"""

import json

import pytest
from mlflow.tracking import MlflowClient

from ml.registry import (
    current_alias_version,
    delete_alias,
    ensure_initial_champion,
    read_audit_rows,
    set_alias,
    write_audit_row,
)

MODEL_NAME = "test-readmission"


@pytest.fixture
def client(tmp_path) -> MlflowClient:
    return MlflowClient(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")


@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "registry_audit.jsonl"


@pytest.fixture
def registered_version(client, tmp_path) -> str:
    """A registered version that points at a placeholder source.

    Alias handling is independent of what the source contains, so these tests
    avoid the cost of fitting and logging a real model.
    """
    client.create_registered_model(MODEL_NAME)
    version = client.create_model_version(
        name=MODEL_NAME, source=(tmp_path / "model").as_uri(), run_id=None
    )
    # str: the SQLite store returns an int here while the HTTP server returns a
    # string, and ml.registry normalises to str.
    return str(version.version)


def test_write_audit_row_appends_and_timestamps(audit_path):
    write_audit_row({"action": "first"}, audit_path=audit_path)
    write_audit_row({"action": "second"}, audit_path=audit_path)

    rows = read_audit_rows(audit_path)
    assert [row["action"] for row in rows] == ["first", "second"]
    assert all("timestamp_utc" in row for row in rows)


def test_audit_log_is_append_only(audit_path):
    """A second write must never rewrite the first line."""
    write_audit_row({"action": "first"}, audit_path=audit_path)
    original = audit_path.read_text(encoding="utf-8")

    write_audit_row({"action": "second"}, audit_path=audit_path)
    assert audit_path.read_text(encoding="utf-8").startswith(original)


def test_read_audit_rows_returns_empty_when_absent(tmp_path):
    assert read_audit_rows(tmp_path / "nothing.jsonl") == []


def test_set_alias_moves_the_alias_and_records_it(client, registered_version, audit_path):
    row = set_alias(
        client,
        model_name=MODEL_NAME,
        alias="champion",
        version=registered_version,
        run_id="run-123",
        reason="unit test",
        git_commit="abc123",
        audit_path=audit_path,
    )

    assert current_alias_version(client, MODEL_NAME, "champion") == registered_version
    assert row["action"] == "set_alias"
    assert row["from_version"] is None
    assert row["to_version"] == str(registered_version)
    assert row["run_id"] == "run-123"
    assert row["git_commit"] == "abc123"
    assert row["actor"]
    assert json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])["alias"] == "champion"


def test_set_alias_rejects_an_unknown_alias(client, registered_version, audit_path):
    with pytest.raises(ValueError, match="unknown alias"):
        set_alias(
            client,
            model_name=MODEL_NAME,
            alias="production",
            version=registered_version,
            run_id="run-123",
            reason="should not happen",
            audit_path=audit_path,
        )
    assert read_audit_rows(audit_path) == []


def test_current_alias_version_is_none_when_unset(client, registered_version):
    assert current_alias_version(client, MODEL_NAME, "shadow") is None


def test_ensure_initial_champion_sets_champion_once(client, registered_version, audit_path):
    first = ensure_initial_champion(
        client,
        model_name=MODEL_NAME,
        version=registered_version,
        run_id="run-1",
        audit_path=audit_path,
    )
    assert first is not None
    assert current_alias_version(client, MODEL_NAME, "champion") == registered_version

    second_version = str(
        client.create_model_version(name=MODEL_NAME, source=first["run_id"], run_id=None).version
    )
    second = ensure_initial_champion(
        client,
        model_name=MODEL_NAME,
        version=second_version,
        run_id="run-2",
        audit_path=audit_path,
    )

    # Promotion is a governed act for Weeks 8-9; Week 4 must not perform it.
    assert second is None
    assert current_alias_version(client, MODEL_NAME, "champion") == registered_version
    assert len(read_audit_rows(audit_path)) == 1


# ---------------------------------------------------------------------------
# Week 9: who moved it, and removing an alias
# ---------------------------------------------------------------------------
def _set(client, version, audit_path, **kwargs):
    return set_alias(
        client,
        model_name=MODEL_NAME,
        alias="shadow",
        version=version,
        run_id="run-1",
        reason="unit test",
        audit_path=audit_path,
        **kwargs,
    )


def test_set_alias_records_a_named_actor(client, registered_version, audit_path):
    """A promotion is made in the approver's name, not the API process's OS user."""
    row = _set(client, registered_version, audit_path, actor="ops-alice")
    assert row["actor"] == "ops-alice"


def test_delete_alias_removes_it_and_records_the_removal(client, registered_version, audit_path):
    _set(client, registered_version, audit_path)

    row = delete_alias(
        client,
        model_name=MODEL_NAME,
        alias="shadow",
        run_id="run-1",
        reason="promotion rejected",
        audit_path=audit_path,
        actor="ops-alice",
    )

    assert current_alias_version(client, MODEL_NAME, "shadow") is None
    assert row["action"] == "delete_alias"
    assert row["from_version"] == registered_version
    assert row["to_version"] is None
    assert row["actor"] == "ops-alice"
    assert [r["action"] for r in read_audit_rows(audit_path)] == ["set_alias", "delete_alias"]


def test_deleting_an_unset_alias_records_nothing(client, registered_version, audit_path):
    assert (
        delete_alias(
            client,
            model_name=MODEL_NAME,
            alias="shadow",
            run_id="run-1",
            reason="nothing to do",
            audit_path=audit_path,
        )
        is None
    )
    assert read_audit_rows(audit_path) == []


def test_delete_alias_rejects_an_unknown_alias(client, audit_path):
    with pytest.raises(ValueError, match="unknown alias"):
        delete_alias(
            client,
            model_name=MODEL_NAME,
            alias="production",
            run_id="run-1",
            reason="should not happen",
            audit_path=audit_path,
        )
