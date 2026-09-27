"""Week 11: `scripts/reset_demo.py` -- archive first, prove it restores, then wipe.

Docker is never called: a fake runner records every command and answers the
way `docker compose` and `psql` would. What is asserted is ordering and
refusal -- nothing destructive runs before a verified archive exists, and a
failed archive stops the reset with the state untouched.
"""

import json
import subprocess
from datetime import UTC, datetime

import pytest

from scripts import reset_demo
from scripts.reset_demo import Compose, ResetError, reset

DUMP = b"-- PostgreSQL database dump\nCOPY ...\n-- PostgreSQL database dump complete\n"
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
LIVE = {"approvals": "8", "decision_cards": "42", "predictions": "201"}


def done(stdout=b"", code=0, stderr=b""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class FakeDocker:
    """Answers `docker compose ...` like a running stack whose dump restores."""

    def __init__(self, *, dump=DUMP, restored=None, fail=None, cp_empty=False):
        self.calls: list[list[str]] = []
        self.dump = dump
        self.restored = dict(LIVE) if restored is None else restored
        self.fail = fail or ()
        self.cp_empty = cp_empty

    def __call__(self, args, input=None):
        self.calls.append(list(args))
        command = " ".join(args)
        for needle in self.fail:
            if needle in command:
                return done(code=1, stderr=f"{needle} exploded".encode())
        if args[:3] == ["docker", "compose", "config"]:
            return done(json.dumps({"name": "demoproj"}).encode())
        if "pg_dump" in args:
            return done(self.dump)
        if "cp" in args[2:3]:
            target = args[-1]
            from pathlib import Path

            Path(target).mkdir(parents=True)
            if not self.cp_empty:
                (Path(target) / "mlflow.db").write_bytes(b"sqlite")
            return done()
        if "-c" in args:
            sql = args[args.index("-c") + 1]
            database = args[args.index("-d") + 1]
            counts = self.restored if database == reset_demo.RESTORE_CHECK_DB else LIVE
            if sql.startswith("SELECT table_name"):
                return done("\n".join(sorted(counts)).encode())
            if sql.startswith("SELECT count(*)"):
                table = sql.split('"')[1]
                return done(counts[table].encode())
            return done()
        if args[:3] == ["docker", "volume", "rm"]:
            return done(args[3].encode())
        return done()

    def index(self, *needle) -> int:
        for position, call in enumerate(self.calls):
            if all(part in call for part in needle):
                return position
        return -1

    def ran(self, *needle) -> bool:
        return self.index(*needle) >= 0


@pytest.fixture
def audit(tmp_path):
    path = tmp_path / "mlflow" / "registry_audit.jsonl"
    path.parent.mkdir()
    path.write_text('{"alias": "champion", "to_version": "5"}\n', encoding="utf-8")
    return path


def do_reset(docker, tmp_path, audit, seeder=lambda: 0):
    compose = Compose(docker, db_user="vitalloop", db_name="vitalloop")
    return reset(
        compose,
        archive_root=tmp_path / "backups",
        registry_audit=audit,
        seeder=seeder,
        now=NOW,
        log=lambda *a: None,
    )


def test_a_reset_archives_verifies_wipes_starts_and_seeds_in_that_order(tmp_path, audit):
    docker, seeded = FakeDocker(), []
    archive_dir = do_reset(docker, tmp_path, audit, seeder=lambda: seeded.append(1) or 0)

    dump = docker.index("pg_dump")
    restore = docker.index("-q")  # the restore-check psql fed the dump on stdin
    copy = docker.index("cp", "mlflow:/mlflow")
    down = docker.index("down")
    first_rm = docker.index("volume", "rm")
    up_all = max(
        i for i, c in enumerate(docker.calls) if c[-4:] == ["up", "-d", "--build", "--wait"]
    )

    assert -1 < dump < restore < copy < down < first_rm < up_all
    assert seeded == [1]
    assert archive_dir.name == "demo-reset-20260927T120000Z"


def test_the_archive_holds_the_dump_the_store_the_trail_and_a_manifest(tmp_path, audit):
    original = audit.read_bytes()
    archive_dir = do_reset(FakeDocker(), tmp_path, audit)

    assert (archive_dir / "postgres.sql").read_bytes() == DUMP
    assert (archive_dir / "mlflow-server" / "mlflow.db").exists()
    assert (archive_dir / "registry_audit.jsonl").read_bytes() == original
    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["restore_verified"] is True
    assert manifest["table_row_counts"] == {k: int(v) for k, v in LIVE.items()}
    assert set(manifest["files_sha256"]) == {"postgres.sql", "registry_audit.jsonl"}
    assert "psql" in (archive_dir / "README.txt").read_text(encoding="utf-8")


def test_the_live_audit_trail_restarts_empty_only_after_it_is_archived(tmp_path, audit):
    archive_dir = do_reset(FakeDocker(), tmp_path, audit)
    assert audit.read_bytes() == b""
    assert (archive_dir / "registry_audit.jsonl").stat().st_size > 0


def test_only_the_two_state_volumes_are_removed_never_the_ollama_models(tmp_path, audit):
    docker = FakeDocker()
    do_reset(docker, tmp_path, audit)
    removed = [c[3] for c in docker.calls if c[:3] == ["docker", "volume", "rm"]]
    assert removed == ["demoproj_pgdata", "demoproj_mlflow-data"]
    assert not any("-v" in c for c in docker.calls if "down" in c)
    assert not any("ollama" in " ".join(c) for c in docker.calls)


@pytest.mark.parametrize(
    ("docker", "message"),
    [
        (FakeDocker(dump=b"-- truncated"), "incomplete dump"),
        (
            FakeDocker(restored={"approvals": "8", "decision_cards": "0", "predictions": "201"}),
            "does not restore",
        ),
        (FakeDocker(fail=("pg_dump",)), "pg_dump failed"),
        (FakeDocker(cp_empty=True), "MLflow store copy is empty"),
        (FakeDocker(fail=("--wait postgres",)), "failed"),
    ],
)
def test_a_failed_archive_stops_before_anything_is_destroyed(tmp_path, audit, docker, message):
    before = audit.read_bytes()
    with pytest.raises(ResetError, match=message):
        do_reset(docker, tmp_path, audit)
    assert not docker.ran("down")
    assert not docker.ran("volume", "rm")
    assert audit.read_bytes() == before


def test_the_scratch_restore_database_is_always_dropped(tmp_path, audit):
    docker = FakeDocker(restored={"approvals": "0"})
    with pytest.raises(ResetError):
        do_reset(docker, tmp_path, audit)
    drops = [c for c in docker.calls if any("DROP DATABASE" in part for part in c)]
    assert len(drops) == 2  # before creating it, and in the finally


def test_a_trail_that_changes_after_archiving_is_left_in_place(tmp_path, audit):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "registry_audit.jsonl").write_text("older copy\n", encoding="utf-8")
    with pytest.raises(ResetError, match="changed after it was archived"):
        reset_demo.rotate_registry_audit(archive_dir, audit)
    assert audit.read_text(encoding="utf-8").startswith('{"alias"')


def test_a_missing_volume_is_not_an_error_but_another_failure_is(tmp_path):
    compose = Compose(
        lambda args, input=None: done(code=1, stderr=b"Error: No such volume"),
        db_user="u",
        db_name="d",
    )
    compose.runner = lambda args, input=None: (
        done() if "down" in args else done(code=1, stderr=b"Error: No such volume: x")
    )
    assert reset_demo.wipe(compose, "p") == []

    compose.runner = lambda args, input=None: (
        done() if "down" in args else done(code=1, stderr=b"volume is in use")
    )
    with pytest.raises(ResetError, match="in use"):
        reset_demo.wipe(compose, "p")


def test_a_failed_seed_is_reported_with_the_archive(tmp_path, audit):
    with pytest.raises(ResetError, match="seeding exited 2"):
        do_reset(FakeDocker(), tmp_path, audit, seeder=lambda: 2)


def test_without_yes_nothing_runs(capsys):
    def refuse(args, input=None):
        raise AssertionError(f"ran {args}")

    assert reset_demo.main([], runner=refuse) == 0
    out = capsys.readouterr().out
    assert "Nothing has been changed" in out and "--yes" in out


def test_with_yes_a_failure_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("scripts.seed_demo.load_env_file", lambda: [])
    docker = FakeDocker(fail=("config",))
    assert reset_demo.main(["--yes"], runner=docker) == 2
    assert "reset stopped" in capsys.readouterr().err
    assert not docker.ran("down")
