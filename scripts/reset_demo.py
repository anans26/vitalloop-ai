"""`scripts/reset_demo.py` -- one command back to a clean, seeded demo state.

RISK_ANALYSIS.md §4, "State corruption between demo runs": *"One-command reset
(Compose down -v + seed); demo scenarios are idempotent seeded scripts."* This
is that command, with one addition the audit design requires: **nothing is
destroyed before it is archived and the archive is proven to restore.**

    python -m scripts.reset_demo            # print the plan; change nothing
    python -m scripts.reset_demo --yes      # archive, wipe, start, seed

What `--yes` does, in order -- and it stops at the first step that fails:

1. Starts `postgres` and `mlflow` if they are not running (they must be, to be
   archived).
2. **Archives** into `backups/demo-reset-<UTC timestamp>/` (gitignored):
   `postgres.sql` (a `pg_dump` of the audit database), `mlflow-server/` (the
   MLflow server's whole store -- tracking DB, registry, artifacts) and
   `registry_audit.jsonl` (the append-only alias-move trail).
3. **Verifies** the dump by restoring it into a scratch database inside the
   same Postgres container and comparing every table's row count with the
   live database. A dump that does not restore aborts the reset.
4. Writes `manifest.json`: row counts, aliases, file hashes, git commit.
5. Only then: `docker compose down`, and removes exactly the two state
   volumes (`<project>_pgdata`, `<project>_mlflow-data`). Unlike a bare
   `down -v` it leaves the optional `ollama-models` volume alone -- a
   multi-gigabyte model cache is not demo state.
6. Rotates `mlflow/registry_audit.jsonl`: its archived copy is hash-verified,
   then the live file starts empty, because its rows describe model versions
   that no longer exist in the new registry. The old trail stays whole in the
   archive, next to the database it describes.
7. `docker compose up -d --build --wait` (so the demo runs the checked-out
   code), then `scripts.seed_demo`.

Historical rows are therefore never silently deleted: they are moved, intact
and verified, to an archive a person can restore (`README.txt` in the
archive says how).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.config import PROJECT_ROOT

DOCKER_DIR = PROJECT_ROOT / "docker"
BACKUPS_DIR = PROJECT_ROOT / "backups"
REGISTRY_AUDIT = PROJECT_ROOT / "mlflow" / "registry_audit.jsonl"
STATE_VOLUMES = ("pgdata", "mlflow-data")
RESTORE_CHECK_DB = "vitalloop_restore_check"

RESTORE_README = """\
This directory is one archived demo state, taken by `python -m scripts.reset_demo --yes`
immediately before the Compose state volumes were removed. See manifest.json.

Restore it onto a stack whose volumes are fresh (from docker/):

    docker compose up -d postgres mlflow
    docker compose exec -T postgres psql -U <user> -d <db> < <this dir>/postgres.sql
    docker compose cp <this dir>/mlflow-server/. mlflow:/mlflow
    docker compose restart mlflow
    copy <this dir>/registry_audit.jsonl over mlflow/registry_audit.jsonl
    docker compose up -d
"""


class ResetError(RuntimeError):
    """The reset stopped. Nothing after the failing step was done."""


Runner = Callable[..., subprocess.CompletedProcess]


def run(args: Sequence[str], *, input: bytes | None = None, cwd: Path = DOCKER_DIR):
    """Runs one command, captured. The default runner; tests pass their own."""
    return subprocess.run(list(args), input=input, cwd=cwd, capture_output=True)


def _check(result: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ResetError(f"{what} failed (exit {result.returncode}): {detail[-800:]}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Compose:
    """`docker compose` from `docker/`, where `.env` and the project name live."""

    def __init__(self, runner: Runner = run, *, db_user: str, db_name: str):
        self.runner = runner
        self.db_user = db_user
        self.db_name = db_name

    def __call__(self, *args: str, input: bytes | None = None, what: str | None = None):
        return _check(
            self.runner(["docker", "compose", *args], input=input),
            what or "docker compose " + " ".join(args),
        )

    def project_name(self) -> str:
        return json.loads(self("config", "--format", "json").stdout)["name"]

    def psql(self, sql: str, *, database: str | None = None) -> str:
        result = self(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            self.db_user,
            "-d",
            database or self.db_name,
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-c",
            sql,
            what=f"psql ({sql[:40]}...)",
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    def table_counts(self, database: str | None = None) -> dict[str, int]:
        tables = self.psql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name",
            database=database,
        ).split()
        counts = {}
        for table in tables:
            counts[table] = int(self.psql(f'SELECT count(*) FROM "{table}"', database=database))
        return counts


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------
def archive(compose: Compose, archive_dir: Path, *, registry_audit: Path = REGISTRY_AUDIT) -> dict:
    """Steps 1-4. Returns the manifest. Raises before anything is destroyed."""
    compose("up", "-d", "--wait", "postgres", "mlflow")
    archive_dir.mkdir(parents=True, exist_ok=False)

    dump = compose(
        "exec",
        "-T",
        "postgres",
        "pg_dump",
        "-U",
        compose.db_user,
        "-d",
        compose.db_name,
        "--no-owner",
        "--no-privileges",
        what="pg_dump",
    ).stdout
    if b"PostgreSQL database dump complete" not in dump:
        raise ResetError("pg_dump produced an incomplete dump; nothing was removed")
    (archive_dir / "postgres.sql").write_bytes(dump)

    live_counts = compose.table_counts()
    restored_counts = verify_restore(compose, dump)
    if restored_counts != live_counts:
        raise ResetError(
            f"the archived dump does not restore to the live row counts "
            f"(live {live_counts}, restored {restored_counts}); nothing was removed"
        )

    compose("cp", "mlflow:/mlflow", str(archive_dir / "mlflow-server"), what="copy MLflow store")
    if not any((archive_dir / "mlflow-server").rglob("*")):
        raise ResetError("the MLflow store copy is empty; nothing was removed")

    files = {"postgres.sql": _sha256(archive_dir / "postgres.sql")}
    if registry_audit.exists():
        shutil.copy2(registry_audit, archive_dir / "registry_audit.jsonl")
        if _sha256(archive_dir / "registry_audit.jsonl") != _sha256(registry_audit):
            raise ResetError("the registry audit copy does not match; nothing was removed")
        files["registry_audit.jsonl"] = _sha256(registry_audit)

    from ml.tracking import git_lineage

    manifest = {
        "archived_at": archive_dir.name.removeprefix("demo-reset-"),
        "reason": "scripts.reset_demo: archive before removing the Compose state volumes",
        "git_commit": git_lineage().get("git_commit"),
        "database": compose.db_name,
        "table_row_counts": live_counts,
        "restore_verified": True,
        "files_sha256": files,
        "mlflow_server_store": "mlflow-server/ (docker compose cp mlflow:/mlflow)",
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (archive_dir / "README.txt").write_text(RESTORE_README, encoding="utf-8")
    return manifest


def verify_restore(compose: Compose, dump: bytes) -> dict[str, int]:
    """Restores the dump into a scratch database and counts it; always drops it."""
    drop = f"DROP DATABASE IF EXISTS {RESTORE_CHECK_DB}"
    compose.psql(drop, database="postgres")
    compose.psql(f"CREATE DATABASE {RESTORE_CHECK_DB}", database="postgres")
    try:
        compose(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            compose.db_user,
            "-d",
            RESTORE_CHECK_DB,
            "-v",
            "ON_ERROR_STOP=1",
            "-q",
            input=dump,
            what="restore check",
        )
        return compose.table_counts(database=RESTORE_CHECK_DB)
    finally:
        compose.psql(drop, database="postgres")


def wipe(compose: Compose, project: str) -> list[str]:
    """Step 5: containers down, then exactly the two state volumes."""
    compose("down")
    removed = []
    for volume in STATE_VOLUMES:
        name = f"{project}_{volume}"
        result = compose.runner(["docker", "volume", "rm", name])
        if result.returncode == 0:
            removed.append(name)
        elif b"no such volume" not in (result.stderr or b"").lower():
            _check(result, f"docker volume rm {name}")
    return removed


def rotate_registry_audit(archive_dir: Path, registry_audit: Path = REGISTRY_AUDIT) -> bool:
    """Step 6: the live trail restarts empty; its verified copy is in the archive."""
    if not registry_audit.exists():
        return False
    archived = archive_dir / "registry_audit.jsonl"
    if not archived.exists() or _sha256(archived) != _sha256(registry_audit):
        raise ResetError(
            f"{registry_audit} changed after it was archived; it was left in place. "
            "Archive and rotate it by hand before seeding."
        )
    registry_audit.write_bytes(b"")
    return True


def reset(
    compose: Compose,
    *,
    archive_root: Path = BACKUPS_DIR,
    registry_audit: Path = REGISTRY_AUDIT,
    seeder: Callable[[], int] | None = None,
    now: datetime | None = None,
    log=print,
) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = archive_root / f"demo-reset-{stamp}"
    project = compose.project_name()

    log(f"[1/4] archiving compose project {project!r} into {archive_dir}")
    manifest = archive(compose, archive_dir, registry_audit=registry_audit)
    log(f"      rows archived and restore-verified: {manifest['table_row_counts']}")

    log("[2/4] removing containers and the state volumes")
    removed = wipe(compose, project)
    log(f"      removed volumes: {removed or 'none (already fresh)'}")
    if rotate_registry_audit(archive_dir, registry_audit):
        log(f"      {registry_audit.name} rotated (archived copy verified)")

    log("[3/4] starting the stack")
    compose("up", "-d", "--build", "--wait", what="docker compose up -d --build --wait")

    log("[4/4] seeding")
    if seeder is None:
        from scripts.seed_demo import main as seed_main

        def seeder():
            return seed_main([])

    status = seeder()
    if status != 0:
        raise ResetError(f"the stack was reset but seeding exited {status}; archive: {archive_dir}")
    return archive_dir


def plan_text(project_hint: str = "<compose project>") -> str:
    return (
        "reset_demo would, in order:\n"
        f"  1. archive the audit database, the MLflow server store and {REGISTRY_AUDIT.name}\n"
        f"     into {BACKUPS_DIR}/demo-reset-<UTC timestamp>/ and prove the dump restores\n"
        f"  2. docker compose down; docker volume rm {project_hint}_pgdata "
        f"{project_hint}_mlflow-data\n"
        f"  3. start {REGISTRY_AUDIT.name} empty (its verified copy is archived)\n"
        "  4. docker compose up -d --build --wait; python -m scripts.seed_demo\n"
        "Nothing has been changed. Re-run with --yes to do it."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive the demo state, wipe it, and reseed (RISK_ANALYSIS.md §4)."
    )
    parser.add_argument("--yes", action="store_true", help="actually do it (default: plan only)")
    return parser


def main(argv: list[str] | None = None, *, runner: Runner = run) -> int:
    args = build_parser().parse_args(argv)
    if not args.yes:
        print(plan_text())
        return 0

    import os

    from scripts.seed_demo import load_env_file

    load_env_file()
    compose = Compose(
        runner,
        db_user=os.environ.get("POSTGRES_USER", "vitalloop"),
        db_name=os.environ.get("POSTGRES_DB", "vitalloop"),
    )
    try:
        archive_dir = reset(compose)
    except ResetError as error:
        print(f"reset stopped: {error}", file=sys.stderr)
        return 2
    print(f"reset complete; previous state archived in {archive_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
