"""`scripts/seed_demo.py` -- a fresh stack, brought to the demo's starting state.

IMPLEMENTATION_ROADMAP.md Week 11: "seed script for a fresh-machine demo";
deliverable "`docker compose up` + seed = working demo on a fresh clone".
RISK_ANALYSIS.md §4 names the same script twice: it "rebuilds demo state on a
fresh machine", and with a Compose reset it is the one-command answer to
"state corruption between demo runs" (`scripts/reset_demo.py`).

    python -m scripts.seed_demo              # seed the Compose stack from the host
    python -m scripts.seed_demo --check      # report the state; change nothing

Run from the host after `docker compose up -d` and `dvc repro`. Credentials
come from `docker/.env`, the file Compose itself reads; nothing new is
configured and nothing is written to disk except through the services.

**What "the demo's starting state" is** (WORKFLOW.md §5):

1. **A registered champion** -- the DVC-pinned Week 3 model, logged and
   registered through the Week 4 path (`ml.tracking.log_run`), which sets
   `champion` only when nothing holds it (`ensure_initial_champion`). This is
   step 1's "healthy champion".
2. **A cached challenger for replay** -- step 4 says "Retrain replays a cached
   run", and ARCHITECTURE.md §3.11 defines replay as re-registering "a
   pre-trained challenger". The seed registers the same pinned model artifact
   again as its own run, labelled `role=cached-challenger` and
   `pipeline=week11-demo-seed`, and points `challenger` at it with an audited
   alias move. It is not presented as a retrain: on unchanged data a live
   retrain produces this identical model (RUNNING_THE_PROJECT.md §16.9), and
   the run that later uses it is recorded with `mode='replay'`. It never
   touches `champion` or `shadow` -- those move only through the gate and a
   human approval.
3. **The API serving that champion** -- on a fresh stack the API starts before
   any model is registered, so the seed asks it to re-resolve its aliases
   (`POST /ops/models/reload`, the Week 9 endpoint) under an `ops` token for
   the subject `demo-seed`.
4. **Predictions flowing** -- step 1 again. When the audit table is empty the
   seed replays the first `--traffic` serving-stream encounters through
   `/predict` as the clinician `demo-seed`, so each is a real audit row.

**Idempotent.** Every step checks before it acts: a second run registers
nothing, moves no alias and sends no traffic. It never deletes or edits a row;
a clean slate is `scripts/reset_demo.py`'s job, which archives first.
"""

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ml.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / "docker" / ".env"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TRAFFIC = 20
SEED_SUBJECT = "demo-seed"

CACHED_CHALLENGER_TAGS = {
    "week": "11",
    "pipeline": "week11-demo-seed",
    "role": "cached-challenger",
    "seed_source": "dvc-pinned-model-artifact",
}
CACHED_CHALLENGER_REASON = (
    "week-11 demo seed: cache the DVC-pinned model as the challenger replay mode re-registers "
    "(ARCHITECTURE.md §3.11); not a retrain"
)


class SeedError(RuntimeError):
    """The stack cannot be seeded as it stands. The message says what to do."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_env_file(path: Path = ENV_FILE, environ=None) -> list[str]:
    """Sets each `KEY=VALUE` from `docker/.env` that the environment lacks.

    The file Compose reads is the single source of these credentials; the seed
    reads it too rather than asking for them twice. A variable already set in
    the environment wins. Returns the names that were set, never the values.
    """
    environ = os.environ if environ is None else environ
    if not path.exists():
        return []
    applied = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value
            applied.append(key)
    return applied


# ---------------------------------------------------------------------------
# The model registry
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = ("MODEL_PATH", "METADATA_PATH", "METRICS_PATH", "SHAP_EXAMPLE_PATH")


def check_artifacts() -> None:
    """The DVC outputs the seed registers must exist, and be the pinned bytes."""
    import hashlib

    from ml import config
    from ml.tracking import dvc_lineage

    missing = [
        str(getattr(config, name).relative_to(PROJECT_ROOT))
        for name in REQUIRED_ARTIFACTS
        if not getattr(config, name).exists()
    ]
    if missing:
        raise SeedError(
            f"missing DVC outputs {missing}; build them with `dvc repro` (after "
            "`python -m ml.data.ingest` on a fresh clone) before seeding"
        )
    pinned = dvc_lineage().get("dvc_model_md5")
    actual = hashlib.md5(config.MODEL_PATH.read_bytes()).hexdigest()
    if pinned and actual != pinned:
        raise SeedError(
            f"models/readmission_model.joblib (md5 {actual}) is not the model dvc.lock pins "
            f"({pinned}); run `dvc repro` so the seeded champion is the pinned artifact"
        )


def load_bundle() -> dict:
    """The DVC-tracked model and the reports `ml.tracking.main` logs with it."""
    import json

    from ml import config, train

    base_model, calibrated_model = train.load_models()
    return {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "metadata": json.loads(config.METADATA_PATH.read_text(encoding="utf-8")),
        "report": json.loads(config.METRICS_PATH.read_text(encoding="utf-8")),
        "local_explanation": json.loads(config.SHAP_EXAMPLE_PATH.read_text(encoding="utf-8")),
    }


@dataclass
class RegistryOutcome:
    champion: str | None
    challenger: str | None
    shadow: str | None
    champion_registered: bool = False
    challenger_registered: bool = False


def seed_registry(
    tracking_uri: str,
    *,
    model_name: str | None = None,
    bundle_loader: Callable[[], dict] = load_bundle,
    log_kwargs: dict | None = None,
    audit_path=None,
) -> RegistryOutcome:
    """Ensures a champion and a cached challenger that is not the champion."""
    from mlflow.tracking import MlflowClient

    from ml.registry import current_alias_version, set_alias
    from ml.tracking import git_lineage, log_run, verify_tracking_reachable
    from ml.tracking_config import REGISTERED_MODEL_NAME

    model_name = model_name or REGISTERED_MODEL_NAME
    verify_tracking_reachable(tracking_uri)
    client = MlflowClient(tracking_uri)
    extra = dict(log_kwargs or {})

    def alias(name):
        return current_alias_version(client, model_name, name)

    bundle = None
    champion_registered = challenger_registered = False

    if alias("champion") is None:
        bundle = bundle_loader()
        log_run(**bundle, tracking_uri=tracking_uri, model_name=model_name, **extra)
        champion_registered = True
    champion = alias("champion")

    challenger = alias("challenger")
    if challenger is None or challenger == champion:
        bundle = bundle or bundle_loader()
        summary = log_run(
            **bundle,
            tracking_uri=tracking_uri,
            model_name=model_name,
            tags=CACHED_CHALLENGER_TAGS,
            **extra,
        )
        set_alias(
            client,
            model_name=model_name,
            alias="challenger",
            version=summary["registered_version"],
            run_id=summary["run_id"],
            reason=CACHED_CHALLENGER_REASON,
            git_commit=git_lineage().get("git_commit"),
            audit_path=audit_path,
        )
        challenger_registered = True
        challenger = alias("challenger")

    return RegistryOutcome(
        champion=champion,
        challenger=challenger,
        shadow=alias("shadow"),
        champion_registered=champion_registered,
        challenger_registered=challenger_registered,
    )


def read_aliases(tracking_uri: str, model_name: str | None = None) -> dict:
    from mlflow.tracking import MlflowClient

    from ml.registry import current_alias_version
    from ml.tracking_config import REGISTERED_MODEL_NAME

    client = MlflowClient(tracking_uri)
    name = model_name or REGISTERED_MODEL_NAME
    return {a: current_alias_version(client, name, a) for a in ("champion", "challenger", "shadow")}


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def wait_for_api(api, timeout_seconds: float = 120.0, poll_seconds: float = 2.0) -> dict:
    """The API's `/ready` body once it answers at all (it answers 503 unready)."""
    from dashboard.api_client import ApiError

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return api.ready()
        except ApiError as error:
            if time.monotonic() >= deadline:
                raise SeedError(
                    f"the API did not answer within {timeout_seconds:.0f}s ({error.detail}); "
                    "is `docker compose up -d` running?"
                ) from error
        time.sleep(poll_seconds)


def serve_champion(ops_api, champion: str | None) -> dict:
    """Makes the API serve the registered champion, reloading it if it does not."""
    ready = wait_for_api(ops_api)
    if champion is not None and str(ready.get("model_version")) != str(champion):
        ops_api.reload()
        ready = ops_api.ready()
    if str(ready.get("model_version")) != str(champion):
        raise SeedError(
            f"the API serves {ready.get('model_version')!r}, not champion v{champion}: {ready}"
        )
    return ready


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------
@dataclass
class DatabaseCounts:
    predictions: int
    drift_events: int
    decision_cards: int
    retrain_runs: int
    approvals: int
    shadow_predictions: int


def database_counts(session) -> DatabaseCounts:
    from sqlalchemy import func, select

    from db.models import (
        Approval,
        DecisionCard,
        DriftEvent,
        Prediction,
        RetrainRun,
        ShadowPrediction,
    )

    def count(model):
        return session.scalar(select(func.count()).select_from(model))

    return DatabaseCounts(
        predictions=count(Prediction),
        drift_events=count(DriftEvent),
        decision_cards=count(DecisionCard),
        retrain_runs=count(RetrainRun),
        approvals=count(Approval),
        shadow_predictions=count(ShadowPrediction),
    )


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------
@dataclass
class SeedReport:
    registry: RegistryOutcome
    serving: str | None
    counts: DatabaseCounts
    traffic_sent: int = 0
    notes: list[str] = field(default_factory=list)

    def problems(self) -> list[str]:
        """Why this state cannot run the WORKFLOW.md §5 demo; empty when it can."""
        registry, found = self.registry, []
        if registry.champion is None:
            found.append("no champion is registered")
        if registry.challenger is None:
            found.append("no cached challenger for replay")
        elif registry.challenger == registry.champion:
            found.append(
                f"the cached challenger v{registry.challenger} is the champion, so replay "
                "could never be promoted"
            )
        if registry.shadow is not None:
            found.append(f"shadow already points at v{registry.shadow} (a gated run is pending)")
        if registry.champion is not None and self.serving != registry.champion:
            found.append(f"the API serves {self.serving!r}, not champion v{registry.champion}")
        return found

    def lines(self) -> list[str]:
        r, c = self.registry, self.counts
        out = [
            f"champion     : {_v(r.champion)}"
            + (" (registered now)" if r.champion_registered else ""),
            f"challenger   : {_v(r.challenger)} -- cached for replay"
            + (" (registered now)" if r.challenger_registered else ""),
            f"shadow       : {_v(r.shadow)}",
            f"API serving  : v{self.serving}" if self.serving else "API serving  : -",
            f"predictions  : {c.predictions}"
            + (f" ({self.traffic_sent} seeded now)" if self.traffic_sent else ""),
            f"drift windows: {c.drift_events}   cards: {c.decision_cards}   "
            f"retrain runs: {c.retrain_runs}   approvals: {c.approvals}",
        ]
        return out + [f"note         : {note}" for note in self.notes]


def seed(
    *,
    tracking_uri: str,
    api_url: str,
    traffic: int = DEFAULT_TRAFFIC,
    check_only: bool = False,
    session_factory=None,
    api_factory=None,
    registry_seeder=seed_registry,
    alias_reader=read_aliases,
    traffic_sender=None,
    artifact_check=check_artifacts,
) -> SeedReport:
    """Brings the stack to the demo's starting state, or (`check_only`) reports it."""
    from dashboard.api_client import OpsApi

    if session_factory is None:
        from loop.monitor.database import open_session as session_factory
    if api_factory is None:

        def api_factory(role: str):
            from api.auth import create_access_token
            from api.config import get_settings

            token = create_access_token(SEED_SUBJECT, role=role, settings=get_settings())
            return OpsApi(api_url, token)

    notes: list[str] = []
    if check_only:
        aliases = alias_reader(tracking_uri)
        registry = RegistryOutcome(**aliases)
    else:
        artifact_check()
        registry = seed_registry_or_explain(registry_seeder, tracking_uri)

    session = session_factory()
    try:
        counts = database_counts(session)
    finally:
        session.close()

    ops_api = api_factory("ops")
    if check_only:
        serving = wait_for_api(ops_api, timeout_seconds=5).get("model_version")
    else:
        serving = serve_champion(ops_api, registry.champion).get("model_version")

    sent = 0
    if not check_only and traffic > 0:
        if counts.predictions == 0:
            sender = traffic_sender or _send_traffic
            sent = sender(api_factory("clinician"), traffic)
            session = session_factory()
            try:
                counts = database_counts(session)
            finally:
                session.close()
        else:
            notes.append(
                f"{counts.predictions} predictions already on record; no baseline traffic sent"
            )

    return SeedReport(
        registry=registry,
        serving=None if serving is None else str(serving),
        counts=counts,
        traffic_sent=sent,
        notes=notes,
    )


def _v(version: str | None) -> str:
    return f"v{version}" if version else "unset"


def seed_registry_or_explain(registry_seeder, tracking_uri: str) -> RegistryOutcome:
    try:
        return registry_seeder(tracking_uri)
    except RuntimeError as error:
        if isinstance(error, SeedError):
            raise
        raise SeedError(f"MLflow at {tracking_uri}: {error}") from error


def _send_traffic(clinician_api, count: int) -> int:
    from dashboard.actions import send_demo_traffic

    result = send_demo_traffic(clinician_api, count=count, offset=0)
    if result.failed:
        raise SeedError(
            f"baseline traffic: {result.scored} scored, {result.failed} failed: "
            + "; ".join(result.errors)
        )
    return result.scored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed a fresh VitalLoop stack for the demo.")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=f"MLflow server (default: $MLFLOW_TRACKING_URI, else {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="the serving API")
    parser.add_argument(
        "--traffic",
        type=int,
        default=DEFAULT_TRAFFIC,
        help="baseline /predict calls when the audit table is empty (0 to skip)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the stack is demo-ready; change nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ml.tracking import make_console_encoding_safe

    make_console_encoding_safe()
    load_env_file()
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    tracking_uri = (
        args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )

    try:
        report = seed(
            tracking_uri=tracking_uri,
            api_url=args.api_url,
            traffic=args.traffic,
            check_only=args.check,
        )
    except SeedError as error:
        print(f"seed failed: {error}", file=sys.stderr)
        return 2

    print("\n".join(report.lines()))
    problems = report.problems()
    if problems:
        print("NOT demo-ready: " + "; ".join(problems))
        return 1
    print("demo-ready: open http://localhost:8501 and follow the 3-minute demo path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
