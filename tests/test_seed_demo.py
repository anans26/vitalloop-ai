"""Week 11: `scripts/seed_demo.py` -- the fresh-machine demo seed.

Everything runs against a throwaway SQLite MLflow store, a SQLite audit
database and a fake API: no Compose service, no network, no real dataset.
What is asserted is the seed's contract: it reaches the demo's starting state,
it is idempotent, and it never moves `champion` or `shadow` past their first
registration.
"""

import pytest
from mlflow.tracking import MlflowClient

from dashboard.api_client import ApiError
from ml.data.features import split_features_target
from ml.evaluate import build_report
from ml.registry import read_audit_rows
from ml.train import build_base_model, build_calibrated_model, build_metadata
from scripts import seed_demo
from scripts.seed_demo import (
    CACHED_CHALLENGER_TAGS,
    DatabaseCounts,
    RegistryOutcome,
    SeedError,
    SeedReport,
    load_env_file,
    seed,
    seed_registry,
)

MODEL = "test-model"
PIP = ["scikit-learn", "lightgbm", "cloudpickle"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    base = build_base_model(n_estimators=5).fit(X, y)
    calibrated = build_calibrated_model(build_base_model(n_estimators=5)).fit(X, y)
    report = build_report(
        synthetic_clean_df,
        y,
        base.predict_proba(X)[:, 1],
        calibrated.predict_proba(X)[:, 1],
        evaluation_slice="synthetic",
        subgroup_columns=["gender"],
    )
    return {
        "base_model": base,
        "calibrated_model": calibrated,
        "metadata": build_metadata(synthetic_clean_df, synthetic_clean_df, n_estimators=5),
        "report": report,
        "local_explanation": {"row_position": 0, "top_contributions": []},
    }


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    MlflowClient(uri).create_experiment("test-experiment", artifact_location=artifacts.as_uri())
    audit = tmp_path / "registry_audit.jsonl"
    monkeypatch.setattr("ml.registry.REGISTRY_AUDIT_PATH", audit)
    return uri, audit


def seeder(bundle, tracking, calls=None):
    uri, _ = tracking

    def load():
        if calls is not None:
            calls.append("load")
        return bundle

    return seed_registry(
        uri,
        model_name=MODEL,
        bundle_loader=load,
        log_kwargs={"experiment_name": "test-experiment", "artifacts": (), "pip_requirements": PIP},
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_a_fresh_registry_gets_a_champion_and_a_distinct_cached_challenger(bundle, tracking):
    outcome = seeder(bundle, tracking)

    assert (outcome.champion, outcome.challenger, outcome.shadow) == ("1", "2", None)
    assert outcome.champion_registered and outcome.challenger_registered

    client = MlflowClient(tracking[0])
    run = client.get_run(client.get_model_version(MODEL, "2").run_id)
    for key, value in CACHED_CHALLENGER_TAGS.items():
        assert run.data.tags[key] == value
    # The champion's run is the ordinary Week 4 baseline run, not a seed run.
    champion_run = client.get_run(client.get_model_version(MODEL, "1").run_id)
    assert champion_run.data.tags["pipeline"] == "week3-readmission"


def test_every_alias_move_is_audited_and_names_the_seed(bundle, tracking):
    seeder(bundle, tracking)
    rows = read_audit_rows(tracking[1])
    assert [(r["alias"], r["to_version"]) for r in rows] == [("champion", "1"), ("challenger", "2")]
    assert "not a retrain" in rows[1]["reason"]


def test_seeding_twice_changes_nothing(bundle, tracking):
    seeder(bundle, tracking)
    calls = []
    again = seeder(bundle, tracking, calls)

    assert (again.champion, again.challenger) == ("1", "2")
    assert not again.champion_registered and not again.challenger_registered
    assert calls == []  # not even the model is loaded
    assert len(MlflowClient(tracking[0]).search_model_versions(f"name='{MODEL}'")) == 2
    assert len(read_audit_rows(tracking[1])) == 2


def test_after_a_promotion_the_seed_caches_a_new_challenger_and_leaves_champion(bundle, tracking):
    from ml.registry import set_alias

    seeder(bundle, tracking)
    client = MlflowClient(tracking[0])
    # What a demo leaves behind: the replayed challenger v2 was promoted.
    set_alias(client, MODEL, "champion", "2", run_id="r", reason="test promotion")

    outcome = seeder(bundle, tracking)
    assert (outcome.champion, outcome.challenger) == ("2", "3")
    assert outcome.challenger_registered and not outcome.champion_registered


def test_the_seed_never_sets_shadow(bundle, tracking):
    outcome = seeder(bundle, tracking)
    assert outcome.shadow is None
    assert all(row["alias"] != "shadow" for row in read_audit_rows(tracking[1]))


def test_an_unreachable_server_is_an_error_not_a_local_fallback():
    with pytest.raises(SeedError, match="MLflow at"):
        seed_demo.seed_registry_or_explain(
            lambda uri: (_ for _ in ()).throw(RuntimeError("not reachable")), "http://x:1"
        )


# ---------------------------------------------------------------------------
# The artifacts it registers
# ---------------------------------------------------------------------------
def test_missing_dvc_outputs_are_refused(monkeypatch, tmp_path):
    from ml import config

    monkeypatch.setattr(config, "MODEL_PATH", config.PROJECT_ROOT / "models" / "absent.joblib")
    with pytest.raises(SeedError, match="dvc repro"):
        seed_demo.check_artifacts()


def test_a_model_that_is_not_the_pinned_bytes_is_refused(monkeypatch, tmp_path):
    from ml import config

    fake = {}
    for name in seed_demo.REQUIRED_ARTIFACTS:
        path = config.PROJECT_ROOT / "reports" / f"_seed_test_{name}.tmp"
        fake[name] = path
    try:
        for name, path in fake.items():
            path.write_bytes(b"not the pinned model")
            monkeypatch.setattr(config, name, path)
        monkeypatch.setattr("ml.tracking.dvc_lineage", lambda: {"dvc_model_md5": "0" * 32})
        with pytest.raises(SeedError, match="not the model dvc.lock pins"):
            seed_demo.check_artifacts()
    finally:
        for path in fake.values():
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The whole seed, with fakes
# ---------------------------------------------------------------------------
class FakeApi:
    def __init__(self, serving=None):
        self.serving = serving
        self.champion = None
        self.reloads = 0

    def ready(self):
        return {"status": "ready", "model_version": self.serving}

    def reload(self):
        self.reloads += 1
        self.serving = self.champion
        return {"champion_version": self.serving}


@pytest.fixture
def seed_db(tmp_path):
    from db.session import configure_engine, get_session, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{(tmp_path / 'seed.db').as_posix()}")
    init_db()
    yield get_session
    reset_engine()


def run_seed(seed_db, api, *, predictions_before=0, traffic=20, check_only=False, registry=None):
    sent = []
    registry = registry or RegistryOutcome(champion="1", challenger="2", shadow=None)

    def registry_seeder(uri):
        api.champion = registry.champion
        return registry

    def sender(clinician_api, count):
        from tests.dashboard.conftest import seed_prediction

        session = seed_db()
        for i in range(count):
            seed_prediction(session, f"seeded-{i}", version=api.serving)
        session.close()
        sent.append(count)
        return count

    if predictions_before:
        from tests.dashboard.conftest import seed_prediction

        session = seed_db()
        for i in range(predictions_before):
            seed_prediction(session, f"earlier-{i}")
        session.close()

    report = seed(
        tracking_uri="sqlite:///unused",
        api_url="http://api",
        traffic=traffic,
        check_only=check_only,
        session_factory=seed_db,
        api_factory=lambda role: api,
        registry_seeder=registry_seeder,
        alias_reader=lambda uri: {"champion": "1", "challenger": "2", "shadow": None},
        traffic_sender=sender,
        artifact_check=lambda: None,
    )
    return report, sent


def test_a_fresh_stack_is_brought_to_the_demo_state(seed_db):
    api = FakeApi(serving=None)  # the API started before any model existed
    report, sent = run_seed(seed_db, api)

    assert api.reloads == 1
    assert report.serving == "1"
    assert sent == [20]
    assert report.counts.predictions == 20
    assert report.problems() == []


def test_existing_traffic_is_not_duplicated(seed_db):
    report, sent = run_seed(seed_db, FakeApi(serving="1"), predictions_before=3)
    assert sent == []
    assert report.counts.predictions == 3
    assert any("no baseline traffic" in note for note in report.notes)


def test_an_api_already_serving_the_champion_is_not_reloaded(seed_db):
    api = FakeApi(serving="1")
    run_seed(seed_db, api, traffic=0)
    assert api.reloads == 0


def test_check_mode_registers_nothing_and_sends_nothing(seed_db):
    def refuse(uri):
        raise AssertionError("check mode must not register")

    api = FakeApi(serving="1")
    report = seed(
        tracking_uri="sqlite:///unused",
        api_url="http://api",
        check_only=True,
        session_factory=seed_db,
        api_factory=lambda role: api,
        registry_seeder=refuse,
        alias_reader=lambda uri: {"champion": "1", "challenger": "2", "shadow": None},
        traffic_sender=lambda *a: pytest.fail("check mode must not send traffic"),
        artifact_check=lambda: pytest.fail("check mode needs no artifacts"),
    )
    assert report.problems() == []
    assert api.reloads == 0


def test_an_api_that_will_not_serve_the_champion_fails_the_seed(seed_db):
    class Stuck(FakeApi):
        def reload(self):
            self.reloads += 1
            return {}

    with pytest.raises(SeedError, match="not champion v1"):
        run_seed(seed_db, Stuck(serving=None))


def test_an_api_that_never_answers_fails_with_a_hint(monkeypatch):
    class Down:
        def ready(self):
            raise ApiError(0, "API unreachable")

    monkeypatch.setattr("scripts.seed_demo.time.sleep", lambda s: None)
    with pytest.raises(SeedError, match="docker compose up"):
        seed_demo.wait_for_api(Down(), timeout_seconds=0)


@pytest.mark.parametrize(
    ("registry", "serving", "problem"),
    [
        (RegistryOutcome(None, None, None), None, "no champion"),
        (RegistryOutcome("1", None, None), "1", "no cached challenger"),
        (RegistryOutcome("2", "2", None), "2", "is the champion"),
        (RegistryOutcome("1", "2", "2"), "1", "shadow already points"),
        (RegistryOutcome("2", "3", None), "1", "not champion v2"),
    ],
)
def test_problems_name_what_blocks_the_demo(registry, serving, problem):
    report = SeedReport(registry=registry, serving=serving, counts=DatabaseCounts(0, 0, 0, 0, 0, 0))
    assert any(problem in p for p in report.problems())


# ---------------------------------------------------------------------------
# Configuration and CLI
# ---------------------------------------------------------------------------
def test_env_file_fills_only_what_is_unset(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nPOSTGRES_PASSWORD=from-file\nPOSTGRES_USER='quoted'\n\nNOT A LINE\n",
        encoding="utf-8",
    )
    environ = {"POSTGRES_PASSWORD": "already-set"}
    applied = load_env_file(env, environ)

    assert environ == {"POSTGRES_PASSWORD": "already-set", "POSTGRES_USER": "quoted"}
    assert applied == ["POSTGRES_USER"]  # names only, never values


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "absent", {}) == []


def test_cli_exit_codes(monkeypatch, capsys):
    ready = SeedReport(
        registry=RegistryOutcome("1", "2", None),
        serving="1",
        counts=DatabaseCounts(20, 0, 0, 0, 0, 0),
        traffic_sent=20,
    )
    monkeypatch.setattr(seed_demo, "load_env_file", lambda: [])
    monkeypatch.setattr(seed_demo, "seed", lambda **kwargs: ready)
    assert seed_demo.main([]) == 0
    out = capsys.readouterr().out
    assert "demo-ready" in out and "cached for replay" in out

    not_ready = SeedReport(
        registry=RegistryOutcome("2", "2", None), serving="2", counts=ready.counts
    )
    monkeypatch.setattr(seed_demo, "seed", lambda **kwargs: not_ready)
    assert seed_demo.main(["--check"]) == 1
    assert "NOT demo-ready" in capsys.readouterr().out

    def failing(**kwargs):
        raise SeedError("no stack")

    monkeypatch.setattr(seed_demo, "seed", failing)
    assert seed_demo.main([]) == 2
    assert "seed failed: no stack" in capsys.readouterr().err


def test_the_seed_prints_no_secret(monkeypatch, capsys):
    monkeypatch.setenv("VITALLOOP_JWT_SECRET", "s3cr3t-value-for-test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-s3cr3t-for-test")
    report = SeedReport(
        registry=RegistryOutcome("1", "2", None),
        serving="1",
        counts=DatabaseCounts(1, 0, 0, 0, 0, 0),
    )
    monkeypatch.setattr(seed_demo, "seed", lambda **kwargs: report)
    seed_demo.main([])
    out = capsys.readouterr()
    assert "s3cr3t" not in out.out + out.err
