"""The retraining pipeline (ARCHITECTURE.md §3.11).

Everything here runs against a temporary SQLite MLflow store, so the suite
never needs the Compose MLflow service -- the same choice `tests/test_tracking.py`
made in Week 4, for the same reason: a test that silently depended on a server
would be exactly the hidden external dependency this project removes.

The contracts asserted are §3.11's two sentences. The retrain uses the *same*
training entry point as the champion; the DVC hash the card pinned is enforced
rather than decorative; the output is a `challenger` run, never a champion move;
and replay is labeled as replay.
"""

import json
from pathlib import Path

import pytest
import yaml
from mlflow.tracking import MlflowClient

from ml.data.features import split_features_target
from ml.registry import current_alias_version, read_audit_rows, set_alias
from ml.retrain import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODE_LIVE,
    MODE_REPLAY,
    MODES,
    ChallengerRun,
    RetrainError,
    current_data_version,
    load_champion,
    log_challenger,
    replay_challenger,
    retrain_from_card,
    train_challenger,
    verify_pinned_data_version,
)
from ml.train import build_base_model, build_calibrated_model, build_metadata

MODEL_NAME = "test-readmission"
PINNED = "abc123def456"

# Pinned so MLflow skips dependency inference, which dominates test runtime.
TEST_PIP_REQUIREMENTS = ["scikit-learn", "lightgbm", "cloudpickle"]


@pytest.fixture
def lock_path(tmp_path) -> Path:
    """A `dvc.lock` whose training-data hash is `PINNED`."""
    path = tmp_path / "dvc.lock"
    path.write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "split": {"outs": [{"path": "datasets/processed/train.csv", "md5": PINNED}]}
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture
def small_models(synthetic_clean_df):
    X, y = split_features_target(synthetic_clean_df)
    base = build_base_model(n_estimators=3)
    base.fit(X, y)
    calibrated = build_calibrated_model(build_base_model(n_estimators=3))
    calibrated.fit(X, y)
    return base, calibrated


@pytest.fixture
def report(synthetic_clean_df, small_models):
    from ml.evaluate import build_report

    base, calibrated = small_models
    X, y = split_features_target(synthetic_clean_df)
    return build_report(
        synthetic_clean_df,
        y,
        base.predict_proba(X)[:, 1],
        calibrated.predict_proba(X)[:, 1],
        evaluation_slice="synthetic",
        subgroup_columns=("age", "gender", "race"),
    )


@pytest.fixture
def metadata(synthetic_clean_df):
    return build_metadata(synthetic_clean_df, synthetic_clean_df, 3)


def card(**overrides) -> dict:
    payload = {
        "card_id": "dc-2026-01-01-aaaabbbb",
        "action": "FULL_RETRAIN",
        "disposition": "AUTO_PROCEED_SHADOW",
        "candidate_data_version": PINNED,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# The pinned data version
# ---------------------------------------------------------------------------
def test_the_pinned_hash_is_read_from_dvc_lock(lock_path):
    assert current_data_version(lock_path) == PINNED


def test_a_matching_pin_is_accepted(lock_path):
    assert verify_pinned_data_version(PINNED, lock_path=lock_path) == PINNED


def test_a_mismatched_pin_refuses_the_retrain(lock_path):
    """The whole value of pinning: the retrain cannot run on other bytes."""
    with pytest.raises(RetrainError, match="the working tree holds"):
        verify_pinned_data_version("a-different-hash", lock_path=lock_path)


def test_a_pin_with_no_lock_file_refuses_the_retrain(tmp_path):
    with pytest.raises(RetrainError, match="dvc.lock records none"):
        verify_pinned_data_version(PINNED, lock_path=tmp_path / "absent.lock")


def test_a_card_that_pinned_nothing_is_allowed_through(lock_path):
    """Week 7 records None when `dvc.lock` is absent; refusing there is untestable."""
    assert verify_pinned_data_version(None, lock_path=lock_path) is None


def test_a_mismatched_pin_stops_retrain_from_card_before_any_training(lock_path):
    with pytest.raises(RetrainError, match="a retrain must run on the bytes"):
        retrain_from_card(card(candidate_data_version="wrong"), lock_path=lock_path)


# ---------------------------------------------------------------------------
# ChallengerRun
# ---------------------------------------------------------------------------
def test_an_unknown_mode_is_refused():
    with pytest.raises(RetrainError, match="unknown retrain mode"):
        ChallengerRun(mode="teleport", calibrated_model=None)


def test_a_replay_run_knows_it_is_a_replay(small_models):
    _, calibrated = small_models
    assert ChallengerRun(mode=MODE_REPLAY, calibrated_model=calibrated).is_replay
    assert not ChallengerRun(mode=MODE_LIVE, calibrated_model=calibrated).is_replay


def test_the_record_carries_metadata_and_no_model(small_models):
    _, calibrated = small_models
    record = ChallengerRun(
        mode=MODE_LIVE, calibrated_model=calibrated, mlflow_run="r1", registered_version="2"
    ).to_record()

    assert record == {
        "mode": MODE_LIVE,
        "mlflow_run": "r1",
        "registered_version": "2",
        "data_version": None,
        "replayed_from": None,
        "lineage": {},
    }


def test_retrain_from_card_refuses_an_unknown_mode(lock_path):
    with pytest.raises(RetrainError, match="unknown retrain mode"):
        retrain_from_card(card(), mode="teleport", lock_path=lock_path)


# ---------------------------------------------------------------------------
# The same training entry point
# ---------------------------------------------------------------------------
def test_training_a_challenger_uses_the_week_three_entry_point(synthetic_clean_df):
    """§3.11: "The *same* training entry point used for the initial model"."""
    base, calibrated, metadata, report = train_challenger(synthetic_clean_df, synthetic_clean_df)

    assert metadata["model_type"] == "lightgbm.LGBMClassifier"
    assert metadata["feature_pipeline"] == "ml.data.features.build_feature_pipeline"
    assert report["calibrated"]["roc_auc"] > 0
    assert hasattr(calibrated, "predict_proba") and hasattr(base, "predict_proba")


def test_the_challenger_report_is_the_champions_report_shape(synthetic_clean_df):
    """A challenger measured by a second reporting path would not be comparable."""
    _, _, _, report = train_challenger(synthetic_clean_df, synthetic_clean_df)
    assert {"raw", "calibrated", "calibration_effect", "subgroups_calibrated"} <= set(report)


def test_training_is_deterministic(synthetic_clean_df):
    """The gate compares two models; a non-reproducible challenger is ungateable."""
    X, _ = split_features_target(synthetic_clean_df)
    _, first, _, _ = train_challenger(synthetic_clean_df, synthetic_clean_df)
    _, second, _, _ = train_challenger(synthetic_clean_df, synthetic_clean_df)
    assert (first.predict_proba(X) == second.predict_proba(X)).all()


# ---------------------------------------------------------------------------
# The logged challenger run
# ---------------------------------------------------------------------------
@pytest.fixture
def logged(small_models, metadata, report, tracking_uri, tmp_path, monkeypatch):
    monkeypatch.setattr("ml.tracking_config.REGISTRY_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr("ml.registry.REGISTRY_AUDIT_PATH", tmp_path / "audit.jsonl")
    base, calibrated = small_models
    return log_challenger(
        base,
        calibrated,
        metadata,
        report,
        card_id="dc-2026-01-01-aaaabbbb",
        data_version=PINNED,
        tracking_uri=tracking_uri,
        model_name=MODEL_NAME,
        pip_requirements=TEST_PIP_REQUIREMENTS,
    )


def test_a_logged_challenger_is_a_live_run_with_a_version(logged):
    assert logged.mode == MODE_LIVE
    assert logged.mlflow_run and logged.registered_version
    assert logged.data_version == PINNED


def test_the_run_carries_the_lineage_the_document_asks_for(logged, tracking_uri):
    """§3.11: "full lineage (data hash, git commit, params, metrics)"."""
    run = MlflowClient(tracking_uri).get_run(logged.mlflow_run)

    assert run.data.params["model_type"] == "lightgbm.LGBMClassifier"
    assert run.data.metrics
    assert "git_commit" in run.data.params or logged.lineage.get("git_commit") is None


def test_the_run_names_the_card_that_authorised_it(logged, tracking_uri):
    tags = MlflowClient(tracking_uri).get_run(logged.mlflow_run).data.tags
    assert tags["decision_card_id"] == "dc-2026-01-01-aaaabbbb"
    assert tags["role"] == CHALLENGER_ALIAS
    assert tags["retrain_mode"] == MODE_LIVE
    assert tags["pinned_data_version"] == PINNED


def test_the_challenger_alias_points_at_the_new_version(logged, tracking_uri):
    client = MlflowClient(tracking_uri)
    assert current_alias_version(client, MODEL_NAME, CHALLENGER_ALIAS) == logged.registered_version


def test_the_champion_alias_is_not_created_by_a_retrain(logged, tracking_uri):
    """§3.13 reserves `champion` for a logged human approval in Week 9."""
    client = MlflowClient(tracking_uri)
    assert current_alias_version(client, MODEL_NAME, CHAMPION_ALIAS) is None


def test_the_alias_move_is_audited(logged, tmp_path):
    """§3.5: "every move writes an audit row"."""
    rows = read_audit_rows(tmp_path / "audit.jsonl")
    assert [row["alias"] for row in rows] == [CHALLENGER_ALIAS]
    assert "dc-2026-01-01-aaaabbbb" in rows[0]["reason"]


def test_logging_without_registering_moves_no_alias(small_models, metadata, report, tracking_uri):
    base, calibrated = small_models
    run = log_challenger(
        base,
        calibrated,
        metadata,
        report,
        card_id="dc-x",
        data_version=None,
        tracking_uri=tracking_uri,
        model_name=MODEL_NAME,
        pip_requirements=TEST_PIP_REQUIREMENTS,
        register=False,
    )
    assert run.registered_version is None and run.alias_audit is None


def test_no_patient_row_is_logged_to_the_tracking_server(logged, tracking_uri):
    """§6: the processed CSVs are DVC's responsibility, never MLflow's."""
    client = MlflowClient(tracking_uri)
    artifacts = {item.path for item in client.list_artifacts(logged.mlflow_run)}
    assert not any(name.endswith(".csv") for name in artifacts)


# ---------------------------------------------------------------------------
# Replay (§3.11's demo acceleration)
# ---------------------------------------------------------------------------
def test_replay_reuses_the_registered_challenger(logged, tracking_uri, synthetic_clean_df):
    replayed = replay_challenger(
        card_id="dc-2026-01-01-aaaabbbb", tracking_uri=tracking_uri, model_name=MODEL_NAME
    )

    assert replayed.mode == MODE_REPLAY and replayed.is_replay
    assert replayed.registered_version == logged.registered_version
    assert replayed.replayed_from == f"models:/{MODEL_NAME}/{logged.registered_version}"


def test_replay_trains_nothing_and_says_which_version_it_reused(logged, tracking_uri):
    replayed = replay_challenger(card_id="dc-x", tracking_uri=tracking_uri, model_name=MODEL_NAME)
    assert replayed.lineage["replayed_from_version"] == logged.registered_version
    assert replayed.mlflow_run == logged.mlflow_run


def test_replay_scores_like_the_model_it_reused(logged, tracking_uri, synthetic_clean_df):
    from loop.gate.metrics import score_frame

    replayed = replay_challenger(card_id="dc-x", tracking_uri=tracking_uri, model_name=MODEL_NAME)
    assert (
        score_frame(replayed.calibrated_model, synthetic_clean_df)
        == score_frame(logged.calibrated_model, synthetic_clean_df)
    ).all()


def test_replay_with_no_cached_challenger_fails_loudly(tracking_uri):
    with pytest.raises(RetrainError, match="found no challenger to reuse"):
        replay_challenger(card_id="dc-x", tracking_uri=tracking_uri, model_name=MODEL_NAME)


def test_replay_of_an_unknown_version_fails_loudly(logged, tracking_uri):
    with pytest.raises(RetrainError, match="cannot read"):
        replay_challenger(
            card_id="dc-x", version="999", tracking_uri=tracking_uri, model_name=MODEL_NAME
        )


def test_retrain_from_card_in_replay_mode_is_labeled_replay(logged, tracking_uri, lock_path):
    run = retrain_from_card(
        card(),
        mode=MODE_REPLAY,
        tracking_uri=tracking_uri,
        model_name=MODEL_NAME,
        lock_path=lock_path,
    )
    assert run.mode == MODE_REPLAY
    assert run.data_version == PINNED


# ---------------------------------------------------------------------------
# The champion, for comparison
# ---------------------------------------------------------------------------
def test_loading_the_champion_without_one_fails_loudly(tracking_uri):
    with pytest.raises(RetrainError, match="nothing to gate against"):
        load_champion(tracking_uri=tracking_uri, model_name=MODEL_NAME)


def test_the_champion_is_resolved_through_the_alias(
    logged, tracking_uri, tmp_path, synthetic_clean_df
):
    """§3.12 compares against *the champion* -- whatever the alias points at."""
    client = MlflowClient(tracking_uri)
    set_alias(
        client,
        model_name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=logged.registered_version,
        run_id=logged.mlflow_run,
        reason="test champion",
        audit_path=tmp_path / "audit.jsonl",
    )

    model, version = load_champion(tracking_uri=tracking_uri, model_name=MODEL_NAME)
    assert version == logged.registered_version
    assert model.predict_proba(
        synthetic_clean_df.drop(columns=["readmitted_30d", "encounter_id", "patient_nbr"])
    ).shape[0] == len(synthetic_clean_df)


# ---------------------------------------------------------------------------
# Portability
# ---------------------------------------------------------------------------
def test_the_modes_are_the_two_the_document_names():
    assert MODES == (MODE_LIVE, MODE_REPLAY)


def test_no_machine_specific_path_is_recorded_on_a_run(logged):
    """A run's own record must travel: no absolute local paths in it."""
    serialised = json.dumps(logged.to_record())
    assert "C:\\\\" not in serialised and "/home/" not in serialised


# ---------------------------------------------------------------------------
# Failure paths: "never swallowed"
# ---------------------------------------------------------------------------
def test_a_challenger_that_cannot_be_loaded_fails_loudly(logged, tracking_uri, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("artifact store is gone")

    monkeypatch.setattr("mlflow.sklearn.load_model", explode)
    with pytest.raises(RetrainError, match="cannot load"):
        replay_challenger(card_id="dc-x", tracking_uri=tracking_uri, model_name=MODEL_NAME)


def test_a_champion_that_cannot_be_loaded_fails_loudly(logged, tracking_uri, tmp_path, monkeypatch):
    set_alias(
        MlflowClient(tracking_uri),
        model_name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=logged.registered_version,
        run_id=logged.mlflow_run,
        reason="test champion",
        audit_path=tmp_path / "audit.jsonl",
    )

    def explode(*args, **kwargs):
        raise OSError("artifact store is gone")

    monkeypatch.setattr("mlflow.sklearn.load_model", explode)
    with pytest.raises(RetrainError, match="cannot load the champion"):
        load_champion(tracking_uri=tracking_uri, model_name=MODEL_NAME)


def test_the_live_path_verifies_the_pin_then_trains_and_logs(lock_path, monkeypatch):
    """§3.11 in order: pin first, the shared training entry point second."""
    calls = []

    monkeypatch.setattr("ml.train.load_split", lambda path: f"split:{path.name}")
    monkeypatch.setattr(
        "ml.retrain.train_challenger",
        lambda train_df, eval_df: (
            calls.append(("train", train_df, eval_df)) or ("base", "calibrated", {"m": 1}, {"r": 1})
        ),
    )
    monkeypatch.setattr(
        "ml.retrain.log_challenger",
        lambda *args, **kwargs: (
            calls.append(("log", kwargs))
            or ChallengerRun(mode=MODE_LIVE, calibrated_model="calibrated")
        ),
    )

    run = retrain_from_card(card(), lock_path=lock_path)

    assert run.mode == MODE_LIVE
    assert calls[0][0] == "train" and calls[0][1] == "split:train.csv"
    assert calls[1][1]["data_version"] == PINNED
    assert calls[1][1]["card_id"] == "dc-2026-01-01-aaaabbbb"


def test_a_challenger_logged_without_a_base_model_is_still_a_valid_run(
    small_models, metadata, report, tracking_uri
):
    """A replayed challenger carries no base model; logging must not require one."""
    _, calibrated = small_models
    run = log_challenger(
        None,
        calibrated,
        metadata,
        report,
        card_id="dc-x",
        data_version=None,
        tracking_uri=tracking_uri,
        model_name=MODEL_NAME,
        pip_requirements=TEST_PIP_REQUIREMENTS,
        register=False,
    )
    assert run.base_model is None and run.mlflow_run
