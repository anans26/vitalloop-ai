"""The retraining pipeline: a Decision Card in, a `challenger` run out.

ARCHITECTURE.md §3.11 specifies this in two sentences, and both are load-bearing:

    "The *same* training entry point used for the initial model, invoked with
     the DVC hash pinned in the Decision Card. Output: a `challenger` run in
     MLflow with full lineage (data hash, git commit, params, metrics)."

    "Demo acceleration: a cached-run replay mode re-registers a pre-trained
     challenger so the live demo completes in seconds (explicitly labeled as
     replay in the UI)."

So there is no second training implementation here. `train_challenger` calls
`ml.train.train_models` -- the identical function that produced the champion --
and the reporting goes through `ml.evaluate.build_report`, so a challenger is
measured by the code the champion's published numbers came from. A retrain that
quietly used a different pipeline would make the gate's comparison meaningless.

**The pinned hash is enforced, not decorative.** A card fixes
`candidate_data_version` at decision time precisely so the retrain cannot
silently run on other bytes; `verify_pinned_data_version` refuses the run when
the working tree's `dvc.lock` disagrees. That check is the whole value of
pinning, so it fails loudly rather than warning.

**The champion is never touched.** This module sets the `challenger` alias and
nothing else. §3.12 moves `shadow` on a gate PASS and §3.13 reserves `champion`
for a human approval in Week 9; `ml.registry.set_alias` remains the only way
any alias moves, so every move here writes an audit row.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from ml.config import EVAL_PATH, PROJECT_ROOT, SUBGROUP_COLUMNS, TRAIN_PATH
from ml.data.features import split_features_target
from ml.evaluate import build_report
from ml.registry import current_alias_version, register_model, set_alias
from ml.tracking import dvc_lineage, git_lineage, resolve_tracking_uri
from ml.tracking_config import REGISTERED_MODEL_NAME

MODE_LIVE = "live"
MODE_REPLAY = "replay"
MODES = (MODE_LIVE, MODE_REPLAY)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# Which `dvc.lock` entry a card's `candidate_data_version` refers to. The
# Decision Engine reads it from `ml.tracking.dvc_lineage()["dvc_train_md5"]`
# (loop/engine/evaluate.py), so the retrain must compare against the same key.
PINNED_DATA_KEY = "dvc_train_md5"


class RetrainError(RuntimeError):
    """The retrain could not be performed as specified. Never swallowed."""


@dataclass(frozen=True)
class ChallengerRun:
    """One challenger, however it was produced, with the lineage that explains it.

    `mode` is carried on the object rather than inferred later because the
    roadmap makes replay "a first-class, labeled feature": a replayed run must
    never be mistaken for one that actually trained.
    """

    mode: str
    calibrated_model: Any
    base_model: Any | None = None
    mlflow_run: str | None = None
    registered_version: str | None = None
    data_version: str | None = None
    replayed_from: str | None = None
    lineage: dict = field(default_factory=dict)
    alias_audit: dict | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise RetrainError(f"unknown retrain mode {self.mode!r}; expected one of {MODES}")

    @property
    def is_replay(self) -> bool:
        return self.mode == MODE_REPLAY

    def to_record(self) -> dict:
        """The metadata a `retrain_runs` row carries. No model, no rows."""
        return {
            "mode": self.mode,
            "mlflow_run": self.mlflow_run,
            "registered_version": self.registered_version,
            "data_version": self.data_version,
            "replayed_from": self.replayed_from,
            "lineage": dict(self.lineage),
        }


# ---------------------------------------------------------------------------
# The pinned data version
# ---------------------------------------------------------------------------
def current_data_version(lock_path: Path | None = None) -> str | None:
    """The training-data hash `dvc.lock` currently records, or None."""
    return dvc_lineage(lock_path).get(PINNED_DATA_KEY)


def verify_pinned_data_version(
    pinned: str | None,
    *,
    lock_path: Path | None = None,
) -> str | None:
    """Confirms the working tree holds the bytes the card pinned.

    Returns the verified hash. A card with no pinned version is allowed through
    -- Week 7 records `None` when `dvc.lock` is absent, and refusing there would
    make the pipeline untestable on a clean clone -- but a card that pinned a
    hash the tree does not have is refused, because running it anyway would
    produce a challenger whose lineage is a lie.
    """
    if pinned is None:
        return None
    present = current_data_version(lock_path)
    if present is None:
        raise RetrainError(
            f"card pins data version {pinned!r} but dvc.lock records none; "
            "run `dvc repro` (or `dvc pull`) before retraining"
        )
    if present != pinned:
        raise RetrainError(
            f"card pins data version {pinned!r} but the working tree holds {present!r}; "
            "a retrain must run on the bytes the decision was taken against"
        )
    return present


# ---------------------------------------------------------------------------
# Live retraining
# ---------------------------------------------------------------------------
def train_challenger(train_df, eval_df) -> tuple[Any, Any, dict, dict]:
    """Fits a challenger with the *same* entry point that fitted the champion.

    Returns `(base_model, calibrated_model, metadata, report)` -- the metadata
    `ml.train` records about how it trained, and the evaluation report
    `ml.evaluate` produces on the frozen slice, so the MLflow run carries the
    params and metrics §3.11 asks for.
    """
    from ml.train import build_metadata, train_models

    base_model, calibrated_model, n_estimators = train_models(train_df)

    X_eval, y_eval = split_features_target(eval_df)
    report = build_report(
        eval_df,
        y_eval,
        base_model.predict_proba(X_eval)[:, 1],
        calibrated_model.predict_proba(X_eval)[:, 1],
        evaluation_slice=EVAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        subgroup_columns=SUBGROUP_COLUMNS,
    )
    return base_model, calibrated_model, build_metadata(train_df, eval_df, n_estimators), report


def log_challenger(
    base_model,
    calibrated_model,
    metadata: dict,
    report: dict,
    *,
    card_id: str,
    data_version: str | None,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    pip_requirements=None,
    register: bool = True,
) -> ChallengerRun:
    """Records the challenger as an MLflow run and points `challenger` at it.

    Tags carry the card that authorised the run, so "which decision produced
    this model version" is answerable from MLflow alone as well as from
    `retrain_runs`.
    """
    import mlflow.sklearn

    import mlflow
    from ml.tracking import build_metrics, build_params, verify_tracking_reachable
    from ml.tracking_config import (
        BASE_MODEL_ARTIFACT,
        CALIBRATED_MODEL_ARTIFACT,
        MLFLOW_EXPERIMENT_NAME,
    )

    tracking_uri = tracking_uri or resolve_tracking_uri()
    verify_tracking_reachable(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    lineage = {**git_lineage(), **dvc_lineage()}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(build_params(metadata))
        mlflow.log_params({k: v for k, v in lineage.items() if v is not None})
        mlflow.set_tags(
            {
                "week": "8",
                "pipeline": "week8-retrain",
                "role": CHALLENGER_ALIAS,
                "retrain_mode": MODE_LIVE,
                "decision_card_id": card_id,
                "pinned_data_version": str(data_version),
                "inference_model": "calibrated",
            }
        )
        mlflow.log_metrics(build_metrics(report))

        # No input_example/signature and no dataset artifacts, for the reason
        # ml/tracking.py gives: inferring a signature embeds a real encounter's
        # feature row, and the processed CSVs are DVC's responsibility.
        calibrated_info = mlflow.sklearn.log_model(
            calibrated_model,
            name=CALIBRATED_MODEL_ARTIFACT,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=pip_requirements,
        )
        if base_model is not None:
            mlflow.sklearn.log_model(
                base_model,
                name=BASE_MODEL_ARTIFACT,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                pip_requirements=pip_requirements,
            )

    version = None
    alias_audit = None
    if register:
        client = MlflowClient(tracking_uri)
        version = register_model(client, run_id, calibrated_info.model_uri, model_name)
        alias_audit = set_alias(
            client,
            model_name=model_name,
            alias=CHALLENGER_ALIAS,
            version=version,
            run_id=run_id,
            reason=f"week-8 retrain authorised by decision card {card_id}",
            git_commit=lineage.get("git_commit"),
        )

    return ChallengerRun(
        mode=MODE_LIVE,
        calibrated_model=calibrated_model,
        base_model=base_model,
        mlflow_run=run_id,
        registered_version=version,
        data_version=data_version,
        lineage=lineage,
        alias_audit=alias_audit,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay_challenger(
    *,
    card_id: str,
    data_version: str | None = None,
    version: str | None = None,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
) -> ChallengerRun:
    """Re-registers a pre-trained challenger instead of fitting one.

    §3.11's demo acceleration. `version` names the registered version to reuse;
    without it the `challenger` alias is followed, which is what a rehearsed
    demo will have left pointing at the cached run. Nothing is trained, nothing
    is logged as new work, and the returned run is labeled `replay` so the
    `retrain_runs` row cannot pass it off as a live retrain.
    """
    import mlflow.sklearn

    import mlflow

    tracking_uri = tracking_uri or resolve_tracking_uri()
    client = MlflowClient(tracking_uri)

    resolved = version or current_alias_version(client, model_name, CHALLENGER_ALIAS)
    if resolved is None:
        raise RetrainError(
            f"replay mode found no challenger to reuse: pass an explicit version, or set the "
            f"{CHALLENGER_ALIAS!r} alias on {model_name!r} with a live retrain first"
        )

    try:
        model_version = client.get_model_version(model_name, resolved)
    except Exception as error:
        raise RetrainError(
            f"replay mode cannot read {model_name!r} version {resolved!r}: {error}"
        ) from error

    mlflow.set_tracking_uri(tracking_uri)
    try:
        calibrated_model = mlflow.sklearn.load_model(f"models:/{model_name}/{resolved}")
    except Exception as error:
        raise RetrainError(
            f"replay mode cannot load {model_name!r} version {resolved!r}: {error}"
        ) from error

    return ChallengerRun(
        mode=MODE_REPLAY,
        calibrated_model=calibrated_model,
        base_model=None,
        mlflow_run=model_version.run_id,
        registered_version=str(resolved),
        data_version=data_version,
        replayed_from=f"models:/{model_name}/{resolved}",
        lineage={"replayed_for_card": card_id, "replayed_from_version": str(resolved)},
    )


# ---------------------------------------------------------------------------
# The champion, for comparison
# ---------------------------------------------------------------------------
def load_champion(
    *,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
) -> tuple[Any, str | None]:
    """The serving model the challenger must be non-inferior to.

    Resolved through the registry alias rather than the local joblib bundle:
    §3.12 compares the challenger against *the champion*, and the champion is
    whatever `champion` points at, not whatever is on this disk.
    """
    import mlflow.sklearn

    import mlflow

    tracking_uri = tracking_uri or resolve_tracking_uri()
    client = MlflowClient(tracking_uri)
    version = current_alias_version(client, model_name, CHAMPION_ALIAS)
    if version is None:
        raise RetrainError(
            f"no {CHAMPION_ALIAS!r} alias on {model_name!r}; there is nothing to gate against"
        )

    mlflow.set_tracking_uri(tracking_uri)
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    except Exception as error:
        raise RetrainError(f"cannot load the champion ({version}): {error}") from error
    return model, str(version)


def retrain_from_card(
    card: dict,
    *,
    mode: str = MODE_LIVE,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    replay_version: str | None = None,
    lock_path: Path | None = None,
    pip_requirements=None,
) -> ChallengerRun:
    """§3.11 end to end: verify the pin, then train (or replay) a challenger.

    `card` is the Decision Card JSON -- the frozen Week 7 contract -- so this
    function couples to the schema and not to the engine that produced it.
    """
    if mode not in MODES:
        raise RetrainError(f"unknown retrain mode {mode!r}; expected one of {MODES}")

    card_id = card["card_id"]
    pinned = card.get("candidate_data_version")
    verified = verify_pinned_data_version(pinned, lock_path=lock_path)

    if mode == MODE_REPLAY:
        return replay_challenger(
            card_id=card_id,
            data_version=verified or pinned,
            version=replay_version,
            tracking_uri=tracking_uri,
            model_name=model_name,
        )

    from ml.train import load_split

    base_model, calibrated_model, metadata, report = train_challenger(
        load_split(TRAIN_PATH), load_split(EVAL_PATH)
    )
    return log_challenger(
        base_model,
        calibrated_model,
        metadata,
        report,
        card_id=card_id,
        data_version=verified or pinned,
        tracking_uri=tracking_uri,
        model_name=model_name,
        pip_requirements=pip_requirements,
    )
