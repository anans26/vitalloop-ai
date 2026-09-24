"""The three buttons the demo needs that are not human decisions.

WORKFLOW.md §5 is a three-minute click-through. Two of its steps are *human
decisions* -- authorising a retrain and approving a promotion -- and those go
through the API (`dashboard/api_client.py`) so the token names the person.
The rest are operational steps the system would otherwise do on its own
schedule, and this module runs them on demand with the **same code** the
worker and the CLIs use:

* **Inject drift** (step 2). The roadmap wires the button to S1 and S2. It
  measures the *next unmeasured* window of the scenario -- the worker's own
  `measure_window` -> `record_window` -> `evaluate_event` chain -- so pressing
  it again measures the window after, and a window already on record is never
  measured twice. (Re-running `scenarios.run_scenario` appends duplicate
  windows, which Week 7's persistence rule then reads as drift that lasted;
  this button cannot do that.)
* **Retrain + gate** (steps 4 and 6): Week 8's `run_card`, in `live`, `replay`
  or `demo-bad` mode. Authorisation is still Week 8/9's: an escalated card
  retrains only after an ops user's recorded `APPROVE`.
* **Demo traffic** (step 1): real, authenticated `/predict` calls, so each one
  leaves an audit row and fills the shadow window exactly as clinician traffic
  would.
"""

from dataclasses import dataclass
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DecisionCard as DecisionCardRow
from db.models import DriftEvent
from loop.monitor.config import BENCHMARK_EPOCH, DRIFT_REPORTS_DIR, WINDOW_DURATION, WINDOW_ROWS

# The roadmap's Week 10 task: "drift-injection button wired to S1/S2".
INJECTABLE_SCENARIOS = ("S1", "S2")


class ActionError(RuntimeError):
    """The action could not be performed as asked. The message is shown verbatim."""


# ---------------------------------------------------------------------------
# Inject drift
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InjectionResult:
    scenario: str
    window_index: int
    event_id: str
    card_id: str | None
    action: str | None
    disposition: str | None
    confidence: float | None
    card_written: bool


def next_window_index(session: Session, scenario: str) -> int:
    """One past the highest window of this scenario already on record."""
    indices = set()
    for start in session.scalars(
        select(DriftEvent.window_start).where(DriftEvent.scenario == scenario)
    ):
        start = start if start.tzinfo else start.replace(tzinfo=UTC)
        indices.add(round((start - BENCHMARK_EPOCH) / WINDOW_DURATION))
    return max(indices) + 1 if indices else 0


def inject_next_window(
    session: Session,
    scenario: str,
    *,
    model=None,
    reference=None,
    stream=None,
    policy=None,
    data_version: str | None = None,
    window_rows: int = WINDOW_ROWS,
    reports_dir=DRIFT_REPORTS_DIR,
) -> InjectionResult:
    """Measures and decides the next window of an injected scenario. One row each."""
    if scenario not in INJECTABLE_SCENARIOS:
        raise ActionError(f"drift injection is wired to {INJECTABLE_SCENARIOS}, not {scenario!r}")

    from loop.engine.evaluate import candidate_data_version, evaluate_event
    from loop.engine.policy import load_policy
    from loop.monitor.persistence import record_window
    from loop.monitor.runner import load_serving_stream, measure_window
    from loop.monitor.windows import iter_windows
    from scenarios.injection import SCENARIO_SEED, apply_scenario

    index = next_window_index(session, scenario)
    if model is None:
        from loop.monitor.scoring import load_scoring_model

        model = load_scoring_model()
    if reference is None:
        from loop.monitor.reference import load_reference

        reference = load_reference(model)
    stream = load_serving_stream() if stream is None else stream

    injected = apply_scenario(stream, scenario, seed=SCENARIO_SEED)
    windows = iter_windows(injected, window_rows=window_rows, max_windows=index + 1)
    if index >= len(windows):
        raise ActionError(
            f"{scenario} has no window {index}: the serving stream holds {len(windows)} "
            "complete windows and every one is already on record"
        )

    result = measure_window(
        windows[index], scenario=scenario, reference=reference, model=model, reports_dir=reports_dir
    )
    event = record_window(session, result)
    card, written = evaluate_event(
        session,
        event,
        policy or load_policy(),
        data_version=data_version if data_version is not None else candidate_data_version(),
    )
    return InjectionResult(
        scenario=scenario,
        window_index=index,
        event_id=event.event_id,
        card_id=card.card_id,
        action=card.action,
        disposition=card.disposition,
        confidence=card.confidence,
        card_written=written,
    )


# ---------------------------------------------------------------------------
# Retrain + gate
# ---------------------------------------------------------------------------
def run_retrain(session: Session, card_id: str, mode: str, *, registry=None, **runner_kwargs):
    """Week 8's `run_card` for one card, with the dashboard's one extra refusal.

    Replay re-registers whatever `challenger` points at. After a promotion that
    *is* the serving champion, and a champion cannot be shadowed against
    itself -- so the run would PASS and then be unpromotable. That is refused
    up front, with the way out named, instead of wasting the demo's time.
    """
    from loop.gate.runner import GateRunnerError, run_card
    from ml.retrain import GATED_MODES, MODE_LIVE, MODE_REPLAY, RetrainError

    if mode not in GATED_MODES:
        raise ActionError(f"unknown mode {mode!r}; expected one of {GATED_MODES}")
    card = session.get(DecisionCardRow, card_id)
    if card is None:
        raise ActionError(f"no decision card {card_id!r}")

    if mode == MODE_REPLAY:
        registry = registry or _default_registry()
        cached, champion = registry.version_of("challenger"), registry.version_of("champion")
        if cached is None:
            raise ActionError("replay has no cached challenger yet; run a live retrain first")
        if cached == champion:
            raise ActionError(
                f"the cached challenger (v{cached}) is the serving champion, so a replay "
                "could never be shadowed or promoted; run a live retrain instead"
            )
    if mode == MODE_LIVE:
        # MLflow prints an emoji run URL; a cp1252 console would lose the run.
        from ml.tracking import make_console_encoding_safe

        make_console_encoding_safe()

    try:
        return run_card(session, card, mode=mode, **runner_kwargs)
    except (GateRunnerError, RetrainError) as error:
        raise ActionError(str(error)) from error


def _default_registry():
    from loop.approval.registry import MlflowAliasRegistry
    from ml.tracking import resolve_tracking_uri
    from ml.tracking_config import REGISTERED_MODEL_NAME

    return MlflowAliasRegistry(resolve_tracking_uri(), REGISTERED_MODEL_NAME)


# ---------------------------------------------------------------------------
# Demo traffic
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficResult:
    sent: int
    scored: int
    failed: int
    versions: tuple[str, ...]
    errors: tuple[str, ...] = ()


def send_demo_traffic(api, *, count: int, offset: int = 0, rows=None) -> TrafficResult:
    """`count` serving-stream encounters through `/predict`, under the caller's token."""
    from dashboard.api_client import ApiError
    from scripts.send_traffic import load_rows, payloads

    frame = load_rows(offset=offset, count=count) if rows is None else rows
    scored, versions, errors = 0, set(), []
    bodies = payloads(frame)
    for body in bodies:
        try:
            versions.add(str(api.predict(body)["model_version"]))
            scored += 1
        except ApiError as error:
            errors.append(str(error))
    return TrafficResult(
        sent=len(bodies),
        scored=scored,
        failed=len(bodies) - scored,
        versions=tuple(sorted(versions)),
        errors=tuple(dict.fromkeys(errors))[:3],
    )
