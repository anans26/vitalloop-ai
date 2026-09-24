"""The ops endpoints: shadow statistics, retrain authorisation, promotion.

Every endpoint here requires the `ops` role. TECH_STACK.md's reason for role
claims is governance separation -- "whoever *sees* a risk score is not
automatically whoever *promotes* a model" -- and Week 5 built `require_role`
for exactly this router.

The approver's identity is the token's subject. Neither decision body has an
"approver" field, so a person cannot record a decision in someone else's name.

This is the API half of §3.13's "human click in the dashboard": the Week 10
Streamlit page will call these endpoints, and until it exists they are the
approval flow. The logic lives in `loop/approval/`; this module maps HTTP onto
it and refreshes the served models after a decision moves an alias.
"""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.auth import OPS_ROLE, Principal, require_role
from api.config import Settings, get_settings
from api.logging_config import get_logger
from api.schemas import ApprovalResponse, DecisionRequest, ErrorResponse
from api.serving_state import refresh_serving_models
from db.models import Approval, RetrainRun
from db.models import DecisionCard as DecisionCardRow
from db.session import session_scope
from loop.approval.persistence import ApprovalError
from loop.approval.promotion import decide_promotion, pending_promotions, promotion_evidence
from loop.approval.registry import AliasRegistry, MlflowAliasRegistry
from loop.approval.retrain import cards_awaiting_authorisation, decide_retrain, retrain_evidence
from loop.approval.rules import PromotionRules, load_rules
from loop.shadow.persistence import window_stats

router = APIRouter(
    prefix="/ops",
    tags=["ops"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing, malformed, or expired token"},
        403: {"model": ErrorResponse, "description": "Token lacks the ops role"},
    },
)
logger = get_logger()

require_ops = require_role(OPS_ROLE)


@lru_cache
def get_promotion_rules() -> PromotionRules:
    return load_rules()


def get_alias_registry(
    request: Request, settings: Settings = Depends(get_settings)
) -> AliasRegistry:
    """The registry the decision moves aliases in.

    Tests put a fake on `app.state.alias_registry`. Otherwise it is MLflow, which
    requires `model_source="mlflow"`: an API serving the local artifact has no
    aliases to move, and pretending otherwise would record promotions that
    changed nothing.
    """
    injected = getattr(request.app.state, "alias_registry", None)
    if injected is not None:
        return injected
    if settings.model_source != "mlflow":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promotion requires the model registry (VITALLOOP_MODEL_SOURCE=mlflow).",
        )
    return MlflowAliasRegistry(settings.mlflow_tracking_uri, settings.registered_model_name)


def _refused(error: ApprovalError) -> HTTPException:
    return HTTPException(status_code=error.status, detail=str(error))


def _approval_response(row: Approval, serving: dict | None = None) -> ApprovalResponse:
    return ApprovalResponse(
        approval_id=row.approval_id,
        kind=row.kind,
        card_id=row.card_id,
        run_id=row.run_id,
        decision=row.decision,
        approver=row.approver,
        reason=row.reason,
        rules_version=row.rules_version,
        challenger_version=row.challenger_version,
        champion_version_before=row.champion_version_before,
        champion_version_after=row.champion_version_after,
        champion_alias_moved=bool(row.champion_alias_moved),
        shadow_alias_cleared=bool(row.shadow_alias_cleared),
        serving=serving,
    )


# ---------------------------------------------------------------------------
# Serving and shadow
# ---------------------------------------------------------------------------
@router.get(
    "/shadow",
    summary="Shadow agreement and stability statistics",
    description=(
        "The models this instance serves, and §3.13's agreement statistics for the "
        "current champion/shadow pair over every request both have scored."
    ),
)
def shadow_status(
    request: Request,
    principal: Principal = Depends(require_ops),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(session_scope),
) -> dict:
    champion = getattr(request.app.state, "model_bundle", None)
    shadow = getattr(request.app.state, "shadow_bundle", None)
    body = {
        "champion_version": champion.model_version if champion else None,
        "shadow_version": shadow.model_version if shadow else None,
        "shadow_active": shadow is not None,
        "stats": None,
    }
    if champion is not None and shadow is not None:
        body["stats"] = window_stats(
            session,
            champion.model_version,
            shadow.model_version,
            threshold=settings.decision_threshold,
        ).to_record()
    return body


@router.post(
    "/models/reload",
    summary="Re-resolve the champion and shadow models",
    description=(
        "Re-reads the `champion` and `shadow` aliases and loads what they point at. "
        "A champion that fails to load keeps the previous one serving."
    ),
)
def reload_models(
    request: Request,
    principal: Principal = Depends(require_ops),
    settings: Settings = Depends(get_settings),
) -> dict:
    served = refresh_serving_models(request.app.state, settings)
    logger.info("models_reloaded", caller=principal.subject)
    return served


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
@router.get(
    "/cards/{card_id}/narrative",
    summary="A Decision Card's plain-English narrative",
    description=(
        "The narrative the card was emitted with, or -- for a card emitted before "
        "narration existed -- the deterministic template's rendering. Never calls an "
        "LLM at request time. Includes the grounding check's verdict."
    ),
    responses={404: {"model": ErrorResponse, "description": "No such card"}},
)
def card_narrative(
    card_id: str,
    principal: Principal = Depends(require_ops),
    session: Session = Depends(session_scope),
) -> dict:
    from loop.narrate.grounding import check_grounding
    from loop.narrate.template import TEMPLATE_SOURCE, render_template

    row = session.get(DecisionCardRow, card_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No decision card {card_id!r}.")
    card = dict(row.card_json)
    stored = bool(card.get("narrative"))
    text = card["narrative"] if stored else render_template(card)
    return {
        "card_id": card_id,
        "narrative": text,
        "narrative_source": card.get("narrative_source") if stored else TEMPLATE_SOURCE,
        "stored": stored,
        "grounding": check_grounding(text, card).to_record(),
    }


# ---------------------------------------------------------------------------
# Retrain authorisation (escalated cards)
# ---------------------------------------------------------------------------
@router.get(
    "/retrains/pending",
    summary="Escalated cards waiting for a person",
    description="Retrain cards the policy escalated, with no decision and no retrain yet.",
)
def pending_retrains(
    principal: Principal = Depends(require_ops),
    session: Session = Depends(session_scope),
) -> list[dict]:
    return [retrain_evidence(row) for row in cards_awaiting_authorisation(session)]


@router.post(
    "/retrains/{card_id}/decision",
    response_model=ApprovalResponse,
    summary="Authorise or reject an escalated retrain",
    description=(
        "Records the decision in `approvals`. APPROVE authorises the retrain, which "
        "`python -m loop.gate.runner` then performs; nothing trains inside this request."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "No such card"},
        409: {"model": ErrorResponse, "description": "Not decidable, or already decided"},
        422: {"model": ErrorResponse, "description": "Invalid decision or reason"},
    },
)
def decide_retrain_endpoint(
    card_id: str,
    body: DecisionRequest,
    principal: Principal = Depends(require_ops),
    rules: PromotionRules = Depends(get_promotion_rules),
    session: Session = Depends(session_scope),
) -> ApprovalResponse:
    try:
        row = decide_retrain(
            session,
            card_id,
            decision=body.decision,
            approver=principal.subject,
            approver_role=principal.role,
            reason=body.reason,
            rules=rules,
        )
    except ApprovalError as error:
        raise _refused(error) from error
    logger.info("retrain_decided", card_id=card_id, decision=row.decision, caller=principal.subject)
    return _approval_response(row)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
@router.get(
    "/promotions/pending",
    summary="Gated challengers in shadow, waiting for a person",
    description=(
        "Each entry carries the full evidence chain -- card, gate result, shadow "
        "statistics, aliases, and which promotion preconditions currently hold."
    ),
)
def pending_promotion_list(
    principal: Principal = Depends(require_ops),
    rules: PromotionRules = Depends(get_promotion_rules),
    registry: AliasRegistry = Depends(get_alias_registry),
    session: Session = Depends(session_scope),
) -> list[dict]:
    return [
        {
            "run_id": run.run_id,
            "card_id": run.card_id,
            "challenger_version": run.challenger_version,
            "evidence": promotion_evidence(session, run, registry, rules),
        }
        for run in pending_promotions(session)
    ]


@router.get(
    "/promotions/{run_id}",
    summary="The evidence for one promotion",
    responses={404: {"model": ErrorResponse, "description": "No such retrain run"}},
)
def promotion_detail(
    run_id: str,
    principal: Principal = Depends(require_ops),
    rules: PromotionRules = Depends(get_promotion_rules),
    registry: AliasRegistry = Depends(get_alias_registry),
    session: Session = Depends(session_scope),
) -> dict:
    run = session.get(RetrainRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No retrain run {run_id!r}.")
    return {
        "run_id": run.run_id,
        "card_id": run.card_id,
        "challenger_version": run.challenger_version,
        "evidence": promotion_evidence(session, run, registry, rules),
    }


@router.post(
    "/promotions/{run_id}/decision",
    response_model=ApprovalResponse,
    summary="Approve or reject a promotion to champion",
    description=(
        "APPROVE moves `champion` to the challenger -- only when the gate passed, the "
        "challenger is still in shadow, the champion is the one it beat, and the shadow "
        "window is complete -- then clears `shadow` and reloads this instance's models. "
        "REJECT moves nothing but clears `shadow`. Both are recorded in `approvals`."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "No such retrain run"},
        409: {"model": ErrorResponse, "description": "Preconditions unmet, or already decided"},
        422: {"model": ErrorResponse, "description": "Invalid decision or reason"},
        503: {"model": ErrorResponse, "description": "No model registry configured"},
    },
)
def decide_promotion_endpoint(
    run_id: str,
    body: DecisionRequest,
    request: Request,
    principal: Principal = Depends(require_ops),
    settings: Settings = Depends(get_settings),
    rules: PromotionRules = Depends(get_promotion_rules),
    registry: AliasRegistry = Depends(get_alias_registry),
    session: Session = Depends(session_scope),
) -> ApprovalResponse:
    try:
        result = decide_promotion(
            session,
            run_id,
            decision=body.decision,
            approver=principal.subject,
            approver_role=principal.role,
            reason=body.reason,
            registry=registry,
            rules=rules,
        )
    except ApprovalError as error:
        raise _refused(error) from error

    serving = None
    if result.champion_alias_moved or result.shadow_alias_cleared:
        serving = refresh_serving_models(request.app.state, settings)

    logger.info(
        "promotion_decided",
        run_id=run_id,
        decision=result.approval.decision,
        champion_alias_moved=result.champion_alias_moved,
        caller=principal.subject,
    )
    return _approval_response(result.approval, serving)
