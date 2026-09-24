"""Which models this API process is serving, and how that changes.

Week 5 resolved the champion once, at startup. Week 9 adds two reasons to
resolve again while running: a gate PASS moves `shadow`, and an approved
promotion moves `champion`. `refresh_serving_models` re-resolves both from the
registry and swaps them onto `app.state` together; the approval endpoint calls
it right after it moves the alias, which is what makes the roadmap's "champion
version changes live" true without a restart.

A refresh that cannot load the new champion keeps the old one. Serving the
previous champion is the safe failure; serving nothing is not, and a half-loaded
bundle is worse than either.
"""

from api.config import Settings
from api.logging_config import get_logger
from api.model_loader import ModelUnavailableError, load_model_bundle
from api.shadow import load_shadow_bundle

logger = get_logger()


def refresh_serving_models(state, settings: Settings) -> dict:
    """Re-resolves champion and shadow onto `state`. Returns what is now served."""
    previous = getattr(state, "model_bundle", None)
    champion_error = None
    try:
        bundle = load_model_bundle(settings)
    except ModelUnavailableError as error:
        bundle = previous
        champion_error = type(error).__name__
        logger.error("champion_reload_failed", error_category=champion_error)

    state.model_bundle = bundle
    champion_version = bundle.model_version if bundle else None
    state.shadow_bundle = load_shadow_bundle(settings, champion_version)

    served = {
        "champion_version": champion_version,
        "shadow_version": state.shadow_bundle.model_version if state.shadow_bundle else None,
        "champion_reload_failed": champion_error is not None,
    }
    logger.info("serving_models_resolved", **{k: v for k, v in served.items() if v is not None})
    return served
