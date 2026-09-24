"""Reading and writing `shadow_predictions`.

The write happens in the API, after the response (see `api/shadow.py`); the
reads happen wherever a person needs the window's statistics -- the ops
endpoints and the promotion check. Both go through here so there is one
definition of "the shadow window for this pair of models".

Append-only, like every audit table: nothing here updates or deletes.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import ShadowPrediction
from loop.shadow.stats import ShadowStats, shadow_stats

STATUS_SCORED = "scored"
STATUS_FAILED = "shadow_failed"


def persist_shadow_row(session: Session, row: ShadowPrediction) -> ShadowPrediction:
    """Writes the row. Failures propagate; the caller decides they are not fatal."""
    try:
        session.add(row)
        session.commit()
        return row
    except Exception:
        session.rollback()
        raise


def window_count(session: Session, champion_version: str, shadow_version: str) -> int:
    """Requests the shadow has seen alongside this champion -- scored or failed."""
    statement = (
        select(func.count())
        .select_from(ShadowPrediction)
        .where(ShadowPrediction.champion_version == champion_version)
        .where(ShadowPrediction.shadow_version == shadow_version)
    )
    return int(session.scalar(statement) or 0)


def window_stats(
    session: Session,
    champion_version: str,
    shadow_version: str,
    *,
    threshold: float | None = None,
) -> ShadowStats:
    """§3.13's statistics over every request this pair of models has scored."""
    statement = (
        select(
            ShadowPrediction.champion_score,
            ShadowPrediction.shadow_score,
            ShadowPrediction.decision_threshold,
        )
        .where(ShadowPrediction.champion_version == champion_version)
        .where(ShadowPrediction.shadow_version == shadow_version)
        .order_by(ShadowPrediction.ts, ShadowPrediction.id)
    )
    rows = session.execute(statement).all()
    if threshold is None:
        threshold = rows[-1].decision_threshold if rows else 0.5
    return shadow_stats(
        ((row.champion_score, row.shadow_score) for row in rows),
        threshold=threshold,
        champion_version=champion_version,
        shadow_version=shadow_version,
    )
