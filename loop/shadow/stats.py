"""Shadow agreement and stability statistics, as a pure function.

ARCHITECTURE.md §3.13: "Shadow model dual-scores live traffic for a configured
window ... Dashboard shows score-agreement and stability statistics." This is
that computation, over `(champion_score, shadow_score)` pairs and nothing else
-- no database, no model, no clock -- for the same reason the gate is a pure
function: the numbers a person approves a promotion on should be re-derivable
from the rows they came from.

The statistics answer the questions a reviewer asks before letting a model
reach clinicians:

* **Would clinicians see different decisions?** `decision_agreement`: the
  fraction of requests on which both models land on the same side of the
  serving threshold, and `flips_to_positive` / `flips_to_negative` counting the
  requests that would change class, in each direction.
* **How far apart are the scores?** Mean, 95th percentile and maximum of the
  absolute difference.
* **Do the models rank patients alike?** Spearman rank correlation. Two models
  can disagree on scores yet agree on who is most at risk, which is what a
  top-decile outreach list depends on.
* **Is the shadow stable?** Mean score of each model and the shift between
  them: a shadow that is systematically higher or lower moves every clinician's
  number, even where decisions agree.

A failed shadow score is counted, not dropped (`failed`): a shadow that errors
on some inputs is a finding about that shadow.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ShadowStats:
    champion_version: str | None
    shadow_version: str | None
    requests: int
    scored: int
    failed: int
    decision_threshold: float | None
    decision_agreement: float | None
    flips_to_positive: int
    flips_to_negative: int
    mean_abs_diff: float | None
    p95_abs_diff: float | None
    max_abs_diff: float | None
    spearman: float | None
    champion_mean_score: float | None
    shadow_mean_score: float | None
    mean_shift: float | None

    def to_record(self) -> dict:
        return asdict(self)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def shadow_stats(
    pairs: Iterable[tuple[float, float | None]],
    *,
    threshold: float,
    champion_version: str | None = None,
    shadow_version: str | None = None,
) -> ShadowStats:
    """Agreement between champion and shadow over one window of requests.

    `pairs` holds `(champion_score, shadow_score)`, with `shadow_score` None
    where the shadow failed to score.
    """
    pairs = list(pairs)
    scored = [(c, s) for c, s in pairs if s is not None]
    failed = len(pairs) - len(scored)

    if not scored:
        return ShadowStats(
            champion_version=champion_version,
            shadow_version=shadow_version,
            requests=len(pairs),
            scored=0,
            failed=failed,
            decision_threshold=threshold,
            decision_agreement=None,
            flips_to_positive=0,
            flips_to_negative=0,
            mean_abs_diff=None,
            p95_abs_diff=None,
            max_abs_diff=None,
            spearman=None,
            champion_mean_score=None,
            shadow_mean_score=None,
            mean_shift=None,
        )

    champion = np.asarray([c for c, _ in scored], dtype=float)
    shadow = np.asarray([s for _, s in scored], dtype=float)
    diff = np.abs(shadow - champion)

    champion_positive = champion >= threshold
    shadow_positive = shadow >= threshold

    # Rank correlation is undefined when either side is constant.
    spearman = None
    if len(scored) >= 2 and np.ptp(champion) > 0 and np.ptp(shadow) > 0:
        spearman = spearmanr(champion, shadow).statistic

    return ShadowStats(
        champion_version=champion_version,
        shadow_version=shadow_version,
        requests=len(pairs),
        scored=len(scored),
        failed=failed,
        decision_threshold=threshold,
        decision_agreement=_round(np.mean(champion_positive == shadow_positive)),
        flips_to_positive=int(np.sum(~champion_positive & shadow_positive)),
        flips_to_negative=int(np.sum(champion_positive & ~shadow_positive)),
        mean_abs_diff=_round(np.mean(diff)),
        p95_abs_diff=_round(np.percentile(diff, 95)),
        max_abs_diff=_round(np.max(diff)),
        spearman=_round(spearman),
        champion_mean_score=_round(np.mean(champion)),
        shadow_mean_score=_round(np.mean(shadow)),
        mean_shift=_round(np.mean(shadow) - np.mean(champion)),
    )
