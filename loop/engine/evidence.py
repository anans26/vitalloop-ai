"""The engine's input: one measured window, plus what came before it.

ARCHITECTURE.md §3.8 lists the Decision Engine's inputs as "latest drift event,
drift history (persistence), label maturity, champion health metrics,
time-since-last-retrain, retrain budget". `DriftEvidence` is that list as one
immutable value.

Two things it deliberately is not:

* **It is not a re-measurement.** Every PSI and KS value here was produced by
  Week 6 and read back off a `drift_events` row. The engine never opens a
  dataset, never scores a model, and never recomputes a statistic -- if the
  monitor measured it wrong, the card is wrong in the same way, and that is the
  correct behaviour for an audit trail.
* **It is not a database query.** Building evidence from rows is
  `history.py`'s job. Keeping the struct pure is what lets every rule branch be
  tested without a database.

`breaching_features` is the set of features whose recorded PSI reached the
policy's breach threshold. It is recomputed from `feature_stats` against the
*policy's* threshold rather than trusting the row's `breaching` flag, because
the row was written under whatever threshold was in force at measurement time
and the policy may differ -- the two are separate facts and the card records
both.
"""

from dataclasses import dataclass, field
from datetime import datetime

LEADING_INDICATORS_ONLY = "leading_indicators_only"
MATURED_LABELS = "matured_labels"


@dataclass(frozen=True)
class FeatureBreach:
    """One breaching feature, as the card's `trigger.breaching_features` records it."""

    feature: str
    psi: float
    ks_p: float | None = None

    def to_record(self) -> dict:
        """Exactly the §3.9 shape -- a feature name and two statistics."""
        return {"feature": self.feature, "psi": self.psi, "ks_p": self.ks_p}


@dataclass(frozen=True)
class DriftEvidence:
    """Everything policy-v1 is allowed to look at, for one window."""

    drift_event_id: str
    scenario: str
    window_start: datetime
    window_end: datetime

    # Measured by Week 6, read back verbatim.
    feature_stats: tuple[dict, ...] = ()
    prediction_drift: bool = False
    prediction_psi: float | None = None
    max_psi: float | None = None
    monitored_feature_count: int = 0
    measured_thresholds: dict = field(default_factory=dict)

    # Derived from the window's predecessors (see `history.py`).
    consecutive_breaching_windows: int = 0
    persistent_features: tuple[str, ...] = ()

    # Label maturity. ARCHITECTURE.md §3.7 places matured-label performance
    # after the 30-day label latency, and nothing in the repository produces it
    # yet, so this is `leading_indicators_only` for every real window today.
    label_maturity: str = LEADING_INDICATORS_ONLY
    matured_auroc_drop: float | None = None

    # Retrain history, for rule 6. No `retrain_runs` table exists until Week 8,
    # so both are empty in Week 7 and the rule is exercised by unit tests.
    days_since_last_retrain: float | None = None
    retrains_in_cooldown: int = 0

    # Lineage carried onto the card.
    model_version: str | None = None
    data_version: str | None = None
    candidate_data_version: str | None = None

    def breaches(self, psi_breach: float) -> tuple[FeatureBreach, ...]:
        """Breaching features at the policy's threshold, worst first.

        Sorted by PSI descending then by name, so two evaluations of the same
        window list the same features in the same order -- a card that reorders
        its own evidence is not re-derivable.
        """
        found = [
            FeatureBreach(
                feature=str(stat.get("feature")),
                psi=float(stat.get("psi")),
                ks_p=None if stat.get("ks_p_value") is None else float(stat["ks_p_value"]),
            )
            for stat in self.feature_stats
            if stat.get("feature") is not None
            and stat.get("psi") is not None
            and float(stat["psi"]) >= psi_breach
        ]
        return tuple(sorted(found, key=lambda breach: (-breach.psi, breach.feature)))

    def observed_max_psi(self) -> float:
        """The window's worst PSI. Falls back to the stats when the row omits it."""
        if self.max_psi is not None:
            return float(self.max_psi)
        values = [float(stat["psi"]) for stat in self.feature_stats if stat.get("psi") is not None]
        return max(values) if values else 0.0

    @property
    def labels_matured(self) -> bool:
        return self.label_maturity == MATURED_LABELS
