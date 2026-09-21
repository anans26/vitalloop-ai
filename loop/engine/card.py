"""The Decision Card: the contract Weeks 8-9 read.

ARCHITECTURE.md §3.9 specifies it as an "immutable row in `decision_cards`,
validated by a Pydantic model", and RISK_ANALYSIS.md §1 names it as one of the
two contracts that must be **frozen in week 7** so the later weeks cannot
develop hidden coupling. That is why fields Week 7 cannot fill are present and
null rather than absent: `narrative` and `narrative_source` belong to Week 9's
narration layer, and a Week 9 that had to widen this schema would be a Week 9
that broke every card already emitted.

RESEARCH_NOVELTY.md C1 lists what the card must bind together: "the trigger
evidence, the exact policy version that evaluated it, the action, a decomposed
confidence score, the pinned data version a retrain will use, and the
acceptance criteria the challenger must meet **before** training starts". All
six are below.

**It carries statistics only.** The trigger holds feature *names* with their
PSI and KS values -- the same aggregate shape `drift_events.feature_stats`
holds -- and there is no field on any model here that can hold a patient row.
The narration layer's PHI-safety argument in §3.10 ("structurally incapable of
seeing PHI") is a claim about this schema, so the schema has to earn it.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CARD_STATUS_OPEN = "OPEN"
CARD_STATUS_CLOSED = "CLOSED"

# §3.9's example carries `"status": "EXECUTED"` and §3.12 closes a blocked card
# as `BLOCKED`; both are transitions Week 8's gate performs. A card at emission
# is in neither state, and the documents do not name the state it is in, so
# Week 7 records the only distinction it can defend: whether the action leaves
# downstream work to do.
CARD_STATUSES = (CARD_STATUS_OPEN, CARD_STATUS_CLOSED)

Action = Literal["NO_OP", "ALERT_ONLY", "INCREMENTAL_RETRAIN", "FULL_RETRAIN"]
Disposition = Literal["NONE", "AUTO_PROCEED_SHADOW", "ESCALATE_HUMAN"]
CardStatus = Literal["OPEN", "CLOSED"]
LabelMaturity = Literal["leading_indicators_only", "matured_labels"]


class BreachingFeature(BaseModel):
    """One breaching feature, exactly the §3.9 shape: a name and two statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    psi: float
    ks_p: float | None = None


class CardTrigger(BaseModel):
    """The evidence that produced the decision, traceable back to one window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drift_event_id: str
    scenario: str
    window_start: datetime
    window_end: datetime

    breaching_features: tuple[BreachingFeature, ...] = ()
    prediction_drift: bool = False
    prediction_psi: float | None = None
    max_psi: float | None = None
    monitored_feature_count: int = 0

    consecutive_breaching_windows: int = 0
    persistent_features: tuple[str, ...] = ()

    label_maturity: LabelMaturity = "leading_indicators_only"
    matured_auroc_drop: float | None = None

    # What the *monitor* was measuring against when the window was recorded,
    # which is a separate fact from what the policy compares it to.
    measured_thresholds: dict = Field(default_factory=dict)


class ConfidenceBreakdownModel(BaseModel):
    """The four terms, so the headline number can be re-derived by hand."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: float
    breadth: float
    persistence: float
    evidence: float


class DecisionCard(BaseModel):
    """One decision, immutable, re-derivable from its own contents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str
    created_at: datetime
    policy_version: str

    trigger: CardTrigger

    action: Action
    disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdownModel

    # Which row of the §3.8 table decided, and why in one sentence. `rule_id`
    # is an int for rules 1-6 and the string "uncovered" for the documented
    # gap-filler, so a reader can always trace a card to a line of the table.
    rule_id: int | str
    rationale: str
    downgraded_from: Action | None = None

    # Lineage. `candidate_data_version` is §3.11's "DVC hash pinned in the
    # Decision Card" -- the data a retrain would run on, fixed at decision time
    # so the retrain cannot quietly use something else.
    candidate_data_version: str | None = None
    model_version: str | None = None
    data_version: str | None = None

    # The bar the challenger must clear, recorded before any training starts.
    acceptance_criteria: dict[str, float]
    policy_thresholds: dict[str, float]

    # Week 9 fills these. Present and null so the contract does not move.
    narrative: str | None = None
    narrative_source: str | None = None

    status: CardStatus

    def to_json_dict(self) -> dict:
        """The JSON blob stored in `decision_cards.card_json`."""
        return self.model_dump(mode="json")
