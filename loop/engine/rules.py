"""The policy-v1 rule table, as a pure function.

ARCHITECTURE.md §3.8 / PROJECT_DESIGN.md §4.2:

| # | Condition                                    | Action                | Disposition           |
|---|----------------------------------------------|-----------------------|-----------------------|
| 1 | No feature PSI >= 0.10, no prediction drift  | `NO_OP`               | --                    |
| 2 | 1-2 features 0.10 <= PSI < 0.25, no          | `ALERT_ONLY`          | --                    |
|   | prediction drift, first window               |                       |                       |
| 3 | Same features breach >= 2 consecutive windows| `INCREMENTAL_RETRAIN` | auto -> shadow        |
| 4 | Any PSI >= 0.25 **or** prediction drift      | `FULL_RETRAIN`        | auto -> shadow if     |
|   |                                              |                       | confidence >= 0.75,   |
|   |                                              |                       | else escalate         |
| 5 | Matured-label AUROC drop > 0.03 vs launch    | `FULL_RETRAIN`        | escalate to human     |
| 6 | Retrained < X days ago **or** budget spent   | downgrade `ALERT_ONLY`| escalate              |

Every threshold above is read from the `Policy`, never from this file. The only
thing hard-coded here is the *shape* of the table, which is what "freeze
policy-v1 at 6 rules" (RISK_ANALYSIS.md §1) means: policy-v2 changes numbers in
a YAML diff a reviewer reads, not branches in Python.

The table's conditions overlap, so evaluating it needs a precedence; §3.8 does
not state one, so `policy.precedence` carries it and `configs/policy-v1.yaml`
explains the reading. The table is also not total -- three or more features
breaching mildly in their first window match no rule -- so
`policy.uncovered_breach_action` names what happens there, documented as a
gap-filler rather than a seventh rule.
"""

from dataclasses import dataclass

from loop.engine.confidence import ConfidenceBreakdown
from loop.engine.evidence import DriftEvidence
from loop.engine.policy import Policy

# Actions, in the order §3.8 escalates through them.
NO_OP = "NO_OP"
ALERT_ONLY = "ALERT_ONLY"
INCREMENTAL_RETRAIN = "INCREMENTAL_RETRAIN"
FULL_RETRAIN = "FULL_RETRAIN"

ACTIONS = (NO_OP, ALERT_ONLY, INCREMENTAL_RETRAIN, FULL_RETRAIN)
RETRAIN_ACTIONS = (INCREMENTAL_RETRAIN, FULL_RETRAIN)

# Dispositions. `AUTO_PROCEED_SHADOW` and `ESCALATE_HUMAN` are the two the
# documents name (§3.9's example card and WORKFLOW.md §4); `NONE` is the
# table's "--", spelled so a card always has a value in the column.
DISPOSITION_NONE = "NONE"
AUTO_PROCEED_SHADOW = "AUTO_PROCEED_SHADOW"
ESCALATE_HUMAN = "ESCALATE_HUMAN"

DISPOSITIONS = (DISPOSITION_NONE, AUTO_PROCEED_SHADOW, ESCALATE_HUMAN)

RULE_UNCOVERED = "uncovered"
RULE_DOWNGRADE = 6


@dataclass(frozen=True)
class RuleOutcome:
    """What the table decided, and which row of it decided."""

    action: str
    disposition: str
    rule_id: int | str
    rationale: str
    downgraded_from: str | None = None

    @property
    def requires_retrain(self) -> bool:
        return self.action in RETRAIN_ACTIONS


# ---------------------------------------------------------------------------
# The individual rules. Each returns a RuleOutcome or None ("does not apply").
# ---------------------------------------------------------------------------
def rule_1_no_drift(evidence: DriftEvidence, policy: Policy, _: ConfidenceBreakdown):
    """No feature breached and the score distribution held."""
    if evidence.breaches(policy.psi_breach) or evidence.prediction_drift:
        return None
    return RuleOutcome(
        action=NO_OP,
        disposition=DISPOSITION_NONE,
        rule_id=1,
        rationale=(
            f"No feature reached PSI {policy.psi_breach} and no prediction drift was "
            "detected. Silence recorded as a decision."
        ),
    )


def rule_2_first_window_mild(evidence: DriftEvidence, policy: Policy, _: ConfidenceBreakdown):
    """At most N features breaching mildly, no prediction drift, first such window."""
    breaches = evidence.breaches(policy.psi_breach)
    if not breaches or evidence.prediction_drift:
        return None
    if len(breaches) > policy.alert_only_max_features:
        return None
    if max(breach.psi for breach in breaches) >= policy.psi_severe:
        return None
    if evidence.consecutive_breaching_windows >= policy.consecutive_windows:
        return None
    return RuleOutcome(
        action=ALERT_ONLY,
        disposition=DISPOSITION_NONE,
        rule_id=2,
        rationale=(
            f"{len(breaches)} feature(s) between PSI {policy.psi_breach} and "
            f"{policy.psi_severe} in their first breaching window, with no prediction "
            "drift. Alerting without acting; the persistence rule needs a second window."
        ),
    )


def rule_3_persistent_breach(evidence: DriftEvidence, policy: Policy, _: ConfidenceBreakdown):
    """The same features have breached for N consecutive windows."""
    if evidence.consecutive_breaching_windows < policy.consecutive_windows:
        return None
    if not evidence.persistent_features:
        return None
    return RuleOutcome(
        action=INCREMENTAL_RETRAIN,
        disposition=AUTO_PROCEED_SHADOW,
        rule_id=3,
        rationale=(
            f"{', '.join(evidence.persistent_features)} breached for "
            f"{evidence.consecutive_breaching_windows} consecutive windows. Persistent "
            "drift is not a blip."
        ),
    )


def rule_4_severe_or_prediction(
    evidence: DriftEvidence, policy: Policy, confidence: ConfidenceBreakdown
):
    """Any feature at or above the severe threshold, or the scores moved."""
    breaches = evidence.breaches(policy.psi_breach)
    severe = [breach for breach in breaches if breach.psi >= policy.psi_severe]
    if not severe and not evidence.prediction_drift:
        return None

    auto = confidence.confidence >= policy.auto_proceed_confidence
    if severe:
        trigger = f"{severe[0].feature} reached PSI {severe[0].psi:.4f}"
    else:
        trigger = f"prediction drift detected (PSI {evidence.prediction_psi})"
    return RuleOutcome(
        action=FULL_RETRAIN,
        disposition=AUTO_PROCEED_SHADOW if auto else ESCALATE_HUMAN,
        rule_id=4,
        rationale=(
            f"{trigger}. Confidence {confidence.confidence:.4f} is "
            f"{'at or above' if auto else 'below'} the "
            f"{policy.auto_proceed_confidence} auto-proceed threshold."
        ),
    )


def rule_5_matured_regression(evidence: DriftEvidence, policy: Policy, _: ConfidenceBreakdown):
    """Matured labels confirm the champion has degraded. Never handled silently."""
    if not evidence.labels_matured or evidence.matured_auroc_drop is None:
        return None
    if evidence.matured_auroc_drop <= policy.matured_auroc_drop:
        return None
    return RuleOutcome(
        action=FULL_RETRAIN,
        disposition=ESCALATE_HUMAN,
        rule_id=5,
        rationale=(
            f"Matured labels show an AUROC drop of {evidence.matured_auroc_drop:.4f} "
            f"against launch, above the {policy.matured_auroc_drop} limit. A confirmed "
            "performance regression always reaches a human."
        ),
    )


RULES = {
    1: rule_1_no_drift,
    2: rule_2_first_window_mild,
    3: rule_3_persistent_breach,
    4: rule_4_severe_or_prediction,
    5: rule_5_matured_regression,
}


def uncovered_breach(evidence: DriftEvidence, policy: Policy) -> RuleOutcome:
    """The table's uncovered region: breaching evidence that no rule claims."""
    breaches = evidence.breaches(policy.psi_breach)
    return RuleOutcome(
        action=policy.uncovered_breach_action,
        disposition=DISPOSITION_NONE,
        rule_id=RULE_UNCOVERED,
        rationale=(
            f"{len(breaches)} feature(s) breached PSI {policy.psi_breach} without "
            f"matching a policy-{policy.version.split('-')[-1]} rule. Taking the "
            f"table's weakest non-silent action ({policy.uncovered_breach_action})."
        ),
    )


def rule_6_downgrade(outcome: RuleOutcome, evidence: DriftEvidence, policy: Policy) -> RuleOutcome:
    """Rule 6 is a modifier, not an action: it downgrades a retrain that just happened.

    Phrased in §3.8 as "downgrade to `ALERT_ONLY`", so it applies to whatever
    the table selected rather than competing with it, and only when that
    selection would retrain -- there is nothing to downgrade about `NO_OP`.
    """
    if not outcome.requires_retrain:
        return outcome

    # §3.8 joins the two conditions with "or", so either one downgrades.
    too_recent = (
        evidence.days_since_last_retrain is not None
        and evidence.days_since_last_retrain < policy.retrain_cooldown_days
    )
    exhausted = evidence.retrains_in_cooldown >= policy.retrain_budget
    if not (too_recent or exhausted):
        return outcome

    if too_recent:
        reason = (
            f"the last retrain was {evidence.days_since_last_retrain:.2f} days ago, "
            f"inside the {policy.retrain_cooldown_days}-day cooldown"
        )
    else:
        reason = (
            f"{evidence.retrains_in_cooldown} of {policy.retrain_budget} retrains in the "
            "budget have been used"
        )
    return RuleOutcome(
        action=ALERT_ONLY,
        disposition=ESCALATE_HUMAN,
        rule_id=RULE_DOWNGRADE,
        rationale=(
            f"{outcome.action} downgraded: {reason}. Escalating instead of retraining again."
        ),
        downgraded_from=outcome.action,
    )


def apply_policy(
    evidence: DriftEvidence, policy: Policy, confidence: ConfidenceBreakdown
) -> RuleOutcome:
    """Runs the table in the policy's precedence, then applies rule 6.

    Deterministic and side-effect free: the same evidence and policy always
    select the same row of the table.
    """
    for rule_id in policy.precedence:
        rule = RULES.get(rule_id)
        if rule is None:  # pragma: no cover - policy validation rejects this first
            raise ValueError(f"{policy.version} names an unknown rule {rule_id!r}")
        outcome = rule(evidence, policy, confidence)
        if outcome is not None:
            return rule_6_downgrade(outcome, evidence, policy)

    return rule_6_downgrade(uncovered_breach(evidence, policy), evidence, policy)
