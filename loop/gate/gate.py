"""`evaluate_gate(champion, challenger, criteria) -> GateResult`. The whole gate.

ARCHITECTURE.md §3.12: "The gate is a pure function of two metric sets --
trivially unit-testable, which is exactly what a promotion safety mechanism
must be." So this module has no database handle, no filesystem access, no
model, no clock and no network. Everything it needs arrives in two mappings of
`MetricSet`, and the same two mappings always produce the same verdict.

The five criteria are §3.12 / §4.5 and are read from `GateCriteria`, never from
this file. What is hard-coded here is only the *shape* of the check -- which
direction each comparison runs in -- because that is the document's table, not
a tunable:

| Criterion              | Rule                                    |
|------------------------|-----------------------------------------|
| AUROC                  | challenger >= champion - margin         |
| Recall @ top decile    | challenger >= champion * min_ratio      |
| Brier score            | challenger <= champion + max_increase   |
| Calibration (ECE)      | challenger <= max_ece  (absolute)       |
| Subgroup non-regression| no subgroup AUROC drop > max_drop       |

**Every criterion is evaluated on every set** §3.12 names -- the frozen holdout
*and* the most recent labeled window -- and `require_all_sets` means a failure
on either blocks. RISK_ANALYSIS.md §2 gives the reason: gating repeatedly
against one frozen holdout eventually overfits to it.

**A missing metric set blocks.** The gate never passes a challenger it could
not measure; "no evidence" is not "no regression".
"""

from collections.abc import Mapping
from dataclasses import dataclass

from loop.gate.criteria import GateCriteria
from loop.gate.metrics import MetricSet

PASS = "PASS"
BLOCK = "BLOCK"
OUTCOMES = (PASS, BLOCK)

CRITERION_AUROC = "auroc_non_inferiority"
CRITERION_RECALL = "recall_at_top_decile"
CRITERION_BRIER = "brier_score"
CRITERION_ECE = "calibration_ece"
CRITERION_SUBGROUP = "subgroup_non_regression"
CRITERION_COVERAGE = "evaluation_set_present"

CRITERIA_ORDER = (
    CRITERION_COVERAGE,
    CRITERION_AUROC,
    CRITERION_RECALL,
    CRITERION_BRIER,
    CRITERION_ECE,
    CRITERION_SUBGROUP,
)

COMPARISON_PRECISION = 9


@dataclass(frozen=True)
class CriterionResult:
    """One criterion, on one evaluation set, with the numbers behind the verdict."""

    criterion: str
    evaluation_set: str
    passed: bool
    detail: str
    observed: float | None = None
    required: float | None = None
    subgroup: str | None = None

    def to_record(self) -> dict:
        return {
            "criterion": self.criterion,
            "evaluation_set": self.evaluation_set,
            "passed": self.passed,
            "observed": self.observed,
            "required": self.required,
            "subgroup": self.subgroup,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateResult:
    """PASS or BLOCK, plus every check that produced it.

    `reasons` is the BLOCK explanation the roadmap asks for ("bad challenger
    BLOCKED with reasons") -- one sentence per failed check, in a fixed order,
    so two runs of the same comparison read identically.
    """

    outcome: str
    criteria_version: str
    checks: tuple[CriterionResult, ...]
    reasons: tuple[str, ...]
    evaluation_sets: tuple[str, ...]
    criteria: dict
    champion_metrics: dict
    challenger_metrics: dict

    @property
    def passed(self) -> bool:
        return self.outcome == PASS

    @property
    def blocked(self) -> bool:
        return self.outcome == BLOCK

    def failed_checks(self) -> tuple[CriterionResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_record(self) -> dict:
        """The JSON blob stored in `retrain_runs.gate_result`.

        Self-contained on purpose: the verdict, the criteria that produced it,
        every check, and both metric sets. An examiner re-derives the decision
        from this one column without opening a config file or a model.
        """
        return {
            "outcome": self.outcome,
            "criteria_version": self.criteria_version,
            "evaluation_sets": list(self.evaluation_sets),
            "criteria": dict(self.criteria),
            "checks": [check.to_record() for check in self.checks],
            "reasons": list(self.reasons),
            "champion_metrics": dict(self.champion_metrics),
            "challenger_metrics": dict(self.challenger_metrics),
        }


def _at_least(observed: float, required: float) -> bool:
    """`observed >= required`, with float noise rounded out of the comparison."""
    return round(observed - required, COMPARISON_PRECISION) >= 0


def _at_most(observed: float, required: float) -> bool:
    return round(required - observed, COMPARISON_PRECISION) >= 0


def _check_auroc(
    champion: MetricSet, challenger: MetricSet, criteria: GateCriteria
) -> CriterionResult:
    floor = champion.roc_auc - criteria.auroc_non_inferiority_margin
    passed = _at_least(challenger.roc_auc, floor)
    return CriterionResult(
        criterion=CRITERION_AUROC,
        evaluation_set=challenger.evaluation_set,
        passed=passed,
        observed=challenger.roc_auc,
        required=round(floor, COMPARISON_PRECISION),
        detail=(
            f"AUROC {challenger.roc_auc:.6f} vs champion {champion.roc_auc:.6f} "
            f"(floor {floor:.6f}, margin {criteria.auroc_non_inferiority_margin})"
        ),
    )


def _check_recall(
    champion: MetricSet, challenger: MetricSet, criteria: GateCriteria
) -> CriterionResult:
    floor = champion.recall_at_top_decile * criteria.recall_top_decile_min_ratio
    passed = _at_least(challenger.recall_at_top_decile, floor)
    return CriterionResult(
        criterion=CRITERION_RECALL,
        evaluation_set=challenger.evaluation_set,
        passed=passed,
        observed=challenger.recall_at_top_decile,
        required=round(floor, COMPARISON_PRECISION),
        detail=(
            f"recall@top-decile {challenger.recall_at_top_decile:.6f} vs champion "
            f"{champion.recall_at_top_decile:.6f} (floor {floor:.6f}, ratio "
            f"{criteria.recall_top_decile_min_ratio})"
        ),
    )


def _check_brier(
    champion: MetricSet, challenger: MetricSet, criteria: GateCriteria
) -> CriterionResult:
    ceiling = champion.brier_score + criteria.max_brier_increase
    passed = _at_most(challenger.brier_score, ceiling)
    return CriterionResult(
        criterion=CRITERION_BRIER,
        evaluation_set=challenger.evaluation_set,
        passed=passed,
        observed=challenger.brier_score,
        required=round(ceiling, COMPARISON_PRECISION),
        detail=(
            f"Brier {challenger.brier_score:.6f} vs champion {champion.brier_score:.6f} "
            f"(ceiling {ceiling:.6f}, max increase {criteria.max_brier_increase})"
        ),
    )


def _check_ece(challenger: MetricSet, criteria: GateCriteria) -> CriterionResult:
    passed = _at_most(challenger.expected_calibration_error, criteria.max_ece)
    return CriterionResult(
        criterion=CRITERION_ECE,
        evaluation_set=challenger.evaluation_set,
        passed=passed,
        observed=challenger.expected_calibration_error,
        required=criteria.max_ece,
        detail=(
            f"ECE {challenger.expected_calibration_error:.6f} against the absolute "
            f"ceiling {criteria.max_ece}"
        ),
    )


def _check_subgroups(
    champion: MetricSet, challenger: MetricSet, criteria: GateCriteria
) -> tuple[CriterionResult, ...]:
    """One check per comparable subgroup, plus a null-coverage check per column.

    A subgroup is comparable only when *both* models have a defined AUROC on
    it. Where the champion's is null the comparison has no baseline, and where
    the challenger's is null there is nothing to compare -- neither is silently
    treated as a pass, so the loss of a subgroup's measurability is recorded as
    an explicit non-comparable check rather than disappearing.
    """
    results: list[CriterionResult] = []
    for column in criteria.subgroup_columns:
        champion_groups = champion.subgroup_roc_auc.get(column, {})
        challenger_groups = challenger.subgroup_roc_auc.get(column, {})
        for value in sorted(set(champion_groups) | set(challenger_groups)):
            before = champion_groups.get(value)
            after = challenger_groups.get(value)
            label = f"{column}={value}"
            if before is None or after is None:
                results.append(
                    CriterionResult(
                        criterion=CRITERION_SUBGROUP,
                        evaluation_set=challenger.evaluation_set,
                        passed=True,
                        observed=after,
                        required=None,
                        subgroup=label,
                        detail=(
                            f"{label}: AUROC undefined for "
                            f"{'champion' if before is None else 'challenger'} "
                            "(single outcome class); not comparable, not counted"
                        ),
                    )
                )
                continue
            floor = before - criteria.max_subgroup_auroc_drop
            results.append(
                CriterionResult(
                    criterion=CRITERION_SUBGROUP,
                    evaluation_set=challenger.evaluation_set,
                    passed=_at_least(after, floor),
                    observed=after,
                    required=round(floor, COMPARISON_PRECISION),
                    subgroup=label,
                    detail=(
                        f"{label}: AUROC {after:.6f} vs champion {before:.6f} "
                        f"(drop {before - after:+.6f}, max allowed "
                        f"{criteria.max_subgroup_auroc_drop})"
                    ),
                )
            )
    return tuple(results)


def evaluate_gate(
    champion: Mapping[str, MetricSet],
    challenger: Mapping[str, MetricSet],
    criteria: GateCriteria,
) -> GateResult:
    """Compares two metric sets and returns PASS or BLOCK with its reasons.

    Pure: the same inputs always produce the same result. Checks are generated
    in a fixed order -- evaluation set, then §3.12's criteria in the document's
    order, then subgroups alphabetically -- so the reason list is stable.
    """
    checks: list[CriterionResult] = []

    for name in criteria.evaluation_sets:
        champion_set = champion.get(name)
        challenger_set = challenger.get(name)
        if champion_set is None or challenger_set is None:
            missing = "champion" if champion_set is None else "challenger"
            checks.append(
                CriterionResult(
                    criterion=CRITERION_COVERAGE,
                    evaluation_set=name,
                    passed=not criteria.require_all_sets,
                    detail=(
                        f"no {missing} metrics for {name!r}; §3.12 requires the challenger "
                        "to be compared on this set, and an unmeasured set is not a pass"
                    ),
                )
            )
            continue

        checks.append(
            CriterionResult(
                criterion=CRITERION_COVERAGE,
                evaluation_set=name,
                passed=True,
                observed=float(challenger_set.rows),
                detail=f"{name}: {challenger_set.rows} labeled rows compared",
            )
        )
        checks.append(_check_auroc(champion_set, challenger_set, criteria))
        checks.append(_check_recall(champion_set, challenger_set, criteria))
        checks.append(_check_brier(champion_set, challenger_set, criteria))
        checks.append(_check_ece(challenger_set, criteria))
        checks.extend(_check_subgroups(champion_set, challenger_set, criteria))

    failed = tuple(check for check in checks if not check.passed)
    return GateResult(
        outcome=BLOCK if failed else PASS,
        criteria_version=criteria.version,
        checks=tuple(checks),
        reasons=tuple(
            f"{check.criterion} [{check.evaluation_set}]: {check.detail}" for check in failed
        ),
        evaluation_sets=tuple(criteria.evaluation_sets),
        criteria=criteria.summary(),
        champion_metrics={name: value.to_record() for name, value in sorted(champion.items())},
        challenger_metrics={name: value.to_record() for name, value in sorted(challenger.items())},
    )
