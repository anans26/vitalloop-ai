"""The validation gate, as a pure function (ARCHITECTURE.md §3.12).

Every criterion in §3.12 / §4.5 is tested on both sides of its own boundary,
and every threshold is expressed *relative to the criteria under test* rather
than as a literal -- the discipline Week 7's rule tests established, for the
same reason: a test that hard-coded `0.005` would stop testing the branch it
names the moment the criterion moved.

The function is pure, so these tests need no model, no database and no files.
That is the property §3.12 asks for ("trivially unit-testable, which is exactly
what a promotion safety mechanism must be"), and asserting it is the point.
"""

import pytest

from loop.gate.criteria import FROZEN_HOLDOUT, RECENT_LABELED_WINDOW
from loop.gate.gate import (
    BLOCK,
    CRITERION_AUROC,
    CRITERION_BRIER,
    CRITERION_COVERAGE,
    CRITERION_ECE,
    CRITERION_RECALL,
    CRITERION_SUBGROUP,
    PASS,
    evaluate_gate,
)

from .conftest import both_sets, make_metric_set

EPSILON = 1e-4


def criteria_of(result, criterion):
    return [check for check in result.checks if check.criterion == criterion]


def failed_criteria(result):
    return {check.criterion for check in result.failed_checks()}


# ---------------------------------------------------------------------------
# The trivial verdicts
# ---------------------------------------------------------------------------
def test_an_identical_challenger_passes(champion_sets, equal_challenger_sets, criteria):
    """A challenger that matches the champion regresses on nothing."""
    result = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    assert result.outcome == PASS
    assert result.passed and not result.blocked
    assert result.reasons == ()


def test_a_better_challenger_passes(champion_sets, criteria):
    challenger = both_sets(
        roc_auc=0.72,
        recall_at_top_decile=0.30,
        brier_score=0.085,
        expected_calibration_error=0.01,
    )
    assert evaluate_gate(champion_sets, challenger, criteria).outcome == PASS


def test_the_verdict_records_the_criteria_version_that_produced_it(
    champion_sets, equal_challenger_sets, criteria
):
    result = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    assert result.criteria_version == criteria.version


def test_the_gate_is_a_pure_function(champion_sets, equal_challenger_sets, criteria):
    """Same two metric sets, same verdict -- every time, in every field."""
    first = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    second = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    assert first.to_record() == second.to_record()


# ---------------------------------------------------------------------------
# Criterion 1: AUROC non-inferiority
# ---------------------------------------------------------------------------
def test_auroc_exactly_on_the_non_inferiority_floor_passes(champion_sets, criteria):
    """§4.5: challenger >= champion - margin. The boundary is inclusive."""
    floor = champion_sets[FROZEN_HOLDOUT].roc_auc - criteria.auroc_non_inferiority_margin
    result = evaluate_gate(champion_sets, both_sets(roc_auc=floor), criteria)
    assert result.outcome == PASS


def test_auroc_below_the_non_inferiority_floor_blocks(champion_sets, criteria):
    floor = champion_sets[FROZEN_HOLDOUT].roc_auc - criteria.auroc_non_inferiority_margin
    result = evaluate_gate(champion_sets, both_sets(roc_auc=floor - EPSILON), criteria)
    assert result.outcome == BLOCK
    assert CRITERION_AUROC in failed_criteria(result)


def test_a_blocked_auroc_reports_both_numbers_and_the_floor(champion_sets, criteria):
    floor = champion_sets[FROZEN_HOLDOUT].roc_auc - criteria.auroc_non_inferiority_margin
    observed = floor - 0.05
    result = evaluate_gate(champion_sets, both_sets(roc_auc=observed), criteria)
    check = next(c for c in criteria_of(result, CRITERION_AUROC) if not c.passed)
    assert check.observed == pytest.approx(observed)
    assert check.required == pytest.approx(floor)
    assert f"{champion_sets[FROZEN_HOLDOUT].roc_auc:.6f}" in check.detail


# ---------------------------------------------------------------------------
# Criterion 2: recall at the top decile
# ---------------------------------------------------------------------------
def test_recall_equal_to_the_champion_passes(champion_sets, criteria):
    """§4.5: challenger >= champion. Equality is not a regression."""
    result = evaluate_gate(
        champion_sets,
        both_sets(recall_at_top_decile=champion_sets[FROZEN_HOLDOUT].recall_at_top_decile),
        criteria,
    )
    assert result.outcome == PASS


def test_any_recall_loss_blocks(champion_sets, criteria):
    """The ratio is 1.0, so §4.5 permits no loss of top-decile recall at all."""
    worse = champion_sets[FROZEN_HOLDOUT].recall_at_top_decile - EPSILON
    result = evaluate_gate(champion_sets, both_sets(recall_at_top_decile=worse), criteria)
    assert result.outcome == BLOCK
    assert CRITERION_RECALL in failed_criteria(result)


# ---------------------------------------------------------------------------
# Criterion 3: Brier score
# ---------------------------------------------------------------------------
def test_brier_exactly_on_the_ceiling_passes(champion_sets, criteria):
    ceiling = champion_sets[FROZEN_HOLDOUT].brier_score + criteria.max_brier_increase
    assert evaluate_gate(champion_sets, both_sets(brier_score=ceiling), criteria).outcome == PASS


def test_brier_above_the_ceiling_blocks(champion_sets, criteria):
    ceiling = champion_sets[FROZEN_HOLDOUT].brier_score + criteria.max_brier_increase
    result = evaluate_gate(champion_sets, both_sets(brier_score=ceiling + EPSILON), criteria)
    assert result.outcome == BLOCK
    assert CRITERION_BRIER in failed_criteria(result)


# ---------------------------------------------------------------------------
# Criterion 4: calibration (an absolute ceiling, not a comparison)
# ---------------------------------------------------------------------------
def test_ece_exactly_on_the_ceiling_passes(champion_sets, criteria):
    result = evaluate_gate(
        champion_sets, both_sets(expected_calibration_error=criteria.max_ece), criteria
    )
    assert result.outcome == PASS


def test_ece_above_the_ceiling_blocks_even_when_the_champion_is_worse(criteria):
    """§3.12 makes ECE an absolute ceiling: "<= 0.05 absolute", not "<= champion".

    RISK_ANALYSIS.md §2 is the reason -- a miscalibrated score presented as a
    probability is its own clinical hazard, independent of what it replaced.
    """
    champion = both_sets(expected_calibration_error=criteria.max_ece * 4)
    challenger = both_sets(expected_calibration_error=criteria.max_ece + EPSILON)
    result = evaluate_gate(champion, challenger, criteria)
    assert result.outcome == BLOCK
    assert CRITERION_ECE in failed_criteria(result)


# ---------------------------------------------------------------------------
# Criterion 5: subgroup non-regression (contribution C4)
# ---------------------------------------------------------------------------
def test_a_subgroup_drop_inside_the_allowance_passes(champion_sets, criteria):
    subgroups = {
        column: {value: auc - criteria.max_subgroup_auroc_drop for value, auc in groups.items()}
        for column, groups in champion_sets[FROZEN_HOLDOUT].subgroup_roc_auc.items()
    }
    result = evaluate_gate(champion_sets, both_sets(subgroup_roc_auc=subgroups), criteria)
    assert result.outcome == PASS


def test_one_subgroup_regression_blocks_an_otherwise_better_challenger(champion_sets, criteria):
    """Fairness is a blocking criterion, not a dashboard: the average is irrelevant."""
    subgroups = {
        column: dict(groups)
        for column, groups in champion_sets[FROZEN_HOLDOUT].subgroup_roc_auc.items()
    }
    subgroups["race"]["AfricanAmerican"] -= criteria.max_subgroup_auroc_drop + EPSILON

    result = evaluate_gate(
        champion_sets,
        both_sets(roc_auc=0.75, recall_at_top_decile=0.40, subgroup_roc_auc=subgroups),
        criteria,
    )
    assert result.outcome == BLOCK
    failed = [c for c in criteria_of(result, CRITERION_SUBGROUP) if not c.passed]
    assert {check.subgroup for check in failed} == {"race=AfricanAmerican"}


def test_an_incomparable_subgroup_is_recorded_rather_than_silently_skipped(champion_sets, criteria):
    """A subgroup that loses its AUROC stays visible instead of vanishing."""
    subgroups = {
        column: dict(groups)
        for column, groups in champion_sets[FROZEN_HOLDOUT].subgroup_roc_auc.items()
    }
    subgroups["gender"]["Male"] = None

    result = evaluate_gate(champion_sets, both_sets(subgroup_roc_auc=subgroups), criteria)
    check = next(c for c in criteria_of(result, CRITERION_SUBGROUP) if c.subgroup == "gender=Male")
    assert check.passed and check.required is None
    assert "not comparable" in check.detail
    assert result.outcome == PASS


def test_every_configured_subgroup_column_is_checked(
    champion_sets, equal_challenger_sets, criteria
):
    result = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    checked = {check.subgroup.split("=")[0] for check in criteria_of(result, CRITERION_SUBGROUP)}
    assert checked == set(criteria.subgroup_columns)


# ---------------------------------------------------------------------------
# Both evaluation sets (§3.12(a) and (b))
# ---------------------------------------------------------------------------
def test_every_criterion_is_evaluated_on_every_configured_set(
    champion_sets, equal_challenger_sets, criteria
):
    result = evaluate_gate(champion_sets, equal_challenger_sets, criteria)
    for criterion in (CRITERION_AUROC, CRITERION_RECALL, CRITERION_BRIER, CRITERION_ECE):
        sets = {check.evaluation_set for check in criteria_of(result, criterion)}
        assert sets == set(criteria.evaluation_sets)


def test_a_regression_on_the_live_window_alone_blocks(champion_sets, criteria):
    """RISK_ANALYSIS.md §2: gating on the frozen holdout alone overfits to it."""
    challenger = both_sets()
    floor = champion_sets[RECENT_LABELED_WINDOW].roc_auc - criteria.auroc_non_inferiority_margin
    challenger[RECENT_LABELED_WINDOW] = make_metric_set(RECENT_LABELED_WINDOW, roc_auc=floor - 0.05)

    result = evaluate_gate(champion_sets, challenger, criteria)
    assert result.outcome == BLOCK
    assert {check.evaluation_set for check in result.failed_checks()} == {RECENT_LABELED_WINDOW}


def test_a_regression_on_the_frozen_holdout_alone_blocks(champion_sets, criteria):
    challenger = both_sets()
    floor = champion_sets[FROZEN_HOLDOUT].roc_auc - criteria.auroc_non_inferiority_margin
    challenger[FROZEN_HOLDOUT] = make_metric_set(FROZEN_HOLDOUT, roc_auc=floor - 0.05)

    result = evaluate_gate(champion_sets, challenger, criteria)
    assert result.outcome == BLOCK
    assert {check.evaluation_set for check in result.failed_checks()} == {FROZEN_HOLDOUT}


def test_a_missing_challenger_set_blocks(champion_sets, criteria):
    """ "No evidence" is never "no regression": an unmeasured set cannot pass."""
    result = evaluate_gate(champion_sets, {FROZEN_HOLDOUT: make_metric_set()}, criteria)
    assert result.outcome == BLOCK
    assert CRITERION_COVERAGE in failed_criteria(result)
    assert any("challenger" in reason for reason in result.reasons)


def test_a_missing_champion_set_blocks(equal_challenger_sets, criteria):
    result = evaluate_gate({FROZEN_HOLDOUT: make_metric_set()}, equal_challenger_sets, criteria)
    assert result.outcome == BLOCK
    assert any("champion" in reason for reason in result.reasons)


def test_a_missing_set_is_tolerated_when_the_criteria_do_not_require_all(champion_sets, criteria):
    relaxed = type(criteria)(**{**criteria.__dict__, "require_all_sets": False})
    result = evaluate_gate(champion_sets, {FROZEN_HOLDOUT: make_metric_set()}, relaxed)
    assert result.outcome == PASS


# ---------------------------------------------------------------------------
# The BLOCK explanation the roadmap asks for
# ---------------------------------------------------------------------------
def test_a_blocked_challenger_reports_one_reason_per_failed_check(champion_sets, criteria):
    """ "bad challenger BLOCKED with reasons" -- the roadmap's Week 8 deliverable."""
    challenger = both_sets(roc_auc=0.40, recall_at_top_decile=0.05, brier_score=0.30)
    result = evaluate_gate(champion_sets, challenger, criteria)

    assert result.outcome == BLOCK
    assert len(result.reasons) == len(result.failed_checks())
    assert {CRITERION_AUROC, CRITERION_RECALL, CRITERION_BRIER} <= failed_criteria(result)
    for reason in result.reasons:
        assert any(name in reason for name in criteria.evaluation_sets)


def test_the_reason_order_is_stable(champion_sets, criteria):
    challenger = both_sets(roc_auc=0.40, recall_at_top_decile=0.05)
    first = evaluate_gate(champion_sets, challenger, criteria).reasons
    second = evaluate_gate(champion_sets, challenger, criteria).reasons
    assert first == second


def test_the_record_is_self_contained(champion_sets, criteria):
    """An examiner re-derives the verdict from `gate_result` alone."""
    challenger = both_sets(roc_auc=0.40)
    record = evaluate_gate(champion_sets, challenger, criteria).to_record()

    assert record["outcome"] == BLOCK
    assert record["criteria"]["auroc_non_inferiority_margin"] == (
        criteria.auroc_non_inferiority_margin
    )
    assert set(record["champion_metrics"]) == set(criteria.evaluation_sets)
    assert set(record["challenger_metrics"]) == set(criteria.evaluation_sets)
    assert record["checks"] and record["reasons"]


def test_the_record_carries_no_patient_data(champion_sets, criteria):
    """§6: only aggregate statistics leave the predictions store."""
    record = evaluate_gate(champion_sets, both_sets(), criteria).to_record()
    for metrics in (*record["champion_metrics"].values(), *record["challenger_metrics"].values()):
        assert set(metrics) == {
            "evaluation_set",
            "rows",
            "positive_rate",
            "roc_auc",
            "recall_at_top_decile",
            "brier_score",
            "expected_calibration_error",
            "subgroup_roc_auc",
        }
