"""The gate criteria artifact, and its agreement with the documents.

Two things are asserted here that no other test can assert. First, that
`configs/gate-v1.yaml` still says what ARCHITECTURE.md §3.12 and
PROJECT_DESIGN.md §4.5 say -- a promotion bar that drifted away from the
specification silently would be the worst kind of bug this project can have.
Second, that it agrees with the `acceptance_criteria` block every policy pins
onto a card, so the two artifacts that both claim to hold the gate's numbers
cannot diverge.
"""

import pytest
import yaml

from loop.engine.policy import available_policies, load_policy
from loop.gate.criteria import (
    CRITERION_KEYS,
    DEFAULT_CRITERIA_VERSION,
    FROZEN_HOLDOUT,
    RECENT_LABELED_WINDOW,
    CriteriaError,
    available_criteria,
    criteria_path,
    from_card,
    load_criteria,
    parse_criteria,
)

# ARCHITECTURE.md §3.12 / PROJECT_DESIGN.md §4.5, transcribed here so the test
# fails if either the file or the document's reading of it changes.
DOCUMENTED_CRITERIA = {
    "auroc_non_inferiority_margin": 0.005,
    "recall_top_decile_min_ratio": 1.0,
    "max_brier_increase": 0.005,
    "max_ece": 0.05,
    "max_subgroup_auroc_drop": 0.01,
}


@pytest.fixture
def document():
    return yaml.safe_load(criteria_path().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Agreement with the specification
# ---------------------------------------------------------------------------
def test_the_shipped_criteria_are_the_documented_ones():
    assert load_criteria().thresholds() == DOCUMENTED_CRITERIA


def test_every_shipped_policy_pins_the_same_criteria_onto_its_cards():
    """A card's `acceptance_criteria` and `configs/gate-v1.yaml` are one bar.

    §3.9 records the criteria on the card at decision time and §10 keeps a
    criteria file in `configs/`; if those two ever disagreed, "the challenger
    met the acceptance criteria" would have two different meanings.
    """
    for version in available_policies():
        assert load_policy(version).acceptance_criteria == DOCUMENTED_CRITERIA


def test_the_documented_subgroup_columns_are_the_ones_week_three_measured():
    """§3.12's "(age, gender, race)" -- the same tuple `ml/config.py` reports on."""
    from ml.config import SUBGROUP_COLUMNS

    assert load_criteria().subgroup_columns == SUBGROUP_COLUMNS


def test_both_evaluation_sets_are_required():
    """§3.12 gates on a frozen holdout *and* the most recent labeled window."""
    criteria = load_criteria()
    assert criteria.evaluation_sets == (FROZEN_HOLDOUT, RECENT_LABELED_WINDOW)
    assert criteria.require_all_sets is True


def test_the_top_decile_fraction_matches_the_published_operating_point():
    from ml.config import TOP_DECILE_FRACTION

    assert load_criteria().top_decile_fraction == TOP_DECILE_FRACTION


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
def test_the_default_version_is_on_disk():
    assert DEFAULT_CRITERIA_VERSION in available_criteria()


def test_a_criteria_file_names_itself():
    """`gate_result.criteria_version` must name a file a reader can open."""
    for version in available_criteria():
        assert load_criteria(version).version == version


def test_loading_an_unknown_version_fails_loudly():
    with pytest.raises(CriteriaError, match="No gate criteria file"):
        load_criteria("gate-v999")


def test_a_filename_version_mismatch_is_refused(tmp_path, document):
    (tmp_path / "gate-v9.yaml").write_text(
        yaml.safe_dump({**document, "version": "gate-v8"}), encoding="utf-8"
    )
    with pytest.raises(CriteriaError, match="does not match its filename"):
        load_criteria("gate-v9", directory=tmp_path)


def test_invalid_yaml_is_refused(tmp_path):
    (tmp_path / "gate-v9.yaml").write_text("criteria: [unclosed", encoding="utf-8")
    with pytest.raises(CriteriaError, match="not valid YAML"):
        load_criteria("gate-v9", directory=tmp_path)


def test_a_non_mapping_document_is_refused():
    with pytest.raises(CriteriaError, match="mapping at the top level"):
        parse_criteria(["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# Validation: a bar the gate could not defend is refused at load time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["version", "criteria", "evaluation"])
def test_a_missing_top_level_key_is_refused(document, key):
    document.pop(key)
    with pytest.raises(CriteriaError, match=f"missing required key '{key}'"):
        parse_criteria(document)


@pytest.mark.parametrize("key", CRITERION_KEYS)
def test_a_missing_criterion_is_refused_rather_than_defaulted(document, key):
    """There is no default threshold anywhere in `loop/gate/`."""
    document["criteria"].pop(key)
    with pytest.raises(CriteriaError, match=f"missing required key '{key}'"):
        parse_criteria(document)


@pytest.mark.parametrize(
    "key", ["sets", "require_all_sets", "subgroup_columns", "top_decile_fraction"]
)
def test_a_missing_evaluation_setting_is_refused(document, key):
    document["evaluation"].pop(key)
    with pytest.raises(CriteriaError, match=f"missing required key '{key}'"):
        parse_criteria(document)


def test_an_empty_version_is_refused(document):
    document["version"] = ""
    with pytest.raises(CriteriaError, match="empty version"):
        parse_criteria(document)


def test_a_negative_auroc_margin_is_refused(document):
    document["criteria"]["auroc_non_inferiority_margin"] = -0.01
    with pytest.raises(CriteriaError, match="must not be negative"):
        parse_criteria(document)


def test_a_negative_brier_allowance_is_refused(document):
    document["criteria"]["max_brier_increase"] = -0.01
    with pytest.raises(CriteriaError, match="max_brier_increase must not be negative"):
        parse_criteria(document)


def test_a_negative_subgroup_allowance_is_refused(document):
    document["criteria"]["max_subgroup_auroc_drop"] = -0.01
    with pytest.raises(CriteriaError, match="max_subgroup_auroc_drop must not be negative"):
        parse_criteria(document)


@pytest.mark.parametrize("value", [0.0, 1.5])
def test_an_out_of_range_ece_ceiling_is_refused(document, value):
    document["criteria"]["max_ece"] = value
    with pytest.raises(CriteriaError, match="max_ece must be within"):
        parse_criteria(document)


def test_a_non_positive_recall_ratio_is_refused(document):
    document["criteria"]["recall_top_decile_min_ratio"] = 0.0
    with pytest.raises(CriteriaError, match="recall_top_decile_min_ratio must be positive"):
        parse_criteria(document)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_an_out_of_range_top_decile_fraction_is_refused(document, value):
    document["evaluation"]["top_decile_fraction"] = value
    with pytest.raises(CriteriaError, match="top_decile_fraction must be within"):
        parse_criteria(document)


def test_an_empty_evaluation_set_list_is_refused(document):
    document["evaluation"]["sets"] = []
    with pytest.raises(CriteriaError, match="must not be empty"):
        parse_criteria(document)


def test_an_unknown_evaluation_set_is_refused(document):
    document["evaluation"]["sets"] = ["some_other_slice"]
    with pytest.raises(CriteriaError, match="unknown evaluation set"):
        parse_criteria(document)


def test_empty_subgroup_columns_are_refused(document):
    """Subgroup non-regression is blocking, so it cannot be configured away."""
    document["evaluation"]["subgroup_columns"] = []
    with pytest.raises(CriteriaError, match="subgroup_columns must not be empty"):
        parse_criteria(document)


# ---------------------------------------------------------------------------
# A card's own criteria
# ---------------------------------------------------------------------------
def test_a_card_supplies_the_thresholds_and_the_file_supplies_the_protocol():
    card_criteria = {**DOCUMENTED_CRITERIA, "auroc_non_inferiority_margin": 0.02}
    criteria = from_card(card_criteria, card_id="dc-2026-01-01-aaaabbbb")

    assert criteria.auroc_non_inferiority_margin == 0.02
    assert criteria.evaluation_sets == load_criteria().evaluation_sets
    assert criteria.subgroup_columns == load_criteria().subgroup_columns


def test_a_card_driven_verdict_names_the_card_it_was_judged_by():
    criteria = from_card(DOCUMENTED_CRITERIA, card_id="dc-2026-01-01-aaaabbbb")
    assert criteria.version == "card:dc-2026-01-01-aaaabbbb"


def test_a_card_missing_its_acceptance_criteria_cannot_gate_anything():
    """A challenger is never judged against a bar that was never recorded."""
    with pytest.raises(CriteriaError, match="missing gate criteria"):
        from_card({}, card_id="dc-2026-01-01-aaaabbbb")


def test_a_card_with_an_indefensible_bar_is_refused():
    broken = {**DOCUMENTED_CRITERIA, "max_ece": 0.0}
    with pytest.raises(CriteriaError, match="max_ece must be within"):
        from_card(broken, card_id="dc-2026-01-01-aaaabbbb")


def test_the_summary_records_what_a_reader_needs():
    summary = load_criteria().summary()
    assert summary["criteria_version"] == DEFAULT_CRITERIA_VERSION
    assert set(CRITERION_KEYS) <= set(summary)
    assert summary["evaluation_sets"] == [FROZEN_HOLDOUT, RECENT_LABELED_WINDOW]
