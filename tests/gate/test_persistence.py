"""`retrain_runs`: the fourth table, and the history row a card's status lives in.

Two contracts matter here. The table is **append-only** -- §4.6 -- so the
status transition §3.12 describes ("card status set to `BLOCKED`") is an
appended run rather than an edit to the immutable card. And the write is
**idempotent**, like `decision_cards` before it, so a worker that died between
gating and persisting is safe to restart.
"""

import pytest
from sqlalchemy import inspect

from db.models import DecisionCard as DecisionCardRow
from db.models import RetrainRun
from loop.gate.criteria import load_criteria
from loop.gate.gate import BLOCK, PASS, evaluate_gate
from loop.gate.persistence import (
    RetrainPersistenceError,
    build_retrain_row,
    cards_awaiting_retrain,
    find_existing,
    latest_run_for,
    persist_run,
    record_retrain_run,
    retrain_run_id,
    runs_for_card,
)
from ml.retrain import MODE_LIVE, MODE_REPLAY

from .conftest import both_sets, seed_card

CARD_ID = "dc-2026-01-01-aaaabbbb"


@pytest.fixture
def criteria():
    return load_criteria()


@pytest.fixture
def passing_result(criteria):
    return evaluate_gate(both_sets(), both_sets(), criteria)


@pytest.fixture
def blocking_result(criteria):
    return evaluate_gate(both_sets(), both_sets(roc_auc=0.40), criteria)


def row_for(result, *, card_id=CARD_ID, mode=MODE_LIVE, version="2", **overrides):
    return build_retrain_row(
        run_id=retrain_run_id(card_id, mode, version),
        card_id=card_id,
        mode=mode,
        gate_result=result,
        challenger_version=version,
        **overrides,
    )


# ---------------------------------------------------------------------------
# The run id
# ---------------------------------------------------------------------------
def test_the_run_id_is_deterministic():
    assert retrain_run_id(CARD_ID, MODE_LIVE, "2") == retrain_run_id(CARD_ID, MODE_LIVE, "2")


def test_the_run_id_names_the_card_it_belongs_to():
    """A reader ties a run to its decision without a join."""
    assert CARD_ID.removeprefix("dc-") in retrain_run_id(CARD_ID, MODE_LIVE, "2")


@pytest.mark.parametrize(
    "first,second",
    [
        ((CARD_ID, MODE_LIVE, "2"), (CARD_ID, MODE_LIVE, "3")),
        ((CARD_ID, MODE_LIVE, "2"), (CARD_ID, MODE_REPLAY, "2")),
        ((CARD_ID, MODE_LIVE, "2"), ("dc-2026-01-02-ccccdddd", MODE_LIVE, "2")),
    ],
)
def test_a_different_challenger_mode_or_card_gets_its_own_id(first, second):
    assert retrain_run_id(*first) != retrain_run_id(*second)


# ---------------------------------------------------------------------------
# The row
# ---------------------------------------------------------------------------
def test_the_row_projects_the_verdict_rather_than_recomputing_it(blocking_result):
    row = row_for(blocking_result)
    assert row.outcome == blocking_result.outcome == BLOCK
    assert row.criteria_version == blocking_result.criteria_version
    assert row.failed_criteria_count == len(blocking_result.failed_checks())
    assert row.gate_result == blocking_result.to_record()


def test_a_pass_row_records_pass(passing_result):
    assert row_for(passing_result).outcome == PASS


def test_the_table_is_created_by_the_documented_init_path(gate_db):
    tables = set(inspect(gate_db.get_bind()).get_table_names())
    assert "retrain_runs" in tables


def test_the_stored_columns_match_the_model(gate_db):
    columns = {c["name"] for c in inspect(gate_db.get_bind()).get_columns("retrain_runs")}
    assert columns == set(RetrainRun.__table__.columns.keys())
    # The §4.6 ERD's own column list.
    assert {"run_id", "card_id", "mlflow_run", "data_version", "gate_result", "outcome"} <= columns


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_a_verdict_is_written_once(gate_db, card_row, blocking_result):
    first, created = record_retrain_run(gate_db, row_for(blocking_result))
    assert created and first.outcome == BLOCK

    second, created_again = record_retrain_run(gate_db, row_for(blocking_result))
    assert not created_again
    assert second.run_id == first.run_id
    assert gate_db.query(RetrainRun).count() == 1


def test_re_gating_the_same_challenger_does_not_append_a_second_verdict(
    gate_db, card_row, passing_result, blocking_result
):
    """Idempotency is about the challenger, not about the verdict."""
    record_retrain_run(gate_db, row_for(passing_result))
    stored, created = record_retrain_run(gate_db, row_for(blocking_result))

    assert not created
    assert stored.outcome == PASS
    assert gate_db.query(RetrainRun).count() == 1


def test_a_different_challenger_on_the_same_card_is_a_distinct_run(
    gate_db, card_row, passing_result, blocking_result
):
    """What the deliberately-bad-challenger demonstration needs."""
    record_retrain_run(gate_db, row_for(passing_result, version="2"))
    record_retrain_run(gate_db, row_for(blocking_result, version="3"))

    runs = runs_for_card(gate_db, card_row.card_id)
    assert len(runs) == 2
    assert {run.outcome for run in runs} == {PASS, BLOCK}


def test_a_replay_of_the_same_challenger_is_a_distinct_run(gate_db, card_row, passing_result):
    """Replay is labeled, so it can never be mistaken for the live run."""
    record_retrain_run(gate_db, row_for(passing_result, mode=MODE_LIVE))
    record_retrain_run(gate_db, row_for(passing_result, mode=MODE_REPLAY))

    assert {run.mode for run in runs_for_card(gate_db, card_row.card_id)} == {
        MODE_LIVE,
        MODE_REPLAY,
    }


def test_a_write_failure_is_raised_rather_than_swallowed(gate_db, card_row, passing_result):
    row = row_for(passing_result)
    row.outcome = None  # NOT NULL
    with pytest.raises(RetrainPersistenceError):
        persist_run(gate_db, row)


def test_finding_an_existing_run(gate_db, card_row, passing_result):
    assert find_existing(gate_db, card_row.card_id, MODE_LIVE, "2") is None
    record_retrain_run(gate_db, row_for(passing_result))
    assert find_existing(gate_db, card_row.card_id, MODE_LIVE, "2") is not None


# ---------------------------------------------------------------------------
# The restart check
# ---------------------------------------------------------------------------
def test_the_latest_run_is_found_by_card_mode_and_data_version(gate_db, card_row, passing_result):
    record_retrain_run(gate_db, row_for(passing_result, data_version="dvc-train-hash"))

    assert latest_run_for(gate_db, card_row.card_id) is not None
    assert latest_run_for(gate_db, card_row.card_id, mode=MODE_LIVE) is not None
    assert latest_run_for(gate_db, card_row.card_id, data_version="dvc-train-hash") is not None
    assert latest_run_for(gate_db, card_row.card_id, mode=MODE_REPLAY) is None
    assert latest_run_for(gate_db, card_row.card_id, data_version="other") is None


def test_no_run_for_an_untouched_card(gate_db, card_row):
    assert latest_run_for(gate_db, card_row.card_id) is None
    assert runs_for_card(gate_db, card_row.card_id) == ()


# ---------------------------------------------------------------------------
# The backlog
# ---------------------------------------------------------------------------
def test_a_card_awaiting_a_retrain_is_found(gate_db, card_row):
    assert [row.card_id for row in cards_awaiting_retrain(gate_db)] == [card_row.card_id]


def test_a_gated_card_leaves_the_backlog(gate_db, card_row, passing_result):
    record_retrain_run(gate_db, row_for(passing_result))
    assert cards_awaiting_retrain(gate_db) == ()


@pytest.mark.parametrize("action", ["NO_OP", "ALERT_ONLY"])
def test_a_card_that_authorises_nothing_is_never_in_the_backlog(gate_db, action):
    seed_card(
        gate_db,
        card_id="dc-2026-01-02-ccccdddd",
        action=action,
        disposition="NONE",
    )
    assert cards_awaiting_retrain(gate_db) == ()


def test_an_escalated_card_is_not_swept_up_automatically(gate_db):
    """WORKFLOW.md §4: nothing retrains until an ops user acts."""
    seed_card(
        gate_db,
        card_id="dc-2026-01-02-ccccdddd",
        disposition="ESCALATE_HUMAN",
        confidence=0.46,
    )
    assert cards_awaiting_retrain(gate_db) == ()
    assert len(cards_awaiting_retrain(gate_db, automated_only=False)) == 1


def test_the_backlog_can_be_filtered_and_limited(gate_db, card_row):
    seed_card(gate_db, card_id="dc-2026-01-02-ccccdddd")

    assert len(cards_awaiting_retrain(gate_db)) == 2
    assert len(cards_awaiting_retrain(gate_db, limit=1)) == 1
    assert len(cards_awaiting_retrain(gate_db, scenario="S1")) == 2
    assert cards_awaiting_retrain(gate_db, scenario="S9") == ()
    assert cards_awaiting_retrain(gate_db, policy_version="policy-v0") == ()


# ---------------------------------------------------------------------------
# Append-only: the card is never edited
# ---------------------------------------------------------------------------
def test_a_block_leaves_the_decision_card_untouched(gate_db, card_row, blocking_result):
    """§3.12 closes a blocked card; §4.6 says that transition appends a row.

    So the card's own `status` and `card_json` are exactly what Week 7 wrote,
    and the BLOCK is read by joining to `retrain_runs`.
    """
    before_status = card_row.status
    before_json = dict(card_row.card_json)

    record_retrain_run(gate_db, row_for(blocking_result))
    gate_db.expire_all()

    after = gate_db.get(DecisionCardRow, card_row.card_id)
    assert after.status == before_status
    assert after.card_json == before_json
    assert runs_for_card(gate_db, card_row.card_id)[0].outcome == BLOCK


def test_no_module_in_loop_gate_issues_an_update_or_delete():
    """§4.6's immutability claim, kept by there being no code that could break it.

    A blocked challenger "closes" its card by appending a `retrain_runs` row, so
    no module here may mutate a row that is already written. Docstrings and
    comments are stripped before the check, because these modules *discuss*
    UPDATE at length -- it is the executable code that must not contain one.
    """
    import ast
    from pathlib import Path

    from ml.config import PROJECT_ROOT

    forbidden = (
        ".delete(",  # session.delete / Query.delete
        "sqlalchemy.delete",
        "sqlalchemy.update",
        "delete from ",  # raw SQL
        "update ",
    )

    def executable_code(text: str) -> str:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
        # `unparse` drops comments, so only real statements survive.
        return ast.unparse(tree).lower()

    for path in sorted((PROJECT_ROOT / "loop" / "gate").glob("*.py")):
        code = executable_code(Path(path).read_text(encoding="utf-8"))
        found = [token for token in forbidden if token in code]
        assert not found, f"{path.name} contains {found}"
