"""Schema initialisation: `init_db` is the only documented way a table appears.

`db/session.py` uses `create_all` rather than a migration system, which is a
deliberate choice while the schema is still being added to week by week. The
cost of that choice is that "the model exists" and "the table exists" are two
different facts, and only the second one matters to a running system. These
tests assert the second.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from db.models import Base, DriftEvent, Prediction
from db.session import init_db

EXPECTED_TABLES = {"predictions", "drift_events"}


@pytest.fixture
def fresh_engine(tmp_path: Path):
    """An empty database that has never seen this project's schema."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    yield engine
    engine.dispose()


def test_a_fresh_database_starts_with_no_tables(fresh_engine):
    assert inspect(fresh_engine).get_table_names() == []


def test_init_db_creates_every_table_in_the_metadata(fresh_engine):
    init_db(fresh_engine)
    assert set(inspect(fresh_engine).get_table_names()) >= EXPECTED_TABLES


def test_drift_events_is_registered_in_the_shared_metadata():
    """A model in its own module is only created if it is on the same Base."""
    assert DriftEvent.__table__.metadata is Base.metadata
    assert Prediction.__table__.metadata is Base.metadata
    assert EXPECTED_TABLES <= set(Base.metadata.tables)


def test_init_db_creates_the_drift_events_columns(fresh_engine):
    init_db(fresh_engine)
    columns = {c["name"] for c in inspect(fresh_engine).get_columns("drift_events")}
    assert columns == set(DriftEvent.__table__.columns.keys())
    assert {"event_id", "window_start", "window_end", "scenario", "feature_stats"} <= columns


def test_init_db_creates_the_composite_scenario_index(fresh_engine):
    """The index the dashboard's per-scenario timeline query depends on."""
    init_db(fresh_engine)
    found = inspect(fresh_engine).get_indexes("drift_events")
    indexes = {index["name"]: index["column_names"] for index in found}
    assert "ix_drift_events_scenario_created" in indexes
    assert indexes["ix_drift_events_scenario_created"] == ["scenario", "created_at"]


def test_event_id_is_the_primary_key(fresh_engine):
    init_db(fresh_engine)
    assert inspect(fresh_engine).get_pk_constraint("drift_events")["constrained_columns"] == [
        "event_id"
    ]


def test_init_db_is_safe_to_run_again_on_an_existing_database(fresh_engine):
    """The worker and the API both call it at startup; the second call must be a no-op."""
    init_db(fresh_engine)
    before = set(inspect(fresh_engine).get_table_names())
    init_db(fresh_engine)
    assert set(inspect(fresh_engine).get_table_names()) == before


def test_init_db_adds_drift_events_to_a_database_that_predates_week_6(fresh_engine):
    """The upgrade path for the existing Postgres instance: predictions only, then both."""
    Prediction.__table__.create(bind=fresh_engine)
    assert set(inspect(fresh_engine).get_table_names()) == {"predictions"}

    init_db(fresh_engine)
    assert set(inspect(fresh_engine).get_table_names()) >= EXPECTED_TABLES


def test_init_db_leaves_existing_rows_alone(fresh_engine):
    """`create_all` must not be a reset: Week 5 audit rows survive a Week 6 startup."""
    from sqlalchemy.orm import Session

    Prediction.__table__.create(bind=fresh_engine)
    with Session(fresh_engine) as session:
        session.add(
            Prediction(
                request_id="req-1",
                caller="dr-test",
                caller_role="clinician",
                model_name="test",
                model_version="1",
                model_source="local",
                input_hash="0" * 64,
                status="success",
            )
        )
        session.commit()

    init_db(fresh_engine)
    with Session(fresh_engine) as session:
        assert session.query(Prediction).count() == 1
