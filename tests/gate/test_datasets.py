"""The two evaluation sets §3.12 gates on.

The contract here is reproducibility. A gate result is only re-derivable if the
rows it was computed on can be found again, so the "most recent labeled window"
is the window the *card* names -- a deterministic slice of a seeded stream --
rather than whatever happens to be newest when the gate runs.

These tests use a synthetic stream, so they need neither `future_stream.csv`
nor `eval_frozen.csv`, which is also how Week 6's own tests stay CI-safe.
"""

from datetime import UTC, datetime, timedelta

import pytest

from loop.gate.criteria import FROZEN_HOLDOUT, RECENT_LABELED_WINDOW
from loop.gate.datasets import (
    EvaluationSetError,
    evaluation_frames,
    frozen_holdout,
    labeled_window,
)
from loop.monitor.config import BENCHMARK_EPOCH
from ml.data.clean import TARGET_COLUMN

WINDOW_ROWS = 50
DURATION = timedelta(days=1)


@pytest.fixture(scope="module")
def stream():
    """A synthetic serving stream, long enough for a real `WINDOW_ROWS` window.

    Sized past `loop.monitor.config.WINDOW_ROWS` so the default window geometry
    -- the one production actually uses -- can be exercised, not only the small
    one these tests pass explicitly.
    """
    from loop.monitor.config import WINDOW_ROWS as PRODUCTION_WINDOW_ROWS
    from tests.conftest import make_synthetic_clean_df

    return make_synthetic_clean_df(n_rows=PRODUCTION_WINDOW_ROWS + 500, seed=23)


def window_kwargs(**overrides):
    defaults = {"window_rows": WINDOW_ROWS, "epoch": BENCHMARK_EPOCH, "duration": DURATION}
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# (a) the frozen holdout
# ---------------------------------------------------------------------------
def test_the_frozen_holdout_is_read_from_the_week_two_slice(tmp_path, synthetic_clean_df):
    path = tmp_path / "eval_frozen.csv"
    synthetic_clean_df.to_csv(path, index=False)

    frame = frozen_holdout(path)
    assert len(frame) == len(synthetic_clean_df)
    assert TARGET_COLUMN in frame.columns


def test_a_missing_frozen_holdout_fails_loudly(tmp_path):
    with pytest.raises(EvaluationSetError, match="dvc repro"):
        frozen_holdout(tmp_path / "absent.csv")


def test_an_unlabeled_frozen_holdout_is_refused(tmp_path, synthetic_clean_df):
    path = tmp_path / "eval_frozen.csv"
    synthetic_clean_df.drop(columns=[TARGET_COLUMN]).to_csv(path, index=False)
    with pytest.raises(EvaluationSetError, match="carries no"):
        frozen_holdout(path)


# ---------------------------------------------------------------------------
# (b) the most recent labeled window
# ---------------------------------------------------------------------------
def test_a_window_is_the_slice_the_card_names(stream):
    frame = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    assert len(frame) == WINDOW_ROWS
    assert TARGET_COLUMN in frame.columns


def test_the_window_carries_its_labels(stream):
    """The gate compares models on labeled data; an unlabeled window is useless."""
    frame = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    assert frame[TARGET_COLUMN].notna().all()


def test_a_later_window_is_a_different_slice(stream):
    first = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    second = labeled_window("live", BENCHMARK_EPOCH + DURATION, stream=stream, **window_kwargs())
    assert list(first.index) != list(second.index)


def test_the_same_window_is_returned_every_time(stream):
    """Determinism: re-gating a card next month compares the same rows."""
    first = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    second = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    assert first.equals(second)


def test_a_scenario_window_has_the_injection_applied(stream):
    """The window the drift event measured, not a similar slice of clean traffic."""
    live = labeled_window("live", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    injected = labeled_window("S1", BENCHMARK_EPOCH, stream=stream, **window_kwargs())

    assert len(injected) == len(live)
    assert not injected["num_lab_procedures"].equals(live["num_lab_procedures"])


def test_the_scenario_seed_is_week_sixs_seed(stream):
    """Same seed, same injected window -- which is what makes the benchmark replayable."""
    from scenarios.injection import SCENARIO_SEED

    default = labeled_window("S1", BENCHMARK_EPOCH, stream=stream, **window_kwargs())
    explicit = labeled_window(
        "S1", BENCHMARK_EPOCH, stream=stream, seed=SCENARIO_SEED, **window_kwargs()
    )
    assert default.equals(explicit)


def test_a_window_outside_the_stream_fails_loudly(stream):
    with pytest.raises(EvaluationSetError, match="no window starting"):
        labeled_window("live", datetime(2030, 1, 1, tzinfo=UTC), stream=stream, **window_kwargs())


# ---------------------------------------------------------------------------
# Both together
# ---------------------------------------------------------------------------
def test_evaluation_frames_builds_both_sets(tmp_path, stream, synthetic_clean_df):
    path = tmp_path / "eval_frozen.csv"
    synthetic_clean_df.to_csv(path, index=False)

    frames = evaluation_frames("live", BENCHMARK_EPOCH, stream=stream, eval_path=path)
    assert set(frames) == {FROZEN_HOLDOUT, RECENT_LABELED_WINDOW}
    assert all(TARGET_COLUMN in frame.columns for frame in frames.values())


def test_only_the_requested_sets_are_built(tmp_path, synthetic_clean_df):
    path = tmp_path / "eval_frozen.csv"
    synthetic_clean_df.to_csv(path, index=False)

    frames = evaluation_frames("live", BENCHMARK_EPOCH, sets=(FROZEN_HOLDOUT,), eval_path=path)
    assert set(frames) == {FROZEN_HOLDOUT}


def test_an_unknown_set_is_refused(tmp_path, synthetic_clean_df):
    path = tmp_path / "eval_frozen.csv"
    synthetic_clean_df.to_csv(path, index=False)
    with pytest.raises(EvaluationSetError, match="unknown evaluation set"):
        evaluation_frames("live", BENCHMARK_EPOCH, sets=("something_else",), eval_path=path)
