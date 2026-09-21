"""Window geometry: contiguous, non-overlapping, position-derived bounds."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from loop.monitor.config import BENCHMARK_EPOCH, WINDOW_DURATION
from loop.monitor.windows import iter_windows, window_bounds


def _stream(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"encounter_id": range(rows), "value": range(rows)})


def test_windows_are_contiguous_and_cover_the_stream_in_order():
    windows = iter_windows(_stream(600), window_rows=200)
    assert [w.index for w in windows] == [0, 1, 2]
    assert [w.rows for w in windows] == [200, 200, 200]
    assert windows[0].frame["encounter_id"].tolist()[:2] == [0, 1]
    assert windows[2].frame["encounter_id"].tolist()[-1] == 599


def test_no_window_overlaps_the_next_one():
    windows = iter_windows(_stream(600), window_rows=200)
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert earlier.end == later.start
        assert set(earlier.frame["encounter_id"]).isdisjoint(later.frame["encounter_id"])


def test_a_trailing_partial_window_is_dropped():
    """PSI on a short sample is noise; a measured stub would read as a false trigger."""
    windows = iter_windows(_stream(450), window_rows=200)
    assert [w.rows for w in windows] == [200, 200]


def test_a_partial_window_can_be_kept_explicitly():
    windows = iter_windows(_stream(450), window_rows=200, drop_partial=False)
    assert [w.rows for w in windows] == [200, 200, 50]


def test_max_windows_caps_the_run():
    assert len(iter_windows(_stream(2000), window_rows=200, max_windows=3)) == 3


def test_max_windows_zero_measures_nothing():
    assert iter_windows(_stream(2000), window_rows=200, max_windows=0) == []


def test_a_stream_shorter_than_one_window_yields_nothing():
    assert iter_windows(_stream(50), window_rows=200) == []


def test_window_bounds_are_derived_from_position_not_the_wall_clock():
    """The same window replays to the same bounds tomorrow -- that is the benchmark."""
    start, end = window_bounds(3)
    assert start == BENCHMARK_EPOCH + 3 * WINDOW_DURATION
    assert end == start + WINDOW_DURATION
    assert window_bounds(3) == window_bounds(3)


def test_window_bounds_are_timezone_aware():
    start, end = window_bounds(0)
    assert start.tzinfo is not None
    assert start.utcoffset() == timedelta(0)
    assert end.tzinfo is not None


def test_window_bounds_accept_a_different_epoch_and_duration():
    epoch = datetime(2030, 6, 1, tzinfo=UTC)
    start, end = window_bounds(2, epoch=epoch, duration=timedelta(hours=6))
    assert start == datetime(2030, 6, 1, 12, tzinfo=UTC)
    assert end == datetime(2030, 6, 1, 18, tzinfo=UTC)


@pytest.mark.parametrize("kwargs", [{"window_rows": 0}, {"window_rows": -5}, {"max_windows": -1}])
def test_nonsensical_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        iter_windows(_stream(100), **kwargs)


def test_a_negative_window_index_is_rejected():
    with pytest.raises(ValueError):
        window_bounds(-1)
