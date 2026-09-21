"""The Week 6 deliverable, end to end, on synthetic data.

IMPLEMENTATION_ROADMAP.md Week 6: "injected drift produces a `drift_events` row
and an Evidently report; no-drift control stays quiet." These tests are that
sentence, run against a small model fitted on the synthetic fixture so CI needs
no dataset, no trained artifact and no database service.

Each scenario is measured once for the module. An Evidently run over 25 columns
is the expensive thing here, and re-running one per assertion would make the
suite slow without testing anything new.
"""

import numpy as np
import pytest
from sqlalchemy import select

from db.models import DriftEvent
from loop.monitor.reference import build_reference
from loop.monitor.runner import (
    StreamUnavailableError,
    load_serving_stream,
    report_path_for,
    run_scenario,
    summarise,
)
from loop.monitor.scoring import ScoringModel, model_inputs

# 4000 rows at 1000 per window keeps per-feature PSI below the 0.10 breach line
# on an undrifted sample. At 300 rows a window, sampling noise alone puts a
# feature over it -- which is the same reason `WINDOW_ROWS` is 2000 in
# `loop/monitor/config.py`.
STREAM_ROWS = 4000
WINDOW_ROWS = 1000


@pytest.fixture(scope="module")
def stream():
    from tests.conftest import make_synthetic_clean_df

    frame = make_synthetic_clean_df(n_rows=STREAM_ROWS, seed=21)
    frame.loc[frame.index[::3], "A1Cresult"] = None
    return frame


@pytest.fixture(scope="module")
def model(stream):
    """A real fitted pipeline, small enough to be cheap: no second model path."""
    from ml.data.features import split_features_target
    from ml.train import build_base_model

    X, y = split_features_target(stream)
    base = build_base_model(n_estimators=5)
    base.fit(X, y)
    return ScoringModel(model=base, version="test-model", source="local")


@pytest.fixture(scope="module")
def reference(stream, model):
    half = STREAM_ROWS // 2
    return build_reference(stream.iloc[:half], model, rows=half, data_version="test-data")


@pytest.fixture(scope="module")
def serving(stream):
    return stream.iloc[STREAM_ROWS // 2 :]


def _run(scenario, reference, model, serving, **kwargs):
    kwargs.setdefault("reports_dir", None)
    return run_scenario(
        scenario,
        reference=reference,
        model=model,
        stream=serving,
        window_rows=WINDOW_ROWS,
        **kwargs,
    )


@pytest.fixture(scope="module")
def control(reference, model, serving):
    return _run("S5", reference, model, serving)


@pytest.fixture(scope="module")
def covariate_shift(reference, model, serving):
    return _run("S1", reference, model, serving)


@pytest.fixture(scope="module")
def label_drift(reference, model, serving):
    return _run("S4", reference, model, serving)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
def test_a_scenario_produces_one_result_per_window(control):
    assert len(control.results) == 2
    assert [r.window_index for r in control.results] == [0, 1]
    assert all(r.scenario == "S5" for r in control.results)


def test_the_window_count_can_be_capped(reference, model, serving):
    assert len(_run("S5", reference, model, serving, windows=1).results) == 1


def test_the_reference_lineage_lands_on_every_result(control, reference):
    for result in control.results:
        assert result.model_version == "test-model"
        assert result.data_version == "test-data"
        assert result.reference_rows == reference.rows


def test_windows_are_measured_in_order_and_do_not_overlap_in_time(control):
    for earlier, later in zip(control.results, control.results[1:], strict=False):
        assert earlier.window_start < earlier.window_end == later.window_start


def test_the_control_stays_quiet(control):
    """The Week 6 contract for S5, measured against its own era."""
    assert control.quiet
    assert all(r.breaching_feature_count == 0 for r in control.results)
    assert not any(r.prediction_drift for r in control.results)


def test_injected_covariate_drift_is_detected(covariate_shift):
    assert not covariate_shift.quiet
    breaching = {s.feature for r in covariate_shift.results for s in r.breaching_features}
    assert "num_lab_procedures" in breaching


def test_injected_label_drift_measures_the_same_as_the_control(label_drift, control):
    """S4 moves labels only -- input drift must not react, and must not pretend to."""
    assert label_drift.quiet
    assert [r.max_psi for r in label_drift.results] == [r.max_psi for r in control.results]


def test_the_live_scenario_measures_the_untransformed_stream(reference, model, serving, control):
    live = _run("live", reference, model, serving)
    assert [r.max_psi for r in live.results] == [r.max_psi for r in control.results]
    assert live.results[0].scenario == "live"


def test_a_replayed_scenario_measures_to_the_same_numbers(
    reference, model, serving, covariate_shift
):
    """Determinism: a reviewer re-running S1 must get the numbers in this repo."""
    replay = _run("S1", reference, model, serving)
    assert [r.max_psi for r in replay.results] == [r.max_psi for r in covariate_shift.results]
    assert [r.prediction_psi for r in replay.results] == [
        r.prediction_psi for r in covariate_shift.results
    ]


# ---------------------------------------------------------------------------
# Persisting
# ---------------------------------------------------------------------------
def test_without_a_session_nothing_is_written(control):
    assert control.persisted_event_ids == ()


def test_every_window_is_persisted_including_the_quiet_ones(reference, model, serving, drift_db):
    outcome = _run("S5", reference, model, serving, session=drift_db)
    rows = list(drift_db.scalars(select(DriftEvent)))
    assert len(rows) == len(outcome.results) == 2
    assert {row.event_id for row in rows} == set(outcome.persisted_event_ids)
    assert all(row.breaching_feature_count == 0 for row in rows)


def test_an_injected_scenario_writes_a_row_carrying_the_breach(reference, model, serving, drift_db):
    _run("S1", reference, model, serving, session=drift_db, windows=1)
    row = drift_db.scalars(select(DriftEvent).where(DriftEvent.scenario == "S1")).one()
    assert row.breaching_feature_count >= 1
    assert row.max_psi >= 0.10
    assert row.window_start < row.window_end
    assert row.policy_thresholds["psi_breach"] == 0.10
    for record in row.feature_stats:
        assert set(record) == {"feature", "kind", "psi", "ks_p_value", "breaching", "severe"}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def test_an_evidently_report_is_written_per_window(reference, model, serving, tmp_path):
    outcome = _run("S1", reference, model, serving, windows=1, reports_dir=tmp_path)
    expected = tmp_path / "s1" / "window_000.html"
    assert expected.exists()
    assert expected.stat().st_size > 0
    assert outcome.results[0].report_uri == expected.as_posix()


def test_the_report_path_is_namespaced_by_scenario_and_window(tmp_path):
    assert report_path_for("S2", 7, tmp_path) == tmp_path / "s2" / "window_007.html"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def test_model_inputs_drop_identifiers_and_labels(stream):
    inputs = model_inputs(stream)
    for forbidden in ("encounter_id", "patient_nbr", "readmitted_30d"):
        assert forbidden not in inputs.columns


def test_scoring_returns_one_probability_per_row(stream, model):
    scores = model.score(stream.head(10))
    assert scores.shape == (10,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_a_missing_serving_stream_is_reported_not_guessed(tmp_path):
    with pytest.raises(StreamUnavailableError):
        load_serving_stream(tmp_path / "nope.csv")


def test_the_summary_names_each_window(covariate_shift):
    text = summarise(covariate_shift)
    assert "window 0" in text
    assert "max_psi=" in text
    assert "prediction_drift=" in text
