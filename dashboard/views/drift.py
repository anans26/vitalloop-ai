"""Drift Monitor: Evidently panels per window (ARCHITECTURE.md §3.14), and the injection button.

The button is the roadmap's "drift-injection button wired to S1/S2" and
WORKFLOW.md §5 step 2. It measures the next unmeasured window of the chosen
scenario and decides it on the spot -- see `dashboard/actions.py` for why it
can never measure the same window twice.
"""

import pandas as pd
import streamlit as st

from dashboard import actions, data

SCENARIO_NOTES = {
    "S1": "covariate shift: num_lab_procedures (+20%) and num_medications",
    "S2": "coding change: a fraction of HbA1c results re-mapped",
}


def render(ctx) -> None:
    st.title("Drift Monitor")

    with st.container(border=True):
        st.subheader("Inject drift")
        scenario = st.radio(
            "Scenario",
            actions.INJECTABLE_SCENARIOS,
            format_func=lambda s: f"{s} - {SCENARIO_NOTES.get(s, '')}",
            horizontal=True,
            key="inject_scenario",
        )
        if st.button("Inject drift: measure and decide the next window", key="inject"):
            with st.spinner(f"Measuring the next {scenario} window..."):
                try:
                    with ctx.session() as session:
                        result = actions.inject_next_window(session, scenario)
                except actions.ActionError as error:
                    st.error(str(error))
                    result = None
            if result is not None:
                st.success(
                    f"{scenario} window {result.window_index} recorded ({result.event_id}). "
                    f"Card **{result.card_id}**: {result.action} / {result.disposition} "
                    f"(confidence {result.confidence:.4f})."
                )

    with ctx.session() as session:
        available = data.scenarios(session)
        if not available:
            st.info("No monitoring window has been recorded yet.")
            return
        chosen = st.selectbox("Stream", available, key="drift_scenario")
        events = data.drift_events(session, chosen)

    table = pd.DataFrame(
        [
            {
                "window": f"{e.window_start:%Y-%m-%d}",
                "event": e.event_id,
                "max PSI": e.max_psi,
                "breaching features": e.breaching_feature_count,
                "prediction PSI": e.prediction_psi,
                "prediction drift": e.prediction_drift,
                "rows": e.current_rows,
                "recorded": f"{e.created_at:%Y-%m-%d %H:%M}",
            }
            for e in events
        ]
    )
    st.line_chart(table.set_index("event")[["max PSI", "prediction PSI"]])
    st.dataframe(table, hide_index=True, width="stretch")

    event_id = st.selectbox("Window detail", [e.event_id for e in events], key="drift_event")
    event = next(e for e in events if e.event_id == event_id)
    stats = pd.DataFrame(event.feature_stats or [])
    if not stats.empty:
        st.subheader("Per-feature statistics")
        st.dataframe(stats.sort_values("psi", ascending=False), hide_index=True, width="stretch")

    st.subheader("Evidently report")
    path = data.report_path(event)
    if path is None:
        st.caption(
            "No Evidently report on disk for this window "
            f"({event.report_uri or 'none recorded'}); "
            "the statistics above are the persisted record."
        )
    else:
        st.iframe(path, height=900)
