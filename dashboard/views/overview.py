"""Overview: model health, versions, traffic (ARCHITECTURE.md §3.14) -- and the demo path.

WORKFLOW.md §5 step 1: "Dashboard shows healthy champion; predictions flowing,
each stamped with model + data version." The traffic button sends real,
authenticated `/predict` calls, so "flowing" means audit rows, not a chart of
invented numbers.
"""

import pandas as pd
import streamlit as st

from dashboard import actions, data
from dashboard.api_client import ApiError
from dashboard.context import alias_versions

DEMO_PATH = """
1. **Overview** -- the champion is serving; send some traffic.
2. **Drift Monitor** -- *Inject drift* (S1: `num_lab_procedures` shift). The monitor
   measures the next window and the Decision Engine emits a narrated card.
3. **Decision Cards** -- read the card and its plain-English rationale. On this data
   the S1 card escalates, so it waits for a person.
4. **Approvals** -- authorise the retrain. **Champion vs Challenger** -- *Retrain + gate*
   (replay, labelled as such) -> gate PASS -> shadow.
5. **Overview** -- send 50+ requests to fill the shadow window. **Approvals** -- approve
   the promotion; the champion version changes live.
6. **Champion vs Challenger** -- re-run the card with the *deliberately bad challenger*
   -> gate BLOCK. **Audit** -- export the audit PDF for the card.
"""


def render(ctx) -> None:
    st.title("Overview")

    try:
        ready = ctx.api.ready()
    except ApiError as error:
        ready = {"status": "unreachable", "model_version": None}
        st.error(error.detail)
    aliases = alias_versions()

    with ctx.session() as session:
        view = data.overview(session)

    cols = st.columns(4)
    cols[0].metric(
        "Serving (API)",
        f"v{ready.get('model_version')}" if ready.get("model_version") else "-",
        help=f"API status: {ready.get('status')}",
    )
    cols[1].metric("champion alias", _v(aliases.get("champion")))
    cols[2].metric("shadow alias", _v(aliases.get("shadow")))
    cols[3].metric("challenger alias", _v(aliases.get("challenger")))

    cols = st.columns(4)
    cols[0].metric("Predictions audited", view.predictions)
    cols[1].metric("Successful", view.successful_predictions)
    cols[2].metric(
        "Median latency (ms)", f"{view.median_latency_ms:.0f}" if view.median_latency_ms else "-"
    )
    cols[3].metric("Shadow scores", view.shadow_scores)

    cols = st.columns(3)
    cols[0].metric("Drift windows", view.drift_windows)
    cols[1].metric("Windows with a breach", view.breaching_windows)
    cols[2].metric("Decision Cards", view.cards)

    left, right = st.columns(2)
    with left:
        st.subheader("Traffic by model version")
        if view.predictions_by_version:
            st.bar_chart(pd.Series(view.predictions_by_version, name="predictions"))
        else:
            st.caption("No predictions yet.")
        if view.last_prediction_at:
            st.caption(f"Last prediction: {view.last_prediction_at:%Y-%m-%d %H:%M:%S}")
    with right:
        st.subheader("Cards by lifecycle state")
        st.dataframe(
            pd.DataFrame(list(view.cards_by_state.items()), columns=["state", "cards"]),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Demo traffic")
    st.caption(
        "Replays encounters from the held-out serving stream through `/predict` under your "
        "token. Each one writes an audit row; while a shadow is loaded, each is also shadow-scored."
    )
    count = st.number_input(
        "Requests", min_value=1, max_value=200, value=55, step=5, key="traffic_count"
    )
    if st.button("Send requests", key="send_traffic"):
        with st.spinner("Scoring..."):
            result = actions.send_demo_traffic(ctx.api, count=int(count), offset=view.predictions)
        if result.failed:
            st.warning(
                f"{result.scored} scored, {result.failed} failed: {'; '.join(result.errors)}"
            )
        else:
            st.success(f"{result.scored} scored by model version(s) {', '.join(result.versions)}")

    with st.expander("The 3-minute demo path", expanded=False):
        st.markdown(DEMO_PATH)


def _v(version) -> str:
    return f"v{version}" if version else "unset"
