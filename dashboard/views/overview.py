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
0. **Start clean** (terminal, before the audience arrives): `python -m scripts.reset_demo --yes`
   archives the current state, wipes it and reseeds; on a fresh stack `python -m scripts.seed_demo`
   alone does the seeding. The *Demo state* panel above must say **Replay ready**.
1. **Overview** -- the champion is serving; send some traffic.
2. **Drift Monitor** -- *Inject drift* (S1: `num_lab_procedures` shift). The monitor
   measures the next window and the Decision Engine emits a narrated card.
3. **Decision Cards** -- read the card and its plain-English rationale. On this data
   the S1 card escalates, so it waits for a person.
4. **Approvals** -- authorise the retrain. **Champion vs Challenger** -- *Retrain + gate*
   (replay: re-registers the seeded cached challenger, labelled REPLAY) -> gate PASS -> shadow.
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

    st.subheader("Demo state")
    ready, message = replay_status(aliases)
    (st.success if ready else st.warning)(message)
    with st.expander("The 3-minute demo path", expanded=False):
        st.markdown(DEMO_PATH)


def replay_status(aliases: dict) -> tuple[bool, str]:
    """Whether step 4's replay can produce a promotable challenger, and why.

    Replay re-registers whatever `challenger` points at (ARCHITECTURE.md §3.11).
    After a promotion that is the serving champion, which can never be shadowed
    or promoted -- so the fast path needs a cached challenger that is *not* the
    champion, which is exactly what `scripts.seed_demo` provides.
    """
    champion, challenger = aliases.get("champion"), aliases.get("challenger")
    if champion is None:
        return False, (
            "No champion is registered: this stack has not been seeded. "
            "Run `python -m scripts.seed_demo`."
        )
    if challenger is not None and challenger != champion:
        return True, (
            f"Replay ready: the cached challenger v{challenger} is not the serving champion "
            f"(v{champion}). A replay re-registers it and is recorded as REPLAY, never as a "
            "retrain."
        )
    detail = "none is cached" if challenger is None else f"v{challenger} is already the champion"
    return False, (
        f"Replay unavailable ({detail}): use a live retrain (~70-90 s), or reseed a cached "
        "challenger with `python -m scripts.seed_demo` (`python -m scripts.reset_demo --yes` "
        "for a clean state; it archives first)."
    )


def _v(version) -> str:
    return f"v{version}" if version else "unset"
