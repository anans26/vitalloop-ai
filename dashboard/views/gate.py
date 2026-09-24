"""Champion vs Challenger: gate results (ARCHITECTURE.md §3.14), and the retrain button.

WORKFLOW.md §5 step 4 -- "Retrain replays a cached run -> challenger appears ->
gate PASS -> shadow" -- and step 6 -- "Re-run with the deliberately bad
challenger -> gate BLOCK". Replay is labelled as replay here, as §3.11 and
RISK_ANALYSIS.md require, and the constructed bad challenger is labelled as
exactly that; neither can be mistaken for a live retrain.
"""

import pandas as pd
import streamlit as st

from dashboard import actions, data
from dashboard.api_client import ApiError
from dashboard.context import alias_versions
from loop.engine.rules import RETRAIN_ACTIONS
from ml.retrain import MODE_DEMO_BAD, MODE_LIVE, MODE_REPLAY

MODES = {
    MODE_REPLAY: "Replay - re-register the cached challenger (seconds; labelled REPLAY)",
    MODE_LIVE: "Live retrain - the champion's own training pipeline on the pinned data (~90 s)",
    MODE_DEMO_BAD: (
        "Deliberately bad challenger - the champion's ranking inverted (demo; never registered)"
    ),
}
MODE_BADGES = {MODE_REPLAY: "REPLAY", MODE_LIVE: "live", MODE_DEMO_BAD: "CONSTRUCTED BAD (demo)"}


def render(ctx) -> None:
    st.title("Champion vs Challenger")

    aliases = alias_versions()
    cols = st.columns(3)
    for col, alias in zip(cols, ("champion", "challenger", "shadow"), strict=True):
        col.metric(alias, f"v{aliases[alias]}" if aliases.get(alias) else "unset")

    _retrain_panel(ctx)

    with ctx.session() as session:
        runs = data.retrain_runs(session)
    if not runs:
        st.info("No retrain has been gated yet.")
        return

    st.subheader("Gate verdicts")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "run": r.run_id,
                    "card": r.card_id,
                    "mode": MODE_BADGES.get(r.mode, r.mode),
                    "challenger": r.challenger_version,
                    "champion": r.champion_version,
                    "verdict": r.outcome,
                    "failed checks": r.failed_criteria_count,
                    "criteria": r.criteria_version,
                    "shadow moved": r.shadow_alias_moved,
                    "authorised by": r.authorized_by,
                    "recorded": f"{r.created_at:%Y-%m-%d %H:%M}",
                }
                for r in runs
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    run_id = st.selectbox("Verdict detail", [r.run_id for r in runs], key="gate_run")
    run = next(r for r in runs if r.run_id == run_id)
    if run.outcome == "PASS":
        st.success(
            f"PASS - {run.mode} challenger {run.challenger_version} "
            f"vs champion {run.champion_version}"
        )
    else:
        st.error(f"BLOCK - {run.failed_criteria_count} failed checks; nothing changed in serving")
        for reason in (run.gate_result or {}).get("reasons") or []:
            st.markdown(f"- {reason}")

    st.markdown("**Champion beside challenger**")
    st.dataframe(pd.DataFrame(data.headline_metrics(run)), hide_index=True, width="stretch")
    with st.expander("Every check the gate made"):
        st.dataframe(pd.DataFrame(data.gate_checks(run)), hide_index=True, width="stretch")


def _retrain_panel(ctx) -> None:
    with st.container(border=True):
        st.subheader("Retrain + gate")
        with ctx.session() as session:
            candidates = [
                (card, state)
                for card, state in data.cards_with_state(session)
                if card.action in RETRAIN_ACTIONS
                and state != data.STATE_AWAITING_AUTHORISATION
                and state != data.STATE_RETRAIN_REJECTED
            ]
        if not candidates:
            st.caption(
                "No card is authorised to retrain. An escalated card needs an ops user's "
                "APPROVE on the Approvals page first."
            )
            return
        labels = {
            card.card_id: f"{card.card_id} - {card.scenario} - {state}"
            for card, state in candidates
        }
        card_id = st.selectbox("Card", list(labels), format_func=labels.get, key="retrain_card")
        mode = st.radio("Challenger", list(MODES), format_func=MODES.get, key="retrain_mode")
        if st.button("Retrain + gate", key="retrain"):
            with st.spinner("Retraining and gating..."):
                try:
                    with ctx.session() as session:
                        run = actions.run_retrain(session, card_id, mode)
                except actions.ActionError as error:
                    st.error(str(error))
                    return
            if run.resumed:
                st.info(f"Already on record: {run.outcome} ({run.run_id}). Nothing was retrained.")
            elif run.passed:
                st.success(
                    f"Gate PASS: {MODE_BADGES[mode]} challenger v{run.challenger_version} "
                    "is now in shadow."
                )
            else:
                st.error(f"Gate BLOCK: {len(run.gate_result.reasons)} reasons. Nothing moved.")
            if run.shadow_alias_moved:
                try:
                    served = ctx.api.reload()
                    st.caption(
                        f"API now serving v{served['champion_version']}, "
                        f"shadowing v{served['shadow_version']}."
                    )
                except ApiError as error:
                    st.warning(f"Shadow moved, but the API did not reload: {error.detail}")
