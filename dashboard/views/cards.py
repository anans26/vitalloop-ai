"""Decision Cards: list + detail with narrative (ARCHITECTURE.md §3.14).

WORKFLOW.md §5 step 3: the card "renders ... narrative panel explains why in
plain English". The narrative is the one stored with the card; a card emitted
before narration existed shows the deterministic template's rendering, labelled
as such. Either way the grounding verdict is shown beside it.
"""

import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.context import alias_audit_rows


def render(ctx) -> None:
    st.title("Decision Cards")

    with ctx.session() as session:
        scenarios = [""] + data.scenarios(session)
        c1, c2, c3 = st.columns(3)
        scenario = c1.selectbox(
            "Stream", scenarios, format_func=lambda s: s or "all", key="cards_scenario"
        )
        action = c2.selectbox(
            "Action",
            ["", "NO_OP", "ALERT_ONLY", "INCREMENTAL_RETRAIN", "FULL_RETRAIN"],
            format_func=lambda a: a or "all",
            key="cards_action",
        )
        rows = data.cards_with_state(session, scenario=scenario or None, action=action or None)
        states = sorted({state for _, state in rows})
        state = c3.selectbox(
            "State", [""] + states, format_func=lambda s: s or "all", key="cards_state"
        )
        if state:
            rows = [(card, s) for card, s in rows if s == state]

        if not rows:
            st.info("No Decision Card matches.")
            return

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "card": card.card_id,
                        "stream": card.scenario,
                        "window": f"{card.window_start:%Y-%m-%d}",
                        "policy": card.policy_version,
                        "rule": card.rule_id,
                        "action": card.action,
                        "disposition": card.disposition,
                        "confidence": card.confidence,
                        "state": s,
                    }
                    for card, s in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        card_id = st.selectbox(
            "Card detail", [card.card_id for card, _ in rows], key="cards_detail"
        )
        history = data.card_history(session, card_id, alias_audit_rows())

    render_card(history)


def render_card(history: data.CardHistory) -> None:
    from loop.narrate.grounding import check_grounding
    from loop.narrate.template import TEMPLATE_SOURCE, render_template

    card = dict(history.card.card_json or {})
    trigger = card.get("trigger") or {}

    st.subheader(f"{history.card.card_id} - {history.state}")
    cols = st.columns(4)
    cols[0].metric("Action", card.get("action"))
    cols[1].metric("Disposition", card.get("disposition"))
    cols[2].metric("Confidence", f"{card.get('confidence', 0):.4f}")
    cols[3].metric("Rule", f"{card.get('policy_version')} #{card.get('rule_id')}")

    stored = bool(card.get("narrative"))
    text = card["narrative"] if stored else render_template(card)
    source = (
        card.get("narrative_source")
        if stored
        else f"{TEMPLATE_SOURCE} (rendered; card predates narration)"
    )
    with st.container(border=True):
        st.markdown("**Narrative**")
        for paragraph in text.split("\n\n"):
            st.write(paragraph)
        grounded = check_grounding(text, card).grounded
        st.caption(f"Source: {source} - grounding check: {'passed' if grounded else 'FAILED'}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Trigger evidence**")
        st.dataframe(
            pd.DataFrame(trigger.get("breaching_features") or []), hide_index=True, width="stretch"
        )
        st.json(
            {
                k: trigger.get(k)
                for k in (
                    "drift_event_id",
                    "prediction_drift",
                    "prediction_psi",
                    "max_psi",
                    "consecutive_breaching_windows",
                    "label_maturity",
                    "matured_auroc_drop",
                )
            },
            expanded=False,
        )
    with right:
        st.markdown("**Confidence breakdown**")
        st.dataframe(
            pd.Series(card.get("confidence_breakdown") or {}, name="value"), width="stretch"
        )
        st.markdown("**Acceptance criteria** (pinned before any training)")
        st.dataframe(
            pd.Series(card.get("acceptance_criteria") or {}, name="value"), width="stretch"
        )
    st.caption(card.get("rationale") or "")

    if history.runs:
        st.markdown("**Retrain runs**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": r.run_id,
                        "mode": r.mode,
                        "challenger": r.challenger_version,
                        "champion": r.champion_version,
                        "outcome": r.outcome,
                        "failed checks": r.failed_criteria_count,
                        "authorised by": r.authorized_by,
                    }
                    for r in history.runs
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if history.approvals:
        st.markdown("**Human decisions**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "when": f"{a.ts:%Y-%m-%d %H:%M}",
                        "kind": a.kind,
                        "decision": a.decision,
                        "approver": a.approver,
                        "reason": a.reason,
                    }
                    for a in history.approvals
                ]
            ),
            hide_index=True,
            width="stretch",
        )
