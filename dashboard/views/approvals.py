"""Approvals: pending promotions, approve/reject (ARCHITECTURE.md §3.14) -- and escalated retrains.

§3.13: "Promotion to `champion` requires a **human click in the dashboard** (ops
role), which writes an approval row (who, when, card reference)." This is that
click. RISK_ANALYSIS.md §3 adds how it must look: "the full evidence chain
(card, gate result, shadow stats, subgroup deltas), not a bare 'Approve'
button; rejection is one click and fully logged." So every decision here shows
the evidence first and requires a written reason.

Escalated retrain cards are decided here too. §3.14 lists only promotions, but
WORKFLOW.md §4 ("nothing retrains until an ops user acts") makes the retrain
authorisation a human decision of the same kind, and without it the demo's S1
card -- which escalates -- could not proceed. It stays on this page rather than
becoming a seventh one.

Every button calls the API with the signed-in token: the approver recorded is
the token's subject, and the API enforces every precondition and returns its
refusals verbatim.
"""

import pandas as pd
import streamlit as st

from dashboard.api_client import ApiError


def render(ctx) -> None:
    st.title("Approvals")
    st.caption(f"Decisions are recorded as **{ctx.identity['subject']}**.")

    try:
        retrains = ctx.api.pending_retrains()
        promotions = ctx.api.pending_promotions()
        shadow = ctx.api.shadow()
    except ApiError as error:
        st.error(f"The API refused: {error.detail}")
        return

    st.subheader("Pending promotions")
    if not promotions:
        st.caption("No gated challenger is waiting in shadow.")
    for entry in promotions:
        _promotion(ctx, entry)

    st.subheader("Escalated retrains")
    if not retrains:
        st.caption("No escalated retrain card is waiting for a person.")
    for entry in retrains:
        _retrain(ctx, entry["card"])

    with st.expander("Current shadow window"):
        st.json(shadow)


def _decision_form(key: str) -> tuple[str | None, str]:
    reason = st.text_area(
        "Reason (recorded verbatim; required for approve and reject)",
        key=f"reason_{key}",
        help="Do not include patient information: this text is stored in the audit trail.",
    )
    c1, c2 = st.columns(2)
    if c1.button("Approve", key=f"approve_{key}", type="primary"):
        return "APPROVE", reason
    if c2.button("Reject", key=f"reject_{key}"):
        return "REJECT", reason
    return None, reason


def _promotion(ctx, entry: dict) -> None:
    evidence = entry["evidence"]
    card, gate, window = evidence["card"], evidence["gate"], evidence["shadow_window"]
    title = (
        f"{entry['run_id']} - challenger v{entry['challenger_version']} "
        f"({'ready' if evidence['ready_to_approve'] else 'preconditions unmet'})"
    )
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(
            f"Card {card.get('card_id')} ({card.get('action')}, {card.get('disposition')}, "
            f"confidence {card.get('confidence')}) - gate {gate['outcome']} under "
            f"{gate['criteria_version']}, mode **{gate['mode']}**, "
            f"authorised by {gate.get('authorized_by')}"
        )
        if card.get("narrative"):
            with st.expander("Narrative"):
                st.write(card["narrative"])

        rows = [
            {"set": s, "metric": m, **values}
            for s, metrics in gate["headline"].items()
            for m, values in metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if gate.get("worst_subgroup_auroc_drop"):
            st.caption(
                "Worst subgroup AUROC drop: "
                + "; ".join(
                    f"{k}: {v['subgroup']} {v['auroc_drop']}"
                    for k, v in gate["worst_subgroup_auroc_drop"].items()
                )
            )

        st.markdown(
            f"Shadow window: **{window.get('requests', 0)} / "
            f"{window['required_requests']}** requests - "
            f"decision agreement {window.get('decision_agreement')}, "
            f"mean |diff| {window.get('mean_abs_diff')}, "
            f"Spearman {window.get('spearman')}, flips +{window.get('flips_to_positive', 0)} / "
            f"-{window.get('flips_to_negative', 0)}"
        )
        for check in evidence["preconditions"]:
            st.markdown(
                f"{'✅' if check['satisfied'] else '❌'} `{check['name']}` - {check['detail']}"
            )

        decision, reason = _decision_form(entry["run_id"])
        if decision:
            try:
                result = ctx.api.decide_promotion(entry["run_id"], decision, reason)
            except ApiError as error:
                st.error(f"Refused ({error.status}): {error.detail}")
                return
            if result["champion_alias_moved"]:
                st.success(
                    f"Promoted: champion v{result['champion_version_before']} -> "
                    f"v{result['champion_version_after']}. The API now serves "
                    f"v{(result.get('serving') or {}).get('champion_version')}."
                )
            else:
                st.info(f"{decision} recorded ({result['approval_id']}). Champion unchanged.")


def _retrain(ctx, card: dict) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{card['card_id']}** - {card['scenario']} window {card['window_start'][:10]} - "
            f"{card['action']}, confidence {card['confidence']:.4f} "
            f"({card['policy_version']} rule {card['rule_id']})"
        )
        st.caption(card.get("rationale") or "")
        if card.get("narrative"):
            with st.expander("Narrative"):
                st.write(card["narrative"])
        features = card.get("breaching_features") or []
        if features:
            st.dataframe(pd.DataFrame(features), hide_index=True, width="stretch")

        decision, reason = _decision_form(card["card_id"])
        if decision:
            try:
                result = ctx.api.decide_retrain(card["card_id"], decision, reason)
            except ApiError as error:
                st.error(f"Refused ({error.status}): {error.detail}")
                return
            st.success(
                f"{decision} recorded ({result['approval_id']})."
                + (" Retrain it from Champion vs Challenger." if decision == "APPROVE" else "")
            )
