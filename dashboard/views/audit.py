"""Audit: searchable log + one-click PDF export (ARCHITECTURE.md §3.14).

One time-ordered log across every audit table -- drift windows, cards, retrain
runs, human decisions, registry alias moves -- searchable by any text in it,
plus a lookup of individual prediction audit rows by request id, input hash or
caller. A prediction row holds a hash of its input, never the input, so the
lookup proves which request produced a score without showing clinical data.

WORKFLOW.md step 18: "one click renders the CMS-style audit PDF for any
Decision Card." The PDF is built from the same rows (`dashboard/audit_pdf.py`).
"""

import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.audit_pdf import render_pdf, report_for_card
from dashboard.context import alias_audit_rows


def render(ctx) -> None:
    st.title("Audit")
    alias_rows = alias_audit_rows()

    st.subheader("Audit log")
    c1, c2 = st.columns([2, 3])
    query = c1.text_input(
        "Search", key="audit_query", placeholder="card id, run id, approver, feature..."
    )
    kinds = c2.multiselect(
        "Kinds", data.EVENT_KINDS, default=list(data.EVENT_KINDS), key="audit_kinds"
    )
    with ctx.session() as session:
        events = data.audit_events(session, alias_rows, query=query, kinds=tuple(kinds))
        card_ids = [card.card_id for card, _ in data.cards_with_state(session)]
    st.dataframe(
        pd.DataFrame(
            [{**e, "ts": f"{e['ts']:%Y-%m-%d %H:%M:%S}" if e["ts"] else ""} for e in events],
            columns=["ts", "kind", "ref", "subject", "summary"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"{len(events)} event(s)")

    st.subheader("Audit PDF")
    if not card_ids:
        st.caption("No Decision Card to report on yet.")
    else:
        card_id = st.selectbox("Decision Card", card_ids, key="audit_card")
        if st.button("Build audit PDF", key="build_pdf"):
            with ctx.session() as session:
                report = report_for_card(session, card_id, alias_rows=alias_rows)
            st.session_state["audit_pdf"] = (card_id, render_pdf(report))
        built = st.session_state.get("audit_pdf")
        if built and built[0] == card_id:
            st.download_button(
                f"Download {card_id}.pdf",
                data=built[1],
                file_name=f"{card_id}.pdf",
                mime="application/pdf",
                key="download_pdf",
            )

    st.subheader("Prediction audit rows")
    key = st.text_input("Request id, input hash, or caller", key="prediction_lookup")
    if key:
        with ctx.session() as session:
            rows = data.find_predictions(session, key)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ts": f"{p.ts:%Y-%m-%d %H:%M:%S}",
                        "request": p.request_id,
                        "caller": p.caller,
                        "model": p.model_version,
                        "data version": p.data_version,
                        "input hash": p.input_hash,
                        "risk": p.risk_score,
                        "status": p.status,
                        "top factors": ", ".join(f["feature"] for f in (p.top_shap or [])),
                    }
                    for p in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if not rows:
            st.caption("No prediction audit row matches.")
