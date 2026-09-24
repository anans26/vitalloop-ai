"""VitalLoop dashboard: the loop, made visible.

    streamlit run dashboard/app.py          # http://localhost:8501

ARCHITECTURE.md §3.14 names six pages and the roadmap caps it there ("6 pages,
no more"): Overview, Drift Monitor, Decision Cards, Champion vs Challenger,
Approvals, Audit. Each page is a `render(ctx)` function in `dashboard/views/`,
and each reads through `dashboard/data.py` -- the pages hold layout, not
queries -- and writes, where it writes at all, through `dashboard/actions.py`
or the API.
"""

import sys
from pathlib import Path

# `streamlit run dashboard/app.py` puts dashboard/ on the path, not the project
# root; the project's packages (db, loop, ml, api, scripts) live one level up.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.context import authenticate  # noqa: E402
from dashboard.views import approvals, audit, cards, drift, gate, overview  # noqa: E402

PAGES = (
    ("Overview", "overview", overview.render),
    ("Drift Monitor", "drift", drift.render),
    ("Decision Cards", "cards", cards.render),
    ("Champion vs Challenger", "gate", gate.render),
    ("Approvals", "approvals", approvals.render),
    ("Audit", "audit", audit.render),
)

st.set_page_config(page_title="VitalLoop", layout="wide")

ctx = authenticate()
if ctx is None:
    st.title("VitalLoop")
    st.info(
        "Paste an **ops** token in the sidebar to open the dashboard. Issue one with\n\n"
        "`python -m scripts.issue_dev_token --subject <your-name> --role ops`\n\n"
        "The API verifies it; every decision you make here is recorded under that name."
    )
    st.stop()


def _page(render, name: str):
    def run():
        render(ctx)

    run.__name__ = name  # Streamlit identifies callable pages by name
    return run


navigation = st.navigation(
    [st.Page(_page(render, path), title=title, url_path=path) for title, path, render in PAGES]
)
navigation.run()
