"""A one-page Streamlit script for `AppTest`: renders `STATE.page` with `STATE.ctx`.

`AppTest` runs a script in this same process, so a test sets the page and the
context (a fake API, a session factory on the test database) on `STATE` and
the script picks them up -- no network, no Postgres, no token.
"""

from types import SimpleNamespace

STATE = SimpleNamespace(page=None, ctx=None)

if __name__ == "__main__":
    import importlib

    from tests.dashboard.page_harness import STATE as shared

    view = importlib.import_module(f"dashboard.views.{shared.page}")
    view.render(shared.ctx)
