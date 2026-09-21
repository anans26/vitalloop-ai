"""The seeded drift benchmark: S1-S5.

`project_docs/PROJECT_DESIGN.md` §10 gives this directory its own place in the
tree because the benchmark is a contribution, not a test fixture (C6: "a
reproducible drift-response benchmark"). Every scenario is a seeded, in-memory
transformation of the Week 2 serving stream, so any claimed detection can be
re-run by a reviewer from a clean clone.

    python -m scenarios.run_scenario S1 --windows 3

`injection.py` holds the transformations; `run_scenario.py` is the CLI that
feeds them through the monitor.
"""
