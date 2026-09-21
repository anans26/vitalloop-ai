"""Drift monitoring: the system's senses.

`project_docs/PROJECT_DESIGN.md` §6 describes this layer as "the system's
senses; every window persisted". That second clause is the design: a window in
which nothing drifted is a `drift_events` row with
`breaching_feature_count = 0`, not an absent row, because Week 7's persistence
rule ("same features breach for >= 2 consecutive windows") can only be
evaluated against an unbroken history of windows.

Reading order:

* `config.py`     -- monitored features, thresholds, window geometry
* `psi.py`        -- hand-rolled PSI/KS, the independent cross-check
* `windows.py`    -- slicing a serving stream into monitoring windows
* `reference.py`  -- the training reference every window is compared against
* `scoring.py`    -- champion scores, for prediction drift
* `drift.py`      -- the Evidently run and the aggregates it produces
* `persistence.py`-- aggregates -> a `drift_events` row
* `runner.py`     -- one window, or a whole scenario, end to end
* `worker.py`     -- the APScheduler worker that runs `runner` on a schedule
"""
