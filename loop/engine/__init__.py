"""The Decision Engine: drift evidence in, an auditable Decision Card out.

`project_docs/PROJECT_DESIGN.md` §4.2 calls this "the core of v2", and
`ARCHITECTURE.md` §3.8 says what it must be: "A pure-Python, fully unit-tested
policy module. **No LLM, no randomness, no network.**"

The separation that matters: Week 6's monitor *measures* -- it produces PSI, KS
and prediction-drift numbers and writes them to `drift_events`. This package
*interprets* -- it reads those measurements back and decides what they mean
operationally. No PSI is recomputed here, and no policy threshold lives in
`loop/monitor/`.

Reading order:

* `policy.py`      -- loads and validates `configs/policy-v*.yaml`
* `evidence.py`    -- the engine's input: one window plus its breach history
* `confidence.py`  -- the deterministic confidence formula (§3.8)
* `rules.py`       -- the six-rule table (§3.8), as a pure function
* `card.py`        -- the Pydantic Decision Card, the contract Weeks 8-9 read
* `engine.py`      -- `decide(evidence, policy) -> DecisionCard`, pure
* `history.py`     -- reads `drift_events` and builds evidence
* `persistence.py` -- a card -> a `decision_cards` row, idempotently
* `evaluate.py`    -- the CLI that evaluates windows that have no card yet

`engine.decide` touches no database, no clock and no filesystem: the same
evidence and the same policy produce the same card, byte for byte, which is
what makes RESEARCH_NOVELTY.md C1's "every card can be re-derived by hand"
testable rather than aspirational.
"""
