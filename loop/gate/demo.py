"""The deliberately bad challenger, for the clickable demo.

WORKFLOW.md §5 step 6: "Re-run with the deliberately bad challenger -> **gate
BLOCK** -> export the audit PDF for the blocked card." Week 8 built this as a
test fixture (`tests/gate/conftest.py::InvertedChallenger`) and said wiring it
into the demo was Week 10's job. This is that wiring, and it keeps the
fixture's defining property: the challenger is bad **by construction**, not by
an unlucky seed. It reverses the champion's ranking, so it fails
discrimination, top-decile recall, calibration and every comparable subgroup
at once, deterministically.

What it is not allowed to be mistaken for:

* It is labelled `demo-bad` on its `retrain_runs` row -- not `live`, not
  `replay`.
* It is never trained, never logged to MLflow and never registered, so it has
  no registry version and can never hold an alias. Its `challenger_version`
  is a descriptive label (`inverted-v<champion>`), which also keeps the Week 8
  idempotency key meaningful: clicking twice returns the verdict on record.
* It goes through the **same** gate, with the card's pinned criteria. Nothing
  about the bar is relaxed or special-cased for it -- the BLOCK is the gate
  working, not the demo scripting a result.
"""

from typing import Any

import numpy as np

from ml.retrain import MODE_DEMO_BAD, ChallengerRun

BAD_CHALLENGER_PREFIX = "inverted-v"


class InvertedModel:
    """Reverses a model's ranking: every probability p becomes 1 - p."""

    def __init__(self, model: Any):
        self._model = model

    def predict_proba(self, frame):
        good = np.asarray(self._model.predict_proba(frame), dtype=float)[:, 1]
        bad = 1.0 - good
        return np.column_stack([1.0 - bad, bad])


def bad_challenger(
    champion_model: Any, champion_version: str | None, *, data_version: str | None = None
) -> ChallengerRun:
    """The constructed bad challenger for one champion. Deterministic; no I/O."""
    return ChallengerRun(
        mode=MODE_DEMO_BAD,
        calibrated_model=InvertedModel(champion_model),
        base_model=None,
        mlflow_run=None,
        registered_version=f"{BAD_CHALLENGER_PREFIX}{champion_version}",
        data_version=data_version,
        lineage={
            "construction": "champion ranking inverted (p -> 1 - p)",
            "constructed_from_champion": str(champion_version),
            "trained": "false",
            "registered": "false",
        },
    )
