"""S1-S5: the seeded drift-injection transformations.

Defined in `project_docs/DATASET_ANALYSIS.md` ("Drift Simulation Plan"):

| Scenario | Mechanism                                            | Simulates                  |
|----------|------------------------------------------------------|----------------------------|
| S1       | shift `num_lab_procedures` / `num_medications`       | new protocol / EHR upgrade |
| S2       | re-map a fraction of HbA1c categories                | coding-practice change     |
| S3       | resample toward elderly + high-utilisation           | population / case-mix drift|
| S4       | degrade the feature-label relationship               | delayed performance decay  |
| S5       | untouched stream                                     | the no-drift control       |

Three properties hold for all five:

* **Seeded.** The same scenario and seed produce a byte-identical stream on any
  machine, which is what makes the benchmark a benchmark.
* **In memory.** No transformed stream is written to disk. The injected data is
  derived from DVC-tracked patient rows, and a second copy on disk would be a
  second copy of clinical data to protect for no benefit.
* **Schema-valid.** Injected values stay inside the domains
  `ml/data/schema.py` declares, so an injected stream is still a stream the
  pipeline would accept.

S4 is deliberately the odd one out: it moves *labels*, not features, so its
input drift is quiet by design. Detecting it needs matured-label performance
monitoring, which ARCHITECTURE.md §3.7 places after the 30-day label latency --
Week 6 ships the injection and measures the (correctly quiet) input signal.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ml.data.clean import TARGET_COLUMN

SCENARIO_SEED = 42

# --- S1 ---------------------------------------------------------------------
# DATASET_ANALYSIS.md: "shift num_lab_procedures distribution (+20%)". The same
# multiplier is applied to num_medications, which the demo script in
# WORKFLOW.md §5 names alongside it.
S1_COLUMNS = ("num_lab_procedures", "num_medications")
S1_SCALE = 1.20
S1_JITTER_SD = 1.0

# --- S2 ---------------------------------------------------------------------
# A coding-practice change: a hospital that previously left HbA1c unrecorded
# starts reporting it. The affected rows are the ones with no result.
S2_COLUMN = "A1Cresult"
S2_FRACTION = 0.40
S2_INTRODUCED_VALUES = (">7", ">8", "Norm")
S2_INTRODUCED_WEIGHTS = (0.25, 0.20, 0.55)

# --- S3 ---------------------------------------------------------------------
# Case-mix drift: resample the window toward older, higher-utilisation patients.
S3_ELDERLY_AGE_BANDS = ("[70-80)", "[80-90)", "[90-100)")
S3_ELDERLY_WEIGHT = 4.0
S3_UTILISATION_WEIGHT = 3.0

# --- S4 ---------------------------------------------------------------------
# Concept drift: the relationship decays, the inputs do not move.
S4_FRACTION = 0.30


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def scenario_s1_covariate_shift(frame: pd.DataFrame, seed: int = SCENARIO_SEED) -> pd.DataFrame:
    """Shift the lab-procedure and medication count distributions upward by 20%."""
    rng = _rng(seed)
    shifted = frame.copy()
    for column in S1_COLUMNS:
        values = shifted[column].to_numpy(dtype=float)
        jitter = rng.normal(0.0, S1_JITTER_SD, size=values.size)
        # Clipped at the column's own observed floor so the injection cannot
        # produce a negative count, which `ml/data/schema.py` forbids.
        shifted[column] = np.clip(np.rint(values * S1_SCALE + jitter), values.min(), None).astype(
            int
        )
    return shifted


def scenario_s2_coding_change(frame: pd.DataFrame, seed: int = SCENARIO_SEED) -> pd.DataFrame:
    """Start recording HbA1c for a fraction of the encounters that had no result."""
    rng = _rng(seed)
    recoded = frame.copy()

    unrecorded = recoded.index[recoded[S2_COLUMN].isna()]
    if len(unrecorded) == 0:
        return recoded

    count = int(round(len(unrecorded) * S2_FRACTION))
    selected = rng.choice(unrecorded, size=count, replace=False)
    recoded.loc[selected, S2_COLUMN] = rng.choice(
        list(S2_INTRODUCED_VALUES), size=count, p=list(S2_INTRODUCED_WEIGHTS)
    )
    return recoded


def scenario_s3_prevalence_shift(frame: pd.DataFrame, seed: int = SCENARIO_SEED) -> pd.DataFrame:
    """Resample the window toward elderly, high-utilisation patients.

    Sampling *with* replacement keeps the window the same size, which matters:
    a window that also changed row count would confound case-mix drift with a
    traffic-volume change.
    """
    rng = _rng(seed)

    weights = np.ones(len(frame), dtype=float)
    weights[frame["age"].isin(S3_ELDERLY_AGE_BANDS).to_numpy()] *= S3_ELDERLY_WEIGHT
    utilisation = (
        frame["number_inpatient"] + frame["number_emergency"] + frame["number_outpatient"]
    ).to_numpy()
    weights[utilisation >= 1] *= S3_UTILISATION_WEIGHT

    selected = rng.choice(len(frame), size=len(frame), replace=True, p=weights / weights.sum())
    resampled = frame.iloc[np.sort(selected)].reset_index(drop=True)
    # Encounter ids are the chronology proxy and must stay unique after a
    # with-replacement draw, so the window keeps its own ordered ids.
    resampled["encounter_id"] = frame["encounter_id"].to_numpy()
    return resampled


def scenario_s4_label_drift(frame: pd.DataFrame, seed: int = SCENARIO_SEED) -> pd.DataFrame:
    """Decay the feature-label relationship, leaving every feature untouched.

    Labels are reassigned at the window's own base rate for a random fraction of
    rows, which severs the association the champion learned without changing
    class prevalence -- so the damage is invisible to input drift and shows up
    only once the 30-day labels mature.
    """
    rng = _rng(seed)
    degraded = frame.copy()
    if TARGET_COLUMN not in degraded.columns:
        raise KeyError(
            f"{TARGET_COLUMN!r} not found; S4 degrades labels and needs a labelled stream"
        )

    count = int(round(len(degraded) * S4_FRACTION))
    if count == 0:
        return degraded

    selected = rng.choice(degraded.index.to_numpy(), size=count, replace=False)
    base_rate = float(degraded[TARGET_COLUMN].mean())
    degraded.loc[selected, TARGET_COLUMN] = (rng.random(count) < base_rate).astype(int)
    return degraded


def scenario_s5_no_drift(frame: pd.DataFrame, seed: int = SCENARIO_SEED) -> pd.DataFrame:
    """The control. Untouched -- and it must stay that way.

    S5 is the scenario the thresholds are calibrated on
    (IMPLEMENTATION_ROADMAP.md Week 6: "calibrate PSI thresholds on the no-drift
    control week-6, not demo-day"), so anything done to the stream here would
    quietly invalidate every other scenario's result.
    """
    return frame.copy()


SCENARIOS: dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]] = {
    "S1": scenario_s1_covariate_shift,
    "S2": scenario_s2_coding_change,
    "S3": scenario_s3_prevalence_shift,
    "S4": scenario_s4_label_drift,
    "S5": scenario_s5_no_drift,
}

SCENARIO_DESCRIPTIONS = {
    "S1": "covariate shift -- num_lab_procedures / num_medications +20%",
    "S2": "coding change -- HbA1c recorded for 40% of previously blank encounters",
    "S3": "prevalence shift -- resampled toward elderly, high-utilisation patients",
    "S4": "label drift -- feature/label relationship degraded, features untouched",
    "S5": "no-drift control -- the untouched stream",
}


def apply_scenario(
    frame: pd.DataFrame, scenario: str, *, seed: int = SCENARIO_SEED
) -> pd.DataFrame:
    """Applies one scenario by name. Never mutates the caller's frame."""
    key = scenario.upper()
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; expected one of {', '.join(SCENARIOS)}")
    return SCENARIOS[key](frame, seed)
