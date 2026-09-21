"""What the monitor watches, how often, and at what thresholds.

Plain module constants, for the same reason `ml/config.py` gives: the versioned
`configs/policy-v*.yaml` files described in `project_docs/ARCHITECTURE.md` are a
Week 7 governance artifact, and inventing the policy-versioning convention here
would pre-empt the week that owns it. Every value below is a *measurement*
setting; none of them decides anything.

The thresholds are the ones the Week 7 rule table will read (ARCHITECTURE.md
§3.8): 0.10 is where a feature counts as breaching, 0.25 is where a single
feature is severe on its own. They are recorded on every `drift_events` row
(`policy_thresholds`) so a stored window can always be re-read against the
thresholds that actually produced it.
"""

from datetime import UTC, datetime, timedelta

from ml.config import PROCESSED_DIR, REPORTS_DIR
from ml.data.features import ENGINEERED_NUMERIC_COLUMNS, MEDICATION_COLUMNS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# `ml/config.py` deliberately exposes no constant for the future stream -- that
# is how "reserved for Week 6" was enforced rather than merely documented, and
# it is also a declared DVC stage dependency, so adding a path to it would
# invalidate dvc.lock and force a retrain. Week 6 is the week that opens it, so
# the constant lives here instead.
TRAIN_PATH = PROCESSED_DIR / "train.csv"
EVAL_FROZEN_PATH = PROCESSED_DIR / "eval_frozen.csv"
SERVING_STREAM_PATH = PROCESSED_DIR / "future_stream.csv"

# The reference every window is measured against: the **frozen evaluation
# slice**, not the training slice.
#
# ARCHITECTURE.md §3.7 says "vs the training reference", and that is the right
# instinct -- but on this dataset the training slice is not one distribution. It
# spans 1999-2008 in encounter order, and splitting it in half and measuring the
# halves against each other gives payer_code PSI 1.68, medical_specialty 0.52,
# admission_source_group 0.29: the training data drifts violently *within
# itself*. A serving window compared against that mixture reports the difference
# between eras the model was already trained across, which is not news, and the
# signal is buried: on the S5 control every window then shows 4-7 breaching
# features at PSI 0.7-0.9, swamping an injected S1 shift of 0.30.
#
# The frozen evaluation slice is internally coherent by the same measurement --
# its halves differ by at most 0.054, and any 2000-row window of it scores under
# 0.06 against the whole. It is also the distribution on which the champion's
# published metrics were established and which ARCHITECTURE.md §3.8 rule 5 calls
# "launch", and it is never trained on, so using it introduces no leakage.
#
# Regenerate the comparison with `python -m scenarios.run_scenario S5`.
REFERENCE_PATH = EVAL_FROZEN_PATH

# ---------------------------------------------------------------------------
# Monitored features
# ---------------------------------------------------------------------------
# The engineered feature frame, not the one-hot matrix: PSI on
# `num_lab_procedures` is a sentence a clinician can read, PSI on
# `cat__diag_2_chapter_Neoplasms_onehot_17` is not. These names are also the
# ones the Decision Card quotes in ARCHITECTURE.md §3.9.
MONITORED_NUMERIC_FEATURES: tuple[str, ...] = tuple(ENGINEERED_NUMERIC_COLUMNS)

# The 23 individual medication columns are deliberately *not* monitored. Their
# signal is already carried by `num_med_changes` and `insulin_changed`, and
# almost all of them are ~99% "No", so including them would add 23 permanently
# quiet features to the denominator of the Week 7 `breadth` term -- diluting a
# real two-feature breach from 2/25 to 2/48 for no added evidence.
MONITORED_CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    column
    for column in (
        "race",
        "gender",
        "age",
        "payer_code",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
        "diag_1_chapter",
        "diag_2_chapter",
        "diag_3_chapter",
        "admission_source_group",
    )
    if column not in MEDICATION_COLUMNS
)

MONITORED_FEATURES: tuple[str, ...] = (
    *MONITORED_NUMERIC_FEATURES,
    *MONITORED_CATEGORICAL_FEATURES,
)

# The champion's calibrated output, compared reference-vs-window to give
# ARCHITECTURE.md §3.7's "prediction drift".
PREDICTION_COLUMN = "risk_score"

# ---------------------------------------------------------------------------
# Thresholds (ARCHITECTURE.md §3.8)
# ---------------------------------------------------------------------------
PSI_BREACH_THRESHOLD = 0.10
PSI_SEVERE_THRESHOLD = 0.25
KS_P_VALUE_THRESHOLD = 0.05
PREDICTION_PSI_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------
# The dataset carries no calendar time (DATASET_ANALYSIS.md), so a window is a
# contiguous block of the encounter-ordered serving stream -- the same
# chronology proxy `ml/data/split.py` uses. 2000 rows keeps per-feature PSI
# stable enough that the no-drift control stays quiet, which is the Week 6
# calibration requirement.
WINDOW_ROWS = 2000

# A trailing block shorter than a full window is dropped rather than measured:
# PSI on a short sample is noisy, and a noisy final window would show up as a
# false trigger in the benchmark.
DROP_PARTIAL_WINDOW = True

# Window bounds are derived from the data position, not the wall clock, so the
# same scenario replays to byte-identical `window_start`/`window_end` values on
# any machine. When the row was written is `created_at`'s job.
BENCHMARK_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DURATION = timedelta(days=1)

# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
# A seeded sample of the training slice rather than all ~49k rows: the feature
# histograms are unchanged at this size, and the champion has to score the
# reference once for prediction drift, which is the expensive half.
REFERENCE_SAMPLE_ROWS = 10_000
REFERENCE_SEED = 42

# ---------------------------------------------------------------------------
# Scenarios and artifacts
# ---------------------------------------------------------------------------
# S1-S5 are defined in DATASET_ANALYSIS.md; "live" is the untransformed stream.
LIVE_SCENARIO = "live"

DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
