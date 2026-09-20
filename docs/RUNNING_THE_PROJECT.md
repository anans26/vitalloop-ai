# Running the Project — VitalLoop 2.0

> **Milestone covered by this document: Week 3 of the 12-week roadmap.**
> This document describes only what exists and runs in the repository today. Features planned for Week 4 onward (MLflow, the API, monitoring, the Decision Engine, the dashboard, LLM narration) are **not implemented** and are not covered here — see `PROJECT_REPOSITORY_GUIDE.md` §11 for the full pending list.

---

## 1. Project Overview

**VitalLoop 2.0** is a self-healing MLOps system for predicting 30-day hospital readmission risk, designed around one principle: *deterministic code decides, an LLM only narrates, a human approves anything clinician-facing*. Full design intent lives in `project_docs/` (`PROJECT_DESIGN.md`, `ARCHITECTURE.md`, etc.) — those are planning documents, not a description of current code.

**Current implementation stage:** end of Week 2 of a 12-week roadmap (`project_docs/IMPLEMENTATION_ROADMAP.md`). At this stage the repository contains **only the data ingestion, validation, cleaning, feature-engineering, and versioning layer**, plus a Week 1 baseline model recorded in a notebook. There is no trained production model, no API, no database schema, no monitoring, no decision engine, and no LLM integration of any kind yet.

**Current milestone (Week 2) exit criteria, both met:**
- `dvc repro` deterministically rebuilds the hashed, cleaned, split dataset from the raw CSV.
- The patient-level, time-sliced split design is implemented and unit-tested for zero patient leakage across splits.

---

## 2. Technology Stack

Only technologies **actually present in the repository today** are listed. Everything else named in `project_docs/TECH_STACK.md` (FastAPI, MLflow, LightGBM, Evidently, Streamlit, Ollama, etc.) belongs to later weeks and is not installed or used yet.

| Technology | Version (installed) | Why chosen | Purpose in this project today |
|---|---|---|---|
| **Python** | 3.12.10 (via `py -3.12`) | Stable, well-supported wheel availability for the ML stack; the machine's *default* `python` resolves to 3.14, which is too new for reliable wheels as of this build — a project-local venv pinned to 3.12 avoids that trap | Runtime for all data/ML code |
| **pandas** | 3.0.5 | Standard dataframe library; required by every module in `ml/data/` | Loading, cleaning, transforming tabular data |
| **numpy** | 2.5.1 | pandas/scikit-learn dependency | Numerical operations |
| **scikit-learn** | 1.9.0 | Provides `Pipeline`, `ColumnTransformer`, imputers/encoders, and the baseline `LogisticRegression` | The single train/serve feature-transformation path; Week 1 baseline model |
| **pandera** | 0.32.1 | Schema-as-code validation living next to the pipeline (lighter than Great Expectations) — see `project_docs/TECH_STACK.md` for the full rationale | Validates the raw CSV before anything touches it (`ml/data/schema.py`) |
| **DVC** | 3.67.1 | Hash-pinned data versioning without cloud credentials | Versions the raw and processed datasets via a **local** remote |
| **ucimlrepo** | 0.0.7 | Official UCI ML Repository fetch helper | Downloads the primary dataset programmatically (`ml/data/ingest.py`) |
| **Jupyter / ipykernel** | 1.0+ / 6.29+ | Interactive EDA | Runs `notebooks/01_eda.ipynb` |
| **pytest** | 9.1.1 | Standard Python test runner | 70 tests: Week 2 data contract plus Week 3 model, calibration, and SHAP contracts |
| **ruff** | 0.16.8 | Fast combined linter/formatter | Enforced via `pyproject.toml`, the pre-commit hook, and CI (`check` + `format --check`) |
| **pre-commit** | 4.6.2 | Runs ruff automatically on commit | Installed (`.git/hooks/pre-commit`); hook pinned to ruff v0.16.8 to match the pinned CLI |
| **LightGBM** | 4.7.0 | Gradient boosting is the right tool for mid-size tabular clinical data (`project_docs/TECH_STACK.md`) | The Week 3 readmission classifier |
| **SHAP** | 0.52.0 | Fast, exact attributions for tree models | Global + per-prediction explanations (`ml/explain.py`) |
| **matplotlib** | 3.11.2 | Figure rendering | Calibration curve and SHAP summary PNGs |
| **joblib** | 1.6.0 | sklearn's own serialisation format | Persists the fitted model bundle |
| **Docker + Docker Compose** | 29.5.2 / v5.1.4 | Documented Windows-friction mitigation; offline-by-design | Runs a single `postgres:16` container (empty — no application schema yet) |
| **Git** | 2.53.0 | Version control | Repository initialized; `origin` configured; Week 1–2 history committed |
| **Node.js** | 24.14.0 | Pre-existing in the repo before this build | Used *only* by `project_docs/build_pdf.mjs` to render the planning-doc PDF; unrelated to the application and not required to run anything in this document |

**Backend / Frontend / AI Model / Database schema:** none exist yet. See §12 (Current Limitations).

---

## 3. Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| **Python 3.12** | All data pipeline code | Confirm available via `py -0p` (Windows) before creating the venv; do not use the system default if it resolves to a newer/older version |
| **Git** | Version control | Any recent version |
| **Docker Desktop** (with Compose v2+) | PostgreSQL container | Must be **running** (the Docker Desktop application, not just the CLI) before `docker compose up` |
| **~2 GB free disk** | venv + raw/processed CSVs + DVC cache | The raw dataset alone is ~18 MB; the venv with all packages is the bulk of this |
| **4 GB+ RAM recommended** | Jupyter + pandas in memory | The dataset (101,766 rows) is small; this is a comfortable margin, not a hard requirement |
| **Windows, macOS, or Linux** | — | Built and verified on Windows 11; nothing in the current code is Windows-specific except the encoding note in §11 |
| **Internet access (one-time)** | Downloading the UCI dataset via `ucimlrepo` | Not needed again once `datasets/raw/diabetic_data.csv` exists locally and/or is pulled from the DVC remote |
| **IDE** | Any | VS Code with the Python extension is a reasonable default; no project-specific extensions are required yet |

**Not required yet:** Node.js (unless rebuilding the planning-doc PDF), any Google Cloud account, any Gemini API key, any GPU. See §7 for why.

---

## 4. Required Downloads

| Software | Version used | Download link | Purpose | Verification command |
|---|---|---|---|---|
| Python | 3.12.x | https://www.python.org/downloads/ | Project runtime | `py -3.12 --version` (Windows) or `python3.12 --version` |
| Git | any recent | https://git-scm.com/downloads | Version control | `git --version` |
| Docker Desktop | any recent | https://www.docker.com/products/docker-desktop/ | Runs PostgreSQL locally | `docker --version` && `docker compose version` |
| VS Code *(optional)* | any recent | https://code.visualstudio.com/ | Recommended editor | `code --version` |

Nothing else needs to be downloaded manually — all Python packages install via `pip` from `requirements.txt` (§5).

---

## 5. Repository Setup

```bash
# 1. Clone
git clone https://github.com/anans26/vitalloop-ai.git
cd vitalloop-ai

# 2. Create and activate a Python 3.12 virtual environment
py -3.12 -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt

# 4. Start PostgreSQL (empty container — no schema yet, see §8)
cd docker
cp .env.example .env   # then edit .env and set a real local POSTGRES_PASSWORD
docker compose up -d
cd ..

# 5. Fetch the dataset (choose ONE):
#    (a) Download fresh from UCI:
python -m ml.data.ingest
#    (b) Or, if a DVC remote with the data already exists and is reachable:
dvc pull

# 6. Build the versioned, cleaned dataset
dvc repro
```

There is no separate "frontend" install step, no `package.json` for the application itself, and no build step — this is a pure-Python data pipeline at this stage.

### 5.1 Dependency Management

Two files, each with one job, no duplicated hand-maintained lists:

| File | Role | Maintained by |
|---|---|---|
| `requirements.txt` | **What** the project depends on — direct packages with lower bounds | Edited by hand |
| `constraints.txt` | **Which exact versions** a known-good environment resolved to, including transitive dependencies | Generated by `pip freeze` |

Always install with both, so your environment matches CI exactly:

```bash
pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` is a pip *constraints* file rather than a lockfile on purpose: a constraint pins a package's version only if something actually pulls that package in. The captured set is from Windows and includes `pywin32`/`pywinpty`, which cannot install on Linux — as constraints they are simply ignored there, so **the same file works unchanged in CI on `ubuntu-latest`**. Installing it with `-r` instead would fail on Linux.

**To add or upgrade a dependency:**

```bash
# 1. edit requirements.txt (add the package, or raise its lower bound)
pip install -r requirements.txt --upgrade
pip freeze --exclude-editable > constraints.txt
# 2. restore the explanatory header at the top of constraints.txt
# 3. verify nothing regressed
pytest -q && ruff check . && ruff format --check . && dvc repro
```

Re-run `dvc repro` after any dependency change: a library upgrade can alter the bytes of the processed datasets, and that must show up as a reviewed `dvc.lock` diff rather than a surprise later.

---

## 6. Environment Variables

Only the PostgreSQL container's credentials exist as environment variables today, sourced from `docker/.env` (gitignored — never committed). A template lives at `docker/.env.example` and **is** committed.

| Variable | Purpose | Required value | Example | Mandatory/Optional |
|---|---|---|---|---|
| `POSTGRES_USER` | Local dev DB user | any string | `vitalloop` | Optional — defaults to `vitalloop` if unset |
| `POSTGRES_PASSWORD` | Local dev DB password | any string | `changeme` | **Mandatory** — `docker compose up` fails fast with a clear error if unset, rather than silently using a default password |
| `POSTGRES_DB` | Local dev DB name | any string | `vitalloop` | Optional — defaults to `vitalloop` if unset |

**Setup:** `cp docker/.env.example docker/.env` and set a real local value for `POSTGRES_PASSWORD` before running `docker compose up`. No other environment variables exist yet anywhere in the codebase — the FastAPI service (Week 5) will introduce its own set (e.g. a JWT signing secret) when it's built; do not anticipate those variables here.

---

## 7. API Configuration — Gemini Only

**Mandatory project constraint:** this project uses **only** Google's Gemini API, specifically:

```
model = "gemini-2.5-flash"
```

**Current state as of Week 2: no AI/LLM API integration exists anywhere in the codebase.** There is no narration layer, no Ollama client, no Gemini client, and no API key handling of any kind yet. The LLM narration layer is Week 9 work per `project_docs/IMPLEMENTATION_ROADMAP.md`.

**Explicit exclusions (already true, and to remain true for the life of the project):**
- ❌ No Claude API calls
- ❌ No Anthropic SDK
- ❌ No Claude API keys, anywhere, including in examples or comments
- ❌ No Claude/Anthropic package dependencies in `requirements.txt`

**⚠️ Documented conflict to resolve before Week 9:** `project_docs/ARCHITECTURE.md` (§3.10) and `project_docs/TECH_STACK.md` specify a **self-hosted, offline** LLM (Ollama running Llama 3.1 8B) for the narration layer specifically *because* the docs argue that a cloud API "sends metadata off-box and breaks the offline demo — wrong trade" for a PHI-adjacent system. The mandatory Gemini-only constraint requires a **cloud** API instead. This is a genuine, unresolved conflict between the project's own architecture documents and the AI-model mandate — it does not block anything through Week 2 (no LLM code exists yet), but it must be explicitly decided (and the offline/PHI-safety trade-off re-documented) before Week 9 implementation begins. It is not silently resolved in this document.

---

## 8. Database Setup

- **Creation:** `docker compose up -d` (from `docker/`) starts a `postgres:16` container named `docker-postgres-1`, reachable at `localhost:5432`.
- **Schema:** **none exists yet.** No tables, no SQLAlchemy models, no migrations. The `db/` directory described in `project_docs/ARCHITECTURE.md` §10 (repo structure) has not been created — it is Week 4–5 work (MLflow registry + FastAPI audit rows).
- **Seeding:** not applicable yet — there is nothing to seed.
- **Verification:**
  ```bash
  docker compose ps
  # Expect: docker-postgres-1 ... Up ... 0.0.0.0:5432->5432/tcp
  docker exec -it docker-postgres-1 psql -U vitalloop -d vitalloop -c "\dt"
  # Expect: "Did not find any relations." -- this is correct at Week 2, not an error.
  ```

---

## 9. Running the Project

There is no application server to "run" yet — only the data pipeline. In order:

```bash
# 1. Activate the venv (every new shell)
source .venv/Scripts/activate

# 2. Ensure Docker's Postgres container is up (optional at this stage, but part of the Week 1 deliverable)
cd docker && docker compose up -d && cd ..

# 3. Rebuild/verify the versioned dataset
dvc repro
# Expected output on an unchanged repo:
#   'datasets\raw\diabetic_data.csv.dvc' didn't change, skipping
#   Stage 'build_dataset' didn't change, skipping
#   Data and pipelines are up to date.

# 4. Run the EDA notebook end-to-end (regenerates all outputs)
python -m jupyter execute --inplace notebooks/01_eda.ipynb
```

`dvc repro` now runs four stages: the Week 2 `build_dataset`, then the Week 3
`train_model`, `evaluate_model`, and `explain_model`. A full forced rebuild
(`dvc repro -f`) takes roughly 45 seconds and is byte-reproducible — every
artifact hash is identical run to run.

### 9.1 The Week 3 model workflow

Each stage is also runnable on its own, which is what `dvc repro` calls:

```bash
python -m ml.train      # fits the model, writes models/
python -m ml.evaluate   # scores the frozen eval slice, writes reports/metrics.json
python -m ml.explain    # SHAP global + local explanations, writes reports/
```

**What is trained.** A LightGBM binary classifier on the `readmitted_30d` target,
wrapped in the *same* `ml.data.features` pipeline used everywhere else — Week 3
introduces no second preprocessing path. Two objects are fitted and saved
together in `models/readmission_model.joblib`:

| Object | What it is | Used for |
|---|---|---|
| `base_model` | `Pipeline(features → LGBMClassifier)` | SHAP explanations, raw-probability comparison |
| `calibrated_model` | `CalibratedClassifierCV(base, isotonic, cv=5)` | **inference** — this is the model that produces the reported probabilities |

The tree count is not hand-picked: `select_n_estimators` runs early stopping
against a chronological validation tail taken from the **training** slice (the
last 15% of `train.csv`), and the resulting count is reused for both models. At a
fixed 300 trees the model scored 0.82 AUROC on train against 0.59 on eval; early
stopping cut that to 82 trees and roughly halved the gap. This is one principled
mechanism, not a hyperparameter search — the roadmap caps Week 3 tuning
deliberately.

**Which split is used.**

| Slice | Role in Week 3 |
|---|---|
| `datasets/processed/train.csv` | Model fitting, calibration (via 5-fold CV), and early-stopping validation |
| `datasets/processed/eval_frozen.csv` | Final scoring only — never fitted on |
| `datasets/processed/future_stream.csv` | **Untouched.** Reserved for the Week 6 drift scenarios |

`ml/config.py` deliberately exposes no path constant for `future_stream.csv`, and
a test asserts no Week 3 module contains code referencing it.

**Which metrics are reported.** `reports/metrics.json` carries the full block for
raw *and* calibrated probabilities — ROC-AUC, average precision (PR-AUC), Brier
score, log loss, expected calibration error, recall at the top risk decile, and
classification counts at **two** documented thresholds. No single number is
presented as the score.

The two thresholds matter. At the nominal `0.5` cut the model flags almost
nobody (2 of 10,498 patients), which is the expected behaviour of a calibrated
model on an 8% base rate — not a bug. The **top-decile threshold** (0.139) is the
operating point a readmission worklist actually uses: flag the highest-risk 10%
of patients, capturing 17.1% of true readmissions. Subgroup ROC-AUC by age,
gender, and race is recorded too, because Week 8's promotion gate blocks on
subgroup regression and needs a baseline on record now.

**How calibration works.** `CalibratedClassifierCV` with `method="isotonic"` and
`cv=5`, fitted on the training slice only. Because the wrapper clones and refits
the whole pipeline inside each fold, preprocessing is fitted on fold-train alone
and the isotonic map never sees rows the underlying model was fitted on. The
frozen evaluation slice is read only to compare raw against calibrated
probabilities afterwards. `reports/reliability_curve.csv` holds the binned
predicted-vs-observed table for both variants, and
`reports/figures/calibration_curve.png` plots it.

**How SHAP explanations are generated.** `ml/explain.py` uses
`shap.TreeExplainer` on the **base (uncalibrated)** model. Two reasons: the
explainer needs the tree ensemble itself, which the calibrated wrapper hides
behind five per-fold clones; and isotonic calibration is a monotonic remap of the
score, so it changes the probability a patient receives but not the ranking or
the relative contribution of features.

Every explanation runs on the exact matrix the model consumes. The pipeline
one-hot encodes, so the model never sees `race` — it sees `cat__race_Other` and
friends, and explanations are labelled with the ColumnTransformer's own
`get_feature_names_out()` names (249 transformed features). A test asserts SHAP
additivity: `sigmoid(base_value + sum(shap_values))` must equal the model's
`predict_proba` output, which is what proves the explanation describes *this*
model on *this* encoding.

**Where artifacts are stored.**

| Path | Contents | In git? |
|---|---|---|
| `models/readmission_model.joblib` | Both fitted models | No — DVC output |
| `models/model_metadata.json` | Model type, target, excluded columns, seed, params, tree-count selection, data identity, library versions | No — DVC output |
| `reports/metrics.json` | The metric suite | **Yes** — a DVC metric with `cache: false`, so quality changes show up as reviewable diffs |
| `reports/reliability_curve.csv` | Binned calibration table | No — DVC output |
| `reports/shap_global_importance.csv` | Mean abs SHAP per transformed feature | No — DVC output |
| `reports/shap_local_example.json` | Worked single-patient explanation | No — DVC output |
| `reports/figures/*.png` | Calibration curve, SHAP summary | No — DVC output |

`dvc metrics show` prints the tracked metrics without opening the file.

**No `docker compose up` for an "app" container** — only Postgres exists in `docker-compose.yml` today. There is no frontend dev server, no backend server, and no `npm start`/`uvicorn` command to run, because none of those components exist yet.

---

## 10. Testing the Project

**How to verify the installation:**
```bash
pytest -q
# Expected: 16 passed
ruff check .
# Expected: All checks passed!
```

**How to verify the data pipeline (there is no API yet to verify):**
```bash
python -c "
import pandas as pd
train = pd.read_csv('datasets/processed/train.csv')
ev = pd.read_csv('datasets/processed/eval_frozen.csv')
fut = pd.read_csv('datasets/processed/future_stream.csv')
assert set(train.patient_nbr).isdisjoint(set(ev.patient_nbr))
assert set(train.patient_nbr).isdisjoint(set(fut.patient_nbr))
assert set(ev.patient_nbr).isdisjoint(set(fut.patient_nbr))
print('No patient leakage across splits: OK')
"
```

**How to verify the UI:** not applicable — no UI exists yet.

**How to verify AI integration:** not applicable — no AI/LLM integration exists yet (§7).

**Expected successful state at Week 3:** `pytest` reports 70 passed (50 passed / 20 skipped if the datasets and model artifacts have not been built), `ruff check .` reports clean, `dvc repro` reports "up to date," and the three processed CSVs exist with disjoint `patient_nbr` sets summing to 69,990 rows.

---

## 11. Common Errors

| Problem | Reason | Solution | Verification |
|---|---|---|---|
| `docker compose up` fails with `failed to connect to the docker API at npipe:...` | Docker Desktop application isn't running (the CLI/daemon binary being installed isn't the same as the app being started) | Launch the Docker Desktop application and wait for it to finish starting | `docker info` succeeds |
| `dvc repro` fails with `ModuleNotFoundError: No module named 'pandas'` | DVC shelled out to the system `python` (e.g. 3.14), not the activated venv's `python` | Activate the venv (`source .venv/Scripts/activate`) in the **same shell** before running `dvc repro` | `which python` (or `where python`) points inside `.venv` |
| `jupyter execute` fails with `UnicodeDecodeError: 'charmap' codec can't decode byte ...` | On Windows, `nbformat` opens notebook files using the system codepage (cp1252) by default, which cannot decode UTF-8 characters like em dashes | Set `PYTHONUTF8=1` in the environment before running: `PYTHONUTF8=1 python -m jupyter execute --inplace notebooks/01_eda.ipynb` | Command completes without a traceback |
| `dvc add`/`git add` reports a `.dvc` file is git-ignored | A broad `.gitignore` pattern (e.g. `datasets/raw/*`) also matches DVC's own `.dvc` pointer files | Add explicit negation rules (`!datasets/raw/*.dvc`) to `.gitignore` | `git status` shows the `.dvc` file as trackable, not ignored |
| `ml.data.ingest` output is missing `patient_nbr`/`encounter_id` | `ucimlrepo`'s `dataset.data.features` excludes ID columns by design; they live in `dataset.data.ids` | Already fixed in `ml/data/ingest.py` (joins `dataset.data.ids` with `dataset.data.features`) — if you see this, you have an older copy of the file | `patient_nbr` and `encounter_id` are present in `datasets/raw/diabetic_data.csv` |
| `LogisticRegression` convergence warning in the baseline notebook | Unscaled numeric features slow `lbfgs` convergence | Already fixed (a `StandardScaler` step was added to the baseline's numeric branch) — if you see this, check you're on the current notebook | Re-running the notebook produces no `ConvergenceWarning` |

---

## 12. Current Limitations

(Scoped strictly to Week 3 — this is not a roadmap of what's missing overall, just what a developer running the project today should know.)

- **No backend, no frontend, no API.** Only the data pipeline (`ml/data/`) and the Week 3 model code (`ml/train.py`, `ml/evaluate.py`, `ml/explain.py`) exist; the model is a local artifact, not a served endpoint.
- **No model registry and no serving.** A calibrated LightGBM model now exists in `models/`, but nothing registers, versions, or serves it — MLflow is Week 4 and the FastAPI service is Week 5. Promotion, shadow deployment, and the validation gate do not exist, so this model is not a "champion" in the governance sense.
- **No database schema.** Postgres runs but is empty; nothing writes to it yet.
- **No monitoring, no Decision Engine, no gate, no shadow deployment, no dashboard.** All Week 6+.
- **No LLM/Gemini integration exists yet**, and the offline-LLM-vs-Gemini-mandate conflict (§7) is unresolved.
- **No shared remote history** — commits exist locally and `origin` is configured, but the Week 1–3 history has not been pushed.
- **Single-machine, local-only setup.** DVC's remote is a local directory (`dvc-storage/`, gitignored) — there is no shared/team remote configured.
- **Windows encoding caveat** (§11): non-ASCII characters in notebooks require `PYTHONUTF8=1` on this OS when using `jupyter execute` from a script; interactive Jupyter (browser) does not hit this issue.

---

## 13. Current Project Status

**Completed (Week 1 + Week 2 + Week 3 exit criteria):**
- Repo scaffold, ruff + pre-commit + basic CI (lint + test) configuration
- Docker Compose skeleton running PostgreSQL 16 (empty)
- UCI Diabetes 130-US dataset downloaded and verified (101,766 × 50 columns)
- `notebooks/01_eda.ipynb` — executed, documents missingness/imbalance/leakage findings with real output
- Week 1 baseline logistic regression: AUROC 0.6191
- `ml/data/schema.py` — pandera schema, validated against the full real dataset
- `ml/data/clean.py` — leakage-code removal, chronological patient dedup, missing-value policy, target binarization (raw `readmitted` dropped)
- `ml/data/icd9.py` + `ml/data/features.py` — single sklearn `Pipeline`, confirmed identical train/serve output width
- `ml/data/split.py` — patient-level time-sliced split; zero-leakage unit test passes on real data
- DVC initialized with a local remote; `dvc repro` confirmed idempotent; `dvc push` succeeded
- `ml/config.py`, `ml/train.py` — LightGBM + isotonic calibration, tree count chosen by early stopping on a training-slice tail
- `ml/evaluate.py` — ROC-AUC / PR-AUC / Brier / log loss / ECE / recall@top-decile, counts at two thresholds, subgroup metrics
- `ml/explain.py` — SHAP TreeExplainer global + local explanations over the exact transformed matrix
- Week 3 model on the frozen eval slice: ROC-AUC 0.6024 calibrated (0.5996 raw), Brier 0.0741, ECE 0.0201
- `dvc repro` now runs four stages and is byte-reproducible end to end
- 70/70 pytest tests passing; `ruff check .` and `ruff format --check .` clean

**In Progress:** nothing — Week 3 is a clean stopping point with no partially-built component.

**Open question carried into Week 4:** the Week 3 model scores ~0.60 ROC-AUC,
below the ~0.64–0.69 range `project_docs/DATASET_ANALYSIS.md` cites for this
dataset. Those published figures come from *random* splits, usually without
patient-level deduplication. Under this project's stricter protocol — one
encounter per patient, chronological split, and a prevalence shift across time
(train 9.7% positive, eval 8.1%) — the same logistic-regression baseline that
scored 0.6191 on Week 1's random split scores 0.5820 on the chronological one.
The honest comparison is therefore LightGBM 0.5996 against logistic 0.5820 under
identical conditions. Whether to restate the roadmap's "at or above published
benchmarks" outcome in light of the stricter protocol is a decision for review,
not something Week 3 resolved by tuning.

**Remaining (Week 3 onward — see `PROJECT_REPOSITORY_GUIDE.md` §11 for the full list):** LightGBM model + calibration + SHAP, MLflow tracking/registry, FastAPI serving + JWT + audit rows, Evidently drift monitoring + seeded scenarios, the Decision Engine + Decision Card, the retrain pipeline + validation gate, shadow deployment + human approval + LLM narration, the Streamlit dashboard + audit PDF export, CI/CD hardening, and final reporting/viva prep.
