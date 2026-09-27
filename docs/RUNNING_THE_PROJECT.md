# Running the Project — VitalLoop 2.0

> **Milestone covered by this document: Week 12 — release `v1.0`** (§20; Week 11 in §19, Weeks 9–10 in §17–§18). Results and limitations: `docs/FINAL_REPORT.md`.
> This document describes only what exists and runs in the repository today. For a fresh clone, the root `README.md` quickstart is the shortest path; this document is the detailed reference. The only roadmap deliverable not produced is the demo video — see `PROJECT_REPOSITORY_GUIDE.md` §11.

---

## 1. Project Overview

**VitalLoop 2.0** is a self-healing MLOps system for predicting 30-day hospital readmission risk, designed around one principle: *deterministic code decides, an LLM only narrates, a human approves anything clinician-facing*. Full design intent lives in `project_docs/` (`PROJECT_DESIGN.md`, `ARCHITECTURE.md`, etc.) — those are planning documents, not a description of current code.

**Current implementation stage:** end of Week 8 of a 12-week roadmap (`project_docs/IMPLEMENTATION_ROADMAP.md`). The repository contains the data plane (ingestion, validation, cleaning, features, DVC versioning), the model plane (LightGBM + isotonic calibration + SHAP, tracked and registered in MLflow), the serving plane (authenticated FastAPI `/predict` with fail-closed Postgres audit rows), and the loop as far as shadow — Evidently drift detection with the seeded S1–S5 benchmark, the deterministic Decision Engine that turns each measured window into an auditable Decision Card, and the retrain-and-gate step that turns a card into a challenger and refuses to promote a regression. There is no shadow scoring, no human approval, no dashboard, and no LLM integration of any kind yet.

**Current milestone (Week 8) exit criteria, both met:**
- The card → retrain → gate path runs end to end and is verified on the live stack: a real Decision Card produced a registered challenger, the gate returned `PASS`, and the `shadow` alias moved while `champion` did not (§16.9).
- A deliberately bad challenger is `BLOCK`ed with one named reason per failed criterion (§16.4), proven by fixture rather than by anecdote.

**Week 7 exit criteria (still met):** all five drift scenarios produce the card the policy table predicts, including `NO_OP` on **every** control window (§15.5); branch coverage on the Decision Engine is 99% against the roadmap's ≥ 95% target (§10).

---

## 2. Technology Stack

Only technologies **actually present in the repository today** are listed. Everything else named in `project_docs/TECH_STACK.md` (Streamlit, Ollama, fpdf2) belongs to later weeks and is not installed or used yet.

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
| **pytest** | 9.1.1 | Standard Python test runner | 867 tests: Week 2 data contract, Week 3 model/calibration/SHAP, Week 4 tracking/registry, Week 5 API/auth/audit contracts, Week 6 drift/PSI/scenario/persistence contracts, Week 7 policy/confidence/card/decision contracts (the rule table is re-verified against every shipped policy version), and Week 8 retrain/criteria/gate/persistence contracts |
| **ruff** | 0.16.8 | Fast combined linter/formatter | Enforced via `pyproject.toml`, the pre-commit hook, and CI (`check` + `format --check`) |
| **pre-commit** | 4.6.2 | Runs ruff automatically on commit | Installed (`.git/hooks/pre-commit`); hook pinned to ruff v0.16.8 to match the pinned CLI |
| **LightGBM** | 4.7.0 | Gradient boosting is the right tool for mid-size tabular clinical data (`project_docs/TECH_STACK.md`) | The Week 3 readmission classifier |
| **SHAP** | 0.52.0 | Fast, exact attributions for tree models | Global + per-prediction explanations (`ml/explain.py`) |
| **matplotlib** | 3.11.2 | Figure rendering | Calibration curve and SHAP summary PNGs |
| **joblib** | 1.6.0 | sklearn's own serialisation format | Persists the fitted model bundle |
| **FastAPI / uvicorn** | 0.141.1 / 0.53.0 | Typed request validation at the boundary and OpenAPI for free (`project_docs/TECH_STACK.md`) | The Week 5 serving API |
| **PyJWT** | 2.14.0 | Stateless auth with `clinician`/`ops` role claims | Protects the prediction endpoint |
| **SQLAlchemy / psycopg** | 2.0.54 / 3.3.6 | ORM + Postgres driver; parameterised by construction | The `predictions` audit table |
| **structlog** | 26.1.0 | JSON logs carrying hashes, never PHI | Per-request serving logs |
| **MLflow** | 3.16.1 | Tracking + registry + artifact store in one self-hostable service; aliases model promotion natively (`project_docs/TECH_STACK.md`) | Week 4 experiment tracking, lineage, and the model registry |
| **Docker + Docker Compose** | 29.5.2 / v5.1.4 | Documented Windows-friction mitigation; offline-by-design | Runs `postgres:16` (holding `predictions` and `drift_events`), the Week 4 `mlflow` tracking server, the Week 5 `api` service, and the Week 6 `monitor` worker |
| **Evidently** | 0.7.23 | PSI/KS/prediction drift plus presentable HTML reports from one library (`project_docs/TECH_STACK.md`) | The Week 6 drift engine (`loop/monitor/drift.py`) |
| **APScheduler** | 3.11.3 | In-process scheduling for one periodic job, without an orchestrator deployment | The Week 6 monitor worker (`loop/monitor/worker.py`) |
| **SciPy** | 1.18.1 | Already a scikit-learn dependency; named explicitly because the monitor imports `ks_2samp` directly | KS two-sample test in the hand-rolled cross-check |
| **PyYAML** | 6.0.3 | Arrives with MLflow, but `configs/policy-v*.yaml` is a governance artifact the engine reads directly, so the dependency is declared rather than borrowed | Loading the versioned decision policy |
| **Pydantic** | 2.13.5 | Already pinned by FastAPI; `project_docs/ARCHITECTURE.md` §3.9 specifies the Decision Card as a Pydantic model | The Week 7 Decision Card contract |
| **pytest-cov** | 7.1.0 | Measures the roadmap's 95%-branch-coverage Week 7 deliverable for the engine | `pytest --cov=loop.engine --cov-branch` |
| **Git** | 2.53.0 | Version control | Repository initialized; `origin` configured; Week 1–2 history committed |
| **Node.js** | 24.14.0 | Pre-existing in the repo before this build | Used *only* by `project_docs/build_pdf.mjs` to render the planning-doc PDF; unrelated to the application and not required to run anything in this document |

**Frontend:** none exists yet (Streamlit is Week 10). See §12 (Current Limitations).

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

**Not required:** Node.js (unless rebuilding the planning-doc PDF), any cloud account, any LLM API key, any GPU. **Optional:** [Ollama](https://ollama.com) with `llama3.1:8b` pulled, only if you want LLM narration instead of the default template (§7, §17.6).

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

## 7. LLM Configuration — Ollama Only (decided in Week 9)

**Decision:** the narration layer's only LLM is **Ollama**, self-hosted, running
Llama 3.1 8B (`llama3.1:8b`), exactly as `project_docs/ARCHITECTURE.md` §3.10
and `TECH_STACK.md` specify. There is no cloud or paid LLM anywhere in the
repository — no Gemini client, no API key, no SDK dependency.

**The discrepancy this resolves.** Until Week 9 this section recorded a
"mandatory" constraint that any LLM be Google Gemini (`gemini-2.5-flash`). That
contradicted the architecture documents, which call for a self-hosted, offline
model precisely so that nothing — not even the Decision Card's aggregate
metadata — leaves the machine, and so the demo runs with no network. The
conflict was flagged in Week 2 and deliberately left open. Before Week 9 shipped,
the project owner decided it: **Ollama only**, because the project must be
**free and self-hosted/offline**, with no paid or cloud LLM dependency. (A
Gemini backend was briefly drafted beside Ollama while the decision was pending;
it was removed before commit and never shipped.)

**What that means in practice:**

| Setting | Values | Default |
|---|---|---|
| `VITALLOOP_NARRATION_BACKEND` | `template` or `ollama` | `template` |
| `VITALLOOP_OLLAMA_URL` | Ollama base URL | `http://localhost:11434` (`http://ollama:11434` in Compose) |
| `VITALLOOP_OLLAMA_MODEL` | an Ollama model tag | `llama3.1:8b` |
| `VITALLOOP_OLLAMA_TIMEOUT_SECONDS` | seconds | `60` |

- The **Jinja2 template is the default** narrator and the **mandatory
  fallback**: if Ollama is off, unreachable, has no model pulled, or writes a
  narrative that fails the grounding check, the template narrates instead. The
  system is complete without any LLM.
- The LLM's **only input is the Decision Card** — aggregate statistics and
  metadata, minus its own narrative fields (`loop/narrate/prompt.py`).
- **CI never downloads a model.** Every LLM test runs against a mocked
  transport; the template is the path CI exercises.
- **Explicit exclusions** (unchanged): no Claude API calls, no Anthropic SDK,
  no Claude/Anthropic keys or package dependencies.

To switch Ollama on, see §17.6.

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
| `datasets/processed/future_stream.csv` | **Untouched.** It is the serving stream the Week 6 drift monitor replays (§14) |

`ml/config.py` still exposes no path constant for `future_stream.csv`, and a test
asserts no Week 3 module references it — training must never see the stream. The
monitor opens it from `loop/monitor/config.py` instead.

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

### 9.2 The Week 4 tracked run (MLflow)

MLflow records *what the Week 3 workflow produced*. It does not retrain
differently or change the evaluation protocol, so a tracked run's numbers are
the Week 3 numbers by construction.

**Start the tracking service** (Postgres and MLflow are independent; starting
one does not disturb the other):

```bash
cd docker && docker compose up -d mlflow && cd ..
# UI: http://localhost:5000
```

**Run the tracked workflow** — one command, which runs training, evaluation,
SHAP, then logging:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # PowerShell: $env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -m ml.tracking
```

**Where tracking data lives.** Two independent stores, by design:

| Mode | Tracking URI | Store | Used for |
|---|---|---|---|
| Compose (documented default) | `http://localhost:5000` | SQLite + proxied artifacts on the `mlflow-data` volume | The UI, the registry, the Week 4 deliverable |
| Local (no Docker) | `sqlite:///mlflow/mlflow.db` | `mlflow/` in the repo, gitignored | Offline development and the test suite |

If `MLFLOW_TRACKING_URI` is unset, the local store is used. If it *is* set but
the server is unreachable, the run **fails loudly** rather than quietly writing
somewhere else — a "tracked" run that silently goes nowhere is the one failure
experiment tracking exists to prevent.

The MLflow service uses a SQLite backend rather than the Postgres container
next door: the official image ships no `psycopg2`, so a Postgres backend would
need a custom image or a pip install at container start, and the latter breaks
the offline-demo requirement. `--serve-artifacts` makes the server proxy
artifact uploads, so artifact URIs are server-relative rather than host paths —
that is what avoids the Windows artifact-path problem the roadmap flags as the
Week 4 risk.

**What is recorded.**

| Kind | Contents |
|---|---|
| Params (42) | Model type, every LightGBM parameter actually used, the early-stopping selection, random seed, calibration method and folds, feature-pipeline identity, excluded columns, train/eval paths, row counts and positive rates, library versions |
| Lineage params | `git_commit`, `git_branch`, `git_dirty`, and the `dvc_*_md5` hashes for the raw, train, eval and model artifacts |
| Metrics (69) | Every Week 3 metric, with the raw-vs-calibrated distinction preserved in the names (`raw_roc_auc` / `calibrated_roc_auc`, …), both threshold blocks, and per-subgroup ROC-AUC |
| Artifacts | `metrics.json`, `reliability_curve.csv`, `shap_global_importance.csv`, `model_metadata.json`, both figures, and a **redacted** copy of the local SHAP example |
| Models | `calibrated_model` (registered) and `base_model` (for SHAP), both reloadable |

**What is deliberately not recorded:** the raw and processed patient CSVs (they
belong to DVC — MLflow stores their *hashes*, not their contents), any `.env` or
credential, and the per-encounter `feature_value` fields from the local SHAP
example. No `input_example`/signature is logged either, because inferring one
would embed a real encounter's feature row in the model artifact.

**Inspect a run:**

```bash
# in the UI
open http://localhost:5000            # experiment: vitalloop-readmission

# or from Python
python -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://localhost:5000'); e=c.get_experiment_by_name('vitalloop-readmission'); r=c.search_runs([e.experiment_id])[0]; print(r.info.run_id, r.data.params['git_commit'], r.data.metrics['calibrated_roc_auc'])"
```

**Model registration.** The run registers the calibrated model as
`vitalloop-readmission` and sets the `champion` alias on version 1 — the
roadmap's Week 4 deliverable. Registration is *not* promotion: once a champion
exists, later runs register a new version and leave the alias alone, because
moving a champion is what Week 8's validation gate and Week 9's human approval
exist to authorise. Alias moves are the only way a model changes state
(`ARCHITECTURE.md` §3.5) and every move appends a row to
`mlflow/registry_audit.jsonl` recording the alias, both versions, run id, git
commit, actor, reason and timestamp. That file is JSONL because the Postgres
table that will hold those rows arrives in Week 5.

Load the registered model:

```bash
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); m=mlflow.sklearn.load_model('models:/vitalloop-readmission@champion'); print(type(m).__name__)"
```

**Running the tests without a live MLflow server.** The suite never contacts the
Compose service: every tracking test builds a throwaway SQLite store under
`tmp_path`. SQLite rather than a bare file store because the Model Registry
requires a database backend. Just run `pytest -q` — no `MLFLOW_TRACKING_URI`, no
container, no network.

### 9.3 The Week 5 serving API

An authenticated FastAPI service that scores one encounter and writes an audit
row for every prediction.

**Start the dependencies, then the API:**

```bash
cd docker && docker compose up -d postgres mlflow && cd ..

# The API runs in Compose too:
cd docker && docker compose up -d api && cd ..        # http://localhost:8000/docs

# ...or locally against the same containers, which is handier while developing:
set -a; . docker/.env; set +a
export POSTGRES_HOST=localhost VITALLOOP_MLFLOW_TRACKING_URI=http://localhost:5000
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Configuration.** Everything comes from the environment; `docker/.env.example`
lists the variables and `docker/.env` (gitignored) holds the real values.

| Variable | Purpose |
|---|---|
| `VITALLOOP_JWT_SECRET` | **Required, no default.** An unset secret fails startup rather than falling back to a value in the repository. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `VITALLOOP_JWT_ALGORITHM` / `VITALLOOP_JWT_EXPIRY_MINUTES` | Defaults `HS256` / `30` |
| `VITALLOOP_MODEL_SOURCE` | `mlflow` (default) loads the registered champion by alias; `local` loads the DVC-tracked joblib for offline work |
| `VITALLOOP_REGISTERED_MODEL_NAME` / `VITALLOOP_MODEL_ALIAS` | `vitalloop-readmission` / `champion` |
| `POSTGRES_*` | Compose the audit database URL, or set `VITALLOOP_DATABASE_URL` directly |

**Get a development JWT:**

```bash
python -m scripts.issue_dev_token --subject dr-synthetic --role clinician
```

Roles are `clinician` and `ops`, per `TECH_STACK.md`. Both may predict; the
`ops`-only endpoints arrive with the Week 9 approval flow.

**Call the endpoint** (the body below is synthetic, not a real patient):

```bash
curl -X POST http://localhost:8000/predict   -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json"   -d '{"time_in_hospital":5,"num_lab_procedures":44,"num_procedures":1,
       "num_medications":18,"number_outpatient":0,"number_emergency":1,
       "number_inpatient":2,"number_diagnoses":9,"admission_source_id":7,
       "race":"Caucasian","gender":"Female","age":"[60-70)","payer_code":"MC",
       "medical_specialty":"InternalMedicine","A1Cresult":">8","change":"Ch",
       "diabetesMed":"Yes","diag_1":"428","diag_2":"250.83","diag_3":"401",
       "insulin":"Up","metformin":"Steady"}'
```

```json
{
  "request_id": "7cde91a2-66a4-471c-8787-fffb7d784fdd",
  "readmission_probability": 0.186189,
  "predicted_class": 0,
  "decision_threshold": 0.5,
  "model_name": "vitalloop-readmission",
  "model_version": "1",
  "data_version": "ebbfae6002203ba5e60ce9647b5e9a92",
  "top_factors": [
    {"feature": "num__number_inpatient", "contribution": 0.499293},
    {"feature": "num__service_utilization", "contribution": 0.061021},
    {"feature": "num__number_diagnoses", "contribution": 0.044287}
  ]
}
```

The request body carries **44 fields** -- exactly what the feature pipeline
consumes. Medication fields default to `"No"`, so only the ones that differ need
sending. Two columns present in the processed data are deliberately not part of
the contract: `admission_type_id` and `discharge_disposition_id` are used during
cleaning but never reach the model. Identifiers are not accepted at all, and
unknown fields are rejected.

**Model loading and version behaviour.** The model is loaded **once at startup**
and reused; nothing trains at request time. `mlflow` mode resolves
`models:/vitalloop-readmission@champion` -- an explicit alias, never "latest", so
the served model changes only when somebody moves the alias. The served version
appears in every response and every audit row. If the model cannot be loaded the
service still starts, `/ready` reports `model_loaded: false`, and `/predict`
returns 503.

**Health and readiness.** Neither requires authentication.

| Endpoint | Meaning |
|---|---|
| `GET /health` | The process is up. Returns `{"status":"ok","service":"vitalloop-api"}` and nothing else |
| `GET /ready` | This instance can serve: model loaded **and** audit database reachable. 200 when ready, 503 otherwise |

**The audit table.** One row per scored request, in `predictions` -- the first of
the five tables in `ARCHITECTURE.md` §4.6. It is append-only in application code:
nothing in this repository updates or deletes a row.

| Column | Contents |
|---|---|
| `id`, `request_id`, `ts` | Row id, the client-facing audit id (also returned), timestamp |
| `caller`, `caller_role` | Identity from the **verified token**, never from the body |
| `model_name`, `model_version`, `model_source`, `data_version` | What scored it, and the DVC hash of the training data |
| `input_hash` | SHA-256 of the request body |
| `risk_score`, `predicted_class`, `decision_threshold`, `top_shap` | The result and its top factors |
| `status`, `error_category`, `latency_ms` | How it went |

**What is not stored:** the request body. §3.6 notes the hash "allows later
verification without storing the record twice", so the audit trail never becomes
a second copy of the clinical record. `top_shap` holds feature *names* and
contributions, not the encounter's values. structlog output is restricted to the
request id, caller, input hash, model version, status and latency -- no field
value, no score, no token.

Auth failures and validation failures are logged but produce no audit row: the
table records predictions the model actually made, and a request rejected at the
boundary never reached it.

**Audit failure is fail-closed.** If the row cannot be written, the caller gets
`503` and **no score**. A prediction reaching a clinician with no audit trail is
exactly the silent, unreviewable event this project exists to prevent, so the
audit write is part of serving rather than a best-effort side effect. The same
applies when the model itself errors: the attempt is recorded, and the response
is a 503 rather than a number.

**OpenAPI** is live at `http://localhost:8000/docs`, including the bearer-token
requirement and both schemas.

**Running the API tests** needs no database, no MLflow server, and no model
artifact -- the fixtures build a small model and point the audit trail at a
temporary SQLite file:

```bash
pytest -q tests/test_api_auth.py tests/test_api_predict.py          tests/test_api_audit.py tests/test_api_ops.py
```

---

## 10. Testing the Project

**How to verify the installation:**
```bash
pytest -q
# Expected: 1292 passed (about 22 min on the Windows dev machine)
ruff check .
# Expected: All checks passed!
ruff format --check .
# Expected: <N> files already formatted
```

**How to verify the Decision Engine's branch coverage** (the Week 7 deliverable
is at least 95%):
```bash
pytest tests/engine --cov=loop.engine --cov-branch --cov-report=term-missing -q
# Expected: TOTAL ... 99%  (the single miss is evaluate.py's `__main__` guard)
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

**How to verify the UI:** `pytest tests/dashboard -q` renders all six pages headlessly with Streamlit's `AppTest` and clicks every demo button; §18.7 lists the live walkthrough.

**How to verify AI integration:** `pytest tests/narrate -q` exercises the grounding check, the template on every policy branch, and the Ollama client over a mocked transport — no model download needed. See §17.6 for a live Ollama run.

**Expected successful state at Week 5:** `pytest` reports 164 passed (144 passed / 20 skipped if the datasets and model artifacts have not been built), `ruff check .` reports clean, `dvc repro` reports "up to date," and the three processed CSVs exist with disjoint `patient_nbr` sets summing to 69,990 rows.

---

## 11. Common Errors

| Problem | Reason | Solution | Verification |
|---|---|---|---|
| `docker compose up` fails with `failed to connect to the docker API at npipe:...` | Docker Desktop application isn't running (the CLI/daemon binary being installed isn't the same as the app being started) | Launch the Docker Desktop application and wait for it to finish starting | `docker info` succeeds |
| `dvc repro` fails with `ModuleNotFoundError: No module named 'pandas'` | DVC shelled out to the system `python` (e.g. 3.14), not the activated venv's `python` | Activate the venv (`source .venv/Scripts/activate`) in the **same shell** before running `dvc repro` | `which python` (or `where python`) points inside `.venv` |
| `jupyter execute` fails with `UnicodeDecodeError: 'charmap' codec can't decode byte ...` | On Windows, `nbformat` opens notebook files using the system codepage (cp1252) by default, which cannot decode UTF-8 characters like em dashes | Set `PYTHONUTF8=1` in the environment before running: `PYTHONUTF8=1 python -m jupyter execute --inplace notebooks/01_eda.ipynb` | Command completes without a traceback |
| `dvc add`/`git add` reports a `.dvc` file is git-ignored | A broad `.gitignore` pattern (e.g. `datasets/raw/*`) also matches DVC's own `.dvc` pointer files | Add explicit negation rules (`!datasets/raw/*.dvc`) to `.gitignore` | `git status` shows the `.dvc` file as trackable, not ignored |
| `ml.data.ingest` output is missing `patient_nbr`/`encounter_id` | `ucimlrepo`'s `dataset.data.features` excludes ID columns by design; they live in `dataset.data.ids` | Already fixed in `ml/data/ingest.py` (joins `dataset.data.ids` with `dataset.data.features`) — if you see this, you have an older copy of the file | `patient_nbr` and `encounter_id` are present in `datasets/raw/diabetic_data.csv` |
| The `api` container logs `model_unavailable`; MLflow answers `403 Invalid Host header - possible DNS rebinding attack detected` | Newer MLflow server images reject `Host` headers outside localhost and private IPs, and inside Compose the API calls `mlflow:5000` | Already fixed: the `mlflow` service passes `--allowed-hosts` including `mlflow:5000`. Recreate an older container with `docker compose up -d mlflow` | `curl localhost:8000/ready` shows `"model_loaded": true` |
| `python -m scripts.send_traffic` reports every request failed (422 on `diag_1`) | pandas reads numeric-looking ICD-9 codes (`428`) as floats | Already fixed: the script reads `diag_1..3` as text | `N scored, 0 failed` |
| `LogisticRegression` convergence warning in the baseline notebook | Unscaled numeric features slow `lbfgs` convergence | Already fixed (a `StandardScaler` step was added to the baseline's numeric branch) — if you see this, check you're on the current notebook | Re-running the notebook produces no `ConvergenceWarning` |

---

## 12. Current Limitations

(Scoped strictly to Week 10 — this is not a roadmap of what's missing overall, just what a developer running the project today should know.)

- **The 3-minute demo needs a seeded state.** Replay is only useful while a PASSed challenger is cached and not yet the champion. Since Week 11, `scripts.seed_demo` (or `scripts.reset_demo --yes`) provides that state, and the demo runs in about 95–102 s of machine time (§19.6); after a promotion, reseed or use a live retrain.
- **A fresh clone's `dvc repro` may record a new model md5.** The model pickle's bytes depend on the environment (memoisation), not its predictions, which are identical (§19.6).
- **CI has not run on GitHub yet** — nothing is pushed; its jobs were exercised locally (§19.6).
- **The dashboard has no user accounts.** It trusts the API's verdict on a pasted token (§18.4); token issuance is still the dev script.
- **Serving latency is above target.** A warm `/predict` takes roughly 350 ms against the roadmap's < 200 ms goal; per-request SHAP dominates. The roadmap's own mitigations (batching, a cached explainer path) are not implemented.
- **No benchmark card retrains unattended.** Every S1–S3 card escalates (confidence 0.39–0.67 against the 0.75 auto-proceed line), so the automated path is implemented and tested but is never the path the seeded benchmark takes; a retrain on this data needs an ops user's `APPROVE` (§17.3).
- **Shadow statistics are trivially perfect on this data.** A retrain runs the same seeded pipeline on the same pinned data, so every challenger so far is bit-identical to the champion and the live shadow windows report agreement 1.0 with zero score difference (§17.7). The statistics are tested on non-trivial pairs; the live stack has simply not produced a different model yet.
- **Replay can re-register the serving champion.** Replay re-registers the cached challenger, which after a promotion *is* the champion. The gate PASSes it (it equals itself), but the API never shadows a model against itself, so the window cannot fill and promotion is refused — safe, but a wasted run. A live retrain produces a new version (§17.7).
- **Policy rules 5 and 6 never fire on real data.** Rule 5 needs matured-label evidence that nothing produces yet. Rule 6 needs a retrain history the Decision Engine can read: `retrain_runs` now exists (Week 8), but wiring its rows back into `DriftEvidence` as a cooldown/budget input is not Week 8's deliverable and was not done. Both branches exist and are unit-tested against both shipped policies (§15.3).
- **The S1–S5 benchmark exercises rules 1 and 4 only.** After the control calibration, no benchmark scenario lands in the mild band that rules 2 and 3 read (§15.6). Those rules, and 5 and 6, are covered by unit tests rather than by the five scenarios.
- **No delayed-label performance monitoring.** ARCHITECTURE.md §3.7 lists AUROC/recall on a matured-label window alongside input and prediction drift. Week 6 implements the two leading indicators only; the matured-label arm (and scenario S4's detection, as opposed to its injection) needs the label-latency handling that Weeks 7–8 introduce.
- **The S5 control is not perfectly silent at the measurement layer** — see §14.4. Two administrative features drift genuinely across the serving stream, which is a finding about the UCI extract, not a bug in the monitor. The monitor still records that drift at its own 0.10 threshold; `policy-v2` calibrates the *decision* threshold on the control, as the roadmap requires, so the control produces `NO_OP` on every window without any evidence being altered (§15.6).
- **Ollama narration is optional and not exercised live on this machine.** The client, grounding check and fallback are tested with a mocked transport, and the fallback was verified live (Ollama running with no model pulled → 404 → template). A live `llama3.1:8b` narrative needs the ~4.9 GB model pulled (§17.6), which was not done here. The Gemini question is closed: Ollama only (§7).
- **No shared remote history** — commits exist locally and `origin` is configured, but the Week 1–3 history has not been pushed.
- **Single-machine, local-only setup.** DVC's remote is a local directory (`dvc-storage/`, gitignored) — there is no shared/team remote configured.
- **Windows encoding caveat** (§11): non-ASCII characters in notebooks require `PYTHONUTF8=1` on this OS when using `jupyter execute` from a script; interactive Jupyter (browser) does not hit this issue.

---

## 13. Current Project Status

**Completed (Week 1 through Week 10 exit criteria):**
- Repo scaffold, ruff + pre-commit + basic CI (lint + test) configuration
- Docker Compose running PostgreSQL 16 (holding `predictions`, `drift_events`, `decision_cards`, `retrain_runs`, `approvals` and `shadow_predictions`)
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
- `ml/tracking.py` + `ml/registry.py` — MLflow tracking with git/DVC lineage, artifact logging, and the alias audit trail
- MLflow service in Compose, UI on :5000; `vitalloop-readmission` v1 registered with the `champion` alias
- `api/` + `db/` — authenticated FastAPI serving with JWT role claims, fail-closed PostgreSQL audit rows, health/readiness, and OpenAPI docs
- `loop/monitor/` + `scenarios/` — Evidently drift monitoring per window (PSI/KS/prediction drift) against the frozen-evaluation launch reference, `drift_events` persistence, Evidently HTML report artifacts, an APScheduler worker, and the seeded S1–S5 benchmark
- `configs/policy-v1.yaml`, `configs/policy-v2.yaml` + `loop/engine/` — the deterministic Decision Engine: the versioned six-rule policy table, the decomposed confidence formula, the frozen Pydantic Decision Card, `decision_cards` persistence with idempotent re-evaluation, and the evaluator CLI
- The control calibration the roadmap assigns to Week 6, applied as `policy-v2`: the S1–S5 benchmark now yields `FULL_RETRAIN` on S1–S3 in their first window and `NO_OP` on every control window (§15.6)
- `configs/gate-v1.yaml` + `loop/gate/` + `ml/retrain.py` + `scripts/replay_retrain.py` — the Week 8 retrain-and-gate step: the DVC-pinned challenger run (live or replay), the versioned promotion criteria, the pure validation gate, `retrain_runs` persistence with idempotent re-gating, and the runner CLI
- Week 10: the Streamlit dashboard (`dashboard/`, six pages behind an API-verified ops token, on :8501 in Compose), the S1/S2 drift-injection button, retrain/replay/deliberately-bad-challenger buttons, demo traffic, and the fpdf2 per-card audit PDF
- Week 9: shadow scoring after the response (`api/shadow.py`, `loop/shadow/`, `shadow_predictions`); the ops-role approval flow for escalated retrains and for promotion (`loop/approval/`, `api/routers/ops.py`, `approvals`, `configs/promotion-v1.yaml`) — the only path that moves `champion`; grounded narration (`loop/narrate/`: Jinja2 default and fallback, optional self-hosted Ollama)
- Week 11: the CI pipeline (ruff → pytest + 80% per-package coverage floor → synthetic-sample smoke train → gate check → image build), `scripts/seed_demo.py`, the archiving `scripts/reset_demo.py`, the root `README.md` quickstart rehearsed from a fresh clone, and the Overview *Demo state* panel (§19)
- 1292/1292 pytest tests passing (Week 11: branch coverage 99% `loop/engine`, 99% `loop/gate`, 87% `api`, 85% `scripts/`; Week 10: 91% branch coverage over `dashboard/` + `loop/gate/` + `api/`, 98% on `dashboard/data.py`, 97% on `dashboard/audit_pdf.py`; Week 9: 97% branch coverage on `loop/approval/promotion.py`, 100% on `loop/approval/retrain.py`, `loop/shadow/stats.py`, `loop/narrate/narrator.py` and `ollama.py`) (99% branch coverage on the Decision Engine, 99% on `loop/gate/` and 100% on `ml/retrain.py`); `ruff check .` and `ruff format --check .` clean

**In Progress:** nothing — `v1.0` (Week 12) is a clean stopping point with no partially-built component.

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

**Remaining:** the demo video (script in `docs/FINAL_REPORT.md` Appendix B). Week 12 delivered the benchmark tables, the report and the local `v1.0` tag (§20).

---

## 14. Week 6 — Drift Monitoring

### 14.1 What it does

A monitoring window is a contiguous block of the serving stream
(`datasets/processed/future_stream.csv`). For each window the monitor runs
**Evidently** against a reference and records, per monitored feature, a PSI
value, a KS p-value where KS is defined, and whether the feature breached
`0.10` (breaching) or `0.25` (severe). It also compares the champion's own score
distribution reference-vs-window — ARCHITECTURE.md §3.7's *prediction drift*.

Each window becomes one append-only `drift_events` row plus one Evidently HTML
report under `reports/drift/<scenario>/window_NNN.html`. **Every window is
written, including the quiet ones**: PROJECT_DESIGN.md §6 makes the persisted
history the input to Week 7's "same features breach for ≥ 2 consecutive
windows" rule, and a missing quiet window would read as a gap in the evidence
rather than as evidence of calm.

| Module | Role |
|---|---|
| `loop/monitor/config.py` | monitored features, thresholds, window geometry, reference choice |
| `loop/monitor/psi.py` | hand-rolled PSI/KS — the independent cross-check, never the engine of record |
| `loop/monitor/windows.py` | slicing the stream into windows with position-derived bounds |
| `loop/monitor/reference.py` | the reference feature distributions and champion scores |
| `loop/monitor/scoring.py` | loads the champion (`local` joblib or `mlflow` alias) |
| `loop/monitor/drift.py` | the Evidently run and the aggregates it produces |
| `loop/monitor/persistence.py` | aggregates to a `drift_events` row |
| `loop/monitor/runner.py` | one window, or a whole scenario, end to end |
| `loop/monitor/worker.py` | the APScheduler worker |
| `scenarios/injection.py` | the seeded S1–S5 transformations |
| `scenarios/run_scenario.py` | the CLI a reviewer runs |

### 14.2 Running it

The monitor needs the database variables (and, for `mlflow` scoring, a reachable
tracking server). It deliberately does **not** read `api.config.Settings`, so it
never needs `VITALLOOP_JWT_SECRET` — a worker that issues no tokens should not
hold a signing key.

```bash
# one scenario, three windows, persisted, with HTML reports
python -m scenarios.run_scenario S1 --windows 3

# measure without touching the database or writing artifacts
python -m scenarios.run_scenario S5 --windows 3 --no-persist --no-report

# the scheduled worker (what the `monitor` Compose service runs)
VITALLOOP_MONITOR_SCENARIO=S1 VITALLOOP_MONITOR_INTERVAL_SECONDS=60 python -m loop.monitor.worker
```

`docker compose up monitor` runs the same worker. The datasets, the model and
`reports/` are bind mounts rather than image layers: they are DVC-tracked
artifacts, and clinical data has no business inside a container image.

### 14.3 What is monitored, and what is not

25 features: the 12 engineered numeric columns and 13 non-medication
categoricals. The 23 individual medication columns are excluded — their signal
is already carried by `num_med_changes` and `insulin_changed`, and nearly all of
them are ~99% `No`, so including them would add 23 permanently quiet features to
the denominator of Week 7's `breadth` term.

Nulls in a categorical feature are monitored as the explicit `missing` category,
because that is exactly what the fitted pipeline's
`SimpleImputer(fill_value="missing")` hands the model. Dropping them — which is
what both Evidently and the hand-rolled PSI do with NaN — would monitor a
different variable than the one being scored, and on `max_glu_serum` (~94%
unrecorded) it would reduce a 2000-row window to ~25 observations and turn
sampling noise into a permanent alarm.

**The reference is the frozen evaluation slice, not the training slice.**
ARCHITECTURE.md §3.7 says "vs the training reference", and that is the right
instinct — but on this dataset the training slice is not one distribution.
Splitting `train.csv` in half and measuring the halves against each other gives
`payer_code` PSI **1.68**, `medical_specialty` **0.52**, `admission_source_group`
**0.29**: the training data drifts violently within itself, because it spans
1999–2008 in encounter order. Measured against that mixture, every S5 control
window shows 4–7 breaching features at PSI 0.7–0.9 and an injected S1 shift of
0.30 is invisible underneath. The frozen evaluation slice is internally coherent
by the same measurement — its halves differ by at most **0.054**, and any
2000-row window of it scores under **0.06** against the whole. It is also the
distribution on which the champion's published metrics were established (what
ARCHITECTURE.md §3.8 rule 5 calls "launch"), and it is never trained on.

### 14.4 The benchmark, as measured

Three windows of 2000 rows each, champion v1 via the MLflow `champion` alias,
reference = 10,000 seeded rows of `eval_frozen.csv`. Reproduce with
`python -m scenarios.run_scenario <S> --windows 3`.

| Scenario | max PSI (w0 / w1 / w2) | breaching | prediction drift | top feature |
|---|---|---|---|---|
| **S1** covariate shift | 0.278 / 0.271 / 0.300 | 2 / 4 / 4 | no | `num_lab_procedures` |
| **S2** coding change | 0.409 / 0.423 / 0.393 | 1 / 3 / 3 | no | `A1Cresult` |
| **S3** prevalence shift | 0.372 / 0.398 / 0.452 | 2 / 4 / 5 | **yes** (PSI 0.104 to 0.274) | `age` |
| **S4** label drift | 0.089 / 0.128 / 0.166 | 0 / 2 / 2 | no | *(identical to S5 — correct)* |
| **S5** no-drift control | 0.089 / 0.128 / 0.166 | 0 / 2 / 2 | no | `medical_specialty` |

Each injected scenario lights up exactly the feature it moved, above the control
floor. S3 is the only one that moves the score distribution, which is right: a
case-mix shift changes who is being scored. S4 measures identically to S5 by
design — it degrades the feature/label relationship without touching a feature,
so input drift must not react to it.

**The control is not perfectly silent, and that is a finding, not a bug.**
Window 0 is fully quiet. From window 1, `payer_code` (0.11–0.13) and
`medical_specialty` (0.13–0.17) breach the 0.10 line on the *untouched* stream.
This is real drift: those two administrative-completeness fields move genuinely
across the UCI extract's encounter ordering, and they are not negligible to the
model — `medical_specialty` is the 7th most important feature by global SHAP
(5.96%) and `payer_code` the 11th (3.57%), so excluding them would be
cherry-picking rather than calibration.

What this means for Week 7, recorded here because the roadmap asks for the
control to be the calibration surface: **on this dataset the control floor is
`max_psi` around 0.17 on two administrative features, sustained across
consecutive windows, with prediction drift never firing.** A policy that reads
0.10 as a universal breach line will therefore satisfy ARCHITECTURE.md §3.8
rule 3 ("same features breach for ≥ 2 consecutive windows →
`INCREMENTAL_RETRAIN`") on a stream where nothing was injected. Week 7 owns that
decision — per-feature baselines, a raised threshold, or an explicit exclusion
are all policy choices, and inventing one here would pre-empt the week that has
to defend it. Week 6's job was to measure it and write it down.

### 14.5 Privacy

`drift_events` has sixteen columns and not one of them can hold a patient row.
`feature_stats` entries carry exactly `feature`, `kind`, `psi`, `ks_p_value`,
`breaching`, `severe` — a feature *name* and four statistics. The frame handed
to Evidently is selected by name from `MONITORED_FEATURES`, so `encounter_id`,
`patient_nbr` and the label are structurally absent. `report_uri` is a
repository-relative path to an artifact, never the artifact. No injected stream
is ever written to disk. Tests assert each of these.

### 14.6 Schema initialisation

`db/session.py` uses `create_all`, not Alembic — the schema is still gaining a
table per week and a migration system would be infrastructure nobody is using.
The consequence worth knowing: **declaring the model is not creating the
table.** `init_db()` is the documented call that creates it, and both the API
lifespan and `loop.monitor.database.open_session` invoke it at startup, so a
database created before Week 6 acquires `drift_events` the next time either
process starts. `tests/test_db_init.py` covers the fresh-database path, the
"predictions-only database gains drift_events" upgrade path, and that existing
rows survive it.

---

## 15. Week 7 — The Decision Engine

### 15.1 What it does

Week 6 measures; Week 7 decides. The Decision Engine reads a `drift_events` row
back, evaluates it against a **versioned policy file**, and emits one **Decision
Card** — an immutable record binding the trigger evidence, the policy version
that judged it, the action, a decomposed confidence score, the DVC hash a
retrain would run on, and the acceptance criteria a challenger must clear
(`RESEARCH_NOVELTY.md` C1).

It recomputes nothing. Every PSI and KS value on a card was produced by the
monitor and copied across verbatim, so a card and the window it came from can
never disagree.

`ARCHITECTURE.md` §3.8: *"A pure-Python, fully unit-tested policy module. **No
LLM, no randomness, no network.**"* `engine.decide` has no database handle, no
filesystem access and no clock of its own — `now` is an argument. The same
evidence and the same policy produce the same card, which is what makes C1's
claim ("every card can be re-derived by hand from the policy table") testable.

| Module | Role |
|---|---|
| `configs/policy-v*.yaml` | the policy artifacts: every number the engine decides with. `policy-v2` is in force; `policy-v1` is retained so its cards stay re-derivable |
| `loop/engine/policy.py` | loads and validates a policy version |
| `loop/engine/evidence.py` | the engine's input: one window plus its breach history |
| `loop/engine/confidence.py` | the deterministic confidence formula (§3.8) |
| `loop/engine/rules.py` | the six-rule table, as a pure function |
| `loop/engine/card.py` | the Pydantic Decision Card — the contract Weeks 8–9 read |
| `loop/engine/engine.py` | `decide(evidence, policy) -> DecisionCard` |
| `loop/engine/history.py` | reads `drift_events`, builds evidence, finds undecided windows |
| `loop/engine/persistence.py` | a card → a `decision_cards` row, idempotently |
| `loop/engine/evaluate.py` | the worker's library call, and the CLI |

### 15.2 Running it

The engine runs inside the **existing** monitor worker — `ARCHITECTURE.md` §5
gives that service both jobs ("drift job + decision engine") and §4.7 repeats
it, so there is no second scheduler, no queue, and no new Compose service. Each
tick measures a window, writes its `drift_events` row, then decides it.

```bash
# every window no policy version has decided yet
python -m loop.engine.evaluate

# one stream; or decide without writing
python -m loop.engine.evaluate --scenario S5
python -m loop.engine.evaluate --dry-run

# the worker, which now measures and decides on the same tick
VITALLOOP_MONITOR_SCENARIO=S1 python -m loop.monitor.worker
```

`VITALLOOP_POLICY_VERSION` selects the policy (default `policy-v2`). A policy
that will not load is fatal at worker startup rather than on the first tick: a
monitor that silently stopped deciding looks exactly like a monitor that found
nothing to decide.

### 15.3 The policy, and how it is versioned

`ARCHITECTURE.md` §3.8: *"Every policy version is a tagged code artifact; the
Decision Card records which policy version produced it. Changing a threshold is
a reviewed pull request — that **is** the governance story."*

So every number lives in `configs/policy-v*.yaml` and nowhere else. There is no
default threshold anywhere in `loop/engine/`: a missing key fails the load
rather than falling back to something no reviewer approved. `load_policy` also
refuses a policy the engine could not defend — weights that do not sum to 1.0, a
breach threshold above the severe threshold, a one-window "persistence" rule, a
precedence that does not order exactly rules 1–5.

**A new policy is a new file.** A policy is never edited once cards reference
it, because `decision_cards.policy_version` is only meaningful if the artifact
it names still says what it said at emission. Re-evaluating history under a new
version produces a *distinct* card per window — that is the governance feature,
not a duplicate.

Two policies ship. **`policy-v1`** is ARCHITECTURE.md §3.8's literal
transcription, retained unedited because cards were emitted under it.
**`policy-v2`** is the policy in force: the same six rules with `psi_breach`
calibrated on the no-drift control, as the roadmap requires (§15.6). Select one
with `--policy` or `VITALLOOP_POLICY_VERSION`.

**Reading §3.8 against the policy in force.** `project_docs/` is frozen planning
documentation — the source of truth for *intent*, written before any code existed
— so §3.8's rule table still prints `PSI ≥ 0.10`, and it is deliberately not
edited to match the implementation. Exactly one number in it is superseded: the
engine decides at `psi_breach: 0.20`. Week 7's stated outcome (“the examiner can
read the policy as a table and re-derive any card by hand”) therefore holds
against `configs/policy-v*.yaml` — the artifact every card names in its
`policy_version` field — rather than against §3.8's printed constants. Every
other value, rule, weight and precedence in §3.8 is reproduced exactly. This is
the one residual divergence between the planning documents and the running
system, and it is recorded here rather than resolved by editing either side.

#### The rule table (unchanged across policy versions)

| # | Condition | Action | Disposition |
|---|---|---|---|
| 1 | No feature PSI ≥ `psi_breach`, no prediction drift | `NO_OP` | — |
| 2 | 1–2 features `psi_breach` ≤ PSI < 0.25, no prediction drift, first window | `ALERT_ONLY` | — |
| 3 | Same features breach ≥ 2 consecutive windows | `INCREMENTAL_RETRAIN` | auto → shadow |
| 4 | Any PSI ≥ 0.25 **or** prediction drift | `FULL_RETRAIN` | auto → shadow if confidence ≥ 0.75, else escalate |
| 5 | Matured-label AUROC drop > 0.03 vs launch | `FULL_RETRAIN` | **always** escalate to human |
| 6 | Retrained < 7 days ago **or** budget exhausted | downgrade to `ALERT_ONLY` | escalate |

**Two things §3.8 does not state, decided here and written into the YAML so
they are reviewable rather than buried in an `if`-chain:**

- **Precedence.** The conditions overlap — a window at PSI 0.30 on features that
  also breached last window satisfies rules 3 and 4 at once — so turning the
  table into a function needs an order. `precedence: [5, 4, 3, 2, 1]`, then rule
  6 as a downgrade. The reading is *strongest evidence wins*: a confirmed
  matured-label regression outranks leading indicators (C3's claim), severe
  outranks persistent, persistent outranks new, and rule 6 is phrased as a
  *downgrade* so it applies to whatever the table selected.
- **The uncovered region.** Three or more features breaching mildly in their
  first window match no rule (rule 2 caps at two, rule 3 needs a second window,
  rule 4 needs 0.25 or prediction drift, rule 1 needs no breach at all).
  `uncovered_breach_action: ALERT_ONLY` takes the table's weakest non-silent
  action. It is a gap-filler, not a seventh rule: it selects no action the table
  does not already contain, and it never retrains. Cards from it carry
  `rule_id: "uncovered"` so they are visible rather than disguised as rule 2.

**Rules 5 and 6 are inert on every real card today.** Nothing in the repository
produces matured-label evidence (delayed-label monitoring is later work) and
`retrain_runs` is Week 8, so both branches exist, are unit-tested, and never
fire on Week 6 data. Their inputs are explicit arguments rather than database
reads, which is how the tests reach them.

#### Confidence

```
confidence = 0.35·severity + 0.20·breadth + 0.20·persistence + 0.25·evidence
  severity    = min(max_PSI / 0.5, 1.0)
  breadth     = breaching features / monitored features
  persistence = min(consecutive breaching windows / 3, 1.0)
  evidence    = 1.0 if matured labels confirm degradation, else 0.5
```

The four terms are stored beside the total, so the headline number can be
checked by arithmetic — a confidence that cannot be re-derived is the thing this
formula exists to replace (`PROJECT_DESIGN.md` §4.2).

Because no matured labels exist, `evidence` is 0.5 on every real card, which
caps attainable confidence at **0.875**. The 0.75 auto-proceed line is therefore
reachable but demanding, and in practice every Week 6 window that reaches rule 4
escalates to a human (§15.5). That is the label-latency restraint C3 argues for,
working as designed.

> **A discrepancy in the documents.** The example card in `ARCHITECTURE.md` §3.9
> shows `"confidence": 0.86` beside `{"severity": 0.54, "breadth": 0.18,
> "persistence": 0.67, "evidence": 0.5}`. Under policy-v1's weights those four
> terms sum to **0.484**, not 0.86. The formula is normative (§3.8, §4.3) and
> the example is illustrative, so the engine implements the formula;
> `tests/engine/test_confidence.py` pins that reading.

### 15.4 The Decision Card and its table

`decision_cards` is the third of the five tables in §4.6. The ERD's columns
(`card_id`, `drift_event_id`, `policy_version`, `action`, `confidence`,
`card_json`, `status`, `created_at`) are all present; the rest are a *queryable
projection* of `card_json`, never a second source of truth, so a dashboard need
not open JSON to answer "which windows escalated, under which policy, on which
scenario".

**Idempotency is enforced twice.** `card_id` is derived from the drift event and
the policy version (`dc-<window date>-<8 hex>`), so re-evaluating a window
produces the same primary key; and `(drift_event_id, policy_version)` carries a
unique constraint, so the guarantee survives any future change to how ids are
shaped. `record_decision` checks first and returns the existing card, which makes
re-running the evaluator over a backlog a no-op rather than merely
non-destructive.

Append-only, like the two tables before it. `EXECUTED` and `BLOCKED` are status
transitions Week 8's gate performs. A card at emission is in neither state and
the documents do not name the state it *is* in, so Week 7 records the only
distinction it can defend: `OPEN` when the action leaves downstream work
(`INCREMENTAL_RETRAIN`, `FULL_RETRAIN`), `CLOSED` when it does not.

Week 9's `narrative` and `narrative_source` are already on the schema and null.
`RISK_ANALYSIS.md` §1 freezes this contract in week 7 precisely so a Week 9 that
had to widen it would not break every card already emitted.

### 15.5 The benchmark, as decided

All 17 Week 6 windows, evaluated with `python -m loop.engine.evaluate` against
the recorded `drift_events` rows, under the policy in force (`policy-v2`):

| Scenario | w0 | w1 | w2 |
|---|---|---|---|
| **S1** covariate | rule 4 -> `FULL_RETRAIN` / escalate (0.394) | (0.456) | (0.543) |
| **S2** coding | rule 4 -> `FULL_RETRAIN` / escalate (0.486) | (0.563) | (0.608) |
| **S3** prevalence | rule 4 -> `FULL_RETRAIN` / escalate (0.460) | (0.553) | (0.673) |
| **S4** label drift | rule 1 -> `NO_OP` (0.187) | rule 1 -> `NO_OP` (0.214) | rule 1 -> `NO_OP` (0.241) |
| **S5** control | rule 1 -> `NO_OP` (0.187) | rule 1 -> `NO_OP` (0.214) | rule 1 -> `NO_OP` (0.241) |

This is the Week 7 deliverable in full: every injected scenario is caught by the
rule matching its mechanism, in its **first** window; the control is silent on
every window; and S4 decides identically to S5, because it moves labels rather
than features and rule 5 has no matured labels to read.

**No card auto-proceeds.** Leading-indicator confidence tops out at 0.875, and
the benchmark's maximum is 0.673, so every `FULL_RETRAIN` escalates to a human.

For comparison, the same rows under the uncalibrated `policy-v1` - the reading
that motivated the calibration - are in §15.6.

### 15.6 The control calibration (policy-v2)

Week 6 recorded that the *untouched* S5 stream breaches on `payer_code`
(0.11-0.13) and `medical_specialty` (0.13-0.17) from window 1 onward - real
drift in the UCI extract, in features ranked 7th and 11th by global SHAP
(§14.4). Applying policy-v1 to those rows gives `NO_OP` -> `ALERT_ONLY` ->
`INCREMENTAL_RETRAIN`: **a retrain recommendation on a stream where nothing was
injected.**

That is the "cries wolf" outcome the risk register names, and the authoritative
documents prescribe the remedy in five places:

| Source | Instruction |
|---|---|
| `IMPLEMENTATION_ROADMAP.md`, Week 6 risks | "threshold tuning noise -> **calibrate PSI thresholds on the no-drift control** week-6, not demo-day" |
| `PROJECT_DESIGN.md` §9, Week 6 outcome | "Reproducible drift benchmark; **thresholds calibrated on control**" |
| `PROJECT_DESIGN.md` §12, risks | "Threshold mis-tuning \| ... \| **Calibrated on the no-drift control (S5)**; persistence rule filters noise" |
| `PROJECT_DESIGN.md` §13, expected results | "drift detection within one monitoring window on S1-S3; **zero false triggers on the control**" |
| `RISK_ANALYSIS.md` §2 | "**Thresholds calibrated on the no-drift control (S5) in week 6**; persistence rule (2 consecutive windows) filters noise" |

So the conflict was never between the rule table and the deliverable. It was
that **the calibration step the roadmap assigns was measured in Week 6 and
never applied**; policy-v1 transcribes §3.8's uncalibrated 0.10 verbatim. The
design expects *both* mechanisms - a calibrated threshold **and** the
persistence rule. Persistence is there to filter noise; it was never meant to
be the thing that converts genuine-but-harmless drift into a retrain.

#### What was calibrated

Measured on the untouched stream (S5, S4 and `live` share identical feature
distributions - S4 moves only labels), three 2000-row windows:

| Feature | w0 | w1 | w2 |
|---|---|---|---|
| `medical_specialty` | 0.0622 | 0.1249 | **0.1663** |
| `payer_code` | 0.0885 | 0.1278 | 0.1125 |
| every other feature | < 0.07 | < 0.07 | < 0.07 |

Against the injected signals on the same windows:

| Scenario | feature | w0 | w1 | w2 |
|---|---|---|---|---|
| S1 | `num_lab_procedures` | 0.2777 | 0.2713 | 0.2998 |
| S2 | `A1Cresult` | 0.4086 | 0.4231 | 0.3931 |
| S3 | `age` | 0.3717 | 0.3984 | 0.4519 |

Any threshold in **(0.1663, 0.25)** separates the two. `policy-v2` takes the
control maximum rounded up to the next 0.05 - **`psi_breach: 0.20`** - leaving
~20% headroom for sampling variation while staying strictly below `psi_severe`,
which is left at §3.8's 0.25 untouched.

#### What was not changed

- **The six rules, the precedence, the confidence formula and its weights, the
  consecutive-window count, and `psi_severe`** are byte-identical between the
  two policies. `RISK_ANALYSIS.md` §1 freezes policy-v1 at six rules and sends
  improvements to policy-v2; this is a *threshold* change in a reviewable diff,
  which is exactly what §3.8 calls the governance story.
  `test_policy_v2_differs_from_v1_in_the_breach_threshold_and_nothing_else`
  keeps it that way.
- **Week 6's measurement.** No PSI, KS or prediction-drift calculation was
  touched, no scenario redefined, no drift event deleted, and no threshold in
  `loop/monitor/config.py` moved. The monitor still measures at 0.10, so
  `drift_events` still records `payer_code` and `medical_specialty` as
  breaching. Only the line at which the *policy* calls a breach actionable
  moved.
- **The raw S5 evidence.** The rows are the same rows. Every card carries both
  numbers - `trigger.measured_thresholds` (the monitor's 0.10) and
  `policy_thresholds` (the policy's 0.20) - so the gap between what was
  observed and what was deemed actionable is on the audit trail rather than
  buried in a config file.
- **`configs/policy-v1.yaml`.** Retained unedited and still loadable: cards
  were emitted under it, and §3.8 requires a card's policy version to still say
  what it said at emission. Re-running the evaluator emits a second, distinct
  card per window under policy-v2 - replaying history under a new policy is the
  governance feature, not a duplicate.

#### The cost, stated plainly

Specificity was bought with sensitivity. Under policy-v2 a feature that moves
between 0.10 and 0.20 no longer reaches a card: S1's `num_medications`
(0.12-0.14) is measured and recorded but no longer counted as breaching, so
S1's cards name `num_lab_procedures` alone and carry slightly lower confidence.
Detection is unaffected - every injected scenario is still caught in its first
window - but the benchmark now exercises **rules 1 and 4 only**. Rules 2, 3, 5
and 6 are covered by unit tests against both shipped policies rather than by
the five scenarios.

### 15.7 Privacy

A Decision Card carries feature *names* with their PSI and KS values — the same
aggregate shape `drift_events.feature_stats` holds — plus versions, thresholds
and prose. `decision_cards` has nineteen columns and not one of them can hold a
patient row. Only *breaching* features reach a card, so it is evidence for a
decision rather than a copy of the measurement.

This is what makes §3.10's claim about the Week 9 narration layer
("structurally incapable of seeing PHI") true: the LLM receives the card, and
the card has nowhere to put a patient record. Tests assert the field sets and
the absence of identifiers, and the same check runs against the live PostgreSQL
rows.

### 15.8 What Week 7 deliberately does not do

- **It does not retrain.** `FULL_RETRAIN` and `INCREMENTAL_RETRAIN` are
  *recommendations recorded on a card*. Nothing in `loop/engine/` imports
  `ml.train`, touches an MLflow alias, or writes a model artifact. The retrain
  pipeline and the validation gate are Week 8.
- **It does not monitor delayed-label performance.** Rule 5's branch exists and
  is tested; no producer of matured-label evidence does.
- **It adds no infrastructure.** No new Compose service, no queue, no second
  scheduler, no API endpoint — the roadmap asks for none, and the dashboard that
  will read these rows is Week 10.
- **It does not change Week 6.** No threshold, statistic, scenario or monitored
  feature was altered. The engine reads what the monitor wrote.

---

## 16. Week 8 — Retraining and the Validation Gate

### 16.1 What it does

Week 8 closes the loop. A Decision Card that called for a retrain becomes a
*challenger*, and the challenger is compared against the champion by a pure
function that refuses to promote a regression.

`IMPLEMENTATION_ROADMAP.md`, Week 8: *"drift → card → retrain → gate PASS path
fully automated; bad challenger BLOCKED with reasons."*

| Module | Role |
|---|---|
| `ml/retrain.py` | §3.11's retraining pipeline: card → DVC-pinned `challenger` run, live or replay |
| `configs/gate-v*.yaml` | the gate criteria artifact: every number a promotion is judged by |
| `loop/gate/criteria.py` | loads and validates a criteria version |
| `loop/gate/datasets.py` | the two evaluation sets §3.12 gates on |
| `loop/gate/metrics.py` | builds a `MetricSet` from labeled data, using Week 3's metric suite |
| `loop/gate/gate.py` | `evaluate_gate(champion, challenger, criteria) -> GateResult` — the pure function |
| `loop/gate/persistence.py` | a verdict → a `retrain_runs` row, idempotently |
| `loop/gate/runner.py` | one card end to end, and the CLI |
| `scripts/replay_retrain.py` | §10's named demo entry point for replay mode |

### 16.2 Running it

```bash
# every card awaiting a retrain, trained live
python -m loop.gate.runner

# one card; a card the policy escalated needs a name to record
python -m loop.gate.runner --card dc-2026-01-01-894fe793 --authorized-by ops-alice

# §3.11's cached-run replay: re-register a pre-trained challenger
python -m scripts.replay_retrain --card dc-2026-01-02-f0018348

# gate and print, writing nothing and moving no alias
python -m loop.gate.runner --dry-run
```

On this machine a live retrain takes about **100 seconds** end to end and a
replay about **28 seconds** — which is the entire reason replay exists.

### 16.3 What authorises a retrain

The runner never overrides the engine. A card is acted on only if:

- its **action** is `FULL_RETRAIN` or `INCREMENTAL_RETRAIN`; and
- its **disposition** is `AUTO_PROCEED_SHADOW` — the automated path, which is
  what §3.8 rule 4 means by "auto → shadow"; or
- its disposition is `ESCALATE_HUMAN` **and** the caller supplies
  `--authorized-by`, which is recorded on the run.

`WORKFLOW.md` §4 is explicit about the second case: *"Ambiguous evidence
(confidence < 0.75) | Disposition = `ESCALATE_HUMAN`; nothing retrains until an
ops user acts."* A backlog sweep therefore picks up automated cards only; an
escalated card must be named.

> **On this dataset every benchmark card escalates.** Leading-indicator
> confidence tops out at 0.875 and the S1–S3 cards reach 0.39–0.67 (§15.5), so
> none of them clears the 0.75 auto-proceed line. The automated path is
> implemented and tested; on the seeded benchmark it is simply never the path
> taken, which is the governed-autonomy design working rather than a gap.
> Week 9's approval screen is what will produce the `authorized_by` name in the
> demo.

### 16.4 The gate

`ARCHITECTURE.md` §3.12: *"The gate is a pure function of two metric sets —
trivially unit-testable, which is exactly what a promotion safety mechanism
must be."* `loop/gate/gate.py` has no database handle, no model, no filesystem
access, no clock and no network.

| Criterion | Rule (§3.12 / §4.5) |
|---|---|
| AUROC | challenger ≥ champion − `auroc_non_inferiority_margin` (0.005) |
| Recall @ top decile | challenger ≥ champion × `recall_top_decile_min_ratio` (1.0) |
| Brier score | challenger ≤ champion + `max_brier_increase` (0.005) |
| Calibration (ECE) | challenger ≤ `max_ece` (0.05) — an **absolute** ceiling, not a comparison |
| Subgroup non-regression | no age/gender/race AUROC drop > `max_subgroup_auroc_drop` (0.01) |

Three properties are worth stating explicitly:

- **Both evaluation sets must hold.** §3.12 gates on *"(a) a frozen holdout and
  (b) the most recent labeled window"*, and `RISK_ANALYSIS.md` §2 gives the
  reason — gating repeatedly against one frozen set eventually overfits to it.
  A failure on either blocks.
- **A missing metric set blocks.** "No evidence" is not "no regression".
- **An incomparable subgroup is recorded, not skipped.** Where either model has
  no defined AUROC for a subgroup (a single outcome class), the check is
  recorded as non-comparable rather than silently passing, so a subgroup that
  stops being measurable stays visible.

**Which numbers apply.** `configs/gate-v1.yaml` is the default criteria
artifact, but a card-driven run is judged by the `acceptance_criteria` the
*card* pinned — `RESEARCH_NOVELTY.md` C1 requires the card to record "the
acceptance criteria the challenger must meet **before** training starts". The
verdict records which it used: `gate-v1`, or `card:<card_id>`. A test asserts
that the file and every policy's pinned block hold the same five numbers, so
the two artifacts cannot drift apart.

### 16.5 What a PASS and a BLOCK actually do

- **PASS** → the `shadow` alias moves to the challenger, through
  `ml.registry.set_alias`, which writes an audit row. Nothing else moves.
- **BLOCK** → nothing moves. §3.12's *"nothing changes in serving"* is enforced
  by there being no code path that could.

**`champion` is never touched by Week 8.** §3.13 reserves it for a logged human
approval, which is Week 9. Autonomy stops at shadow.

### 16.6 Where a card's status transition lives

§3.12 says a blocked challenger leaves the card *"closed as `BLOCKED`"*, while
§4.6 requires every table to be append-only in application code — *"no UPDATE
on card contents; status transitions append history rows"*. These read as a
conflict; they are not. **The `retrain_runs` row is the history row.** Nothing
in this repository updates a `decision_cards` row after it is written, and a
card's effective state is read by joining to its retrain runs:

```sql
SELECT c.card_id, c.action, r.mode, r.outcome, r.shadow_alias_moved
FROM decision_cards c LEFT JOIN retrain_runs r ON r.card_id = c.card_id;
```

This also keeps the Week 7 Decision Card contract frozen, which
`RISK_ANALYSIS.md` §1 requires: `CardStatus` is still `OPEN`/`CLOSED`, and no
Week 8 code widened a schema that Week 7 froze.

### 16.7 Idempotency and restart

`run_id` is derived from the card, the mode and the challenger version, and
`(card_id, mode, challenger_version)` carries a unique constraint — the same
two-place guarantee `decision_cards` uses. Before doing any expensive work the
runner looks for an existing verdict for this card, mode and data version; if
one exists it is returned and **nothing is retrained**. A worker that died
between training and persisting is therefore safe to restart.

A *different* challenger on the same card is a distinct row. That is not a
loophole — it is what a second attempt after a BLOCK legitimately is, and what
the deliberately-bad-challenger demonstration needs.

### 16.8 Replay mode

`ARCHITECTURE.md` §3.11: *"a cached-run replay mode re-registers a pre-trained
challenger so the live demo completes in seconds (explicitly labeled as replay
in the UI)."* The mode is a column on `retrain_runs`, not an inference, so a
replayed run can never be mistaken for one that trained.

**Replay accelerates producing a challenger; it never relaxes the bar.** The
gate is the same function in both modes, and a test asserts that a bad
challenger is blocked under replay exactly as under a live run.

### 16.9 Verified on the live stack

Against the running Compose Postgres and MLflow, on real Week 6/7 rows:

| Step | Result |
|---|---|
| `retrain_runs` created by `init_db` on the existing Week 7 database | 17 columns, the ERD's six present |
| Live retrain on card `dc-2026-01-01-894fe793` (S1 w0, policy-v2) | challenger **v2** registered, ~100 s |
| Pinned data version enforced | card pin `ebbfae60…` = `dvc.lock` train md5 |
| Gate | **PASS**, 47 checks over `frozen_holdout` (10,498 rows) + `recent_labeled_window` (2,000 rows) |
| Aliases after | `champion` → 1 (unmoved), `challenger` → 2, `shadow` → 2 |
| Alias moves audited | both, in `mlflow/registry_audit.jsonl` |
| Replay on card `dc-2026-01-02-f0018348` | **PASS**, ~28 s, `mode=replay` |
| Re-running a gated card | `[already on record]`, no retrain, still 2 rows |
| Inverted challenger against the live champion (card `dc-2026-01-03-9bd13a7e`, not persisted) | **BLOCK**, 37 of 46 checks failed — all five criteria tripped with named numbers |

The BLOCK row is the roadmap's "money shot" on real data rather than on a
fixture: against the registered champion and the real evaluation sets, an
inverted challenger fails discrimination
(AUROC 0.397582 vs 0.602418, floor 0.597418), top-decile recall
(0.049180 vs 0.170960), Brier (0.747287 against a 0.079107 ceiling),
calibration (ECE 0.818115 against the absolute 0.05) and every comparable
subgroup. It was computed read-only and deliberately not persisted, so the
live `retrain_runs` table still holds only the two genuine runs above.

The live challenger's metrics are *identical* to the champion's (AUROC 0.602418
on the frozen holdout for both). That is the expected result and a useful one:
the retrain ran the same seeded pipeline on the same pinned data, so a
bit-reproducible training pipeline has to produce an equivalent model.

### 16.10 Privacy

A `gate_result` holds counts and aggregates only: `rows`, `positive_rate`, the
four headline metrics, and per-subgroup AUROC keyed by category label. Verified
against the live rows — no `encounter_id` or `patient_nbr` appears in any
payload, and no raw feature value is carried. The MLflow challenger run logs
params, metrics and tags; it logs **no dataset artifact**, for the reason
`ml/tracking.py` gives — the processed CSVs are DVC's responsibility.

### 16.11 What Week 8 deliberately does not do

- **It does not promote.** No code path moves `champion`. Shadow scoring,
  approval and promotion are Week 9 (§17).
- **It adds no infrastructure.** No new Compose service, no scheduler, no API
  endpoint — §5's topology gains nothing in Week 8, and the roadmap asks for
  none. The gate is a CLI and a library call.
- **It does not change Weeks 6 or 7.** No drift statistic, threshold, scenario,
  policy rule or card field was altered. The one earlier file that changed is
  `ml/evaluate.py`, where the report builder was extracted into `build_report`
  so a challenger is measured by the champion's own reporting code;
  `reports/metrics.json` is byte-identical afterwards, and `dvc.lock` is
  regenerated in the same commit because `ml/evaluate.py` is a stage dependency.
- **It does not add a "deliberately bad challenger" to the registry.** The
  roadmap asks for a *fixture* proving BLOCK, and that is what
  `tests/gate/conftest.py::InvertedChallenger` is — bad by construction rather
  than by seed. Wiring a bad challenger into the clickable demo is Week 10.


---

## 17. Week 9 — Shadow, Approval, Narration

### 17.1 What it does

Week 9 takes a gated challenger the rest of the way, and puts a person at the
two points where the design says a person must act.

`IMPLEMENTATION_ROADMAP.md`, Week 9: *"full lifecycle drift → … → shadow →
human approval → promotion, every step persisted; narration works with and
without Ollama."*

| Module | Role |
|---|---|
| `api/shadow.py` | §3.6 shadow scoring: the `shadow` model scores each request **after** the response is sent |
| `loop/shadow/stats.py` | §3.13 agreement and stability statistics, as a pure function over score pairs |
| `loop/shadow/persistence.py` | `shadow_predictions` rows and the statistics for one champion/shadow pair |
| `loop/approval/retrain.py` | an ops user authorises (or rejects) a retrain the policy escalated |
| `loop/approval/promotion.py` | an ops user approves (or rejects) a gated challenger — **the only code that moves `champion`** |
| `loop/approval/registry.py` | the alias operations a decision may perform, over `ml.registry` |
| `configs/promotion-v1.yaml` | the promotion rules: shadow window size, minimum reason length |
| `api/routers/ops.py` | the `ops`-role endpoints for all of the above |
| `api/serving_state.py` | re-resolves champion and shadow in a running API after a decision |
| `loop/narrate/` | narration: grounding check, Jinja2 template (default + fallback), optional Ollama |
| `scripts/send_traffic.py` | replays serving-stream rows through `/predict` to fill a shadow window |

Two tables join the schema. `approvals` is the fifth table of §4.6's ERD.
`shadow_predictions` is not in the ERD: §3.6 says the shadow score is *logged*,
and a score nobody is ever shown needs its own row rather than a column on the
audit row the clinician's answer was built from. Both are created by `init_db`,
like every table before them, and both are append-only.

### 17.2 Running it

```bash
# the API (Compose) now also exposes /ops/*; an ops token:
python -m scripts.issue_dev_token --subject ops-alice --role ops

# fill a shadow window with real, audited requests
python -m scripts.send_traffic --count 60

# narrate a stored card (read-only); measure LLM faithfulness over every card
python -m loop.narrate --card dc-2026-01-03-9bd13a7e
python -m loop.narrate --all --backend ollama
```

| Endpoint (`ops` role) | Does |
|---|---|
| `GET /ops/retrains/pending` | escalated retrain cards nobody has decided |
| `POST /ops/retrains/{card_id}/decision` | `APPROVE`/`REJECT` + reason → `approvals` row; trains nothing |
| `GET /ops/promotions/pending` | gated challengers in shadow, each with its full evidence chain |
| `GET /ops/promotions/{run_id}` | the evidence for one |
| `POST /ops/promotions/{run_id}/decision` | `APPROVE` moves `champion`; `REJECT` moves nothing; both clear `shadow` |
| `GET /ops/shadow` | what this instance serves, and the current window's statistics |
| `POST /ops/models/reload` | re-resolve champion and shadow (after a gate PASS moved `shadow`) |
| `GET /ops/cards/{card_id}/narrative` | the card's narrative and its grounding verdict |

A clinician token gets `403` on every one of them. The approver is the token's
subject; neither request body has an approver field, so nobody can record a
decision in someone else's name.

### 17.3 Authorising an escalated retrain

Week 8 refused to retrain an `ESCALATE_HUMAN` card without a name to record,
and said Week 9 would supply the name. It is an `approvals` row of kind
`RETRAIN`:

- `APPROVE` authorises. The decision **does not train** — a retrain takes
  minutes and belongs to `python -m loop.gate.runner`, which now sweeps
  approved escalated cards alongside automated ones and records the approver
  as `authorized_by`. Every Week 8 check still applies.
- `REJECT` closes the card, with its reason. The runner then refuses that card
  **even if a caller passes `--authorized-by`**: a person reviewed the evidence
  and said no, and a CLI flag must not quietly overrule them.
- A person cannot turn an `ALERT_ONLY` card into a retrain, cannot authorise an
  automated card (it already carries its own authority), and cannot decide the
  same card twice.

### 17.4 Shadow scoring

`/predict` queues the shadow as a FastAPI background task, which Starlette runs
only **after** the response has been sent. So the clinician's answer, its
latency and its audit row are settled before the shadow runs, and nothing it
does — slow, failing, broken model — can reach them. A shadow that fails is a
row with status `shadow_failed`, not an error anyone sees.

The shadow model is loaded **by version**, not by alias, so the version written
on each row is always the model that scored it. The API never shadows a model
against itself.

§3.13's statistics, per `(champion, shadow)` pair: decision agreement at the
serving threshold, flips in each direction, mean/p95/max absolute score
difference, Spearman rank correlation, mean score of each model and the shift
between them, and failed-score count.

### 17.5 Promotion: what must be true, and what moves

`ARCHITECTURE.md` §6: *"Alias change requires gate PASS + shadow window +
logged human approval."* An `APPROVE` is refused (409, nothing moved, nothing
written) unless **all** of these hold at the moment of the click:

| Precondition | Why |
|---|---|
| `gate_pass` | the run PASSed and actually moved `shadow` |
| `challenger_in_shadow` | `shadow` still points at this challenger — otherwise its window is about a model nobody is watching |
| `champion_unchanged_since_gate` | `champion` is still the model the gate compared against — the verdict was "better than *that* model" |
| `shadow_window_complete` | ≥ `min_shadow_requests` (50) requests dual-scored by exactly this pair |

plus a reason of at least `min_reason_length` (10) characters —
RISK_ANALYSIS.md §3's answer to rubber-stamping, for both outcomes. `REJECT`
needs no evidence threshold: stopping a model reaching clinicians should never
be blocked.

`APPROVE` moves `champion` to the challenger (audited in the approver's name),
clears `shadow`, and reloads the API's models so the new champion serves
immediately. `REJECT` moves nothing but clears `shadow`, so a rejected model
stops scoring. Each decision is final — `(kind, subject)` is unique.

The alias moves before the row is written; if the row cannot be written, the
champion alias is **moved back** before the error propagates. That ordering is
the only one where no champion move can survive without its approval record.
A test scans the source tree and asserts that `loop/approval/promotion.py` is
the only code (besides Week 4's one-time initial registration) that sets
`champion`.

The row stores the evidence the person was shown — card summary and narrative,
the gate's headline numbers champion-beside-challenger, the worst subgroup
AUROC drop, the shadow statistics, the aliases, and each precondition's state.

### 17.6 Narration

`WORKFLOW.md` step 13: *"Emits the Decision Card; LLM narration attached
afterwards."* The engine decides, then the narrator fills the card's
`narrative` and `narrative_source` slots (frozen and null since Week 7), then
the card is written — so the narrative can never influence the decision and
the stored card is never edited. A test asserts that narration changes those
two fields and nothing else.

- **Template (default).** `loop/narrate/templates/decision_card-v1.j2`,
  deterministic, labelled `template/decision_card-v1`. It is grounded on every
  card the engine can emit: a test renders it for every rule branch under both
  shipped policies and runs the grounding check on each.
- **Grounding check** (`grounding.py`, pure). Every number the narrative quotes
  must be a card value (rounding and percentages allowed), a list length, or a
  number inside the card's own rationale; every ISO date must be one of the
  card's dates. Digits inside identifiers (`policy-v2`, `S1`, card ids, hashes)
  are not quantities. Numbers written as words are not checked.
- **Ollama (optional, self-hosted).** The only LLM backend (§7). Its only input
  is the card minus its narrative fields. `temperature` 0, fixed seed, but
  nothing relies on that: an ungrounded answer or any failure falls back to the
  template, and `python -m loop.narrate --all --backend ollama` reports the
  faithfulness rate (RESEARCH_NOVELTY.md C2).

To switch Ollama on:

```bash
# on the host
ollama pull llama3.1:8b           # ~4.9 GB, once
VITALLOOP_NARRATION_BACKEND=ollama python -m loop.engine.evaluate

# or in Compose (the `ollama` profile is not started by default)
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.1:8b
# set VITALLOOP_NARRATION_BACKEND=ollama in docker/.env, then restart `monitor`
```

Cards emitted before Week 9 keep their null narrative — they are immutable.
`GET /ops/cards/{id}/narrative` and `python -m loop.narrate --card` render the
template for them on demand, without writing anything.

### 17.7 Verified on the live stack

Against the running Compose Postgres, MLflow and API, on 2026-09-24:

| Step | Result |
|---|---|
| `init_db` on the existing Week 8 database | `approvals` and `shadow_predictions` created |
| API start with `shadow` → v2 | `model_loaded` v1, `shadow_loaded` v2 |
| Clinician token on `/ops/shadow` | `403` |
| `APPROVE` promotion of `rr-…894fe793` with an empty window | `409` — "0 of 50 requests dual-scored" |
| `send_traffic --count 60` | 60 scored; 60 `shadow_predictions` rows; agreement 1.0, max diff 0.0 |
| `APPROVE` again | `200`; champion 1 → 2, shadow cleared, `/ready` reports v2 without a restart |
| 5 more requests | served by v2; no shadow rows |
| The other Week 8 run (`rr-…f0018348`) | three preconditions unmet; `REJECT` recorded, nothing moved |
| `APPROVE` retrain of escalated card `dc-2026-01-03-9bd13a7e` via the API | `approvals` row; a second click → `409` |
| `loop.gate.runner --card … --mode replay` | PASS, `authorized_by=ops-verifier` read from the approval — but the replayed challenger was **v2, the serving champion** (§12); promotion correctly impossible, `REJECT`ed |
| `loop.gate.runner --card … --mode live` | challenger **v3**, PASS, shadow → v3 (~84 s) |
| reload, 55 requests, `APPROVE` | champion 2 → 3, `/ready` v3, new traffic served by v3 |
| Alias audit | the API container's moves land in the host `mlflow/registry_audit.jsonl`, `actor` = the approver |
| Narration of new cards | stored with `template/decision_card-v1`; grounded |
| Ollama fallback | Ollama running with no model pulled → HTTP 404 → template, card unaffected |
| PHI scan of every `approvals.evidence` and `shadow_predictions` row | no identifier, no feature value |

The approvals trail after these steps, one query:

```sql
SELECT a.kind, a.decision, c.card_id, r.mode, r.challenger_version,
       a.champion_version_before, a.champion_version_after, a.approver
FROM approvals a JOIN decision_cards c ON c.card_id = a.card_id
LEFT JOIN retrain_runs r ON r.run_id = a.run_id ORDER BY a.ts;
```

**Two infrastructure fixes were needed to get there**, both in §11: the MLflow
image now rejects the `mlflow:5000` Host header unless `--allowed-hosts` names
it (so the containerised API could not load any model), and the traffic script
had to read ICD-9 codes as text.

> **Note on the local database.** While verifying narration, `scenarios.run_scenario S1
> --windows 4` was re-run against the live database. Week 6 gives every run fresh
> event ids, so this appended **duplicates of S1 windows 0–2** (`de-s1-w000-2246816e`,
> `de-s1-w001-5c878750`, `de-s1-w002-0eda1717`) plus a new window 3
> (`de-s1-w003-701c835e`), and four policy-v2 cards over them
> (`dc-2026-01-01-d42a3603`, `dc-2026-01-02-b649814a`, `dc-2026-01-03-9d894574`,
> `dc-2026-01-04-368f4418`). Their persistence term counts the duplicate history
> (7 "consecutive windows"), so these four cards are not part of the S1 benchmark.
> The original Week 6/7 rows are untouched. That re-running a scenario appends
> rather than no-ops is a Week 6 behaviour worth knowing before any demo.

### 17.8 Privacy

- A shadow row holds versions, two scores, the threshold, a status and a
  latency — the payload is used in memory for one call and never stored.
- An approval's evidence holds aggregates only: card statistics, metric sets,
  shadow statistics, alias versions.
- The LLM receives the Decision Card and nothing else, and runs on the same
  machine; no card ever leaves it.
- The approval reason is free text written by the ops user, stored verbatim —
  it is the one field where a person *could* type something sensitive, so the
  dashboard should say so when Week 10 builds the form.

### 17.9 What Week 9 deliberately does not do

- **No dashboard.** The Approvals page is Week 10; it will call these endpoints.
- **No automatic promotion.** Nothing promotes without a person. Even the
  automated (`AUTO_PROCEED_SHADOW`) path stops at shadow.
- **No agreement threshold.** The shadow statistics are shown to the approver
  and recorded; they do not auto-block. The documents give the judgement to the
  person, and inventing a number here would take it away. A threshold would be
  a new `promotion-v*` file.
- **No change to Weeks 6–8 decisions.** No drift statistic, policy rule,
  confidence term, gate criterion or card field changed. What Week 9 changed in
  earlier code: `loop/engine/evaluate.py` narrates before persisting;
  `loop/gate/runner.py` and `persistence.py` read `approvals`;
  `ml/registry.py` gained `actor=` and `delete_alias`; `/predict` queues the
  shadow; Compose gained the MLflow `--allowed-hosts` fix, the API's audit
  mount and the optional `ollama` profile.


---

## 18. Week 10 — Dashboard and Audit Report

### 18.1 What it does

Week 10 makes the loop visible and clickable. `IMPLEMENTATION_ROADMAP.md`,
Week 10: *"Streamlit pages — Overview, Drift Monitor, Decision Cards (with
narrative), Champion vs Challenger, Approvals, Audit; drift-injection button
wired to S1/S2; fpdf2 audit-PDF export per card; polish the 3-minute demo
path."* The page list is ARCHITECTURE.md §3.14's, and the roadmap caps it:
*"6 pages, no more."*

| Module | Role |
|---|---|
| `dashboard/app.py` | the Streamlit entry point: sign-in, then the six pages |
| `dashboard/views/*.py` | one `render(ctx)` per page; layout only |
| `dashboard/data.py` | every query a page shows — read-only — and the derived card lifecycle state |
| `dashboard/actions.py` | the demo's non-decision buttons: inject drift, retrain + gate, demo traffic |
| `dashboard/api_client.py` | every human decision, through the API's `/ops` endpoints under the user's token |
| `dashboard/audit_pdf.py` | the per-card audit PDF (fpdf2), and `python -m dashboard.audit_pdf --card <id>` |
| `dashboard/context.py` | the token check (`/ops/whoami`), a private DB engine, registry reads |
| `loop/gate/demo.py` | the deliberately bad challenger for WORKFLOW.md §5 step 6 |
| `docker/Dockerfile.dashboard` + the `dashboard` Compose service | Streamlit on :8501 |

### 18.2 Running it

```bash
docker compose up -d                       # from docker/: now includes `dashboard` on :8501
python -m scripts.issue_dev_token --subject <you> --role ops   # paste into the sidebar

# or, outside Compose (same env as the monitor + MLFLOW_TRACKING_URI + VITALLOOP_API_URL)
streamlit run dashboard/app.py

# the audit PDF from the command line (writes reports/audit/<card>.pdf)
python -m dashboard.audit_pdf --card dc-2026-01-01-294d3a1d
```

### 18.3 The six pages

| Page | Shows | Does |
|---|---|---|
| **Overview** | API-served version, the three aliases, audited predictions by model version, latency, drift windows, cards by lifecycle state, the demo path | *Send requests* — real `/predict` calls under your token |
| **Drift Monitor** | every window per stream (max/prediction PSI, breaches), per-feature PSI/KS, the Evidently HTML report | *Inject drift* (S1 or S2) — measure and decide the next window |
| **Decision Cards** | cards filtered by stream/action/state; detail with the narrative, its source and grounding verdict, evidence, confidence terms, pinned criteria, runs, decisions | — |
| **Champion vs Challenger** | aliases; every gate verdict with its mode (REPLAY / live / CONSTRUCTED BAD) labelled; champion-beside-challenger metrics; BLOCK reasons; every check | *Retrain + gate* — replay, live, or the bad challenger |
| **Approvals** | pending promotions with the full evidence chain (gate numbers, worst subgroup drop, shadow window, each precondition); escalated retrains with their narrative | *Approve* / *Reject* with a required reason |
| **Audit** | one searchable, time-ordered log across drift windows, cards, retrain runs, decisions and alias moves; prediction audit rows by request id, input hash or caller | *Build audit PDF* → download |

### 18.4 Who the dashboard acts as

The dashboard has no user store and **holds no JWT secret** — a dashboard that
could sign tokens could record a decision in anyone's name. You paste an `ops`
token; the dashboard asks the API who it belongs to (`GET /ops/whoami`, the one
endpoint Week 10 adds) and stays locked until the API says `ops`. A clinician
token is refused.

It **reads** Postgres and MLflow directly, as §5 draws it. Every **human
decision** — authorising a retrain, approving or rejecting a promotion — is a
call to the Week 9 `/ops` endpoints with your token, so the approver on the
`approvals` row is the API's verified subject, every Week 9 precondition is
enforced where it lives, the API's refusals are shown verbatim, and an approved
promotion reloads the serving model inside the API: the champion changes live.

### 18.5 The buttons, and why they cannot corrupt the history

- **Inject drift** measures the **next unmeasured window** of S1 or S2 with the
  worker's own chain (`measure_window` → `record_window` → `evaluate_event`),
  then narrates and stores its card. Pressing it again measures the window
  after; a window already on record is never measured twice. (Re-running
  `scenarios.run_scenario` *does* append duplicate windows — §17.7's note — and
  Week 7's persistence rule then reads them as drift that lasted. The button
  cannot.) When the stream's windows are exhausted it says so.
- **Retrain + gate** is Week 8's `run_card`. An escalated card still needs an
  ops user's recorded `APPROVE` first. Three modes, each labelled wherever a
  run is shown:
  - **replay** re-registers the cached challenger (seconds). It is refused up
    front when the cached challenger *is* the serving champion — it could only
    PASS and then never be shadowed — with "run a live retrain instead".
  - **live** trains with the champion's own pipeline on the pinned data (~65–80 s).
  - **demo-bad** is the deliberately bad challenger: the champion's ranking
    inverted, built by `loop/gate/demo.py`, **never trained, logged or
    registered**, labelled `demo-bad` on its `retrain_runs` row, and judged by
    the *same* gate with the card's own criteria. It extends Week 8's
    `live | replay` modes additively; `retrain_from_card` still refuses it.
  After a PASS the page reloads the API so the new shadow starts scoring.
  Clicking twice returns the verdict on record (Week 8 idempotency).
- **Send requests** replays held-out stream rows through `/predict`; each is an
  audit row and, while a shadow is loaded, a shadow score.

### 18.6 The audit PDF

WORKFLOW.md step 18: *"one click renders the CMS-style audit PDF for any
Decision Card."* Nine sections: the decision; the narrative with its source and
grounding verdict (§3.10's "narrative section of the audit report"); the trigger
evidence; the policy's reasoning and confidence terms; the acceptance criteria
pinned before training; every retrain run and gate verdict (replay and the
constructed bad challenger labelled, BLOCK reasons listed); every human decision
with its reason; every registry alias move that names the card, with its actor;
lineage. It ends with the privacy statement.

It is built as data first (`build_report`, tested as data) and laid out second
(`render_pdf`, fpdf2 core fonts, no native dependencies). **Byte-reproducible**:
the same rows and the same `generated_at` give the same bytes.

### 18.7 Verified on the live stack

Against the running Compose Postgres, MLflow, API and the new `dashboard`
container, on 2026-09-24. Pages were driven with Streamlit's `AppTest` against
the live services with a real ops token; the heavy actions were also run inside
the dashboard container to prove its mounts.

| Demo step | Result |
|---|---|
| Sign-in | clinician token → "Token refused"; ops token → Overview (serving v3) |
| All six pages on live data | render without error (1–14 s; the first Overview load includes MLflow lookups) |
| 1. Send 10 requests | 10 scored by v3 |
| 2. Inject drift S1 (**inside the container**) | window 4 measured (next unmeasured), Evidently report written, card `dc-2026-01-05-196d00de` FULL_RETRAIN / ESCALATE_HUMAN, 25 s |
| 3. Decision Cards | card AWAITING AUTHORISATION; template narrative, grounding passed |
| 4. Approvals → authorise | `APPROVE` recorded as `ops-demo` via the API |
| 4. Replay | **refused**: cached challenger v3 is the serving champion |
| 4. Live retrain (**inside the container**) | v4 — **BLOCK**, genuinely: on S1 window 4 the recent-window ECE is 0.0602 for champion *and* challenger, above the absolute 0.05 ceiling |
| 4. Live retrain on S2 card `dc-2026-01-01-294d3a1d` via the page | v5 PASS, shadow → v5, API reloaded ("serving v3, shadowing v5"), 77 s |
| 5. Send 55 requests → Approvals → approve | champion **v3 → v5**, the API serves v5 without a restart; new traffic served by v5 |
| 6. Bad challenger on the same card | **BLOCK**, 36 failed checks, nothing moved; a second click → "Already on record" |
| 6. Audit → Build PDF; and the CLI in the container | download offered; `reports/audit/dc-2026-01-01-294d3a1d.pdf` (4 pages, full lineage) |
| Restart `api` + `dashboard` | API back on champion v5; all pages render |
| Privacy | 42 live cards' reports: 0 identifier or PHI hits, all byte-reproducible; no JWT variable in the dashboard container |

**Timing, honestly.** The machine time of steps 1–6 was about 3½ minutes with a
live retrain (~77 s) and a 55-request shadow window (~60 s at the API's per-request
SHAP latency, §12). Replay instead of live saves ~50 s and brings the path under
three minutes — but only when a PASSed challenger is cached and not yet the
champion, which after any promotion it is not. Rebuilding that demo state on
demand is the seed/reset script, which the roadmap assigns to Week 11.

### 18.8 Privacy

- Pages and the PDF are built from the aggregate audit rows only. The
  prediction lookup shows the input **hash**, never the input.
- The approval-reason box warns that its text is stored in the audit trail.
- The dashboard container holds no JWT secret and no LLM credential; narration
  of injected windows uses the same template-default, Ollama-optional path as
  the monitor, with the Decision Card as the only input.

### 18.9 What Week 10 deliberately does not do

- **No seed/reset script, no README quickstart, no CI changes** — Week 11.
- **No seventh page.** Escalated-retrain authorisation lives on Approvals.
- **No change to Weeks 6–9 decisions.** Additive changes only: the `demo-bad`
  gate mode (and the runner's refusal to alias it), `GET /ops/whoami`,
  `scripts/send_traffic.load_rows`, the `dashboard` service, a root
  `.dockerignore` (the Week 9 API image had shipped a stale `.pyc`), and
  `streamlit`/`fpdf2` in `requirements.txt` (six new pins in `constraints.txt`,
  no existing pin changed).


---

## 19. Week 11 — CI/CD, Hardening, the Demo Seed and Reset

### 19.1 What it does

`IMPLEMENTATION_ROADMAP.md`, Week 11: *"GitHub Actions — ruff → pytest →
training smoke run (sampled data) → gate check → Docker build; Compose profiles
(with/without ollama); seed script for a fresh-machine demo; test-coverage pass
(engine + gate + API ≥ 80%); README quickstart verified on a clean machine."*
Deliverable: *"green pipeline badge; `docker compose up` + seed = working demo
on a fresh clone."* RISK_ANALYSIS.md §4 adds the reset — *"One-command reset
(Compose down -v + seed)"* — and names Week 11 for the clean-clone rehearsal.

| Module | Role |
|---|---|
| `.github/workflows/ci.yml` | four chained jobs: `lint` → `test` (+ per-package coverage floor) → `smoke` → `images` |
| `scripts/ci_smoke.py` | the training smoke run on a 5,000-row synthetic sample, then the gate check |
| `scripts/seed_demo.py` | a running stack → the demo's starting state; idempotent; `--check` reports only |
| `scripts/reset_demo.py` | archive → restore-verify → remove the two state volumes → rebuild → seed |
| `README.md` | the fresh-clone quickstart and the 3-minute demo |
| `dashboard/views/overview.py` | the *Demo state* panel: whether replay can produce a promotable challenger, and what to run if not |

The Compose profiles the roadmap names already existed (Week 9's optional
`ollama` profile); Week 11 validates both renderings in CI and in a test.

### 19.2 Running it

```bash
python -m scripts.seed_demo            # after `docker compose up -d --build --wait` and `dvc repro`
python -m scripts.seed_demo --check    # exit 0 = demo-ready; changes nothing
python -m scripts.reset_demo           # prints the plan only
python -m scripts.reset_demo --yes     # archive, wipe, rebuild, reseed
python -m scripts.ci_smoke             # CI's smoke run + gate check, locally
```

Both demo scripts read `docker/.env` (the file Compose reads) for anything the
environment does not set, and `scripts.issue_dev_token` now does the same, so
the quickstart needs no `export` step.

### 19.3 The demo's starting state, and why it is honest

WORKFLOW.md §5 step 4 says *"Retrain replays a cached run"*, and
ARCHITECTURE.md §3.11 defines replay as re-registering *"a pre-trained
challenger"*. Week 10's timing note (§18.7) was that the fast path only exists
while a PASSed challenger is cached and is *not* the champion. `seed_demo`
creates exactly that and nothing else:

| Step | What | Idempotency key |
|---|---|---|
| champion | the DVC-pinned model (its md5 checked against `dvc.lock`), logged and registered through Week 4's `log_run`; `ensure_initial_champion` sets the alias | `champion` already set → skipped |
| cached challenger | the same pinned artifact, logged again as its own run tagged `pipeline=week11-demo-seed`, `role=cached-challenger`, `seed_source=dvc-pinned-model-artifact`; `challenger` moved to it with an audited reason that says *not a retrain* | `challenger` set and ≠ `champion` → skipped |
| serving | `POST /ops/models/reload` under an `ops` token for `demo-seed`, because on a fresh stack the API starts before any model exists | API already serves the champion → skipped |
| traffic | the first 20 serving-stream rows through `/predict` as the clinician `demo-seed` | audit table not empty → skipped |

It never moves `champion` after the first registration and never sets
`shadow`. The cached challenger is the model a live retrain produces on this
data (§16.9: identical metrics; §17.7: agreement 1.0), so replaying it gates
the same numbers a live retrain would — faster, and recorded as `REPLAY`.
Nothing about the model, the policy, the gate criteria or the promotion rules
changed to make the demo faster.

### 19.4 The reset, and what it does with history

A bare `down -v` would delete the audit trail. `reset_demo --yes` moves it
instead, and refuses to destroy anything it has not proven it can restore:

1. `pg_dump` the audit database into `backups/demo-reset-<UTC>/postgres.sql`;
2. restore that dump into a scratch database in the same container and compare
   every table's row count with the live database — any difference aborts;
3. copy the MLflow server's store (`docker compose cp mlflow:/mlflow`) and the
   alias log `mlflow/registry_audit.jsonl`, with SHA-256s in `manifest.json`
   and restore instructions in `README.txt`;
4. only then `docker compose down` and remove exactly `<project>_pgdata` and
   `<project>_mlflow-data` — never the optional `ollama-models` volume;
5. start the live alias log empty (its verified copy is archived beside the
   database it describes), `docker compose up -d --build --wait`, seed.

`backups/` is gitignored and excluded from the Docker build context.

### 19.5 CI

| Job | Runs | Needs |
|---|---|---|
| `lint` | `ruff check .`, `ruff format --check .` | — |
| `test` | `pytest` with branch coverage; then `coverage report --fail-under=80` separately for `loop/engine`, `loop/gate` and `api` | `lint` |
| `smoke` | `python -m scripts.ci_smoke --rows 5000` (summary uploaded as an artifact) | `test` |
| `images` | `docker compose config` with and without `--profile ollama`; `docker compose build api monitor dashboard` | `smoke` |

**The sample is synthetic, deliberately.** The dataset is DVC-tracked against a
*local* remote, so a runner never has it, and downloading it from UCI would make
every pipeline depend on a third-party server. `ci_smoke` generates a
deterministic frame in the cleaned-split shape (no real record), then runs the
production split, `ml.train.train_models`, `ml.evaluate.build_report`, and the
gate under `configs/gate-v1.yaml`. It fails unless the retrained challenger
PASSes **and** the inverted challenger BLOCKs. At 5,000 rows the absolute ECE
ceiling has headroom (0.0156 and 0.0197 on the two sets against 0.05; seeds
0–9 also all passed, worst 0.0451); at 2,000–4,000 rows the 15% window is too
noisy and the gate genuinely BLOCKs — which is why the roadmap's 5k is kept.

CI needs no Postgres, MLflow server, Ollama, secret, model download or dataset:
data-dependent tests skip, the tracking tests use SQLite, and every LLM test
uses a mocked transport.

### 19.6 Verified

**Fresh-clone rehearsal** — a new `git clone` into a temporary directory with
the Week 11 tree applied, a new Python 3.12 venv, and a separate Compose
project name so the existing stack's volumes were never touched — on
2026-09-27:

| Step | Result |
|---|---|
| `pip install -r requirements.txt -c constraints.txt` into a new venv | clean |
| `python -m ml.data.ingest` + `dvc repro` | 1 min 14 s; every **data** hash and `reports/metrics.json` identical to the committed `dvc.lock` (see the model-hash note below) |
| `docker compose up -d --build --wait` | all five services up (9 min 18 s, three image builds) |
| `seed_demo --check` on the empty stack | *NOT demo-ready: no champion is registered; no cached challenger* |
| `seed_demo` | champion v1, cached challenger v2, API reloaded onto v1, 20 predictions (1 min 51 s) |
| `seed_demo` again | nothing registered, no traffic sent — idempotent |
| The six-step demo, driven through the real pages with `AppTest` against the live services | **102.1 s** machine time: S1 w0 card `FULL_RETRAIN / ESCALATE_HUMAN` (0.3941) → authorised → **replay PASS** → 55 requests → **champion v1 → v2**, served live → bad challenger **BLOCK (37 reasons)** → PDF offered |
| Clinician token on `/ops/*` | 403; refused at dashboard sign-in |
| All six pages | render in 1.4–2.0 s |
| `restart api dashboard` | API back on v2; dashboard health 200 |
| JWT variables in the dashboard / monitor containers | 0 / 0 |
| `reset_demo --yes` on that stack | 2 min 25 s; 200 rows archived and restore-verified; only that project's two volumes removed; reseeded demo-ready |
| The demo again, from the reset state | **96.3 s**; identical card, verdicts and promotion |

**The development stack.** `reset_demo --yes` was then run on the existing
stack, which held the Week 9/10 history — 201 predictions, 24 drift windows
(including §17.7's duplicate S1 windows and S1 window 4), 42 cards, 7 retrain
runs, 8 approvals, 170 shadow scores, champion v5 and a 57-line alias log. All
of it is in `backups/demo-reset-20260927T075525Z/`, restore-verified row for
row; nothing was edited or deleted in place. The demo on the reseeded stack:
**94.9 s**. A PHI-pattern scan of every row of every table, the API logs and
the card's 4-page audit PDF found no identifier and no feature value; the PDF
labels the replay and the constructed bad challenger.

**Three problems the rehearsal found, fixed here:**

- **CI lint would have failed on its first run.** Ruff classified `mlflow` as
  first-party wherever the gitignored local store `mlflow/` exists (every dev
  machine) and third-party on a fresh checkout. `pyproject.toml` now pins it
  as third-party; six import blocks were re-sorted (no behaviour change, no
  DVC stage dependency touched).
- **The test suite wrote to the real alias log.** Two Week 4 tracking tests
  appended `test-model` rows to `mlflow/registry_audit.jsonl` on every run
  (one from 2026-09-24 and two from this week's baseline run are in the
  archive). An autouse fixture in `tests/conftest.py` now gives every test a
  private file, and a test asserts it.
- **`mlflow:latest` was unpinned.** A fresh machine would pull a server newer
  than the one Weeks 4–10 were verified against — the Week 9 `--allowed-hosts`
  break was exactly that. The image is pinned to `v3.14.0`, the version the
  stack ran.

**One finding documented rather than changed:** on a different checkout path,
`dvc repro` writes a model joblib whose bytes differ (pickle memoisation of
column-name strings) and so records a new md5 for
`models/readmission_model.joblib` in `dvc.lock`. The data hashes, the metrics
and every downstream output are identical, and the two models' predictions on
all 10,498 frozen-eval rows differ by exactly 0.0. Making the pickle
byte-stable would mean changing Week 3's training code and re-pinning lineage,
which is outside Week 11's scope.

**CI, run locally.** The `lint`, `test` and `smoke` jobs' commands were run in
a clean `python:3.12-slim` container from the tracked tree only (no dataset,
no model, no `mlflow/` directory): ruff clean; **1272 passed, 20 skipped** —
the skips are the data- and artifact-dependent tests, by design — in 23 min 51 s;
branch coverage `loop/engine` 99%, `loop/gate` 99%, `api` 85%, every floor
met; the smoke run PASSed the retrained challenger and BLOCKed the bad one.
The `images` job's builds are the rehearsal's `docker compose up --build`
above, and `docker compose config` was validated with and without the
`ollama` profile. On Windows with the data present the suite is **1292
passed, 0 skipped** (21 min 35 s); `scripts/` is at 85%.

**Not verified:** the workflow has not run on GitHub, because nothing has been
pushed. The README badge will show the first real run.

### 19.7 Privacy and security

- The smoke sample is generated; no record is read, written or uploaded.
- The seed sends stream rows only through the API's contract (no identifiers)
  and prints no secret; the tokens it mints live in memory for one run.
- Archives contain the audit database, which holds hashes and aggregates only
  (§9.3); they stay local (`backups/` gitignored and excluded from images).
- The dashboard still holds no JWT secret; the seed and reset run on the host.

### 19.8 What Week 11 deliberately does not do

- **No model, policy, gate or promotion change** to make the demo faster —
  the speed comes from replaying a correctly labelled cached challenger.
- **No new page.** The demo state is a panel on Overview.
- **No push.** CI is configured and locally exercised; its first GitHub run is
  the owner's to trigger.
- **No change to the monitor's restart behaviour:** a restarted `monitor`
  begins the `live` stream at window 0 again and appends duplicate live
  windows (the Week 6 behaviour §17.7 describes for scenarios). It affects
  only the quiet `live` stream, not the S1/S2 demo path, which the dashboard
  button measures exactly once per window.


---

## 20. Week 12 — Benchmark Tables, Report, Release

### 20.1 What it does

`IMPLEMENTATION_ROADMAP.md`, Week 12: *"final-year report (reuse
PROJECT_DESIGN.md structure); results tables from the S1–S5 benchmark
(detection latency, false-trigger rate, gate outcomes); demo video recording;
viva Q&A drill …; freeze `v1.0` tag."* Deliverables: *"report draft, demo
video, tagged release."*

| Deliverable | Where | Status |
|---|---|---|
| Report draft | `docs/FINAL_REPORT.md` | done — measured results, limitations, the contribution-by-contribution evidence |
| Benchmark tables | `scenarios/benchmark.py` → `reports/benchmark.json`, `reports/benchmark.md` (tracked) | done |
| Viva Q&A drill | `docs/FINAL_REPORT.md` Appendix A — prepared answers to the five named questions | answers written; the drill itself is a rehearsal for people |
| Demo video | `docs/FINAL_REPORT.md` Appendix B — the timed recording script | **not recorded** |
| `v1.0` tag | local annotated tag on the Week 12 commit | done, **not pushed** |

### 20.2 The benchmark

```bash
python -m scenarios.benchmark            # ~5 min: every window, both policies, the gate
python -m scenarios.benchmark --no-gate  # decisions only
```

Every complete 2,000-row window (5 per scenario) is measured once with the
monitor's `measure_window`, recorded and decided with the worker's own
`record_window` → `evaluate_event` chain under `policy-v1` and `policy-v2`, in
a throwaway SQLite database — nothing touches the live stack. For
`policy-v2`'s retrain cards, one live retrain on the pinned data (the
production `ml.retrain.train_challenger`; every card pins the same hash) and
the deliberately bad challenger are gated per card with
`loop.gate.runner.gate_challenger`, the card's pinned criteria and its own
evaluation sets. Definitions (onset, detection latency, false-trigger rate)
are in the module docstring.

Measured on 2026-09-27 (5 min 1 s):

| | policy-v1 | policy-v2 |
|---|---|---|
| S1–S3 detection latency | 0 windows | 0 windows |
| S5 control false-trigger rate | **80%** (4/5) | **0%** (0/5) |
| S4 label drift | "detected" at window 1 — by the control's administrative drift, not the label drift | not detected (by design: rule 5 needs matured labels) |
| Retrained challenger at the gate | — | 11 PASS, 4 BLOCK (absolute ECE ceiling on late windows) |
| Bad challenger at the gate | — | 15/15 BLOCK |

Window-level rows, confidences and every BLOCK reason are in
`reports/benchmark.md`. Two notes the report repeats: policy-v2's calibration
used the control's windows 0–2, so those are in-sample (windows 3–4 are not,
and also stayed quiet); and the four retrained BLOCKs are windows where the
champion is equally miscalibrated — ECE is an absolute ceiling by design.

A first run of the benchmark stopped at the gate stage on a bug in the new
module (it read `scenario` from the card, which nests it under `trigger`); a
unit test caught the same bug, and the tables above are from the fixed run.

### 20.3 Verified

- The benchmark's first windows reproduce §15.5's recorded confidences exactly
  (S1 0.3941, S2 0.4857, S3 0.4599 under policy-v2).
- `tests/test_benchmark_tables.py`: the metrics as pure functions, and the
  decide/gate loop on injected measurements through the real engine.
- Final end-to-end verification of the release is in §20.4.

### 20.4 Release verification

On 2026-09-27, before the `v1.0` commit:

| Check | Result |
|---|---|
| `pytest` (full suite, data present) | **1299 passed, 0 failed, 0 skipped** (22 min 27 s) |
| Branch coverage | `loop/engine` 99%, `loop/gate` 99%, `api` 87%, `scripts` 85%, `dashboard` 89%, `scenarios` 77% |
| `ruff check .`, `ruff format --check .`, `pre-commit run --all-files` | clean |
| `dvc status` | up to date |
| `docker compose config` (default and `--profile ollama`) | valid; `ollama` only in the profile |
| `reset_demo --yes` on the development stack | 2 min 23 s; the Week 11 demo run's rows archived and restore-verified; reseeded demo-ready |
| The six-step demo through the real pages | **94.5 s**; clinician 403 on `/ops/*`; all six pages 1.4–1.7 s; replay PASS; champion v1 → v2 live; bad challenger BLOCK (37); PDF offered |
| `restart api dashboard monitor` | API back on v2; dashboard and `/docs` 200 |
| Audit chain by one SQL query | authorisation → replay PASS → promotion 1 → 2, approver `ops-demo` |
| Secrets in the dashboard / monitor containers | none (JWT secret, API keys) |
| PHI-pattern scan: every table, API logs, audit PDF, `reports/benchmark.*` | 0 hits |
| Test runs appending to `mlflow/registry_audit.jsonl` | none (5 lines = seed + demo) |

After the tag, the stack was reset once more so it is left in the seeded,
demo-ready state.
