# Running the Project — VitalLoop 2.0

> **Milestone covered by this document: Week 6 of the 12-week roadmap.**
> This document describes only what exists and runs in the repository today. Features planned for Week 7 onward (the Decision Engine, the retrain pipeline and gate, the dashboard, LLM narration) are **not implemented** and are not covered here — see `PROJECT_REPOSITORY_GUIDE.md` §11 for the full pending list.

---

## 1. Project Overview

**VitalLoop 2.0** is a self-healing MLOps system for predicting 30-day hospital readmission risk, designed around one principle: *deterministic code decides, an LLM only narrates, a human approves anything clinician-facing*. Full design intent lives in `project_docs/` (`PROJECT_DESIGN.md`, `ARCHITECTURE.md`, etc.) — those are planning documents, not a description of current code.

**Current implementation stage:** end of Week 6 of a 12-week roadmap (`project_docs/IMPLEMENTATION_ROADMAP.md`). The repository contains the data plane (ingestion, validation, cleaning, features, DVC versioning), the model plane (LightGBM + isotonic calibration + SHAP, tracked and registered in MLflow), the serving plane (authenticated FastAPI `/predict` with fail-closed Postgres audit rows), and the monitoring half of the loop (Evidently drift detection, `drift_events` persistence, the seeded S1–S5 benchmark). There is no decision engine, no retrain pipeline, no gate, no dashboard, and no LLM integration of any kind yet.

**Current milestone (Week 6) exit criteria, both met:**
- An injected drift scenario produces a `drift_events` row and an Evidently report per window (§14.4).
- The no-drift control is measured under the same thresholds, and the residual signal it shows is documented rather than tuned away (§14.4).

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
| **pytest** | 9.1.1 | Standard Python test runner | 328 tests: Week 2 data contract, Week 3 model/calibration/SHAP, Week 4 tracking/registry, Week 5 API/auth/audit contracts, and Week 6 drift/PSI/scenario/persistence contracts |
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
| `LogisticRegression` convergence warning in the baseline notebook | Unscaled numeric features slow `lbfgs` convergence | Already fixed (a `StandardScaler` step was added to the baseline's numeric branch) — if you see this, check you're on the current notebook | Re-running the notebook produces no `ConvergenceWarning` |

---

## 12. Current Limitations

(Scoped strictly to Week 6 — this is not a roadmap of what's missing overall, just what a developer running the project today should know.)

- **No frontend.** The Streamlit dashboard is Week 10; the API's OpenAPI page at `/docs` is the only interactive surface.
- **No governed promotion, and no shadow scoring.** The API serves the `champion` alias, but `challenger` and `shadow` remain unused: shadow dual-scoring is Week 9 middleware, and the validation gate and human approval arrive in Weeks 8–9. `champion` still means "the first registered version", not "a model that passed a gate".
- **Serving latency is above target.** A warm `/predict` takes roughly 350 ms against the roadmap's < 200 ms goal; per-request SHAP dominates. The roadmap's own mitigations (batching, a cached explainer path) are not implemented.
- **No Decision Engine, no gate, no shadow deployment, no dashboard.** All Week 7+. The monitor writes `drift_events` rows; nothing reads them yet.
- **No delayed-label performance monitoring.** ARCHITECTURE.md §3.7 lists AUROC/recall on a matured-label window alongside input and prediction drift. Week 6 implements the two leading indicators only; the matured-label arm (and scenario S4's detection, as opposed to its injection) needs the label-latency handling that Weeks 7–8 introduce.
- **The S5 control is not perfectly silent on this dataset** — see §14.4. Two administrative features drift genuinely across the serving stream, which is a finding about the UCI extract, not a bug in the monitor.
- **No LLM/Gemini integration exists yet**, and the offline-LLM-vs-Gemini-mandate conflict (§7) is unresolved.
- **No shared remote history** — commits exist locally and `origin` is configured, but the Week 1–3 history has not been pushed.
- **Single-machine, local-only setup.** DVC's remote is a local directory (`dvc-storage/`, gitignored) — there is no shared/team remote configured.
- **Windows encoding caveat** (§11): non-ASCII characters in notebooks require `PYTHONUTF8=1` on this OS when using `jupyter execute` from a script; interactive Jupyter (browser) does not hit this issue.

---

## 13. Current Project Status

**Completed (Week 1 + Week 2 + Week 3 + Week 4 + Week 5 + Week 6 exit criteria):**
- Repo scaffold, ruff + pre-commit + basic CI (lint + test) configuration
- Docker Compose running PostgreSQL 16 (holding `predictions` and `drift_events`)
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
- 328/328 pytest tests passing; `ruff check .` and `ruff format --check .` clean

**In Progress:** nothing — Week 6 is a clean stopping point with no partially-built component.

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

**Remaining (Week 7 onward — see `PROJECT_REPOSITORY_GUIDE.md` §11 for the full list):** the Decision Engine + Decision Card, the retrain pipeline + validation gate, shadow deployment + human approval + LLM narration, the Streamlit dashboard + audit PDF export, CI/CD hardening, and final reporting/viva prep.

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
