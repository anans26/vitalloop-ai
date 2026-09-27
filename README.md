# VitalLoop 2.0

[![CI](https://github.com/anans26/vitalloop-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/anans26/vitalloop-ai/actions/workflows/ci.yml)

A self-healing, audit-first MLOps loop for 30-day hospital-readmission risk
(UCI Diabetes 130-US). It watches its own serving stream for drift, decides
what to do with a versioned deterministic policy, retrains against a pinned
dataset, refuses any challenger that is worse, shadows the one that is not, and
promotes it only when a person approves — leaving an audit trail for every step.

> **Deterministic code decides. The LLM only narrates. A human approves anything
> that touches clinicians.**

**Release `v1.0`** (the Week 12 submission). Results, limitations and the
measured benchmark: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

**This is a research and education system, not a medical device.** Its scores
are decision-support risk stratification on a public, de-identified 1999–2008
dataset; they are not clinical advice.

## Quickstart (fresh clone)

**You need:** Python **3.12**, Git, and Docker Desktop with Compose v2
(running). Internet access once, for the Python packages, the Docker base
images and the UCI dataset. No cloud account, API key, GPU or LLM is needed.

```bash
git clone https://github.com/anans26/vitalloop-ai.git
cd vitalloop-ai

# 1. Python environment (Windows: `py -3.12`; macOS/Linux: `python3.12`)
py -3.12 -m venv .venv
source .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
                                       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt

# 2. Data and the model: download UCI once, then rebuild every artifact.
#    Every dataset hash and reports/metrics.json must come out identical to
#    the committed dvc.lock. On a different machine or checkout path the
#    model's pickle bytes (not its predictions) can differ, so dvc.lock may
#    show a new md5 for models/readmission_model.joblib only -- expected.
python -m ml.data.ingest
dvc repro

# 3. Local secrets. Edit docker/.env: set POSTGRES_PASSWORD, and set
#    VITALLOOP_JWT_SECRET to the output of:
#      python -c "import secrets; print(secrets.token_urlsafe(48))"
cp docker/.env.example docker/.env

# 4. The stack: postgres, mlflow, api, monitor, dashboard
cd docker && docker compose up -d --build --wait && cd ..

# 5. The demo's starting state: champion, cached challenger, baseline traffic
python -m scripts.seed_demo

# 6. An ops token to sign in to the dashboard with
python -m scripts.issue_dev_token --subject <your-name> --role ops
```

Open **http://localhost:8501**, paste the token into the sidebar, and check the
Overview's **Demo state** panel says *Replay ready*.

| Service | URL |
|---|---|
| Dashboard (Streamlit) | http://localhost:8501 |
| API + OpenAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

`seed_demo` prints what it did and exits 0 only when the stack is demo-ready;
`python -m scripts.seed_demo --check` reports the state without changing it.
Running it twice changes nothing.

## The 3-minute demo

From a seeded stack (`WORKFLOW.md` §5; the same list is on the Overview page):

1. **Overview** — the champion is serving; send a few requests.
2. **Drift Monitor** — *Inject drift* **S1**. The monitor measures the next
   window and the Decision Engine emits a narrated Decision Card.
3. **Decision Cards** — read the card. On this data it is `FULL_RETRAIN`,
   escalated to a person (confidence below the auto-proceed line).
4. **Approvals** — authorise the retrain. **Champion vs Challenger** —
   *Retrain + gate*, **Replay** → gate **PASS** → the challenger is in shadow.
   Replay re-registers the seeded cached challenger and is recorded as `REPLAY`.
5. **Overview** — send 55 requests to fill the shadow window. **Approvals** —
   approve the promotion: the champion version changes live.
6. **Champion vs Challenger** — the same card with the **deliberately bad
   challenger** → gate **BLOCK**, nothing moves. **Audit** — build the card's
   audit PDF.

**Reset between runs:**

```bash
python -m scripts.reset_demo          # prints the plan; changes nothing
python -m scripts.reset_demo --yes    # archive, wipe, rebuild, reseed
```

The reset never silently deletes history: it `pg_dump`s the audit database,
copies the MLflow store and the alias-move log into `backups/` (gitignored),
proves the dump restores, and only then removes the two state volumes. The
optional Ollama model volume is left alone.

## Optional: LLM narration with Ollama

Narration defaults to a deterministic **Jinja2 template**, which is also the
mandatory fallback. Ollama is optional, self-hosted, and receives only the
aggregate Decision Card — never a patient record. Nothing downloads a model
unless you ask it to:

```bash
cd docker
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.1:8b     # ~4.9 GB, once
# set VITALLOOP_NARRATION_BACKEND=ollama in docker/.env, then:
docker compose up -d monitor dashboard
```

If Ollama is stopped, has no model, or writes a narrative that fails the
grounding check, the template narrates instead and the card is unaffected.

## Development

```bash
pytest -q                                  # the full suite; no services needed
ruff check . && ruff format --check .
python -m scripts.ci_smoke                 # CI's training smoke run + gate check
python -m scenarios.benchmark              # S1-S5 results tables -> reports/benchmark.md (~5 min)
```

CI (`.github/workflows/ci.yml`) runs ruff → pytest with an 80% branch-coverage
floor on the Decision Engine, the gate and the API → a training smoke run on a
5,000-row synthetic sample → a gate check (the retrained challenger must PASS,
the deliberately bad one must BLOCK) → Compose validation and image builds. It
needs no database, MLflow server, Ollama or dataset.

## Documentation

| Where | What |
|---|---|
| [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) | The final report: architecture as built, measured results, limitations, viva notes |
| [`reports/benchmark.md`](reports/benchmark.md) | The S1–S5 benchmark tables (detection latency, false-trigger rate, gate outcomes) |
| [`docs/RUNNING_THE_PROJECT.md`](docs/RUNNING_THE_PROJECT.md) | Everything that is built, how to run it, and what was verified, week by week |
| [`docs/PROJECT_REPOSITORY_GUIDE.md`](docs/PROJECT_REPOSITORY_GUIDE.md) | The repository, file by file, and the project's status |
| [`project_docs/`](project_docs/README.md) | The design: architecture, workflow, roadmap, risks, research contributions |
