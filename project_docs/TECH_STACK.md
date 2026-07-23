# VitalLoop 2.0 — Technology Stack

> Every choice below was re-derived from the problem statement, not inherited from v1.
> Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) · [PROJECT_DESIGN.md](PROJECT_DESIGN.md)

## Stack at a Glance

| Layer | Technology | v1 proposed | Verdict |
|---|---|---|---|
| Frontend / Dashboard | **Streamlit** | React/Next.js + WebSockets | Changed — 3–4 weeks saved |
| Backend / API | **FastAPI + Pydantic v2** | FastAPI | Kept |
| Database | **PostgreSQL 16** | PostgreSQL | Kept |
| Experiment tracking | **MLflow** | MLflow | Kept |
| Model registry | **MLflow Registry (aliases)** | MLflow | Kept |
| Data versioning | **DVC (local remote)** | DVC + S3 | Simplified |
| Monitoring / drift | **Evidently** | Evidently | Kept |
| Data validation | **pandera** | Evidently + pandera | Kept (pandera for schema, Evidently for drift) |
| Model | **LightGBM + sklearn calibration + SHAP** | LightGBM/XGBoost + SHAP | Kept (LightGBM fixed as primary) |
| Decision making | **Deterministic Decision Engine (pure Python)** | LangGraph multi-agent LLM | **Replaced** |
| Reasoning framework | *(none — not needed)* | LangGraph | **Removed** |
| LLM | **Ollama + Llama 3.1 8B — narration only** | Ollama Llama 3.1 8B (decision-maker) | Kept, demoted |
| Scheduling | **APScheduler** (in monitor worker) | — (unspecified) | Added |
| Containerization | **Docker + Docker Compose** | Docker Compose | Kept |
| CI/CD | **GitHub Actions** | GitHub Actions | Kept |
| Authentication | **JWT (PyJWT) + role claims** | JWT | Kept |
| Logging | **structlog (JSON) + Postgres audit rows** | unspecified | Specified |
| Reporting | **fpdf2** (audit PDF) + Evidently HTML | "CMS-style PDF" | Specified |
| Testing | **pytest + ruff** | unspecified | Specified |
| Cloud | **None (design section only)** | AWS ECS/Fargate stretch | Dropped from build scope |

---

## Layer-by-Layer Rationale

### Frontend — Streamlit

- **Why**: Python-native; every dashboard page (drift panels, Decision Card viewer, champion-vs-challenger, approvals, audit log) is buildable in hours, not weeks. The team's effort belongs in the loop, which is what gets judged and examined.
- **Advantages**: zero JS toolchain; direct read of Postgres/MLflow; `st.button` gives human-in-the-loop approval for free; auto-refresh replaces WebSockets.
- **Alternatives**: React/Next.js (better polish, ~3–4 weeks cost — stretch only); Dash (heavier callbacks model); Gradio (too form-oriented for a monitoring UI).

### Backend — FastAPI + Pydantic v2

- **Why**: async, typed request validation at the boundary, automatic OpenAPI docs (excellent for viva), industry standard for ML serving.
- **Advantages**: Pydantic schema = the API contract = the audit contract; dependency-injected auth; trivial to test with `TestClient`.
- **Alternatives**: Flask (no native validation/async), Django (ORM and admin unneeded), BentoML/MLserver (hides exactly the serving logic this project needs to demonstrate).

### Database — PostgreSQL 16

- **Why**: relational lineage across `drift_events → decision_cards → retrain_runs → approvals` is the audit story; JSONB stores card payloads without schema churn.
- **Advantages**: one database for state + audit; mature Docker image; append-only discipline is an application-level convention, no exotic tech needed.
- **Alternatives**: SQLite (fine for dev, no concurrent writers for API+worker), MongoDB (loses relational lineage), TimescaleDB (overkill).

### Experiment Tracking & Registry — MLflow

- **Why**: tracking + registry + artifact store in one self-hostable service; **aliases** (`champion`/`challenger`/`shadow`) model promotion natively.
- **Advantages**: every run pins params, git commit, DVC hash, metrics; the registry UI is a second demo surface for free.
- **Alternatives**: Weights & Biases (SaaS — data leaves the environment; wrong posture for a PHI-adjacent story), DVC-only pipelines (no registry semantics), ZenML (an abstraction layer this project should show it doesn't need).

### Data Versioning — DVC with a local remote

- **Why**: dataset hash pinned to git commits; the Decision Card's `candidate_data_version` is a DVC hash.
- **v1 correction**: the S3 remote added AWS credentials and cost to a system that must demo offline. A local directory remote gives identical reproducibility semantics.
- **Alternatives**: Git LFS (no pipeline semantics), lakeFS (infrastructure-heavy), plain folders + checksums (no tooling, error-prone).

### Drift Detection & Monitoring — Evidently

- **Why**: PSI/KS per feature, prediction drift, data-quality checks, and presentable HTML reports out of one library.
- **Advantages**: report artifacts double as dashboard content and appendix material; metrics are exportable as JSON for the Decision Engine.
- **Alternatives**: NannyML (strong performance-estimation, weaker report surface), Alibi Detect (more powerful detectors, more work), hand-rolled PSI (acceptable fallback and a good unit-test cross-check — worth implementing for 2–3 features as validation).

### Data Validation — pandera

- **Why**: schema-as-code beside the pipeline; fails fast with precise errors.
- **Alternatives**: Great Expectations (much heavier setup for equal benefit here), Pydantic-only (fine at the API boundary, clumsy on dataframes).

### Model — LightGBM + isotonic calibration + SHAP

- **Why**: gradient boosting remains the correct tool for mid-size tabular clinical data; calibration turns margins into trustworthy probabilities; SHAP TreeExplainer is fast and exact for trees.
- **Alternatives**: XGBoost (equivalent; pick one and stop), TabNet/FT-Transformer (worse on this data size, slower, harder to explain), logistic regression (keep as the interpretable baseline in the report).

### Decision Making — deterministic Decision Engine (replaces LangGraph)

- **Why replaced**: the retrain decision must be **reproducible, unit-testable, and explainable to an examiner or regulator**. A five-node LangGraph system makes a threshold comparison non-deterministic, hard to test, and dependent on a local LLM's mood. The actual logic — "PSI over threshold on ≥ k features for ≥ w windows → retrain" — is 200 lines of Python with a truth table.
- **Advantages**: policy versions are code artifacts; every branch has a test; confidence is a formula, not a vibe; runs in microseconds offline.
- **Alternatives**: LangGraph (kept *as vocabulary* — the report explains why it was rejected, which is itself a research point), a rules DSL like Durable Rules (unnecessary indirection).

### LLM — Ollama + Llama 3.1 8B, narration only

- **Is an LLM actually necessary? No — not for any decision.** It is *valuable* for one thing: translating a structured Decision Card into plain English for clinicians, ops, and audit reports. That is a genuine LLM strength and carries zero safety weight because a grounding check validates every quoted number against the card, and a Jinja2 template is the always-available fallback.
- **Advantages**: self-hosted (PHI-safe posture preserved), offline demo, keeps a defensible GenAI component for hackathon/viva.
- **Alternatives**: no LLM + templates only (simplest; loses the GenAI narrative), API LLMs like Claude/GPT (better prose, but sends metadata off-box and breaks the offline demo — wrong trade here).

### Scheduling — APScheduler

- **Why**: in-process scheduler inside the monitor worker; no Airflow/Prefect deployment burden for one periodic job.
- **Alternatives**: cron in container (fine, less visible), Prefect/Airflow (orchestrators for tens of DAGs, not one loop — classic over-engineering trap).

### Containerization — Docker Compose

- **Why**: `docker compose up` = the whole system (api, dashboard, mlflow, postgres, monitor, optional ollama). Reviewer-proof, offline-proof.
- **Alternatives**: Kubernetes/k3s (nothing here needs orchestration; mention in future work), plain venvs (loses the one-command story).

### CI/CD — GitHub Actions

- **Why**: free for public repos; the pipeline *is* part of the project's thesis: lint → unit tests (Decision Engine + gate) → training smoke run → gate check → image build.
- **Alternatives**: GitLab CI (fine, team is on GitHub), Jenkins (self-hosted burden).

### Auth — JWT (PyJWT) with role claims

- **Why**: stateless, standard, and role claims (`clinician`/`ops`) demonstrate governance separation: the person who *sees* risk scores is not automatically the person who *promotes* models.
- **Alternatives**: OAuth2/OIDC via Keycloak (real-world correct, deployment burden too high — future work), API keys (no identity for the approval audit row).

### Logging & Reporting

- **structlog** JSON logs (no PHI — hashes only) for services; **Postgres audit rows** as the authoritative record; **fpdf2** renders the per-card audit PDF (pure-Python, no native dependencies — important on Windows dev machines); Evidently HTML reports attached as artifacts.

### Visualization

- Streamlit-native charts + Plotly for calibration curves and drift timelines; SHAP's built-in plots exported as images for reports.

### Cloud — deliberately none

- The architecture is stateless-service + registry + database and would map 1:1 onto ECS/Cloud Run + RDS + S3. That mapping is a **paper section** (ARCHITECTURE.md future-work) — building it adds cost, credentials, and demo fragility while demonstrating nothing the Compose deployment doesn't.
