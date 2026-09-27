# VitalLoop 2.0 — Final Report (draft)

## Self-Healing, Audit-First MLOps for 30-Day Readmission Risk

> **Status:** Week 12 report draft, `v1.0`. It describes the system as built and
> measured in this repository; every number below is traced to a file, a test or
> a documented run. Where a planned outcome was not achieved, it says so.
> Structure follows `project_docs/PROJECT_DESIGN.md`, as the roadmap asks.
> **Research and education system — not a medical device.**

---

## Summary

VitalLoop 2.0 serves a calibrated LightGBM model of 30-day hospital
readmission (UCI Diabetes 130-US) and governs its own lifecycle. A monitor
measures drift on every window of the serving stream; a **deterministic,
versioned policy** turns each measurement into a **Decision Card**; a retrain
runs only on the card's pinned data; a **validation gate** refuses any
challenger that is worse, miscalibrated or unfair to a subgroup; a gated
challenger is **shadow-scored**, and only a **logged human approval** moves the
champion. An LLM is present only as an optional narrator of the card, behind a
grounding check, with a deterministic template as default and fallback.

What was measured (details in §9):

| Claim | Result |
|---|---|
| Detect S1–S3 within one window | **Met** — latency 0 on every scenario, all 5 windows (policy-v2) |
| Zero false triggers on the no-drift control | **Met under policy-v2 (0/5 windows)**; the uncalibrated policy-v1 triggered on 4/5 |
| Gate blocks a bad challenger | **Met** — 15/15 retrain cards BLOCKed the deliberately bad challenger |
| Gate passes an equivalent retrain | **11/15**; the other 4 BLOCKed on the absolute calibration ceiling (champion equally miscalibrated there) |
| Model at published benchmark (~0.65–0.69 AUROC) | **Not met** — 0.6024 calibrated AUROC under a stricter chronological, patient-level protocol (§6, §9.1) |
| Label-drift scenario S4 detected | **Not detected, by design** — needs matured labels (§9.3, §11) |
| Full audit chain reconstructable | **Met** — card → retrain run → gate → approval → alias move, by SQL and in the audit PDF |
| Demo in under 3 minutes from a seeded state | **Met** — 94.9–102.1 s machine time, three runs (§9.6) |

---

## 1. The Problem

A deployed clinical model degrades as case mix, coding practice and care
protocols change, and the labels that would show it arrive 30 days late. The
engineering problem is to notice, decide, retrain and redeploy without ever
shipping a worse model silently. The research problem is to make the
*retraining decision itself* a governed, reproducible, auditable artifact
rather than an alert followed by undocumented human glue (RESEARCH_NOVELTY.md).

## 2. What v2 Kept and Removed from v1

The original design (v1) let a LangGraph multi-agent LLM system decide when to
retrain. v2 keeps every governance idea — the Decision Card, MLflow, DVC,
Evidently, FastAPI, PostgreSQL, shadow deployment, validation gate, human
approval, audit trail — and replaces the LLM decision-maker with a
deterministic policy engine (PROJECT_DESIGN.md §2–§3). The LLM survives in the
one place it helps: narrating a decision already made.

## 3. The Architecture, as Built

Five planes, one `docker compose up` (ARCHITECTURE.md §5):

| Service | Built | Role |
|---|---|---|
| `postgres` 16 | Week 1/5 | the append-only audit store: `predictions`, `drift_events`, `decision_cards`, `retrain_runs`, `approvals`, `shadow_predictions` |
| `mlflow` (pinned `v3.14.0`) | Week 4 | tracking and the registry; aliases `champion` / `challenger` / `shadow`; every alias move appended to `mlflow/registry_audit.jsonl` |
| `api` (FastAPI) | Weeks 5, 9 | JWT-authenticated `/predict` with a fail-closed audit row; post-response shadow scoring; `ops`-role approval endpoints |
| `monitor` (APScheduler) | Weeks 6–7 | Evidently PSI/KS/prediction drift per 2,000-row window; the Decision Engine on the same tick |
| `dashboard` (Streamlit) | Week 10 | six pages; every human decision goes through the API under the user's token |
| `ollama` | Week 9 | optional Compose profile; not started by default |

The loop (WORKFLOW.md): monitor → `drift_events` → Decision Engine
(`configs/policy-v2.yaml`) → Decision Card → human authorisation when the card
escalates → retrain on the card's pinned DVC hash (`live`, or `replay` of a
cached challenger) → gate (`configs/gate-v1.yaml`) → `shadow` alias → shadow
window (`configs/promotion-v1.yaml`: ≥ 50 dual-scored requests) → human
approval → `champion` alias moves → the API reloads.

**Separation of powers, enforced in code:** only
`loop/approval/promotion.py` moves `champion` after the initial registration
(a test scans the source tree for any other setter); the gate is a pure
function; the narrator receives only the card, minus its narrative fields.

## 4. Decision Engine and Decision Card

Six rules in precedence order, a decomposed confidence score
(severity, breadth, persistence, evidence completeness, with fixed weights),
and a frozen Pydantic card carrying the trigger evidence, policy version,
action, disposition, confidence breakdown, pinned data version and the
acceptance criteria the challenger must meet — fixed *before* training
(RUNNING_THE_PROJECT.md §15). Two policies ship: `policy-v1` transcribes
ARCHITECTURE.md §3.8 verbatim; `policy-v2` differs only in `psi_breach`
(0.10 → 0.20), calibrated on the no-drift control as the roadmap's Week 6 risk
line requires (§15.6). Both remain loadable because stored cards name them.
Leading-indicator evidence caps confidence at 0.875; auto-proceeding needs
0.75, and every benchmark retrain card stayed below it (maximum 0.673), so every
one escalated to a person.

## 5. AI Components

| Component | Implementation | Decides anything? |
|---|---|---|
| Readmission model | LightGBM + isotonic calibration (5-fold), SHAP TreeExplainer for top-3 factors | scores only |
| Drift detection | Evidently PSI/KS + a hand-rolled PSI cross-check | measures only |
| Decision Engine | deterministic rules + formula (`loop/engine/`) | **yes** — the only thing that does |
| Narration | Jinja2 template `decision_card-v1` (default and mandatory fallback); optional Ollama `llama3.1:8b` | no — attached after the decision, never edits it |

**LLM fallback architecture.** `VITALLOOP_NARRATION_BACKEND=template` is the
default everywhere (Compose, `.env.example`, CI). With `ollama`, the narrator
sends only the aggregate card (temperature 0, fixed seed) and runs a grounding
check: every number quoted must be a card value (rounding and percentages
allowed), a list length, or a number in the card's rationale; every date must
be a card date. Ollama down, no model pulled, a timeout, or an ungrounded
answer → the template narrates and the card records `template/decision_card-v1`
as its source. No Gemini, cloud or paid LLM exists in the repository; CI never
downloads a model (every LLM test uses a mocked transport). Verified live:
Ollama running with no model → HTTP 404 → template (RUNNING_THE_PROJECT.md
§17.7). **Not measured:** a live `llama3.1:8b` faithfulness rate, because the
4.9 GB model was never pulled on the build machine.

## 6. Dataset and Evaluation Protocol

UCI Diabetes 130-US (101,766 encounters, 1999–2008), downloaded with
`ucimlrepo`, validated with a pandera schema, hash-pinned with DVC.

**Protocol (stricter than the published benchmarks, deliberately):**

1. Leakage removal — discharges to death/hospice (11, 13, 14, 19, 20, 21)
   dropped; target binarised (`<30` → 1).
2. **One encounter per patient** (the first, chronologically), so no patient
   is in two splits — a unit test asserts zero overlap on the real data.
3. **Chronological split** by `encounter_id` (the dataset has no dates):
   70% train (48,993), 15% frozen evaluation (10,498), 15% held-out serving
   stream (10,499). Positive rate: 9.7% train, **8.1%** eval, 6.3% stream —
   a prevalence shift across time.
4. The frozen evaluation slice is read only to score; the serving stream is
   never opened by training or tracking code (tests scan those modules for it).

## 7. Technology Stack

As `project_docs/TECH_STACK.md`, all self-hosted and free: Python 3.12,
pandas, scikit-learn, LightGBM, SHAP, pandera, DVC (local remote), MLflow,
FastAPI + PyJWT, SQLAlchemy + PostgreSQL, structlog, Evidently, APScheduler,
Streamlit, fpdf2, Jinja2, optional Ollama, Docker Compose, GitHub Actions,
ruff, pytest. Versions are pinned in `constraints.txt`.

## 8. Implementation Timeline

| Week | Delivered |
|---|---|
| 1–2 | EDA, baseline, pandera schema, cleaning, features, patient-level chronological split, DVC |
| 3 | LightGBM + isotonic calibration, metric suite incl. subgroups, SHAP |
| 4 | MLflow tracking with git/DVC lineage; registry; audited alias moves |
| 5 | Authenticated, audited serving API |
| 6 | Evidently monitor, seeded S1–S5 scenarios |
| 7 | Decision Engine, Decision Card, policy-v1 → policy-v2 calibration |
| 8 | Retrain on pinned data, validation gate, replay mode |
| 9 | Shadow scoring, approval flow, grounded narration |
| 10 | Six-page dashboard, per-card audit PDF |
| 11 | CI pipeline, demo seed, archiving reset, README quickstart, clean-clone rehearsal |
| 12 | S1–S5 benchmark tables, this report, `v1.0` |

## 9. Results

### 9.1 The model (`reports/metrics.json`, frozen eval slice, 10,498 rows)

| Metric | Raw LightGBM | Calibrated (served) |
|---|---|---|
| ROC-AUC | 0.5996 | **0.6024** |
| Average precision (AUPRC) | 0.1253 | 0.1301 |
| Brier score | 0.0743 | 0.0741 |
| Log loss | 0.2791 | 0.2787 |
| Expected calibration error | 0.0189 | 0.0201 |
| Recall @ top decile | 0.171 | 0.171 |

Baseline logistic regression: 0.6191 on a random split (Week 1), **0.5820** on
the same chronological protocol — so the like-for-like comparison is LightGBM
0.5996 (raw) vs logistic 0.5820. The published 0.64–0.69 figures use random,
non-deduplicated splits; this project did not tune toward them (Week 3 capped
tuning, RUNNING_THE_PROJECT.md §13). **The roadmap's "at or above published
benchmarks" outcome was not met in absolute terms.**

Calibrated ROC-AUC by subgroup (count): gender F 0.596 (5,493), M 0.609
(5,003); race Caucasian 0.600 (8,560), African American 0.573 (1,149),
Hispanic 0.718 (197), Other 0.533 (158), Asian 0.731 (60); age 40–50 0.598,
50–60 0.615, 60–70 0.618, 70–80 0.573, 80–90 0.541, 90–100 0.468 (321). Small
subgroups are noisy; the model is weakest for the oldest patients.

### 9.2 Monitoring (RUNNING_THE_PROJECT.md §14.4)

Each injected scenario raises exactly the feature it moved above the control
floor (S1 `num_lab_procedures` PSI 0.27–0.30; S2 `A1Cresult` 0.39–0.42; S3
`age` 0.37–0.45, and the only scenario to move the score distribution). The
**untouched stream is not silent**: from window 1, `payer_code` (0.11–0.13)
and `medical_specialty` (0.12–0.17) breach the monitor's 0.10 line — genuine
drift in the UCI extract's ordering. The monitor keeps reporting it; the
policy's breach line was calibrated above it.

### 9.3 Decision benchmark (`reports/benchmark.md`, `python -m scenarios.benchmark`)

Every complete 2,000-row window (5 per scenario) of each seeded scenario,
measured once by the production monitor and decided by the production engine
under both policies. Drift is present from window 0.

| Scenario | Monitor breach rate | policy-v1: acting cards / latency | policy-v2: acting cards / latency |
|---|---|---|---|
| S1 covariate shift | 100% | 100% / 0 | 100% / **0** |
| S2 coding change | 100% | 100% / 0 | 100% / **0** |
| S3 prevalence shift | 100% | 100% / 0 | 100% / **0** |
| S4 label drift | 80% | 80% / 1 (triggered by the control's administrative drift, not by the label drift) | 0% / not detected |
| S5 no-drift control | 80% | **80% false triggers** (ALERT_ONLY, INCREMENTAL_RETRAIN) | **0% false triggers** |

All S1–S3 cards under policy-v2 are `FULL_RETRAIN / ESCALATE_HUMAN` with
confidence 0.39–0.67; none auto-proceeds. **Caveat:** policy-v2's threshold was
calibrated on the control's windows 0–2, so its 0% on those windows is
in-sample; windows 3–4 were not used for calibration and also stayed at 0%.

### 9.4 Gate outcomes (policy-v2's 15 retrain cards)

A live retrain on the pinned data (the production entry point, trained once
because every card pins the same data hash) and the deliberately bad
challenger, each gated against every card's own frozen holdout + labelled
window with the card's pinned criteria:

| | PASS | BLOCK |
|---|---|---|
| Retrained challenger | 11 | 4 — S1 w3 (ECE 0.0504), S1 w4 (0.0602), S2 w4 (0.0597), S3 w4 (0.0643), all on `recent_labeled_window` against the absolute 0.05 ceiling |
| Deliberately bad challenger | 0 | **15** (28–38 failed checks each) |

The four retrained BLOCKs are the gate working as specified, and a finding:
ECE is an absolute ceiling (RISK_ANALYSIS.md §2), so on late drifted windows
where the *champion* is equally miscalibrated, a retrain on the same data
cannot pass. The data a retrain uses is pinned to the training split, so a
retrain cannot adapt to the drifted window; RESEARCH_NOVELTY C3's matured-label
path is where that would change (§11). Gate *precision* against a labelled
ground truth was not measured — the benchmark has no independently known
"should pass" set.

### 9.5 Shadow, approval and promotion

Promotion requires, at the moment of the click: gate PASS, `shadow` still on
the challenger, `champion` unchanged since the gate, ≥ 50 dual-scored requests,
and a reason of ≥ 10 characters; otherwise 409 and nothing moves
(RUNNING_THE_PROJECT.md §17.5). On this data every challenger so far is
bit-identical to the champion, so live shadow windows report agreement 1.0 and
zero score difference; the statistics are tested on non-trivial pairs.

### 9.6 The end-to-end demo (WORKFLOW.md §5)

Driven through the real dashboard pages against the live stack from a seeded
state (RUNNING_THE_PROJECT.md §19.6): inject S1 → card
`FULL_RETRAIN / ESCALATE_HUMAN` (0.3941) → authorise → replay → gate PASS →
55 requests → approve → champion v1 → v2 served live → bad challenger BLOCK
(37 reasons) → audit PDF. Machine time **102.1 s** (fresh clone), **96.3 s**
(after a reset), **94.9 s** (the development stack after its reset).

### 9.7 Engineering evidence

| | |
|---|---|
| Tests | **1,299 passed, 0 failed, 0 skipped** locally with the data (Week 12); in a clean Linux container without data, 1,272 passed and 20 data-dependent tests skipped by design (Week 11 tree) |
| Branch coverage | Decision Engine 99%, gate 99%, API 87%, `scripts/` 85% (CI floor 80% each for engine, gate, API) |
| Lint | ruff check + format clean; pre-commit enforced |
| Reproducibility | `dvc repro` byte-reproducible on the build machine; on a fresh clone every data hash and `metrics.json` match (the model pickle's bytes differ; its predictions do not — §11) |
| CI | ruff → pytest + coverage floor → smoke train → gate check → image build, locally exercised; **not yet run on GitHub** (nothing pushed) |
| Privacy | PHI-pattern scans of every table, API logs and audit PDFs: 0 identifiers, 0 feature values |

## 10. Research Contributions — What the Evidence Supports

| Contribution | Evidence in this repo | Not yet evidenced |
|---|---|---|
| C1 Decision Card | every benchmark card re-derivable from the policy table; tests per rule branch under both policies | — |
| C2 Decide/narrate separation | grounding check; template grounded on every rule branch; fallback verified live | live LLM faithfulness rate; human-readability comparison |
| C3 Label-latency-aware policy | evidence-completeness term caps leading-indicator autonomy; every benchmark card escalated | matured-label evidence (rule 5) never fires; S4 undetected; no with/without comparison |
| C4 Fairness gate | subgroup non-regression is a blocking criterion; the bad challenger trips every comparable subgroup | a constructed challenger that improves aggregate AUROC while hurting one subgroup |
| C5 Shadow-first ladder | automated path stops at shadow; `champion` moves only with logged approval (source-scanned test) | — |
| C6 Benchmark | `scenarios/benchmark.py` regenerates §9.3–§9.4 from code + DVC data | the cron-baseline comparison; premature-retrain rate; gate precision |

## 11. Limitations

- **Model quality** is below the published range under this protocol (§9.1).
- **Serving latency** ≈ 350 ms warm, above the < 200 ms goal; per-request SHAP dominates.
- **No matured-label monitoring**: rule 5 never fires, S4 is undetected, and
  rule 6 (retrain budget) is not fed from `retrain_runs`; rules 2, 3, 5, 6 are
  covered by unit tests, not by the benchmark.
- **Retrains cannot differ from the champion** on this data: same pinned data,
  seeded pipeline — so shadow statistics are trivially perfect and some late
  windows can never pass the absolute ECE ceiling (§9.4).
- **Synthetic drift**: scenarios are seeded transformations of real rows,
  clinically motivated but not observed events.
- **Dataset vintage** (1999–2008, ICD-9) and US-specific coding.
- **Single machine**: local DVC remote, local MLflow, dev-token issuance, no
  user accounts; the dashboard trusts the API's verdict on a pasted token.
- **Fresh-clone model hash**: `dvc repro` elsewhere can record a new model
  md5 (pickle memoisation); predictions are identical.
- **Monitor restart** re-measures the `live` stream from window 0, appending
  duplicate live windows.
- **Not done in Week 12**: the demo video (Appendix B has the recording
  script) and a live Ollama narration run.

## 12. Reproducing Everything

```bash
# fresh clone -> demo: see README.md (Quickstart)
pytest -q && ruff check . && ruff format --check .
python -m scripts.ci_smoke             # CI smoke train + gate check
python -m scenarios.benchmark          # regenerates reports/benchmark.{json,md} (~5 min)
python -m scripts.reset_demo --yes     # archive, wipe, reseed the demo state
```

## 13. Conclusion

The thesis held where it was tested: a deterministic, versioned policy catches
every injected covariate, coding and case-mix shift in its first window,
stays silent on the control once calibrated, refuses every bad challenger,
never promotes without a logged human decision, and leaves a chain that can be
reconstructed from SQL. What it does not yet do — see label drift, beat the
published AUROC under a leakage-free protocol, or measure an LLM's faithfulness
live — is stated above rather than hidden.

---

## Appendix A — Viva preparation: the five roadmap questions

**Why not LangGraph?** Every step v1 gave an agent is a comparison against a
threshold or a pure function; an agent adds latency, failure modes and
non-reproducibility without adding judgement (PROJECT_DESIGN.md §2.2). v2's
decisions are 100% reproducible: the benchmark re-derives every card from the
policy file. The LLM keeps the one job it is good at — narration — and cannot
affect a decision because it runs after `decide()` and only on the card.

**Why a deterministic policy?** A clinical retraining decision must be
reproducible, testable and auditable. The policy is a reviewable YAML file;
changing a threshold is a new version (`policy-v1` → `policy-v2`), every card
names the version that judged it, and each rule branch has unit tests. The
v1 → v2 calibration is itself the governance story: measured on the control
(80% false triggers under v1), changed in one reviewable line, 0% under v2.

**Why isotonic calibration?** The score is used as a probability, and the gate
treats calibration as a blocking criterion (ECE ≤ 0.05). Isotonic regression
is non-parametric, suits a tree ensemble's non-sigmoid distortions, and with
~49k training rows has enough data to avoid overfitting; 5-fold
`CalibratedClassifierCV` keeps it out-of-fold. Honestly measured on the frozen
slice it changed little: AUPRC 0.1253 → 0.1301, Brier 0.0743 → 0.0741, and ECE
0.0189 → 0.0201 (slightly worse) — the raw LightGBM was already well
calibrated here; the gate, not the calibrator, is what guarantees the ceiling.

**How is PHI protected?** The audit row stores a SHA-256 of the request, never
the request; logs carry ids, hashes, versions and latency only; drift rows and
cards carry aggregate statistics; the LLM receives only the card and is
self-hosted; the dashboard shows hashes and holds no signing secret; demo
archives stay local and gitignored. Scans of every table, the logs and the
audit PDFs found no identifier or feature value.

**What happens when the gate blocks?** The run is recorded in `retrain_runs`
with `BLOCK`, one named reason per failed check (e.g. *"calibration_ece
[recent_labeled_window]: ECE 0.060249 against the absolute ceiling 0.05"*),
both metric sets and the criteria version; no alias moves, serving is
untouched, and the card's audit PDF shows the reasons. Re-running the same
card returns the verdict on record rather than retraining. In the benchmark the
bad challenger was blocked 15/15 times.

## Appendix B — Demo video: recording script (video not yet recorded)

The roadmap's demo-video deliverable was **not produced** in this build: it
needs a person, a screen recorder and a narration. The script, timed from the
measured runs:

1. `python -m scripts.reset_demo --yes` beforehand; Overview shows *Replay ready*.
2. 0:00 Overview — champion v1 serving; send 10 requests.
3. 0:10 Drift Monitor — inject S1; show the card id and the Evidently report.
4. 0:30 Decision Cards — read the template narrative and grounding verdict.
5. 0:45 Approvals — authorise; Champion vs Challenger — Replay → PASS → shadow.
6. 1:05 Overview — send 55 requests; Approvals — approve; champion v1 → v2.
7. 1:55 Champion vs Challenger — bad challenger → BLOCK with reasons.
8. 2:10 Audit — build and open the card's audit PDF.
9. 2:30 End on the Audit log: every step, one searchable trail.
