# VitalLoop 2.0 — Risk Analysis

> Companion documents: [PROJECT_DESIGN.md](PROJECT_DESIGN.md) · [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)

Severity = impact if it happens · Likelihood = probability during the 12-week build. Mitigations marked **[built-in]** are structural properties of the v2 architecture rather than process promises.

---

## 1. Technical Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| Two students, 12 weeks — scope overrun | High | High | Scope valve defined in roadmap (drop order: 2nd dataset → LLM narration → mock-FHIR; **loop never cut**); Friday demo rule keeps `main` always demoable |
| Windows dev friction (MLflow paths, Ollama, native deps) | Med | Med | Everything runs in Docker Compose **[built-in]**; fpdf2 chosen over WeasyPrint specifically to avoid native GTK deps |
| Retraining too slow for live demo | Med | High | Cached-run replay mode, clearly labeled as replay in the UI **[built-in]**; small training subset for live runs |
| Ollama unavailable on demo machine | Low | Med | Jinja2 template narrative is the default CI path; LLM is additive, never load-bearing **[built-in]** |
| Integration hell in weeks 8–9 | High | Med | Contracts (Decision Card schema, gate function signature) frozen in week 7; components communicate only via Postgres rows and MLflow aliases — no hidden coupling |
| CI pipeline too slow / flaky | Low | Med | 5k-row smoke sample for the training stage; dependency caching; gate + engine tests are pure functions (fast, deterministic) |

## 2. Model Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| Promoting a worse model | **Critical** | Med | Validation gate blocks on AUROC non-inferiority, calibration ceiling, recall, subgroup non-regression **[built-in]**; demonstrated with a deliberately bad challenger |
| Miscalibrated risk scores trusted as probabilities | High | Med | Isotonic calibration + ECE as a *gate criterion*, not just a report metric **[built-in]** |
| Overfitting to the frozen eval set through repeated gating | Med | Med | Gate evaluates on frozen holdout **and** most-recent labeled window; tuning time-boxed (week 3) |
| Retraining on drifted-but-corrupt data (garbage in) | High | Med | pandera validation quarantines bad batches before they reach the DVC-pinned training set **[built-in]** |
| Threshold mis-tuning → alert fatigue or blindness | Med | High | Thresholds calibrated on the no-drift control (S5) in week 6; persistence rule (2 consecutive windows) filters noise; false-trigger rate is a tracked benchmark metric |
| Class imbalance masking failure (accuracy illusion) | Med | Low | AUPRC + recall@top-decile are primary metrics; accuracy is never reported alone |

## 3. Healthcare-Specific Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| PHI exposure via logs or the LLM | **Critical** | Low | Only SHA-256 input hashes + aggregate statistics leave the predictions store; narration layer receives *only* the Decision Card (aggregates/metadata) — structurally cannot see a patient record; LLM is self-hosted **[built-in]** |
| Silent clinical model swap | **Critical** | Low | Champion alias moves require gate PASS + shadow window + logged human approval; there is no other code path **[built-in]** |
| Automation bias — humans rubber-stamping approvals | High | Med | Approval screen shows the full evidence chain (card, gate result, shadow stats, subgroup deltas), not a bare "Approve" button; rejection is one click and fully logged |
| Delayed 30-day labels → flying blind on true performance | High | Certain | Explicitly modeled: leading indicators cap autonomy at shadow; matured-label regressions always escalate (contribution C3) **[built-in]** |
| Model treated as clinical advice | High | Low | Scope statement in UI and docs: decision-*support* risk stratification, not diagnosis; SHAP factors shown as "contributing factors," never causal claims |

## 4. Deployment Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| Demo-day environment failure | High | Med | Fully offline stack (Compose, local MLflow, local LLM); seed script rebuilds demo state on a fresh machine; demo rehearsed from a clean clone in week 11 |
| Shadow model degrading serving latency | Med | Low | Shadow scoring is post-response/async in middleware; champion path never waits on it |
| State corruption between demo runs | Med | Med | One-command reset (Compose down -v + seed); demo scenarios are idempotent seeded scripts |
| Registry/DB drift between environments | Med | Low | All state in two services (Postgres, MLflow) with volumes; no local files as sources of truth |

## 5. Data Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| Leakage (death/hospice discharges; duplicate patients) | **Critical** | High (if unhandled) | Discharge codes 11/13/14/19/20/21 dropped; one encounter per patient; **unit test asserts no patient overlap across splits** |
| Dataset vintage (1999–2008, ICD-9) questioned at viva | Low | High | Prepared answer: the project's subject is the *lifecycle*, which is dataset-agnostic; benchmark comparability outweighs vintage; MIMIC-IV extension path documented |
| Synthetic drift dismissed as artificial | Med | Med | Scenarios are seeded, published, and clinically motivated (coding change, case-mix shift); eICU multi-hospital split offers a natural-drift complement; framed as a controlled benchmark (C6) |
| High-missingness columns handled naively | Med | Med | Explicit policy per column (drop `weight`; missingness-as-category for payer/specialty), documented and tested |

## 6. Ethical Risks

| Risk | Sev | Lik | Mitigation |
|---|---|---|---|
| Subgroup performance erosion through retraining | High | Med | Fairness is a **blocking gate criterion** (max 0.01 subgroup AUROC drop), not a dashboard (contribution C4) **[built-in]** |
| Historical bias in training data reproduced by the model | High | Certain (data property) | Subgroup metrics reported from week 3 onward; limitations section in the report; race/gender never used as split-worthy features without discussion |
| LLM hallucination in audit narratives | Med | Med | Grounding check rejects narratives quoting numbers not present in the card; template fallback; narrative always labeled with its source (`ollama/...` or `template`) **[built-in]** |
| Over-claiming clinical readiness | Med | Med | REAL vs MOCK table maintained honestly (as in v1 — its best habit); README states this is a research/education system, not a medical device |
| Audit trail tampering | Med | Low | Append-only application discipline; cards reference immutable policy versions, DVC hashes, MLflow runs — cross-referenced records make silent edits detectable |

---

## Top 5 Risks to Actively Manage (weekly review)

1. **Scope overrun** — the only risk that kills the project; the scope valve is the answer, applied without sentimentality.
2. **Leakage** — the only risk that silently invalidates every reported number; killed by tests, not vigilance.
3. **Threshold mis-tuning** — the difference between a credible loop and a demo that cries wolf; calibrated on the control scenario.
4. **Integration in weeks 8–9** — mitigated by frozen contracts and pair-programming the ⭐ weeks.
5. **Demo fragility** — mitigated by full offline stack + clean-clone rehearsal in week 11.
