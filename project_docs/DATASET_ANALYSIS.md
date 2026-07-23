# VitalLoop 2.0 — Dataset Analysis

> All links verified live on 2026-07-13.
> Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) · [PROJECT_DESIGN.md](PROJECT_DESIGN.md)

## Ranking Summary

| Rank | Dataset | Access | Size | Role in this project |
|---|---|---|---|---|
| 1 | **UCI Diabetes 130-US Hospitals (1999–2008)** | Open | 101,766 encounters | **Primary** — trains and serves the readmission model |
| 2 | **MIMIC-IV Clinical Database Demo v2.2** | Open | 100 patients | Secondary — realistic EHR structure for the mock-FHIR ingestion demo |
| 3 | **eICU-CRD Demo v2.0.1** | Open | ~2,500 ICU stays | Generalization evidence — "same engine, different hospital system" |
| 4 | **MIMIC-IV v3.1 (full)** | Credentialed (CITI training + DUA) | 65k+ ICU / 200k+ ED patients | Research extension — only if credentialing completes in time |
| 5 | **Synthea synthetic patients** | Open (generate locally) | Unlimited (generated) | Drift-injection source + native FHIR bundles for the ingestion demo |
| 6 | **UCI Heart Failure Clinical Records** | Open | 299 patients | Mini generalization demo — wrap a second model with the same loop |

**Decision: rank 1 is the build dataset.** It is the only open dataset with real hospital readmission labels at meaningful scale, and it is the standard benchmark for this exact task — reviewers can compare reported AUROC against published baselines (~0.64–0.69 for <30-day readmission).

---

## 1. UCI Diabetes 130-US Hospitals (PRIMARY)

- **Link**: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Source**: UCI Machine Learning Repository (Strack et al., 2014, *BioMed Research International*)
- **Rows**: 101,766 inpatient encounters (diabetic patients, 130 US hospitals, 1999–2008)
- **Features**: 47 — demographics (race, gender, age bands), admission/discharge/source codes, `time_in_hospital`, lab and procedure counts, 23 medication columns, HbA1c and glucose results, prior utilization (outpatient/emergency/inpatient visits), 3 ICD-9 diagnosis codes
- **Target**: `readmitted` ∈ {`<30`, `>30`, `NO`} → binarized to **readmitted within 30 days vs not** (~11% positive)

**Advantages**
- Real hospital data, open license, exactly the 30-day readmission problem in the problem statement
- Large enough to hold out a time-sliced "future" segment for **honest drift simulation**
- Published benchmark results exist → defensible evaluation
- Rich categorical/count structure gives SHAP meaningful clinical features to surface

**Limitations**
- 1999–2008 vintage; ICD-9 not ICD-10 (acceptable — the project is about the *lifecycle*, not the era)
- Class imbalance (~11% positive) — evaluate with AUPRC/recall@k, not accuracy
- `weight` ~97% missing, `payer_code`/`medical_specialty` ~40–50% missing
- No timestamps beyond encounter ordering — drift windows must be constructed synthetically

**Cleaning requirements**
1. Deduplicate to one encounter per `patient_nbr` (first encounter) to prevent identity leakage across train/test
2. **Remove leakage**: drop encounters with discharge disposition = expired/hospice (11, 13, 14, 19, 20, 21) — these patients cannot be readmitted
3. Drop `weight`; treat `payer_code`/`medical_specialty` missingness as an explicit category
4. Map `?` to NaN; map ICD-9 `diag_1..3` to ~18 clinically meaningful chapter groups
5. Binarize target; document the `>30 → negative` choice

**Feature-engineering ideas**
- `service_utilization = number_outpatient + number_emergency + number_inpatient`
- Medication-change intensity: count of the 23 med columns with `Up`/`Down`; `insulin_changed` flag
- `num_med_changes`, `num_meds_prescribed`, per-day procedure rate (`num_procedures / time_in_hospital`)
- Age-band ordinal encoding; admission-source risk grouping (ER vs referral vs transfer)
- **Drift-injection features**: shift `num_lab_procedures` distribution (+20%), re-map a fraction of HbA1c values to simulate a coding-practice change — the two scripted demo scenarios

---

## 2. MIMIC-IV Clinical Database Demo v2.2

- **Link**: https://physionet.org/content/mimic-iv-demo/
- **Source**: PhysioNet / MIT-LCP (Beth Israel Deaconess Medical Center)
- **Rows**: 100 patients (full relational EHR: admissions, diagnoses, labs, prescriptions)
- **Features**: dozens of linked tables mirroring the full MIMIC-IV schema (`hosp` + `icu` modules)
- **Target**: constructible — 30-day readmission from `admissions` (admit/discharge timestamps per subject)

**Advantages**: open access; the *real* structure of a modern EHR — ideal source for the mock-FHIR ingestion demo and for showing the pipeline generalizes beyond a flat CSV.
**Limitations**: 100 patients — far too small to train on; demo/structural use only.
**Cleaning**: join `admissions`+`patients`+`diagnoses_icd`; derive readmission gaps; ICD-9/10 mixed coding.
**FE ideas**: prior-admission counts, length-of-stay, Charlson comorbidity index from ICD codes.

## 3. eICU Collaborative Research Database Demo v2.0.1

- **Link**: https://physionet.org/content/eicu-crd-demo/
- **Source**: PhysioNet (Philips eICU program, 2014–2015, ~20 hospitals)
- **Rows**: ~2,500 ICU stays; CSV (130 MB) + SQLite
- **Target**: constructible — ICU readmission or hospital mortality

**Advantages**: open; **multi-hospital** — the natural dataset for demonstrating cross-site distribution shift (train on subset A of hospitals, monitor on subset B → genuine covariate drift, not synthetic).
**Limitations**: ICU population differs from general inpatient; demo subset is modest.
**Cleaning**: unit harmonization across hospitals; missing-value patterns differ by site (itself a drift story).
**FE ideas**: APACHE-derived severity features; site indicator held out as the drift axis.

## 4. MIMIC-IV v3.1 (full)

- **Link**: https://physionet.org/content/mimiciv/ (DOI: 10.13026/kpb9-mt58)
- **Source**: PhysioNet — **credentialed**: CITI "Data or Specimens Only Research" training + signed DUA (typically 1–3 weeks turnaround)
- **Rows**: 65k+ ICU patients, 200k+ ED patients, 2008–2022
- **Target**: 30-day readmission constructible with real calendar time — enabling **real temporal drift** (pre/post-COVID case mix, ICD-9→10 transition)

**Advantages**: the strongest possible research extension — actual temporal drift instead of injected drift would materially strengthen a paper.
**Limitations**: credentialing latency and data-handling obligations; too risky as the primary plan for a 12-week project. **Start the credentialing application in week 1; treat as a bonus.**

## 5. Synthea (synthetic patients)

- **Link**: https://github.com/synthetichealth/synthea (generator; releases include prebuilt sample sets)
- **Source**: MITRE — open-source synthetic patient generator
- **Rows**: unlimited — generate on demand; outputs **FHIR R4 bundles**, CSV, C-CDA
- **Target**: synthetic encounters/conditions allow a constructed readmission label

**Advantages**: zero PHI by construction; native FHIR output makes it the perfect feed for the mock-FHIR ingestion endpoint; module parameters can be changed between batches to generate *controlled, explainable* drift.
**Limitations**: synthetic correlations are weaker than real data — models trained on it look unrealistically clean; use for ingestion + drift demos, never for headline metrics.

## 6. UCI Heart Failure Clinical Records

- **Link**: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- **Source**: UCI ML Repository (Chicco & Jurman, 2020)
- **Rows**: 299 · **Features**: 12 clinical (ejection fraction, serum creatinine, age, …) · **Target**: `DEATH_EVENT` during follow-up

**Advantages**: tiny and clean — ideal for the "the same engine wraps any tabular clinical model" claim: register a second model in the same registry, monitored by the same loop, in under a day.
**Limitations**: 299 rows — a toy; different target (mortality, not readmission). Positioning: generalization exhibit, one paragraph and one dashboard tab.

---

## Drift Simulation Plan (applies to the primary dataset)

Because dataset 1 lacks calendar timestamps, drift is **scripted and reproducible** — which is a feature, not a hack: it becomes the project's evaluation benchmark.

| Scenario | Mechanism | What it simulates |
|---|---|---|
| S1 Covariate shift | Shift `num_lab_procedures`, `num_medications` distributions in the serving stream | New hospital protocol / EHR upgrade |
| S2 Coding change | Re-map a fraction of HbA1c categories | The diabetes coding-practice change from the v1 demo story |
| S3 Prevalence shift | Resample serving stream to raise elderly + high-utilization mix | Population/case-mix drift |
| S4 Label drift (delayed) | Replay a labeled holdout with degraded relationship | Performance decay visible only after 30-day label maturity |
| S5 No-drift control | Untouched stream | Proves the engine does **not** cry wolf (`NO_OP` cards) |

Each scenario is a seeded script committed to the repo — reviewers can re-run every claimed detection.
