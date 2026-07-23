# VitalLoop 2.0 — Research Novelty & Contributions

> Companion documents: [PROJECT_DESIGN.md](PROJECT_DESIGN.md) · [ARCHITECTURE.md](ARCHITECTURE.md)

The project's research positioning: **most MLOps research automates retraining; almost none makes the retraining decision itself a governed, auditable, testable artifact.** VitalLoop 2.0's contributions all live in that gap.

---

## C1 — The Decision Card as an Auditable Governance Primitive

**Claim**: drift response should produce a *machine-checkable decision record*, not an alert. The Decision Card binds together, in one immutable artifact: the trigger evidence, the exact policy version that evaluated it, the action, a decomposed confidence score, the pinned data version a retrain will use, and the acceptance criteria the challenger must meet **before** training starts.

**Why it matters**: regulators (FDA GMLPs, EU AI Act Article 12 logging obligations, CMS audit expectations) increasingly require exactly this — traceable, reconstructable automated decisions. Existing tools (Evidently, SageMaker Model Monitor, Vertex) stop at detection; the decision layer is undocumented human glue. Making the decision a first-class schema is a small idea with outsized governance value, and it generalizes to any supervised model lifecycle.

**Evaluable**: every card in the S1–S5 benchmark can be re-derived by hand from the policy table — reproducibility *is* the evaluation.

## C2 — Separation of Powers: Deterministic Policy Decides, LLM Narrates

**Claim**: the safe pattern for LLMs in clinical MLOps is **decision/narration separation** — a versioned, unit-tested rule policy owns every consequential action; the LLM's only job is translating the structured decision into human language, with a *grounding check* that rejects any narrative quoting numbers absent from the card.

**Why it matters**: this is a direct, defensible answer to the open question "can LLMs be used in regulated ML pipelines?" — yes, in the presentation layer, never in the control path. It inverts the v1 design (LLM-as-decision-maker) and the inversion itself is the finding: we get 100% decision reproducibility and lose nothing, because the LLM's genuine strength (fluent explanation) is fully retained. The pattern names a reusable architecture, and the grounding check gives it teeth.

**Evaluable**: narrative faithfulness rate (fraction of generated narratives passing the grounding check) and a human-readability comparison vs template output.

## C3 — Label-Latency-Aware Drift Response Policy

**Claim**: clinical labels arrive late by definition (a 30-day readmission label matures 30 days after discharge), so a retraining policy must explicitly reason over *evidence completeness*: leading indicators (input/prediction drift) justify at most shadow-bound autonomy; only matured-label performance evidence justifies escalated response — and confirmed performance regressions always escalate to a human.

**Why it matters**: most drift literature and tooling implicitly assumes labels are available; the delayed-label regime is the *normal* case in healthcare and is under-treated. Encoding evidence completeness as an explicit term in the confidence formula (weight w4 = 0.25) is a concrete, transferable policy mechanism.

**Evaluable**: scenario S4 measures decision quality with and without the evidence-completeness term (premature-retrain rate, detection latency).

## C4 — Fairness as a Hard Promotion Gate, Not a Dashboard

**Claim**: subgroup non-regression (no age/gender/race subgroup loses more than 0.01 AUROC) is enforced as a **blocking criterion** in the validation gate — a challenger that improves aggregate AUROC while degrading one subgroup is refused promotion automatically.

**Why it matters**: fairness monitoring is common; fairness as an automated *deployment veto* in a retraining loop is rare, and the retraining setting is precisely where silent fairness erosion happens (drift is rarely uniform across subgroups). This operationalizes "fairness drift" from a metric into a control.

**Evaluable**: a constructed challenger that trades subgroup performance for aggregate gain, demonstrably BLOCKED with the subgroup evidence in the gate result.

## C5 — Autonomy with Restraint: the Shadow-First Promotion Ladder

**Claim**: a safe autonomy boundary for clinical models is *"the system may retrain and stage on its own; it may never change what clinicians see on its own."* Formally: automation ceiling = shadow alias; crossing to champion requires gate PASS ∧ shadow-window statistics ∧ logged human approval.

**Why it matters**: this gives a crisp, implementable answer to "how autonomous should clinical MLOps be?" — a graduated ladder (observe → decide → retrain → shadow → human-gated promote) where each rung's precondition is machine-checkable. It reframes the human-in-the-loop from a vague principle into a specific, minimal placement: one approval, at the last rung, with full evidence in front of them.

**Evaluable**: end-to-end audit reconstruction — for any champion version, the chain drift_event → card → retrain_run → gate_result → approval is complete and queryable.

## C6 — A Reproducible Drift-Response Benchmark for Tabular Clinical ML

**Claim**: the five seeded drift scenarios (covariate shift, coding change, prevalence shift, delayed-label decay, no-drift control) plus the metric suite — detection latency, false-trigger rate, gate precision, premature-retrain rate — form a small but complete benchmark for evaluating *closed-loop* drift response, not just drift detection.

**Why it matters**: drift *detectors* have benchmarks; drift *response policies* largely don't. Publishing the scenario scripts with fixed seeds lets anyone re-run every claim and compare alternative policies (e.g., policy-v1 vs a cron baseline vs an LLM-decided baseline) on identical evidence — including quantifying exactly what the v1 LangGraph design would have cost in reproducibility.

**Evaluable**: it *is* the evaluation harness; the headline table compares policy-v1 against a naive cron-retrain baseline on all five scenarios.

---

## Positioning Against Existing Work

| Existing work | What it does | Gap VitalLoop 2.0 fills |
|---|---|---|
| Evidently / NannyML / Alibi Detect | Detect drift, render reports | No decision, no action, no loop |
| SageMaker Model Monitor, Vertex Model Monitoring | Detect + alert inside a cloud suite | Decision layer is human glue; no governance artifact |
| Scheduled (cron) retraining literature | Retrain on a calendar | Blind to evidence; can promote regressions; no audit story |
| LLM-agent MLOps demos (incl. VitalLoop v1) | LLM decides when to retrain | Non-reproducible decisions; untestable; indefensible in a regulated setting |
| Fairness-monitoring literature | Measure subgroup metrics over time | Measurement without a deployment veto |

**Paper-shaped output** (stretch, week 12+): *"Decision Cards: Auditable Autonomy for Clinical Model Retraining"* — C1 + C2 + C6 as the spine, C3–C5 as design contributions, the S1–S5 benchmark as evaluation. Suitable venues: MLSys/MLOps workshops, ML4H (Machine Learning for Health), or a national student research conference.
