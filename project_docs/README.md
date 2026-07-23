# VitalLoop 2.0 — Project Documentation

Design documentation for **VitalLoop 2.0**, a self-healing, audit-first MLOps system for 30-day hospital-readmission risk — the redesigned Version 2.0 of the original VitalLoop proposal (`vitalloop.pdf`), produced after a full architectural review.

**Design principle:** *Deterministic code decides. The LLM only narrates. A human approves anything that touches clinicians.*

## Documents

| File | Contents |
|---|---|
| [VITALLOOP_COMPLETE_DOCUMENTATION.md](VITALLOOP_COMPLETE_DOCUMENTATION.md) | **Single-file edition** — the full contents of every document below merged into one navigable Markdown file (12 parts + appendix) |
| [PROJECT_DESIGN.md](PROJECT_DESIGN.md) | **Master document** — problem statement, v1 review, weakness analysis, full v2 architecture, workflow, AI components, datasets, stack, roadmap, repo structure, research contributions, risks, final recommendation |
| [PROJECT_DESIGN.pdf](PROJECT_DESIGN.pdf) | Print-ready render of the master document (cover page, TOC, page numbers) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Component-by-component specification: Decision Engine rule table, Decision Card schema, gate criteria, DB schema, deployment topology, security posture |
| [WORKFLOW.md](WORKFLOW.md) | The 18-step end-to-end lifecycle with flowchart, sequence diagram, failure paths, and the 3-minute demo script |
| [TECH_STACK.md](TECH_STACK.md) | Every technology layer: why chosen, advantages, alternatives, and what changed from v1 |
| [DATASET_ANALYSIS.md](DATASET_ANALYSIS.md) | Six ranked public datasets (links verified), cleaning plans, feature engineering, and the seeded drift-scenario benchmark |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | 12-week plan for two students: objectives, tasks, deliverables, risks, and the scope valve |
| [RESEARCH_NOVELTY.md](RESEARCH_NOVELTY.md) | Six research contributions with claims, positioning against existing work, and evaluation paths |
| [RISK_ANALYSIS.md](RISK_ANALYSIS.md) | Six-category risk register (technical, model, healthcare, deployment, data, ethical) with mitigations |

## Reading Order

1. **PROJECT_DESIGN.md** end-to-end (or the PDF) — the complete story.
2. **ARCHITECTURE.md** + **WORKFLOW.md** — before writing any code.
3. **IMPLEMENTATION_ROADMAP.md** — week 1 starts here.

## Regenerating the PDF

The PDF is rendered from `PROJECT_DESIGN.md` via Markdown → HTML (with Mermaid diagrams) → headless-Edge print. From inside `project_docs/`:

```powershell
npm install marked          # one-time
node build_pdf.mjs          # writes design.html next to the script
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu `
  --virtual-time-budget=25000 --no-pdf-header-footer `
  --print-to-pdf="PROJECT_DESIGN.pdf" "file:///$PWD/design.html"
```

(`build_pdf.mjs` converts the Markdown with `marked` and injects print CSS + Mermaid; the `--virtual-time-budget` gives Mermaid time to render the diagrams before printing. Edit the `SRC` path at the top of the script if the folder moves.)
