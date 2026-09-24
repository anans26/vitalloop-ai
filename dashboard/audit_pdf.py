"""The per-card audit report: one Decision Card and everything that followed it, as a PDF.

WORKFLOW.md step 18: "one click renders the CMS-style audit PDF for any Decision
Card." TECH_STACK.md picks **fpdf2** for it -- "pure-Python, no native
dependencies -- important on Windows dev machines" -- and ARCHITECTURE.md §3.10
says the narration layer writes "the narrative section of the audit report".

The report is built in two steps, deliberately:

1. `build_report` turns the database rows into an `AuditReport` -- plain data:
   sections of paragraphs, key/value lists and tables. This is where every
   claim the PDF makes is decided, and it is tested as data.
2. `render_pdf` lays that out with fpdf2. It adds no facts of its own.

**What is in it** is the §4.6 lineage for one card, three joins deep: the
decision and the policy that made it, the narrative (labelled with its source
and its grounding verdict), the trigger evidence, the acceptance criteria that
were pinned before any training, every retrain attempt and the gate's verdict
on it (replay and the constructed bad challenger labelled as such), every human
decision with its reason, and every registry alias move that names the card.

**What is not in it** is anything about a patient: the report is built from the
same aggregate rows the dashboard shows, and a test scans rendered reports for
identifiers.

**Reproducible.** Given the same rows and the same `generated_at`, the bytes are
identical -- the creation date is pinned to `generated_at`, so an exported
report can be re-derived and compared later.
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from dashboard.data import CardHistory, card_history, headline_metrics
from ml.config import REPORTS_DIR
from ml.retrain import MODE_DEMO_BAD, MODE_REPLAY

AUDIT_DIR = REPORTS_DIR / "audit"
REPORT_VERSION = "audit-report-v1"
PRIVACY_NOTE = (
    "This report contains aggregate statistics and metadata only. No patient record, "
    "identifier or feature value appears in it: prediction inputs are held as SHA-256 "
    "hashes, and drift evidence as per-feature statistics."
)

MODE_LABELS = {
    "live": "live retrain",
    MODE_REPLAY: "REPLAY - cached challenger re-registered, nothing trained",
    MODE_DEMO_BAD: "CONSTRUCTED BAD CHALLENGER (demo) - never trained or registered",
}


# ---------------------------------------------------------------------------
# The report as data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Block:
    kind: str  # "text" | "pairs" | "table"
    text: str = ""
    pairs: tuple[tuple[str, str], ...] = ()
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    widths: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Section:
    heading: str
    blocks: tuple[Block, ...] = ()


@dataclass(frozen=True)
class AuditReport:
    card_id: str
    title: str
    generated_at: datetime
    state: str
    sections: tuple[Section, ...] = field(default=())

    def text(self) -> str:
        """Every string in the report, for search and for the privacy test."""
        parts = [self.title, self.card_id, self.state]
        for section in self.sections:
            parts.append(section.heading)
            for block in section.blocks:
                parts.append(block.text)
                parts.extend(f"{k} {v}" for k, v in block.pairs)
                parts.extend(block.headers)
                parts.extend(" ".join(row) for row in block.rows)
        return "\n".join(parts)


def _s(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value == value else "-"
    if isinstance(value, datetime):
        value = value if value.tzinfo else value.replace(tzinfo=UTC)
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


def _text(value: str) -> Block:
    return Block(kind="text", text=value)


def _pairs(*items) -> Block:
    return Block(kind="pairs", pairs=tuple((k, _s(v)) for k, v in items))


def _table(headers, rows, widths=None) -> Block:
    return Block(
        kind="table",
        headers=tuple(headers),
        rows=tuple(tuple(_s(cell) for cell in row) for row in rows),
        widths=tuple(widths) if widths else None,
    )


def _when(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError:
        return None


def _narrative(card_json: dict) -> tuple[str, str, bool]:
    from loop.narrate.grounding import check_grounding
    from loop.narrate.template import TEMPLATE_SOURCE, render_template

    if card_json.get("narrative"):
        text, source = (
            card_json["narrative"],
            f"{card_json.get('narrative_source')} (stored with the card)",
        )
    else:
        text = render_template(card_json)
        source = f"{TEMPLATE_SOURCE} (rendered for this report; the card predates narration)"
    return text, source, check_grounding(text, card_json).grounded


def build_report(history: CardHistory, *, generated_at: datetime) -> AuditReport:
    card, card_json = history.card, dict(history.card.card_json or {})
    trigger = card_json.get("trigger") or {}
    sections: list[Section] = []

    sections.append(
        Section(
            "1. Decision",
            (
                _pairs(
                    ("Card", card.card_id),
                    ("Current state", history.state),
                    (
                        "Scenario / window",
                        f"{card.scenario}, {_s(card.window_start)} to {_s(card.window_end)}",
                    ),
                    ("Policy version", card.policy_version),
                    ("Rule", card.rule_id),
                    ("Action", card.action),
                    ("Disposition", card.disposition),
                    ("Confidence", card.confidence),
                    ("Card status at emission", card.status),
                    ("Decided at", card.created_at),
                ),
            ),
        )
    )

    text, source, grounded = _narrative(card_json)
    sections.append(
        Section(
            "2. Narrative",
            (
                _text(text),
                _pairs(
                    ("Source", source),
                    (
                        "Grounding check",
                        "passed - every number quoted is in the card" if grounded else "FAILED",
                    ),
                ),
            ),
        )
    )

    features = trigger.get("breaching_features") or []
    event = history.event
    sections.append(
        Section(
            "3. Trigger evidence",
            (
                _table(
                    ("Breaching feature", "PSI", "KS p-value"),
                    [(f.get("feature"), f.get("psi"), f.get("ks_p")) for f in features],
                )
                if features
                else _text("No feature breached the policy's threshold in this window."),
                _pairs(
                    ("Drift event", trigger.get("drift_event_id")),
                    ("Monitored features", trigger.get("monitored_feature_count")),
                    ("Max PSI", trigger.get("max_psi")),
                    ("Prediction drift", trigger.get("prediction_drift")),
                    ("Prediction PSI", trigger.get("prediction_psi")),
                    ("Consecutive breaching windows", trigger.get("consecutive_breaching_windows")),
                    ("Label maturity", trigger.get("label_maturity")),
                    ("Matured AUROC drop", trigger.get("matured_auroc_drop")),
                    (
                        "Rows (reference / window)",
                        f"{_s(event.reference_rows)} / {_s(event.current_rows)}" if event else "-",
                    ),
                    ("Evidently report", event.report_uri if event else None),
                ),
            ),
        )
    )

    breakdown = card_json.get("confidence_breakdown") or {}
    thresholds = card_json.get("policy_thresholds") or {}
    sections.append(
        Section(
            "4. Policy reasoning",
            (
                _text(card_json.get("rationale") or ""),
                _table(("Confidence term", "Value"), sorted(breakdown.items())),
                _table(("Policy threshold", "Value"), sorted(thresholds.items())),
                _pairs(("Downgraded from", card_json.get("downgraded_from"))),
            ),
        )
    )

    criteria = card_json.get("acceptance_criteria") or {}
    sections.append(
        Section(
            "5. Acceptance criteria (pinned before any training)",
            (
                _table(("Criterion", "Value"), sorted(criteria.items())),
                _pairs(("Pinned data version", card.candidate_data_version)),
            ),
        )
    )

    run_blocks: list[Block] = []
    if not history.runs:
        run_blocks.append(_text("No retrain has been run for this card."))
    for run in history.runs:
        reasons = (run.gate_result or {}).get("reasons") or []
        run_blocks.append(
            _pairs(
                ("Retrain run", run.run_id),
                ("Mode", MODE_LABELS.get(run.mode, run.mode)),
                (
                    "Challenger / champion",
                    f"{_s(run.challenger_version)} vs {_s(run.champion_version)}",
                ),
                (
                    "Gate verdict",
                    f"{run.outcome} under {run.criteria_version} "
                    f"({run.failed_criteria_count} failed checks)",
                ),
                ("Shadow alias moved", run.shadow_alias_moved),
                (
                    "Authorised by",
                    run.authorized_by or "the card's own AUTO_PROCEED_SHADOW disposition",
                ),
                ("MLflow run", run.mlflow_run),
                ("Recorded at", run.created_at),
            )
        )
        metrics = headline_metrics(run)
        if metrics:
            run_blocks.append(
                _table(
                    ("Set", "Metric", "Champion", "Challenger", "Delta"),
                    [
                        (
                            m["evaluation_set"],
                            m["metric"],
                            m["champion"],
                            m["challenger"],
                            m["delta"],
                        )
                        for m in metrics
                    ],
                    widths=(30, 44, 20, 20, 14),
                )
            )
        if reasons:
            run_blocks.append(_text("BLOCK reasons:\n" + "\n".join(f"- {r}" for r in reasons)))
    sections.append(Section("6. Retrain runs and gate verdicts", tuple(run_blocks)))

    sections.append(
        Section(
            "7. Human decisions",
            (
                _table(
                    ("When", "Kind", "Decision", "Approver", "Champion before -> after", "Reason"),
                    [
                        (
                            a.ts,
                            a.kind,
                            a.decision,
                            a.approver,
                            f"{_s(a.champion_version_before)} -> {_s(a.champion_version_after)}"
                            if a.kind == "PROMOTION"
                            else "-",
                            a.reason,
                        )
                        for a in history.approvals
                    ],
                    widths=(22, 20, 15, 16, 16, 41),
                )
                if history.approvals
                else _text("No human decision has been recorded for this card."),
            ),
        )
    )

    sections.append(
        Section(
            "8. Model registry moves",
            (
                _table(
                    ("When", "Action", "Alias", "From", "To", "Actor"),
                    [
                        (
                            _when(m.get("timestamp_utc")),
                            m.get("action"),
                            m.get("alias"),
                            m.get("from_version"),
                            m.get("to_version"),
                            m.get("actor"),
                        )
                        for m in history.alias_moves
                    ],
                )
                if history.alias_moves
                else _text("No registry alias move names this card."),
            ),
        )
    )

    sections.append(
        Section(
            "9. Lineage",
            (
                _pairs(
                    ("Policy version", card.policy_version),
                    ("Model version measured", card.model_version),
                    ("Reference data version", card.data_version),
                    ("Candidate data version", card.candidate_data_version),
                    ("Report version", REPORT_VERSION),
                ),
                _text(PRIVACY_NOTE),
            ),
        )
    )

    return AuditReport(
        card_id=card.card_id,
        title="VitalLoop Decision Card Audit Report",
        generated_at=generated_at,
        state=history.state,
        sections=tuple(sections),
    )


def report_for_card(
    session: Session,
    card_id: str,
    *,
    alias_rows: list[dict] | None = None,
    generated_at: datetime | None = None,
) -> AuditReport | None:
    history = card_history(session, card_id, alias_rows)
    if history is None:
        return None
    return build_report(history, generated_at=generated_at or datetime.now(UTC))


# ---------------------------------------------------------------------------
# The PDF
# ---------------------------------------------------------------------------
_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "→": "->",
    "←": "<-",
    "≥": ">=",
    "≤": "<=",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "§": "S",
    "−": "-",
    "×": "x",
}


def latin1(text: str) -> str:
    """fpdf2's core fonts are Latin-1; map the punctuation we use, replace the rest."""
    for char, plain in _REPLACEMENTS.items():
        text = text.replace(char, plain)
    return text.encode("latin-1", "replace").decode("latin-1")


def render_pdf(report: AuditReport) -> bytes:
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    class _Pdf(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", size=7)
            self.set_text_color(90)
            self.cell(
                0,
                5,
                latin1(
                    f"{report.card_id} | generated {report.generated_at.isoformat()} | "
                    f"{REPORT_VERSION} | page {self.page_no()}/{{nb}}"
                ),
                align="C",
            )

    pdf = _Pdf(orientation="P", unit="mm", format="A4")
    pdf.set_creation_date(report.generated_at)
    pdf.set_title(latin1(f"{report.title} - {report.card_id}"))
    pdf.set_author("VitalLoop")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, latin1(report.title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(
        0,
        5,
        latin1(
            f"Card {report.card_id} - current state: {report.state} - "
            f"generated {report.generated_at.isoformat()}"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    heading_style = FontFace(emphasis="BOLD", fill_color=(230, 232, 236))
    for section in report.sections:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(40, 60, 90)
        pdf.set_text_color(255)
        pdf.cell(0, 7, latin1(section.heading), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.set_fill_color(255)
        pdf.ln(1)
        for block in section.blocks:
            pdf.set_font("Helvetica", size=8.5)
            if block.kind == "text":
                pdf.multi_cell(0, 4.5, latin1(block.text), new_x="LMARGIN", new_y="NEXT")
            elif block.kind == "pairs":
                with pdf.table(
                    col_widths=(35, 65), first_row_as_headings=False, line_height=4.5
                ) as table:
                    for key, value in block.pairs:
                        row = table.row()
                        row.cell(latin1(key), style=FontFace(emphasis="BOLD"))
                        row.cell(latin1(value))
            elif block.kind == "table" and block.rows:
                with pdf.table(
                    headings_style=heading_style, line_height=4.5, col_widths=block.widths
                ) as table:
                    header = table.row()
                    for cell in block.headers:
                        header.cell(latin1(cell))
                    for values in block.rows:
                        row = table.row()
                        for cell in values:
                            row.cell(latin1(cell))
            pdf.ln(2)
        pdf.ln(1)

    return bytes(pdf.output())


def export(
    session: Session,
    card_id: str,
    *,
    alias_rows=None,
    out_dir: Path = AUDIT_DIR,
    generated_at: datetime | None = None,
) -> Path:
    report = report_for_card(session, card_id, alias_rows=alias_rows, generated_at=generated_at)
    if report is None:
        raise KeyError(card_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card_id}.pdf"
    path.write_bytes(render_pdf(report))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the audit PDF for one Decision Card.")
    parser.add_argument("--card", required=True, help="decision card id")
    parser.add_argument("--out-dir", default=str(AUDIT_DIR), help="where to write <card>.pdf")
    args = parser.parse_args(argv)

    from loop.monitor.database import open_session
    from ml.registry import read_audit_rows

    session = open_session()
    try:
        path = export(session, args.card, alias_rows=read_audit_rows(), out_dir=Path(args.out_dir))
    except KeyError:
        print(f"no decision card {args.card!r}", file=sys.stderr)
        return 2
    finally:
        session.close()
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
