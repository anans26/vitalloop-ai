"""The per-card audit PDF (WORKFLOW.md step 18): what it says, that it is reproducible,
and that it says nothing about a patient."""

import pytest

from dashboard.audit_pdf import (
    MODE_LABELS,
    PRIVACY_NOTE,
    latin1,
    main,
    render_pdf,
    report_for_card,
)
from dashboard.data import STATE_BLOCKED, STATE_PROMOTED

from .conftest import FIXED_AT

SECTIONS = [
    "1. Decision",
    "2. Narrative",
    "3. Trigger evidence",
    "4. Policy reasoning",
    "5. Acceptance criteria (pinned before any training)",
    "6. Retrain runs and gate verdicts",
    "7. Human decisions",
    "8. Model registry moves",
    "9. Lineage",
]


def _report(session, card_id, **kwargs):
    return report_for_card(session, card_id, generated_at=FIXED_AT, **kwargs)


def test_the_report_has_every_section_in_order(dash_db, lifecycle):
    report = _report(dash_db, lifecycle["promoted"].card_id)
    assert [s.heading for s in report.sections] == SECTIONS
    assert report.state == STATE_PROMOTED


def test_the_report_carries_the_lineage_of_a_promoted_card(dash_db, lifecycle):
    card_id = lifecycle["promoted"].card_id
    alias_rows = [
        {
            "timestamp_utc": "2026-09-24T13:00:00+00:00",
            "action": "set_alias",
            "alias": "champion",
            "from_version": "1",
            "to_version": "2",
            "actor": "ops-alice",
            "reason": f"promotion for decision card {card_id}",
        }
    ]
    text = _report(dash_db, card_id, alias_rows=alias_rows).text()

    for expected in (
        "FULL_RETRAIN",
        "ESCALATE_HUMAN",
        "PASS",
        "APPROVE",
        "ops-alice",
        "auroc_non_inferiority_margin",
        "set_alias",
        PRIVACY_NOTE,
    ):
        assert expected in text


def test_a_blocked_bad_challenger_is_labelled_and_explained(dash_db, lifecycle):
    report = _report(dash_db, lifecycle["blocked"].card_id)
    text = report.text()
    assert report.state == STATE_BLOCKED
    assert MODE_LABELS["demo-bad"] in text
    assert "BLOCK under" in text


def test_a_card_without_a_stored_narrative_is_rendered_and_labelled(dash_db, lifecycle):
    narrative = _report(dash_db, lifecycle["awaiting"].card_id).sections[1]
    pairs = dict(narrative.blocks[1].pairs)
    assert "rendered for this report" in pairs["Source"]
    assert pairs["Grounding check"].startswith("passed")


def test_a_stored_narrative_is_used_verbatim(dash_db, lifecycle):
    from sqlalchemy.orm.attributes import flag_modified

    card = lifecycle["noop"]
    payload = dict(card.card_json)
    payload.update(narrative="Nothing happens.", narrative_source="template/decision_card-v1")
    card.card_json = payload
    flag_modified(card, "card_json")
    dash_db.commit()

    narrative = _report(dash_db, card.card_id).sections[1]
    assert narrative.blocks[0].text == "Nothing happens."
    assert "stored with the card" in dict(narrative.blocks[1].pairs)["Source"]


def test_an_unknown_card_has_no_report(dash_db):
    assert _report(dash_db, "dc-nope") is None


# ---------------------------------------------------------------------------
# The PDF itself
# ---------------------------------------------------------------------------
def test_render_produces_a_pdf(dash_db, lifecycle):
    pdf = render_pdf(_report(dash_db, lifecycle["promoted"].card_id))
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")


def test_the_pdf_is_byte_reproducible(dash_db, lifecycle):
    """Same rows, same generated_at -> the same bytes; an export can be re-derived."""
    card_id = lifecycle["blocked"].card_id
    assert render_pdf(_report(dash_db, card_id)) == render_pdf(_report(dash_db, card_id))


def test_the_report_holds_no_patient_data(dash_db, lifecycle):
    """Aggregate statistics only: no identifier, no raw feature value."""
    for card in lifecycle.values():
        report = _report(dash_db, card.card_id)
        text = report.text().lower()
        for forbidden in ("patient_nbr", "encounter_id", "input_hash", "readmitted", "payer_code"):
            assert forbidden not in text
        pdf = render_pdf(report)
        assert b"patient_nbr" not in pdf and b"encounter_id" not in pdf


@pytest.mark.parametrize(
    ("text", "plain"),
    [
        ("a — b", "a - b"),
        ("x ≥ 0.05", "x >= 0.05"),
        ("1×", "1x"),
        ("→", "->"),
        ("snow ☃", "snow ?"),
    ],
)
def test_text_is_made_safe_for_core_fonts(text, plain):
    assert latin1(text) == plain


def test_export_writes_one_file_per_card(dash_db, lifecycle, tmp_path):
    from dashboard.audit_pdf import export

    path = export(dash_db, lifecycle["shadow"].card_id, out_dir=tmp_path, generated_at=FIXED_AT)
    assert path == tmp_path / f"{lifecycle['shadow'].card_id}.pdf"
    assert path.read_bytes().startswith(b"%PDF-")
    with pytest.raises(KeyError):
        export(dash_db, "dc-nope", out_dir=tmp_path)


def test_the_cli_exports_and_refuses_unknown_cards(
    dash_db, lifecycle, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("loop.monitor.database.open_session", lambda: dash_db)
    monkeypatch.setattr("ml.registry.read_audit_rows", lambda *a, **k: [])
    monkeypatch.setattr(dash_db, "close", lambda: None)

    assert main(["--card", lifecycle["noop"].card_id, "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / f"{lifecycle['noop'].card_id}.pdf").exists()
    assert main(["--card", "dc-nope", "--out-dir", str(tmp_path)]) == 2
