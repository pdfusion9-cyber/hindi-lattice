import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_wikidata_human_review_queue.py"
SPEC = importlib.util.spec_from_file_location("review_queue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(record_id, property_id, object_qid, external=False, imported=False):
    refs = [{"imported_from_items": [{"property_id": "P143", "qid": "Q1"}]}] if imported else []
    return {
        "record_id": record_id, "subject_qid": "Q100", "subject_label_hi": "विषय",
        "property_id": property_id, "relation_label_hi": "संबंध",
        "object_qid": object_qid, "display_value_hi": object_qid,
        "statement_uri": "https://www.wikidata.org/entity/statement/" + record_id,
        "has_external_citation": external, "wikidata_reference_records": refs,
        "reference_enrichment": {"status": "MATCHED"},
    }


def test_priority_citation_gaps_are_queued_and_import_marker_is_not_citation():
    result = MODULE.build_queue([
        row("a", "P2043", "Q2", imported=True),
        row("b", "P403", "Q3"),
        row("c", "P17", "Q4", external=True),
    ])
    assert result["citation_gap_review_count"] == 2
    assert result["citation_gap_by_property"]["P2043"]["citation_gaps"] == 1
    assert result["citation_gap_by_property"]["P403"]["citation_gaps"] == 1
    assert result["training_approved_record_count"] == 0
    assert all(x["decision"] == "NOT_REVIEWED" for x in result["citation_gap_review_list"])


def test_multi_value_group_is_queued_without_automatic_rejection():
    result = MODULE.build_queue([row("a", "P403", "Q2"), row("b", "P403", "Q3")])
    assert result["multi_value_group_count"] == 1
    group = result["multi_value_review_list"][0]
    assert group["decision"] == "NOT_REVIEWED"
    assert {x["object_qid"] for x in group["values"]} == {"Q2", "Q3"}


def test_review_ids_are_stable_and_csv_ledger_is_created(tmp_path):
    rows = [row("a", "P2043", "Q2"), row("b", "P2043", "Q3")]
    first = MODULE.build_queue(rows)
    second = MODULE.build_queue(rows)
    assert first["citation_gap_review_list"][0]["review_id"] == second["citation_gap_review_list"][0]["review_id"]
    ledger = tmp_path / "review.csv"
    MODULE.write_ledger(ledger, first)
    text = ledger.read_text(encoding="utf-8")
    assert "NOT_REVIEWED" in text
    assert "evidence_url" in text
