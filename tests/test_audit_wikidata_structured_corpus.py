import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_wikidata_structured_corpus.py"
SPEC = importlib.util.spec_from_file_location("wikidata_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def row(record_id, subject="Q30", prop="P36", value="Q60", refs=None):
    return {
        "record_id": record_id,
        "subject_qid": subject,
        "subject_label_hi": "देश",
        "property_id": prop,
        "relation_label_hi": "राजधानी",
        "statement_uri": "https://www.wikidata.org/entity/statement/" + record_id,
        "statement_rank": "http://wikiba.se/ontology#NormalRank",
        "object_qid": value,
        "object_lexical": None,
        "display_value_hi": "नगर",
        "source_references": refs if refs is not None else [{"kind": "url", "value": "https://example.org"}],
        "source_endpoint": "https://query.wikidata.org/sparql",
        "source_query_sha256": "a" * 64,
        "retrieved_at_utc": "2026-11-25T00:00:00Z",
        "source_license": "CC0-1.0",
    }


def test_audit_flags_missing_references_and_never_approves_training():
    report = AUDIT.audit_rows([row("a"), row("b", subject="Q31", refs=[])])
    assert report["record_count"] == 2
    assert report["records_with_statement_references"] == 1
    assert report["records_without_statement_references"] == 1
    assert report["issue_counts"]["missing_statement_reference"] == 1
    assert report["training_approved_record_count"] == 0
    assert report["training_approval"] == "NOT_GRANTED_BY_AUTOMATED_AUDIT"


def test_audit_detects_duplicate_ids_and_multi_value_groups():
    report = AUDIT.audit_rows([row("same", value="Q60"), row("same", value="Q61")])
    assert report["duplicate_record_ids"] == ["same"]
    assert report["multi_value_group_count"] == 1
    assert report["multi_value_subject_property_groups"][0]["review_required"] is True


def test_audit_flags_invalid_ids_and_missing_hindi_label():
    bad = row("bad", subject="Q0", prop="P0")
    bad["subject_label_hi"] = ""
    report = AUDIT.audit_rows([bad])
    assert report["issue_counts"]["invalid_subject_qid"] == 1
    assert report["issue_counts"]["invalid_property_id"] == 1
    assert report["issue_counts"]["missing_hindi_subject_label"] == 1


def test_audit_does_not_call_multi_value_facts_false():
    report = AUDIT.audit_rows([row("a", value="Q60"), row("b", value="Q61")])
    group = report["multi_value_subject_property_groups"][0]
    assert group["note"].startswith("Potential multi-valued property")
    assert report["factual_accuracy"] == "NOT_ASSESSED"
