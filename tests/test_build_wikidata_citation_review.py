import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_wikidata_citation_review.py"
SPEC = importlib.util.spec_from_file_location("wikidata_citation_review", SCRIPT)
REVIEW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REVIEW)


def make_row(record_id, prop="P36", value="Q60", refs=None, external=False):
    return {
        "record_id": record_id,
        "subject_qid": "Q30",
        "subject_label_hi": "देश",
        "property_id": prop,
        "relation_label_hi": "राजधानी",
        "object_qid": value,
        "display_value_hi": "नगर",
        "statement_uri": "https://www.wikidata.org/entity/statement/" + record_id,
        "reference_enrichment": {"status": "MATCHED"},
        "wikidata_reference_records": refs or [],
        "has_external_citation": external,
    }


def test_property_coverage_and_import_marker_only_are_separated():
    rows = [
        make_row("a", refs=[{"source_urls": [{"property_id": "P854", "url": "https://www.example.org/a"}],
                             "source_items": [], "imported_from_items": []}], external=True),
        make_row("b", refs=[{"source_urls": [], "source_items": [],
                             "imported_from_items": [{"property_id": "P143", "qid": "Q206855"}]}]),
        make_row("c", prop="P17"),
    ]
    report = REVIEW.build_report(rows)
    assert report["overall"]["records_with_external_citations"] == 1
    assert report["overall"]["records_import_marker_only"] == 1
    assert report["per_property"]["P36"]["records"] == 2
    assert report["per_property"]["P36"]["external_citation_coverage_percent"] == 50.0
    assert report["citation_gap_review_count"] == 2
    assert report["training_approved_record_count"] == 0


def test_multi_value_review_contains_values_without_marking_them_false():
    rows = [
        make_row("a", value="Q60"),
        make_row("b", value="Q61"),
    ]
    report = REVIEW.build_report(rows)
    assert report["multi_value_group_count"] == 1
    group = report["multi_value_groups"][0]
    assert group["decision"] == "PENDING_HUMAN_REVIEW"
    assert {item["object_qid"] for item in group["values"]} == {"Q60", "Q61"}
    assert "no automatic rejection" in group["note"]


def test_url_domains_are_normalized_and_source_properties_counted():
    rows = [make_row("a", refs=[{
        "source_urls": [{"property_id": "P854", "url": "https://www.example.org/path"}],
        "source_items": [{"property_id": "P248", "qid": "Q123"}],
        "imported_from_items": [],
    }], external=True)]
    report = REVIEW.build_report(rows)
    assert report["external_source_domains"] == {"example.org": 1}
    assert report["source_property_counts"] == {"P248": 1, "P854": 1}


def test_markdown_report_is_human_readable_and_keeps_training_closed():
    report = REVIEW.build_report([make_row("a")])
    markdown = REVIEW.markdown_summary(report)
    assert "Coverage by property" in markdown
    assert "Training-approved records: **0**" in markdown
