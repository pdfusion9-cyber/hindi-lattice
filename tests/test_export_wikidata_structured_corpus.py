import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_wikidata_structured_corpus.py"
SPEC = importlib.util.spec_from_file_location("wikidata_exporter", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPORTER)


def sample_binding():
    return {
        "subject": {"type": "uri", "value": "http://www.wikidata.org/entity/Q30"},
        "subject_label_hi": {"type": "literal", "value": "संयुक्त राज्य अमेरिका", "xml:lang": "hi"},
        "subject_label_en": {"type": "literal", "value": "United States", "xml:lang": "en"},
        "statement": {"type": "uri", "value": "http://www.wikidata.org/entity/statement/Q30-ABC"},
        "object": {"type": "uri", "value": "http://www.wikidata.org/entity/Q60"},
        "object_qid": {"type": "literal", "value": "Q60"},
        "object_label_hi": {"type": "literal", "value": "न्यूयॉर्क नगर", "xml:lang": "hi"},
        "object_label_en": {"type": "literal", "value": "New York City", "xml:lang": "en"},
        "reference_url": {"type": "uri", "value": "https://example.org/source"},
        "reference_item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q123"},
    }


def test_build_query_is_bounded_and_scoped():
    query = EXPORTER.build_query("Q6256", "P36", 100, 200)
    assert "wd:Q6256" in query
    assert "p:P36" in query
    assert "LIMIT 100" in query
    assert "OFFSET 200" in query
    assert 'STRAFTER(STR(?object), "http://www.wikidata.org/entity/")' in query
    assert 'STRAFTER(STR(?object), "https://www.wikidata.org/entity/")' in query
    with pytest.raises(ValueError):
        EXPORTER.build_query("Q6256; SERVICE", "P36", 100, 0)
    with pytest.raises(ValueError):
        EXPORTER.build_query("Q6256", "P36 } SERVICE", 100, 0)


def test_row_keeps_qids_statement_and_retrieval_provenance():
    row = EXPORTER.row_from_binding(
        sample_binding(),
        property_id="P36",
        relation_hi="राजधानी",
        class_qid="Q6256",
        endpoint=EXPORTER.ENDPOINT,
        query_sha256="a" * 64,
        retrieved_at_utc="2026-11-25T10:00:00Z",
    )
    assert row["subject_qid"] == "Q30"
    assert row["object_qid"] == "Q60"
    assert row["property_id"] == "P36"
    assert row["statement_uri"].endswith("Q30-ABC")
    assert row["has_statement_reference"] is True
    assert row["source_references"][0]["value"] == "https://example.org/source"
    assert row["retrieved_at_utc"].endswith("Z")
    assert row["source_license"] == "CC0-1.0"


def test_row_without_reference_is_flagged_not_fabricated():
    binding = sample_binding()
    binding.pop("reference_url")
    binding.pop("reference_item")
    row = EXPORTER.row_from_binding(
        binding, property_id="P36", relation_hi="राजधानी",
        class_qid="Q6256", endpoint=EXPORTER.ENDPOINT,
        query_sha256="b" * 64, retrieved_at_utc="2026-11-25T10:00:00Z",
    )
    assert row["source_references"] == []
    assert row["has_statement_reference"] is False


def test_literal_objects_preserve_datatype_and_language():
    binding = sample_binding()
    binding["object"] = {
        "type": "literal", "value": "1200000", "datatype": "http://www.w3.org/2001/XMLSchema#decimal"
    }
    binding["object_qid"] = {"type": "literal", "value": ""}
    binding["object_lexical"] = {"type": "literal", "value": "1200000"}
    binding["object_datatype"] = {
        "type": "literal", "value": "http://www.w3.org/2001/XMLSchema#decimal"
    }
    binding["object_language"] = {"type": "literal", "value": ""}
    binding.pop("object_label_hi")
    row = EXPORTER.row_from_binding(
        binding, property_id="P1082", relation_hi="जनसंख्या",
        class_qid="Q6256", endpoint=EXPORTER.ENDPOINT,
        query_sha256="c" * 64, retrieved_at_utc="2026-11-25T10:00:00Z",
    )
    assert row["object_qid"] is None
    assert row["object_lexical"] == "1200000"
    assert row["object_datatype"].endswith("#decimal")


def test_parse_results_rejects_malformed_response():
    with pytest.raises(ValueError):
        EXPORTER.parse_results(
            {}, property_id="P36", relation_hi="राजधानी",
            class_qid="Q6256", endpoint=EXPORTER.ENDPOINT,
            query_sha256="d" * 64, retrieved_at_utc="2026-11-25T10:00:00Z",
        )


def test_export_pages_deduplicates_and_records_query_metadata():
    calls = []
    binding = sample_binding()

    def fake_fetch(query, endpoint):
        calls.append(query)
        return {"results": {"bindings": [binding]}}

    scope = {"name": "tiny", "class_qid": "Q6256", "properties": (("P36", "राजधानी"),)}
    records, log = EXPORTER.export(
        scopes=[scope], limit_per_property=2, page_size=1,
        endpoint="https://example.test/sparql", request_interval=1,
        sleep=lambda _: None, fetch=fake_fetch,
    )
    assert len(calls) == 2
    assert len(records) == 1
    assert len(log) == 2
    assert log[0]["query_sha256"] != log[1]["query_sha256"]


def test_export_merges_multiple_statement_references():
    first = sample_binding()
    second = sample_binding()
    second["reference_url"] = {"type": "uri", "value": "https://example.org/second"}
    second["reference_item"] = {
        "type": "uri", "value": "http://www.wikidata.org/entity/Q456"
    }

    calls = []

    def fake_fetch(query, endpoint):
        calls.append(query)
        binding = first if len(calls) == 1 else second
        return {"results": {"bindings": [binding]}}

    scope = {"name": "tiny", "class_qid": "Q6256", "properties": (("P36", "राजधानी"),)}
    records, _ = EXPORTER.export(
        scopes=[scope], limit_per_property=2, page_size=1,
        endpoint="https://example.test/sparql", request_interval=1,
        sleep=lambda _: None, fetch=fake_fetch,
    )
    assert len(records) == 1
    assert len(records[0]["source_references"]) == 4
    assert records[0]["has_statement_reference"] is True


def test_export_refuses_aggressive_request_rate():
    scope = {"name": "tiny", "class_qid": "Q6256", "properties": (("P36", "राजधानी"),)}
    with pytest.raises(ValueError):
        EXPORTER.export(
            scopes=[scope], limit_per_property=1, page_size=1,
            endpoint="https://example.test/sparql", request_interval=0.1,
            sleep=lambda _: None, fetch=lambda *a, **k: {},
        )


def test_triple_compatibility_is_not_the_canonical_source():
    # The exporter writes a JSONL record with provenance; triples are only a
    # projection. This assertion protects the required provenance fields.
    row = EXPORTER.row_from_binding(
        sample_binding(), property_id="P36", relation_hi="राजधानी",
        class_qid="Q6256", endpoint=EXPORTER.ENDPOINT,
        query_sha256="e" * 64, retrieved_at_utc="2026-11-25T10:00:00Z",
    )
    assert {"record_id", "subject_qid", "property_id", "statement_uri",
            "source_query_sha256", "retrieved_at_utc", "source_license"} <= set(row)
