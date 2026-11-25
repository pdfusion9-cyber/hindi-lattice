import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "enrich_wikidata_statement_references.py"
SPEC = importlib.util.spec_from_file_location("wikidata_reference_enrichment", SCRIPT)
ENRICH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ENRICH)


def reference(snaks, digest="abc"):
    return {"hash": digest, "snaks": snaks}


def snak(value, datatype="string"):
    return {"snaktype": "value", "datatype": datatype,
            "datavalue": {"type": "string" if datatype == "url" else "wikibase-entityid",
                          "value": value}}


def test_import_marker_is_not_counted_as_external_citation():
    normalized = ENRICH.normalize_reference(reference({
        "P143": [snak({"id": "Q206855"}, "wikibase-item")]
    }))
    assert normalized["imported_from_items"] == [{"property_id": "P143", "qid": "Q206855"}]
    assert normalized["has_external_citation"] is False


def test_url_and_named_source_are_external_citations():
    normalized = ENRICH.normalize_reference(reference({
        "P854": [snak("https://example.org/source", "url")],
        "P248": [snak({"id": "Q123"}, "wikibase-item")],
    }))
    assert normalized["has_external_citation"] is True
    assert normalized["source_urls"][0]["url"] == "https://example.org/source"
    assert normalized["source_items"][0]["qid"] == "Q123"


def test_enrich_matches_statement_and_preserves_revision():
    row = {
        "subject_qid": "Q30",
        "statement_uri": "http://www.wikidata.org/entity/statement/Q30-uuid",
        "source_references": [],
    }
    statement = {
        "id": "Q30$uuid",
        "references": [reference({"P854": [snak("https://example.org/cite", "url")]})],
    }
    entities = {"Q30": {"id": "Q30", "lastrevid": 12345, "claims": {"P36": [statement]}}}
    enriched, summary = ENRICH.enrich_rows(
        [row], fetch=lambda ids, endpoint: entities, sleep=lambda _: None
    )
    assert enriched[0]["reference_enrichment"]["status"] == "MATCHED"
    assert enriched[0]["reference_enrichment"]["entity_revision_id"] == 12345
    assert enriched[0]["has_external_citation"] is True
    assert enriched[0]["source_references"][0]["url"] == "https://example.org/cite"
    assert summary["statements_matched"] == 1


def test_unmatched_statement_is_explicit_not_silently_accepted():
    row = {"subject_qid": "Q30", "statement_uri": "http://www.wikidata.org/entity/statement/Q30-nope"}
    enriched, summary = ENRICH.enrich_rows(
        [row], fetch=lambda ids, endpoint: {"Q30": {"id": "Q30", "claims": {}}},
        sleep=lambda _: None,
    )
    assert enriched[0]["reference_enrichment"]["status"] == "STATEMENT_NOT_MATCHED"
    assert enriched[0]["has_external_citation"] is False
    assert summary["statements_unmatched"] == 1


def test_action_api_batch_limit_is_enforced():
    try:
        ENRICH.fetch_entities(["Q" + str(i) for i in range(1, 52)])
    except ValueError as exc:
        assert "1..50" in str(exc)
    else:
        raise AssertionError("more than 50 QIDs should be rejected")

def test_guid_suffix_fallback_matches_unique_claim_even_when_qid_prefix_differs():
    claim = {"id": "Q30$aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "references": []}
    entity = {"id": "Q30", "claims": {"P36": [claim]}}
    matched, method = ENRICH.match_statement(
        entity,
        "http://www.wikidata.org/entity/statement/Q31-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert matched is claim
    assert method == "UNIQUE_GUID_SUFFIX"


def test_guid_suffix_fallback_rejects_ambiguous_matches():
    first = {"id": "Q30$aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
    second = {"id": "Q31$aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
    entity = {"id": "Q30", "claims": {"P36": [first, second]}}
    matched, method = ENRICH.match_statement(
        entity,
        "http://www.wikidata.org/entity/statement/Q32-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert matched is None
    assert method == "AMBIGUOUS_GUID_SUFFIX"

