import json
from pathlib import Path


def test_external_rights_triage_keeps_all_four_sources_unapproved():
    root = Path(__file__).parent
    report = json.loads((root / "external_dataset_rights_review_2026-11-23.json").read_text(encoding="utf-8"))
    assert len(report["sources"]) == 4
    assert report["approved_rows"] == 0
    assert report["expert_dataset_approval_rows"] == 27
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert report["production_training_gate"] == "CLOSED"
    assert all(source["commercial_training_approved"] is False for source in report["sources"])
    assert all(source["registry_approval_changed"] is False for source in report["sources"])


def test_high_risk_sources_are_not_mislabeled_as_approved():
    root = Path(__file__).parent
    report = json.loads((root / "external_dataset_rights_review_2026-11-23.json").read_text(encoding="utf-8"))
    by_id = {source["dataset_id"]: source for source in report["sources"]}
    assert by_id["HuggingFaceM4/DocumentVQA"]["decision"] == "HOLD_DO_NOT_INGEST"
    assert by_id["teknium/OpenHermes-2.5"]["decision"] == "EXCLUDE_PENDING_SOURCE_BY_SOURCE_CLEARANCE"
    assert "CONDITIONAL_HOLD" in by_id["ai4bharat/IndicCorpV2"]["decision"]
    assert "CONDITIONAL_CANDIDATE" in by_id["ccdv/govreport-summarization"]["decision"]
