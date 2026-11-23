import copy
import json
from pathlib import Path

from dataset_approval_policy import validate_registry


REGISTRY_PATH = Path(__file__).with_name("dataset_approval.json")


def _approved_row(expert_id=0, dataset_id="example/hindi-source"):
    return {
        "expert_id": expert_id,
        "expert_name": "hindi_indic_docs",
        "dataset_id": dataset_id,
        "revision": "a" * 40,
        "config_name": "default",
        "train_split": "train",
        "validation_split": "validation",
        "license_id": "CC-BY-4.0",
        "license_terms_url": "https://example.org/license",
        "commercial_training_allowed": True,
        "public_redistribution_allowed": True,
        "pii_reviewed": True,
        "consent_reviewed": True,
        "schema_reviewed": True,
        "split_reviewed": True,
        "approved": True,
        "reviewer": "qualified-reviewer",
        "review_date": "2026-11-23",
        "rights_holder": "Example rights holder",
        "rights_basis_url": "https://example.org/rights",
        "source_provenance_url": "https://example.org/source",
        "schema_fingerprint": "b" * 64,
        "schema_adapter_version": "1.0.0",
        "text_model_compatible": True,
        "legal_review_reference": "LEGAL-REVIEW-0001",
    }


def _registry(rows, status):
    return {
        "schema_version": "1.0",
        "status": status,
        "repository_visibility": "public",
        "warning": "Public repository artifacts may be downloadable.",
        "datasets": rows,
    }


def test_current_repository_registry_is_valid_but_all_pending():
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    report = validate_registry(payload)

    assert report["registry_valid"] is True
    assert report["status"] == "DATASET_APPROVAL_POLICY_VALID"
    assert report["approval_row_count"] == 27
    assert report["distinct_external_source_count"] == 22
    assert report["approved_row_count"] == 0
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert report["production_training_gate"] == "CLOSED"


def test_complete_human_approval_evidence_is_required_before_corpus_gate_opens():
    payload = _registry([_approved_row()], "APPROVED_FOR_COMMERCIAL_TRAINING")
    report = validate_registry(payload)

    assert report["registry_valid"] is True
    assert report["approved_row_count"] == 1
    assert report["corpus_preparation_gate"] == "READY_FOR_REVIEWED_CORPUS_PREPARATION"
    # This validator never turns on actual model training.
    assert report["production_training_gate"] == "CLOSED"


def test_approved_row_without_pinned_revision_or_evidence_fails_closed():
    row = _approved_row()
    row["revision"] = "main"
    row["license_terms_url"] = ""
    row["consent_reviewed"] = False
    payload = _registry([row], "APPROVED_FOR_COMMERCIAL_TRAINING")

    report = validate_registry(payload)

    assert report["registry_valid"] is False
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert any("40-character lowercase commit SHA" in message for message in report["errors"])
    assert any("license_terms_url" in message for message in report["errors"])
    assert any("consent_reviewed=true" in message for message in report["errors"])


def test_partially_approved_registry_stays_closed():
    payload = _registry(
        [_approved_row(), {**_approved_row(1, "example/other-source"), "approved": False}],
        "PARTIALLY_APPROVED_REVIEW",
    )
    report = validate_registry(payload)

    assert report["registry_valid"] is True
    assert report["approved_row_count"] == 1
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert report["production_training_gate"] == "CLOSED"


def test_status_drift_and_duplicate_rows_are_rejected():
    row = _approved_row()
    payload = _registry([row, copy.deepcopy(row)], "ALL_DATASETS_PENDING_REVIEW")

    report = validate_registry(payload)

    assert report["registry_valid"] is False
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert any("duplicate expert_id/dataset_id" in message for message in report["errors"])
    assert any("registry status mismatch" in message for message in report["errors"])


def test_non_https_rights_evidence_cannot_open_gate():
    row = _approved_row()
    row["rights_basis_url"] = "http://example.org/rights"
    payload = _registry([row], "APPROVED_FOR_COMMERCIAL_TRAINING")

    report = validate_registry(payload)

    assert report["registry_valid"] is False
    assert report["corpus_preparation_gate"] == "CLOSED"
    assert any("rights_basis_url must use HTTPS" in message for message in report["errors"])
