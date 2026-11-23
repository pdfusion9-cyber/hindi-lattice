import json
from pathlib import Path

from build_dataset_review_matrix import build_review_matrix


def _assignment(expert_id, dataset_id, **overrides):
    row = {
        "expert_id": expert_id,
        "expert_name": "test_expert",
        "dataset_id": dataset_id,
        "approved": False,
        "revision": None,
        "license_id": None,
        "license_terms_url": None,
        "pii_reviewed": False,
        "consent_reviewed": False,
        "schema_reviewed": False,
        "split_reviewed": False,
    }
    row.update(overrides)
    return row


def test_current_registry_review_matrix_remains_fail_closed():
    root = Path(__file__).parent
    registry = json.loads((root / "dataset_approval.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "dataset_verification_report.json").read_text(encoding="utf-8"))
    report = build_review_matrix(registry, audit)

    assert report["registry_approval_row_count"] == 27
    assert report["distinct_external_dataset_count"] == 21
    assert report["synthetic_mixture_placeholder_rows"] == 1
    assert report["approved_registry_row_count"] == 0
    assert report["all_source_reviews_blocked_pending_human_review"] is True
    assert report["production_training_gate"] == "CLOSED"
    assert report["corpus_preparation_gate"] == "CLOSED"


def test_multiple_expert_assignments_are_grouped_by_source():
    registry = {
        "status": "ALL_DATASETS_PENDING_REVIEW",
        "datasets": [
            _assignment(1, "example/hindi"),
            _assignment(2, "example/hindi"),
        ],
    }
    audit = {
        "generated_at_utc": "2026-11-23T00:00:00Z",
        "datasets": [{
            "dataset_id": "example/hindi",
            "metadata_status": "available",
            "declared_license": "cc-by-4.0",
            "license_status": "declared_requires_terms_review",
            "revision_status": "not_pinned",
            "resolved_revision": "a" * 40,
            "gated": False,
            "splits_status": "listed",
            "schema_status": "sampled",
            "feature_names": ["text"],
            "sample_rows_returned": 5,
        }],
    }

    report = build_review_matrix(registry, audit)
    source = report["source_reviews"][0]

    assert source["dataset_id"] == "example/hindi"
    assert source["expert_assignment_count"] == 2
    assert source["metadata_evidence"]["declared_license"] == "cc-by-4.0"
    assert source["review_priority"] == "P2_HUMAN_RIGHTS_AND_PROVENANCE_REVIEW"
    assert source["review_gate"] == "BLOCKED_PENDING_HUMAN_REVIEW"
    assert source["automatic_approval_granted"] is False


def test_missing_license_and_gated_access_are_prioritized():
    registry = {
        "status": "ALL_DATASETS_PENDING_REVIEW",
        "datasets": [
            _assignment(1, "example/no-license"),
            _assignment(2, "example/gated"),
        ],
    }
    audit = {
        "datasets": [
            {
                "dataset_id": "example/no-license",
                "metadata_status": "available",
                "declared_license": None,
                "license_status": "missing",
                "revision_status": "not_pinned",
                "gated": False,
                "splits_status": "listed",
                "schema_status": "sampled",
            },
            {
                "dataset_id": "example/gated",
                "metadata_status": "available",
                "declared_license": "cc-by-4.0",
                "license_status": "declared_requires_terms_review",
                "revision_status": "not_pinned",
                "gated": True,
                "splits_status": "unavailable",
                "schema_status": "unknown",
            },
        ],
    }

    report = build_review_matrix(registry, audit)
    by_id = {row["dataset_id"]: row for row in report["source_reviews"]}

    assert by_id["example/no-license"]["review_priority"] == "P0_LICENSE_METADATA_MISSING"
    assert by_id["example/gated"]["review_priority"] == "P1_ACCESS_OR_SCHEMA_REVIEW"
    assert all(row["automatic_approval_granted"] is False for row in report["source_reviews"])


def test_unmatched_source_is_not_assumed_available():
    registry = {
        "status": "ALL_DATASETS_PENDING_REVIEW",
        "datasets": [_assignment(1, "example/missing-from-audit")],
    }
    report = build_review_matrix(registry, {"datasets": []})
    source = report["source_reviews"][0]

    assert source["metadata_evidence"]["metadata_status"] == "not_found_in_audit"
    assert source["review_priority"] == "P0_METADATA_UNAVAILABLE"
    assert source["review_gate"] == "BLOCKED_PENDING_HUMAN_REVIEW"
