#!/usr/bin/env python3
"""Build a human-review matrix from registry rows and HF metadata audit.

This script performs no legal analysis and grants no approvals. It simply
joins expert/source assignments to the latest public-metadata audit report,
marks missing or incomplete evidence, and emits an actionable review checklist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYNTHETIC_MIXTURE_ID = "curated_mixture_from_all_domains"

REVIEW_CHECKLIST = [
    "Identify original rights holders and upstream source works, not only the dataset host.",
    "Read the exact license and linked terms for each included component and content type.",
    "Obtain an explicit, documented basis for commercial model training.",
    "Determine whether public redistribution of raw examples or derived artifacts is allowed; do not infer this from training permission.",
    "Review personal data, sensitive data, minors, confidential records, and applicable privacy obligations.",
    "Verify consent, notice, lawful basis, opt-outs, and any source-specific restrictions.",
    "Pin immutable repository revision, config, train split, and validation split.",
    "Inspect samples and confirm schema adapter, modality, data quality, and provenance.",
    "Record qualified reviewer, review date, evidence URLs, legal review reference, and decision.",
]


def build_review_matrix(registry: Any, metadata_report: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(registry, dict) or not isinstance(registry.get("datasets"), list):
        raise ValueError("approval registry must contain a datasets list")
    if not isinstance(metadata_report, dict) or not isinstance(metadata_report.get("datasets"), list):
        raise ValueError("metadata report must contain a datasets list")

    rows = registry["datasets"]
    audit_rows = metadata_report["datasets"]
    audit_by_id: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        if isinstance(row, dict) and isinstance(row.get("dataset_id"), str):
            audit_by_id[row["dataset_id"]] = row

    assignments: dict[str, list[dict[str, Any]]] = {}
    synthetic_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("non-object approval row encountered")
            continue
        dataset_id = str(row.get("dataset_id") or "").strip()
        assignment = {
            "expert_id": row.get("expert_id"),
            "expert_name": row.get("expert_name"),
            "approved": row.get("approved") is True,
            "revision_pinned_in_registry": bool(row.get("revision")),
            "license_evidence_recorded": bool(row.get("license_id") and row.get("license_terms_url")),
            "pii_reviewed": row.get("pii_reviewed") is True,
            "consent_reviewed": row.get("consent_reviewed") is True,
            "schema_reviewed": row.get("schema_reviewed") is True,
            "split_reviewed": row.get("split_reviewed") is True,
        }
        if dataset_id == SYNTHETIC_MIXTURE_ID:
            synthetic_rows.append(assignment)
            continue
        assignments.setdefault(dataset_id, []).append(assignment)

    source_reviews: list[dict[str, Any]] = []
    for dataset_id in sorted(set(assignments) | set(audit_by_id)):
        audit = audit_by_id.get(dataset_id)
        expert_rows = assignments.get(dataset_id, [])
        if audit is None:
            evidence = {
                "metadata_status": "not_found_in_audit",
                "declared_license": None,
                "license_status": "unverified",
                "revision_status": "not_pinned",
                "resolved_revision": None,
                "gated": None,
                "splits_status": "unverified",
                "schema_status": "unverified",
                "feature_names": [],
                "sample_rows_returned": 0,
            }
        else:
            evidence = {
                "metadata_status": audit.get("metadata_status", "unknown"),
                "declared_license": audit.get("declared_license"),
                "license_status": audit.get("license_status", "unverified"),
                "revision_status": audit.get("revision_status", "not_pinned"),
                "resolved_revision": audit.get("resolved_revision"),
                "last_modified": audit.get("last_modified"),
                "gated": audit.get("gated"),
                "splits_status": audit.get("splits_status", "unknown"),
                "splits": audit.get("splits", []),
                "schema_status": audit.get("schema_status", "unknown"),
                "feature_names": audit.get("feature_names", []),
                "sample_rows_returned": audit.get("sample_rows_returned", 0),
                "evidence_notes": audit.get("evidence", []),
            }

        declared_license = evidence["declared_license"]
        if evidence["metadata_status"] != "available":
            priority = "P0_METADATA_UNAVAILABLE"
        elif not declared_license:
            priority = "P0_LICENSE_METADATA_MISSING"
        elif evidence["gated"] is True or evidence["splits_status"] != "listed" or evidence["schema_status"] != "sampled":
            priority = "P1_ACCESS_OR_SCHEMA_REVIEW"
        else:
            priority = "P2_HUMAN_RIGHTS_AND_PROVENANCE_REVIEW"

        source_reviews.append({
            "dataset_id": dataset_id,
            "expert_assignments": expert_rows,
            "expert_assignment_count": len(expert_rows),
            "metadata_evidence": evidence,
            "review_priority": priority,
            "revision_pinned_in_approval_registry": all(
                row["revision_pinned_in_registry"] for row in expert_rows
            ) if expert_rows else False,
            "license_evidence_recorded_in_approval_registry": all(
                row["license_evidence_recorded"] for row in expert_rows
            ) if expert_rows else False,
            "review_gate": "BLOCKED_PENDING_HUMAN_REVIEW",
            "automatic_approval_granted": False,
            "review_checklist": [
                {"item": item, "status": "PENDING", "reviewer_note": ""}
                for item in REVIEW_CHECKLIST
            ],
        })

    priority_counts: dict[str, int] = {}
    for source in source_reviews:
        priority_counts[source["review_priority"]] = priority_counts.get(source["review_priority"], 0) + 1

    return {
        "report_version": "1.0",
        "source_registry_status": registry.get("status"),
        "metadata_report_generated_at_utc": metadata_report.get("generated_at_utc"),
        "registry_approval_row_count": len(rows),
        "distinct_external_dataset_count": len(assignments),
        "synthetic_mixture_placeholder_rows": len(synthetic_rows),
        "metadata_audit_dataset_count": len(audit_by_id),
        "source_review_priority_counts": priority_counts,
        "approved_registry_row_count": sum(row.get("approved") is True for row in rows if isinstance(row, dict)),
        "all_source_reviews_blocked_pending_human_review": all(
            source["review_gate"] == "BLOCKED_PENDING_HUMAN_REVIEW" for source in source_reviews
        ),
        "synthetic_mixture_placeholder_assignments": synthetic_rows,
        "source_reviews": source_reviews,
        "production_training_gate": "CLOSED",
        "corpus_preparation_gate": "CLOSED",
        "errors": errors,
        "note": (
            "This is a review organizer, not a legal or privacy determination. Hugging Face "
            "metadata/license tags, split listings, sample schemas, and resolved revisions are "
            "discovery evidence only. A resolved revision is not considered pinned in the approval "
            "registry until a human reviewer records it there. No dataset is approved by this report."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-file", type=Path, default=Path(__file__).with_name("dataset_approval.json"))
    parser.add_argument("--metadata-report", type=Path, default=Path(__file__).with_name("dataset_verification_report.json"))
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        registry = json.loads(args.approval_file.read_text(encoding="utf-8"))
        audit = json.loads(args.metadata_report.read_text(encoding="utf-8"))
        report = build_review_matrix(registry, audit)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"REFUSED: cannot build dataset review matrix: {exc}") from exc

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "registry_rows": report["registry_approval_row_count"],
        "external_datasets": report["distinct_external_dataset_count"],
        "priority_counts": report["source_review_priority_counts"],
        "approved_rows": report["approved_registry_row_count"],
        "production_training_gate": report["production_training_gate"],
        "report_path": str(args.report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
