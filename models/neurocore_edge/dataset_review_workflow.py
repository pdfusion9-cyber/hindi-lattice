#!/usr/bin/env python3
"""Generate a default-deny human review packet from dataset_approval.json."""
from __future__ import annotations
import argparse, json
from pathlib import Path

PLACEHOLDER = "curated_mixture_from_all_domains"
SOURCE_CHECKS = [
    "copyright_and_rights_holder", "exact_license_scope",
    "commercial_training_permission", "redistribution_permission_separate_from_training",
    "upstream_component_terms", "privacy_pii_sensitive_data",
    "consent_notice_lawful_basis", "opt_out_takedown",
    "immutable_revision_and_splits", "schema_modality_sample_inspection",
]
ASSIGNMENT_CHECKS = [
    "sample_quality", "schema_adapter", "split_integrity", "task_relevance",
    "language_and_modality_fit", "privacy_handling", "known_limitations",
]

def build_packet(registry):
    if not isinstance(registry, dict) or not isinstance(registry.get("datasets"), list):
        raise ValueError("registry must contain datasets list")
    rows = registry["datasets"]
    if len(rows) != 27:
        raise ValueError(f"expected 27 expert assignments; found {len(rows)}")
    if any(not isinstance(row, dict) or not row.get("dataset_id") for row in rows):
        raise ValueError("every assignment must have a dataset_id")
    external_ids = sorted({row["dataset_id"] for row in rows if row["dataset_id"] != PLACEHOLDER})
    assignments = []
    for i, row in enumerate(rows):
        assignments.append({
            "row_id": f"EXP-{int(row['expert_id']):02d}-{i+1:03d}",
            "expert_id": row["expert_id"], "expert_name": row["expert_name"],
            "dataset_id": row["dataset_id"], "decision": "PENDING",
            "reviewer": "", "review_date": "", "decision_reference": "",
            "scope": "expert-domain suitability, schema adapter and task fit",
            "required_checks": ASSIGNMENT_CHECKS, "evidence_urls": [], "notes": "",
        })
    sources = []
    for i, dataset_id in enumerate(external_ids):
        sources.append({
            "source_review_id": f"SRC-{i+1:03d}", "dataset_id": dataset_id,
            "decision": "PENDING", "rights_reviewer": "", "rights_review_date": "",
            "rights_review_reference": "", "domain_reviewer": "", "domain_review_date": "",
            "domain_review_reference": "",
            "rights_evidence": {
                "canonical_source_url": "", "license_id": "", "license_terms_url": "",
                "rights_holder": "", "commercial_training_basis_url": "",
                "redistribution_basis_url": "", "upstream_inventory_url": "",
                "immutable_revision": "", "config_name": "", "train_split": "",
                "validation_split": "",
            },
            "checks": SOURCE_CHECKS, "evidence_urls": [], "notes": "", "hold_reason": "",
        })
    return {
        "schema_version": "1.0", "title": "NeuroCore dataset human approval workflow",
        "policy": "Default deny. No automated approval. Each source requires independent rights and domain sign-off; each expert/source assignment requires a separate expert-domain decision.",
        "source_count": len(sources), "assignment_count": len(assignments),
        "approved_source_count": 0, "approved_assignment_count": 0,
        "source_reviews": sources, "expert_assignments": assignments,
        "gate_status": {"corpus_preparation": "CLOSED", "production_training": "CLOSED"},
        "instructions": [
            "Missing, conflicting, inaccessible or ambiguous evidence means HOLD or REJECT.",
            "License metadata or a public dataset card alone does not establish commercial training rights.",
            "Training and redistribution permissions are separate.",
            "Source approval does not automatically approve every expert assignment.",
            "Production training is a separate approval and configuration gate.",
        ],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("dataset_approval.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    packet = build_packet(registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": packet["source_count"], "assignments": packet["assignment_count"],
        "approved_sources": 0, "approved_assignments": 0, "gates": packet["gate_status"]}, indent=2))

if __name__ == "__main__":
    main()
