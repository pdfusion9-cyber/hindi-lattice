#!/usr/bin/env python3
"""Validate NeuroCore dataset approvals without granting approvals.

The validator is deliberately fail-closed. It distinguishes a structurally
valid pending registry from an approval-complete registry. It never fetches
datasets, interprets a license, or changes approval fields.

A source is eligible for corpus preparation only when every expert/dataset row
has a pinned immutable revision, split/schema identity, documented rights
basis, privacy/consent review, a human legal-review reference, and explicit
commercial-training and redistribution permissions. Production model training
is a separate gate and remains disabled by config.json.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

HEX_SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATUSES = {
    "ALL_DATASETS_PENDING_REVIEW",
    "PARTIALLY_APPROVED_REVIEW",
    "APPROVED_FOR_COMMERCIAL_TRAINING",
}

BASE_TEXT_FIELDS = ("expert_name", "dataset_id")
BOOL_FIELDS = (
    "commercial_training_allowed",
    "public_redistribution_allowed",
    "pii_reviewed",
    "consent_reviewed",
    "schema_reviewed",
    "split_reviewed",
    "approved",
)
APPROVAL_TEXT_FIELDS = (
    "config_name",
    "train_split",
    "validation_split",
    "license_id",
    "license_terms_url",
    "reviewer",
    "review_date",
    "rights_holder",
    "rights_basis_url",
    "source_provenance_url",
    "schema_fingerprint",
    "schema_adapter_version",
    "legal_review_reference",
    "rights_reviewer",
    "rights_review_date",
    "rights_review_reference",
    "domain_reviewer",
    "domain_review_date",
    "domain_review_reference",
)
APPROVAL_TRUE_FLAGS = (
    "commercial_training_allowed",
    "public_redistribution_allowed",
    "pii_reviewed",
    "consent_reviewed",
    "schema_reviewed",
    "split_reviewed",
)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_https(value: Any) -> bool:
    return _nonempty(value) and value.strip().startswith("https://")


def _expected_status(rows: list[dict[str, Any]]) -> str:
    approved_count = sum(row.get("approved") is True for row in rows)
    if approved_count == 0:
        return "ALL_DATASETS_PENDING_REVIEW"
    if approved_count == len(rows):
        return "APPROVED_FOR_COMMERCIAL_TRAINING"
    return "PARTIALLY_APPROVED_REVIEW"


def validate_registry(payload: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {
            "status": "DATASET_APPROVAL_POLICY_INVALID",
            "registry_valid": False,
            "corpus_preparation_gate": "CLOSED",
            "production_training_gate": "CLOSED",
            "errors": ["registry root must be a JSON object"],
        }

    rows = payload.get("datasets")
    if not isinstance(rows, list) or not rows:
        rows = []
        errors.append("datasets must be a non-empty list")

    if payload.get("schema_version") != "1.0":
        errors.append("schema_version must equal '1.0'")
    if not _nonempty(payload.get("repository_visibility")):
        errors.append("repository_visibility is missing")
    if not _nonempty(payload.get("warning")):
        errors.append("public repository warning is missing")

    declared_status = payload.get("status")
    if declared_status not in ALLOWED_STATUSES:
        errors.append(f"unsupported registry status: {declared_status!r}")

    pair_keys: set[tuple[int, str]] = set()
    approved_rows = 0
    distinct_sources: set[str] = set()
    row_summaries: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        prefix = f"datasets[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue

        row_errors: list[str] = []
        for field in BASE_TEXT_FIELDS:
            if not _nonempty(row.get(field)):
                row_errors.append(f"{field} is required")

        expert_id = row.get("expert_id")
        if not isinstance(expert_id, int) or isinstance(expert_id, bool) or not 0 <= expert_id <= 19:
            row_errors.append("expert_id must be an integer from 0 through 19")

        dataset_id = row.get("dataset_id")
        if _nonempty(dataset_id):
            distinct_sources.add(dataset_id.strip())

        for field in BOOL_FIELDS:
            if not isinstance(row.get(field), bool):
                row_errors.append(f"{field} must be a boolean")

        if isinstance(expert_id, int) and not isinstance(expert_id, bool) and _nonempty(dataset_id):
            pair = (expert_id, dataset_id.strip())
            if pair in pair_keys:
                row_errors.append("duplicate expert_id/dataset_id approval row")
            pair_keys.add(pair)

        if row.get("approved") is True:
            approved_rows += 1
            for field in APPROVAL_TEXT_FIELDS:
                if not _nonempty(row.get(field)):
                    row_errors.append(f"approved row requires {field}")

            if _nonempty(row.get("revision")) and not HEX_SHA40.fullmatch(row["revision"].strip()):
                row_errors.append("revision must be a pinned 40-character lowercase commit SHA")
            if _nonempty(row.get("license_terms_url")) and not _is_https(row.get("license_terms_url")):
                row_errors.append("license_terms_url must use HTTPS")
            for field in ("rights_basis_url", "source_provenance_url"):
                if _nonempty(row.get(field)) and not _is_https(row.get(field)):
                    row_errors.append(f"{field} must use HTTPS")
            for date_field in ("review_date", "rights_review_date", "domain_review_date"):
                if _nonempty(row.get(date_field)):
                    try:
                        parsed = date.fromisoformat(row[date_field].strip())
                        if parsed.isoformat() != row[date_field].strip():
                            row_errors.append(f"{date_field} must use YYYY-MM-DD format")
                    except ValueError:
                        row_errors.append(f"{date_field} must be a valid YYYY-MM-DD date")
            if (
                _nonempty(row.get("rights_reviewer"))
                and _nonempty(row.get("domain_reviewer"))
                and row["rights_reviewer"].strip().casefold()
                    == row["domain_reviewer"].strip().casefold()
            ):
                row_errors.append("rights_reviewer and domain_reviewer must be two distinct people")
            if _nonempty(row.get("schema_fingerprint")) and not SHA256.fullmatch(
                row["schema_fingerprint"].strip()
            ):
                row_errors.append("schema_fingerprint must be a 64-character SHA-256 hex digest")
            for field in APPROVAL_TRUE_FLAGS:
                if row.get(field) is not True:
                    row_errors.append(f"approved row requires {field}=true")
            if row.get("text_model_compatible") is not True:
                row_errors.append("approved row requires text_model_compatible=true")
            if not _nonempty(row.get("schema_adapter_version")):
                row_errors.append("approved row requires schema_adapter_version")

        if row_errors:
            errors.extend(f"{prefix}: {message}" for message in row_errors)

        row_summaries.append({
            "expert_id": expert_id,
            "dataset_id": dataset_id,
            "approved": row.get("approved") is True,
            "validation_errors": row_errors,
        })

    expected_status = _expected_status(rows) if rows else None
    if declared_status != expected_status:
        errors.append(
            f"registry status mismatch: declared {declared_status!r}, expected {expected_status!r}"
        )

    registry_valid = not errors
    all_approved = bool(rows) and approved_rows == len(rows)
    corpus_ready = registry_valid and all_approved

    return {
        "status": "DATASET_APPROVAL_POLICY_VALID" if registry_valid else "DATASET_APPROVAL_POLICY_INVALID",
        "registry_valid": registry_valid,
        "declared_registry_status": declared_status,
        "expected_registry_status": expected_status,
        "approval_row_count": len(rows),
        "distinct_external_source_count": len(distinct_sources),
        "approved_row_count": approved_rows,
        "pending_or_unapproved_row_count": len(rows) - approved_rows,
        "corpus_preparation_gate": "READY_FOR_REVIEWED_CORPUS_PREPARATION" if corpus_ready else "CLOSED",
        "production_training_gate": "CLOSED",
        "row_summaries": row_summaries,
        "errors": errors,
        "note": (
            "This mechanical validator checks completeness of recorded evidence only. "
            "It does not verify that URLs are truthful, interpret license terms, establish "
            "copyright ownership, perform privacy/consent review, or replace qualified human "
            "legal and domain review. Even a complete registry does not enable production "
            "model training; config.json controls that separate gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approval-file",
        type=Path,
        default=Path(__file__).with_name("dataset_approval.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    try:
        payload = json.loads(args.approval_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"REFUSED: cannot read approval registry: {exc}") from exc

    report = validate_registry(payload)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "registry_valid": report["registry_valid"],
        "approval_rows": report["approval_row_count"],
        "approved_rows": report["approved_row_count"],
        "corpus_preparation_gate": report["corpus_preparation_gate"],
        "production_training_gate": report["production_training_gate"],
        "error_count": len(report["errors"]),
        "report_path": str(args.report_path),
    }, ensure_ascii=False, indent=2))
    if not report["registry_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
