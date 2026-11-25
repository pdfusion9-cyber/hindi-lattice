#!/usr/bin/env python3
"""Structural quality audit for traceable Wikidata JSONL exports.

This tool flags review needs; it does not prove factual correctness or approve
records for model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

QID_RE = re.compile(r"^Q[1-9][0-9]*$")
PID_RE = re.compile(r"^P[1-9][0-9]*$")
REQUIRED = {
    "record_id", "subject_qid", "subject_label_hi", "property_id",
    "relation_label_hi", "statement_uri", "statement_rank",
    "display_value_hi", "source_references", "source_endpoint",
    "source_query_sha256", "retrieved_at_utc", "source_license",
}


def audit_rows(rows: list[dict]) -> dict:
    issues = []
    seen_ids = Counter(row.get("record_id") for row in rows)
    duplicate_ids = sorted(key for key, count in seen_ids.items() if key and count > 1)
    groups = defaultdict(list)
    property_counts = Counter()
    reference_count = 0
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED - set(row))
        if missing:
            issues.append({"row": index, "kind": "missing_fields", "fields": missing})
        if not QID_RE.fullmatch(str(row.get("subject_qid", ""))):
            issues.append({"row": index, "kind": "invalid_subject_qid"})
        if not PID_RE.fullmatch(str(row.get("property_id", ""))):
            issues.append({"row": index, "kind": "invalid_property_id"})
        if not row.get("subject_label_hi"):
            issues.append({"row": index, "kind": "missing_hindi_subject_label"})
        if not row.get("relation_label_hi") or not row.get("display_value_hi"):
            issues.append({"row": index, "kind": "missing_hindi_relation_or_value"})
        if row.get("statement_rank", "").endswith("DeprecatedRank"):
            issues.append({"row": index, "kind": "deprecated_rank"})
        refs = row.get("source_references")
        if not isinstance(refs, list):
            issues.append({"row": index, "kind": "invalid_reference_list"})
            refs = []
        if not refs:
            issues.append({"row": index, "kind": "missing_statement_reference"})
        else:
            reference_count += 1
        property_counts[str(row.get("property_id", "UNKNOWN"))] += 1
        groups[(row.get("subject_qid"), row.get("property_id"))].append(row)

    multi_value_groups = []
    for (subject, prop), group in sorted(groups.items(), key=lambda item: str(item[0])):
        values = sorted({str(row.get("object_qid") or row.get("object_lexical") or row.get("display_value_hi")) for row in group})
        if len(values) > 1:
            multi_value_groups.append({
                "subject_qid": subject,
                "property_id": prop,
                "distinct_values": len(values),
                "review_required": True,
                "note": "Potential multi-valued property or conflicting statements; human review required.",
            })
    issue_counts = Counter(item["kind"] for item in issues)
    return {
        "schema_version": "1.0",
        "status": "AUDITED_REQUIRES_HUMAN_REVIEW",
        "record_count": len(rows),
        "unique_record_ids": len({row.get("record_id") for row in rows}),
        "duplicate_record_ids": duplicate_ids,
        "property_distribution": dict(sorted(property_counts.items())),
        "records_with_statement_references": reference_count,
        "records_without_statement_references": len(rows) - reference_count,
        "multi_value_subject_property_groups": multi_value_groups,
        "multi_value_group_count": len(multi_value_groups),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
        "training_approved_record_count": 0,
        "training_approval": "NOT_GRANTED_BY_AUTOMATED_AUDIT",
        "factual_accuracy": "NOT_ASSESSED",
        "privacy_clearance": "NOT_ASSESSED",
        "input_sha256": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every JSONL row must be an object")
    report = audit_rows(rows)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "record_count": report["record_count"],
        "issue_counts": report["issue_counts"],
        "multi_value_group_count": report["multi_value_group_count"],
        "training_approved_record_count": 0,
        "input_sha256": report["input_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
