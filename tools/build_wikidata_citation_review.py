#!/usr/bin/env python3
"""Build a property-wise citation coverage and human-review report.

No citations are invented. This report separates external citations, Wikidata
reference records, import markers, unmatched statements, and multi-value groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


def source_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return "invalid-url"


def build_report(rows: list[dict], input_sha256: str | None = None) -> dict:
    per_property: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    review_rows = []
    domain_counts = Counter()
    citation_property_counts = Counter()
    source_type_counts = Counter()

    for row in rows:
        prop = str(row.get("property_id", "UNKNOWN"))
        summary = per_property.setdefault(prop, {
            "records": 0,
            "reference_records_present": 0,
            "external_citations_present": 0,
            "external_citations_absent": 0,
            "import_marker_only": 0,
            "reference_records_absent": 0,
            "statements_matched": 0,
            "statements_unmatched": 0,
            "url_citations": 0,
            "named_source_items": 0,
        })
        summary["records"] += 1

        references = row.get("wikidata_reference_records")
        if not isinstance(references, list):
            references = []
        has_reference_record = bool(references)
        has_external = bool(row.get("has_external_citation"))
        has_import = any(
            isinstance(reference, dict) and bool(reference.get("imported_from_items"))
            for reference in references
        )
        match_status = row.get("reference_enrichment", {}).get("status")
        summary["reference_records_present"] += int(has_reference_record)
        summary["reference_records_absent"] += int(not has_reference_record)
        summary["external_citations_present"] += int(has_external)
        summary["external_citations_absent"] += int(not has_external)
        summary["import_marker_only"] += int(has_import and not has_external)
        summary["statements_matched"] += int(match_status == "MATCHED")
        summary["statements_unmatched"] += int(match_status != "MATCHED")

        urls = []
        named_items = []
        import_items = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            for item in reference.get("source_urls", []) or []:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    urls.append(item)
                    domain_counts[source_domain(item["url"])] += 1
                    source_type_counts[str(item.get("property_id", "UNKNOWN"))] += 1
            for item in reference.get("source_items", []) or []:
                if isinstance(item, dict):
                    named_items.append(item)
                    source_type_counts[str(item.get("property_id", "UNKNOWN"))] += 1
            for item in reference.get("imported_from_items", []) or []:
                if isinstance(item, dict):
                    import_items.append(item)
                    source_type_counts[str(item.get("property_id", "UNKNOWN"))] += 1
        summary["url_citations"] += len(urls)
        summary["named_source_items"] += len(named_items)
        if has_external:
            citation_property_counts[prop] += 1

        if not has_external:
            if not has_reference_record:
                review_reason = "NO_REFERENCE_RECORD"
            elif has_import:
                review_reason = "IMPORT_MARKER_ONLY"
            else:
                review_reason = "REFERENCE_WITHOUT_EXTERNAL_CITATION"
            review_rows.append({
                "record_id": row.get("record_id"),
                "subject_qid": row.get("subject_qid"),
                "subject_label_hi": row.get("subject_label_hi"),
                "property_id": prop,
                "relation_label_hi": row.get("relation_label_hi"),
                "object_qid": row.get("object_qid"),
                "display_value_hi": row.get("display_value_hi"),
                "review_reason": review_reason,
                "statement_uri": row.get("statement_uri"),
                "reference_enrichment_status": match_status,
                "imported_from_items": import_items,
            })
        groups[(str(row.get("subject_qid", "")), prop)].append(row)

    multi_value_groups = []
    for (subject_qid, property_id), group in sorted(groups.items()):
        unique = {}
        for row in group:
            value_key = str(
                row.get("object_qid")
                or row.get("object_lexical")
                or row.get("display_value_hi")
                or ""
            )
            unique.setdefault(value_key, {
                "object_qid": row.get("object_qid"),
                "object_lexical": row.get("object_lexical"),
                "display_value_hi": row.get("display_value_hi"),
                "record_id": row.get("record_id"),
                "has_external_citation": bool(row.get("has_external_citation")),
                "statement_uri": row.get("statement_uri"),
            })
        if len(unique) > 1:
            first = group[0]
            multi_value_groups.append({
                "subject_qid": subject_qid,
                "subject_label_hi": first.get("subject_label_hi"),
                "property_id": property_id,
                "relation_label_hi": first.get("relation_label_hi"),
                "distinct_values": len(unique),
                "values": [unique[key] for key in sorted(unique)],
                "review_required": True,
                "decision": "PENDING_HUMAN_REVIEW",
                "note": "Can be a valid multi-valued property or a conflict; no automatic rejection.",
            })

    total = len(rows)
    property_report = {}
    for prop, values in sorted(per_property.items()):
        property_report[prop] = {
            **values,
            "external_citation_coverage_percent": round(
                100 * values["external_citations_present"] / values["records"], 1
            ) if values["records"] else 0.0,
            "reference_record_coverage_percent": round(
                100 * values["reference_records_present"] / values["records"], 1
            ) if values["records"] else 0.0,
        }

    return {
        "schema_version": "1.0",
        "status": "HUMAN_REVIEW_REQUIRED",
        "record_count": total,
        "overall": {
            "records_with_external_citations": sum(v["external_citations_present"] for v in per_property.values()),
            "records_without_external_citations": sum(v["external_citations_absent"] for v in per_property.values()),
            "records_with_reference_records": sum(v["reference_records_present"] for v in per_property.values()),
            "records_without_reference_records": sum(v["reference_records_absent"] for v in per_property.values()),
            "records_import_marker_only": sum(v["import_marker_only"] for v in per_property.values()),
            "statements_matched": sum(v["statements_matched"] for v in per_property.values()),
            "statements_unmatched": sum(v["statements_unmatched"] for v in per_property.values()),
            "url_citation_count": sum(v["url_citations"] for v in per_property.values()),
            "named_source_item_count": sum(v["named_source_items"] for v in per_property.values()),
            "external_citation_coverage_percent": round(
                100 * sum(v["external_citations_present"] for v in per_property.values()) / total, 1
            ) if total else 0.0,
        },
        "per_property": property_report,
        "external_source_domains": dict(domain_counts.most_common()),
        "source_property_counts": dict(sorted(source_type_counts.items())),
        "citation_gap_review_list": review_rows,
        "citation_gap_review_count": len(review_rows),
        "multi_value_groups": multi_value_groups,
        "multi_value_group_count": len(multi_value_groups),
        "input_sha256": input_sha256,
        "training_approved_record_count": 0,
        "training_approval": "NOT_GRANTED_BY_CITATION_REVIEW_REPORT",
        "factual_accuracy": "NOT_ASSESSED",
        "privacy_clearance": "NOT_ASSESSED",
    }


def markdown_summary(report: dict) -> str:
    overall = report["overall"]
    lines = [
        "# Wikidata Citation Coverage Review",
        "",
        f"- Records: **{report['record_count']}**",
        f"- External citations: **{overall['records_with_external_citations']} ({overall['external_citation_coverage_percent']}%)**",
        f"- Reference records: **{overall['records_with_reference_records']}**",
        f"- Import-marker-only: **{overall['records_import_marker_only']}**",
        f"- Statement matches: **{overall['statements_matched']} / {report['record_count']}**",
        f"- Citation-gap review rows: **{report['citation_gap_review_count']}**",
        f"- Multi-value groups for review: **{report['multi_value_group_count']}**",
        "- Training-approved records: **0**",
        "",
        "## Coverage by property",
        "",
        "| Property | Relation | Records | With external citation | Coverage | Import-only | No reference |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    # Relation label is attached to review list or source rows; report keeps
    # relation labels in the actual JSON list. For readable markdown, show ID.
    for prop, data in report["per_property"].items():
        lines.append(
            f"| {prop} | — | {data['records']} | {data['external_citations_present']} | "
            f"{data['external_citation_coverage_percent']}% | {data['import_marker_only']} | "
            f"{data['reference_records_absent']} |"
        )
    lines += [
        "",
        "## Multi-value groups requiring human review",
        "",
        "| Subject | Hindi label | Property | Relation | Distinct values |",
        "|---|---|---|---|---:|",
    ]
    for group in report["multi_value_groups"]:
        lines.append(
            f"| {group['subject_qid']} | {group.get('subject_label_hi') or '—'} | "
            f"{group['property_id']} | {group.get('relation_label_hi') or '—'} | "
            f"{group['distinct_values']} |"
        )
    lines += [
        "",
        "## Citation-gap categories",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    reasons = Counter(item["review_reason"] for item in report["citation_gap_review_list"])
    for reason, count in sorted(reasons.items()):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "This report does not add citations or approve records. Multi-valued groups may be legitimate. Human review, factual verification, and privacy review remain required.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every JSONL row must be an object")
    report = build_report(rows, input_sha256=hashlib.sha256(raw).hexdigest())
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = markdown_summary(report)
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "record_count": report["record_count"],
        "overall": report["overall"],
        "per_property": report["per_property"],
        "external_source_domains": report["external_source_domains"],
        "citation_gap_review_count": report["citation_gap_review_count"],
        "multi_value_group_count": report["multi_value_group_count"],
        "training_approved_record_count": 0,
        "input_sha256": report["input_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
