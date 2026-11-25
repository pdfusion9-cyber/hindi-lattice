#!/usr/bin/env python3
"""Export a bounded, traceable Hindi corpus from Wikidata structured data.

The JSONL file is the canonical source-aware output. The optional triple file
is a compatibility view for existing subject | relation | object consumers.
This tool never reads, modifies, or merges data/gyaan_base.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice; traceable structured-data export)"
LICENSE_ID = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
WIKIDATA_LICENSE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"

# Intentionally small, reviewable domain/property allow-list. Expand only by
# code review; labels are Hindi display values, not free-form source text.
SCOPES = (
    {"name": "country", "class_qid": "Q6256", "properties": (
        ("P36", "राजधानी"), ("P30", "महाद्वीप"), ("P37", "राजभाषा"),
        ("P38", "मुद्रा"), ("P1082", "जनसंख्या"), ("P2046", "क्षेत्रफल"),
    )},
    {"name": "city", "class_qid": "Q515", "properties": (
        ("P17", "देश"), ("P131", "प्रशासनिक क्षेत्र"), ("P1082", "जनसंख्या"),
        ("P571", "स्थापना"),
    )},
    {"name": "river", "class_qid": "Q4022", "properties": (
        ("P17", "देश"), ("P403", "मुहाना"), ("P2043", "लंबाई"),
    )},
    {"name": "mountain", "class_qid": "Q8502", "properties": (
        ("P17", "देश"), ("P2044", "ऊँचाई"), ("P4552", "पर्वत श्रृंखला"),
    )},
)

QID_RE = re.compile(r"^Q[1-9][0-9]*$")
PID_RE = re.compile(r"^P[1-9][0-9]*$")
ENTITY_PREFIX = "http://www.wikidata.org/entity/"
ENTITY_PREFIX_HTTPS = "https://www.wikidata.org/entity/"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def uri_tail(value: str | None) -> str | None:
    if not value:
        return None
    for prefix in (ENTITY_PREFIX, ENTITY_PREFIX_HTTPS):
        if value.startswith(prefix):
            tail = value[len(prefix):]
            return tail if QID_RE.fullmatch(tail) else None
    return None


def binding_value(binding: dict[str, Any], key: str) -> str | None:
    item = binding.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    return value if isinstance(value, str) else None


def build_query(class_qid: str, property_id: str, limit: int, offset: int) -> str:
    if not QID_RE.fullmatch(class_qid):
        raise ValueError(f"invalid Wikidata class QID: {class_qid}")
    if not PID_RE.fullmatch(property_id):
        raise ValueError(f"invalid Wikidata property ID: {property_id}")
    if not 1 <= limit <= 5000 or offset < 0:
        raise ValueError("limit must be 1..5000 and offset must be non-negative")

    return f"""SELECT ?subject ?subject_label_hi ?subject_label_en ?statement
                    ?object ?object_label_hi ?object_label_en ?object_qid
                    ?object_lexical ?object_datatype ?object_language
                    ?reference_url ?reference_item
WHERE {{
  ?subject wdt:P31 wd:{class_qid} ;
           rdfs:label ?subject_label_hi ;
           p:{property_id} ?statement .
  FILTER(LANG(?subject_label_hi) = "hi")
  OPTIONAL {{
    ?subject rdfs:label ?subject_label_en .
    FILTER(LANG(?subject_label_en) = "en")
  }}
  ?statement ps:{property_id} ?object .
  OPTIONAL {{
    ?object rdfs:label ?object_label_hi .
    FILTER(LANG(?object_label_hi) = "hi")
  }}
  OPTIONAL {{
    ?object rdfs:label ?object_label_en .
    FILTER(LANG(?object_label_en) = "en")
  }}
  BIND(IF(isIRI(?object), REPLACE(STR(?object), "^https?://www\\\\.wikidata\\\\.org/entity/", ""), "") AS ?object_qid)
  BIND(IF(isLiteral(?object), STR(?object), "") AS ?object_lexical)
  BIND(IF(isLiteral(?object), STR(DATATYPE(?object)), "") AS ?object_datatype)
  BIND(IF(isLiteral(?object), LANG(?object), "") AS ?object_language)
  OPTIONAL {{
    ?statement prov:wasDerivedFrom ?reference .
    OPTIONAL {{ ?reference pr:P854 ?reference_url . }}
    OPTIONAL {{ ?reference pr:P248 ?reference_item . }}
  }}
}}
ORDER BY STR(?subject) STR(?statement) STR(?object) STR(?reference_url)
LIMIT {limit}
OFFSET {offset}"""


def row_from_binding(
    binding: dict[str, Any],
    *,
    property_id: str,
    relation_hi: str,
    class_qid: str,
    endpoint: str,
    query_sha256: str,
    retrieved_at_utc: str,
) -> dict[str, Any] | None:
    subject_uri = binding_value(binding, "subject")
    subject_qid = uri_tail(subject_uri)
    subject_hi = binding_value(binding, "subject_label_hi")
    statement_uri = binding_value(binding, "statement")
    object_uri = binding_value(binding, "object")
    object_qid = uri_tail(binding_value(binding, "object_qid")) or uri_tail(object_uri)
    object_hi = binding_value(binding, "object_label_hi")
    object_en = binding_value(binding, "object_label_en")
    object_lexical = binding_value(binding, "object_lexical")
    object_datatype = binding_value(binding, "object_datatype")
    object_language = binding_value(binding, "object_language")
    statement_id = statement_uri.rsplit("/", 1)[-1] if statement_uri else None

    if not subject_qid or not subject_hi or not statement_id or not object_uri:
        return None
    if object_qid and not QID_RE.fullmatch(object_qid):
        object_qid = None
    display_value = object_hi or object_lexical or object_qid
    if not display_value:
        return None

    reference_url = binding_value(binding, "reference_url")
    reference_item_uri = binding_value(binding, "reference_item")
    reference_item_qid = uri_tail(reference_item_uri)
    source_references = []
    if reference_url:
        source_references.append({
            "kind": "url",
            "value": reference_url,
            "retrieved_at_utc": retrieved_at_utc,
            "source_endpoint": endpoint,
            "source_query_sha256": query_sha256,
        })
    if reference_item_qid:
        source_references.append({
            "kind": "wikidata_item",
            "value": reference_item_qid,
            "url": f"https://www.wikidata.org/wiki/{reference_item_qid}",
            "retrieved_at_utc": retrieved_at_utc,
            "source_endpoint": endpoint,
            "source_query_sha256": query_sha256,
        })

    identity = {
        "subject_qid": subject_qid,
        "property_id": property_id,
        "statement_uri": statement_uri,
        "object_uri": object_uri,
        "object_lexical": object_lexical,
    }
    return {
        "record_id": sha256_text(canonical_json(identity)),
        "subject_qid": subject_qid,
        "subject_label_hi": subject_hi,
        "subject_label_en": binding_value(binding, "subject_label_en"),
        "property_id": property_id,
        "relation_label_hi": relation_hi,
        "class_qid": class_qid,
        "statement_uri": statement_uri,
        "object_qid": object_qid,
        "object_label_hi": object_hi,
        "object_label_en": object_en,
        "object_lexical": object_lexical,
        "object_datatype": object_datatype,
        "object_language": object_language,
        "display_value_hi": display_value,
        "source_references": source_references,
        "has_statement_reference": bool(source_references),
        "subject_url": f"https://www.wikidata.org/wiki/{subject_qid}",
        "object_url": (
            f"https://www.wikidata.org/wiki/{object_qid}" if object_qid else None
        ),
        "retrieved_at_utc": retrieved_at_utc,
        "source_endpoint": endpoint,
        "source_query_sha256": query_sha256,
        "source_license": LICENSE_ID,
        "source_license_url": LICENSE_URL,
        "source_policy_url": WIKIDATA_LICENSE_URL,
    }


def parse_results(
    payload: dict[str, Any],
    *,
    property_id: str,
    relation_hi: str,
    class_qid: str,
    endpoint: str,
    query_sha256: str,
    retrieved_at_utc: str,
) -> list[dict[str, Any]]:
    bindings = payload.get("results", {}).get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("SPARQL response has no results.bindings array")
    rows = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        row = row_from_binding(
            binding,
            property_id=property_id,
            relation_hi=relation_hi,
            class_qid=class_qid,
            endpoint=endpoint,
            query_sha256=query_sha256,
            retrieved_at_utc=retrieved_at_utc,
        )
        if row:
            rows.append(row)
    return rows


def fetch_query(
    query: str,
    *,
    endpoint: str = ENDPOINT,
    timeout: int = 90,
    retries: int = 4,
) -> dict[str, Any]:
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                parsed = json.loads(response.read().decode(charset))
                if not isinstance(parsed, dict):
                    raise ValueError("SPARQL endpoint returned non-object JSON")
                return parsed
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == retries - 1:
                raise RuntimeError(f"Wikidata SPARQL HTTP {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(2, min(60, int(retry_after))) if retry_after else min(60, 5 * (2 ** attempt))
            except ValueError:
                delay = min(60, 5 * (2 ** attempt))
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Wikidata SPARQL request failed: {type(exc).__name__}") from exc
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError("Wikidata SPARQL retries exhausted")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def select_scopes(names: list[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    known = {scope["name"] for scope in SCOPES}
    unknown = wanted - known
    if unknown:
        raise ValueError("unknown scope(s): " + ", ".join(sorted(unknown)))
    return [scope for scope in SCOPES if scope["name"] in wanted]


def export(
    *,
    scopes: list[dict[str, Any]],
    limit_per_property: int,
    page_size: int,
    endpoint: str,
    request_interval: float,
    sleep=time.sleep,
    fetch=fetch_query,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= limit_per_property <= 5000:
        raise ValueError("limit-per-property must be in 1..5000")
    if not 1 <= page_size <= min(500, limit_per_property):
        raise ValueError("page-size must be in 1..min(500, limit-per-property)")
    if request_interval < 1.0:
        raise ValueError("request-interval must be at least 1 second")

    records: dict[str, dict[str, Any]] = {}
    query_log = []
    last_request_started = 0.0

    for scope in scopes:
        for property_id, relation_hi in scope["properties"]:
            offset = 0
            remaining = limit_per_property
            while remaining > 0:
                page_limit = min(page_size, remaining)
                query = build_query(scope["class_qid"], property_id, page_limit, offset)
                elapsed = time.monotonic() - last_request_started if last_request_started else request_interval
                if elapsed < request_interval:
                    sleep(request_interval - elapsed)
                retrieved = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                last_request_started = time.monotonic()
                query_hash = sha256_text(query)
                payload = fetch(query, endpoint=endpoint)
                batch = parse_results(
                    payload,
                    property_id=property_id,
                    relation_hi=relation_hi,
                    class_qid=scope["class_qid"],
                    endpoint=endpoint,
                    query_sha256=query_hash,
                    retrieved_at_utc=retrieved,
                )
                for row in batch:
                    existing = records.get(row["record_id"])
                    if existing is None:
                        records[row["record_id"]] = row
                        continue
                    # One Wikidata statement may have multiple references. Keep
                    # every distinct reference instead of overwriting the row.
                    merged = {
                        canonical_json(reference): reference
                        for reference in (
                            existing["source_references"] + row["source_references"]
                        )
                    }
                    existing["source_references"] = [
                        merged[key] for key in sorted(merged)
                    ]
                    existing["has_statement_reference"] = bool(
                        existing["source_references"]
                    )
                query_log.append({
                    "scope": scope["name"],
                    "class_qid": scope["class_qid"],
                    "property_id": property_id,
                    "query_sha256": query_hash,
                    "offset": offset,
                    "limit": page_limit,
                    "result_rows": len(batch),
                    "retrieved_at_utc": retrieved,
                })
                raw_count = len(payload.get("results", {}).get("bindings", []))
                remaining -= page_limit
                offset += page_limit
                if raw_count < page_limit:
                    break

    return sorted(records.values(), key=lambda row: (
        row["subject_qid"], row["property_id"], row["statement_uri"], row["record_id"]
    )), query_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopes", nargs="+", choices=[s["name"] for s in SCOPES],
                        default=["country", "city", "river", "mountain"])
    parser.add_argument("--limit-per-property", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-interval", type=float, default=1.25)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--jsonl-out", type=Path, default=Path("data/wikidata_structured.jsonl"))
    parser.add_argument("--triples-out", type=Path, default=Path("data/wikidata_structured_triples.txt"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/wikidata_structured_manifest.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned requests without contacting Wikidata")
    args = parser.parse_args()

    selected = select_scopes(args.scopes)
    if args.dry_run:
        planned = [
            {"scope": scope["name"], "class_qid": scope["class_qid"],
             "property_id": prop, "relation_label_hi": label}
            for scope in selected for prop, label in scope["properties"]
        ]
        print(json.dumps({
            "status": "DRY_RUN_NO_NETWORK",
            "endpoint": args.endpoint,
            "planned_query_count": len(planned),
            "limit_per_property": args.limit_per_property,
            "page_size": args.page_size,
            "request_interval_seconds": args.request_interval,
            "planned_queries": planned,
        }, ensure_ascii=False, indent=2))
        return 0

    records, query_log = export(
        scopes=selected,
        limit_per_property=args.limit_per_property,
        page_size=args.page_size,
        endpoint=args.endpoint,
        request_interval=args.request_interval,
    )
    jsonl = "".join(canonical_json(row) + "\n" for row in records)
    triples = set()
    for row in records:
        triples.add(
            f'{row["subject_label_hi"]} | {row["relation_label_hi"]} | {row["display_value_hi"]}'
        )
    triples_text = "".join(line + "\n" for line in sorted(triples))
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "wikidata_structured_hindi_traceable_v1",
        "status": "EXPORTED_NOT_YET_QUALITY_REVIEWED",
        "source": "Wikidata structured data only",
        "endpoint": args.endpoint,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_license": LICENSE_ID,
        "source_license_url": LICENSE_URL,
        "source_policy_url": WIKIDATA_LICENSE_URL,
        "scope_definitions": selected,
        "limits": {
            "limit_per_property": args.limit_per_property,
            "page_size": args.page_size,
            "request_interval_seconds": args.request_interval,
        },
        "record_count": len(records),
        "records_with_statement_references": sum(bool(r["has_statement_reference"]) for r in records),
        "records_without_statement_references": sum(not r["has_statement_reference"] for r in records),
        "query_count": len(query_log),
        "queries": query_log,
        "outputs": {
            "jsonl": {
                "path": str(args.jsonl_out),
                "sha256": sha256_bytes(jsonl.encode("utf-8")),
            },
            "triples": {
                "path": str(args.triples_out),
                "sha256": sha256_bytes(triples_text.encode("utf-8")),
            },
        },
        "notices": [
            "This export contains Wikidata structured statements and labels only; it does not include Wikipedia article text.",
            "CC0 source licensing does not certify factual accuracy, privacy, safety, completeness, or model-training suitability.",
            "Rows without statement references remain visibly flagged; no source citation is invented.",
            "This output is a candidate corpus, not an approval or a training checkpoint.",
        ],
    }
    atomic_write(args.jsonl_out, jsonl)
    atomic_write(args.triples_out, triples_text)
    atomic_write(args.manifest_out, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "record_count": len(records),
        "with_statement_references": manifest["records_with_statement_references"],
        "without_statement_references": manifest["records_without_statement_references"],
        "query_count": len(query_log),
        "jsonl": str(args.jsonl_out),
        "triples": str(args.triples_out),
        "manifest": str(args.manifest_out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
