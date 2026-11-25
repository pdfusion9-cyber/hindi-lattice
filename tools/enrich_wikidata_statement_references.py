#!/usr/bin/env python3
"""Enrich exported Wikidata statements with complete reference snaks via wbgetentities.

Reads and writes JSONL only. It does not edit Wikidata or enable model training.
Entity IDs are fetched in bounded batches of at most 50, per Wikidata API guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ENDPOINT = "https://www.wikidata.org/w/api.php"
USER_AGENT = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice; statement-reference audit)"
QID_RE = re.compile(r"^Q[1-9][0-9]*$")
MAX_IDS_PER_REQUEST = 50


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def statement_api_id(statement_uri: str) -> str:
    tail = statement_uri.rsplit("/", 1)[-1]
    # RDF statement URIs conventionally use QID-UUID, while the Action API
    # claims representation uses QID$UUID.
    if re.match(r"^Q[1-9][0-9]*-", tail):
        qid, suffix = tail.split("-", 1)
        return qid + "$" + suffix
    return tail


def value_from_snak(snak: dict[str, Any]) -> dict[str, Any]:
    datavalue = snak.get("datavalue")
    if not isinstance(datavalue, dict):
        datavalue = {}
    return {
        "snaktype": snak.get("snaktype"),
        "datatype": snak.get("datatype"),
        "value_type": datavalue.get("type"),
        "value": datavalue.get("value"),
    }


def normalize_reference(reference: dict[str, Any]) -> dict[str, Any]:
    raw_snaks = reference.get("snaks")
    if not isinstance(raw_snaks, dict):
        raw_snaks = {}
    snaks = []
    for property_id in sorted(raw_snaks):
        entries = raw_snaks[property_id]
        if not isinstance(entries, list):
            continue
        for snak in entries:
            if isinstance(snak, dict):
                snaks.append({"property_id": property_id, **value_from_snak(snak)})
    urls = []
    source_items = []
    imported_from_items = []
    for snak in snaks:
        prop = snak["property_id"]
        value = snak.get("value")
        if prop in ("P854", "P4656") and isinstance(value, str):
            urls.append({"property_id": prop, "url": value})
        elif prop == "P248" and isinstance(value, dict) and value.get("id"):
            source_items.append({"property_id": prop, "qid": value["id"]})
        elif prop == "P143" and isinstance(value, dict) and value.get("id"):
            imported_from_items.append({"property_id": prop, "qid": value["id"]})
    return {
        "reference_hash": reference.get("hash"),
        "snaks": snaks,
        "source_urls": urls,
        "source_items": source_items,
        "imported_from_items": imported_from_items,
        "has_external_citation": bool(urls or source_items),
    }


def parse_entities_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities = payload.get("entities")
    parsed: dict[str, dict[str, Any]] = {}
    if isinstance(entities, dict):
        for qid, entity in entities.items():
            if isinstance(entity, dict):
                parsed[str(qid)] = entity
    elif isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict) and isinstance(entity.get("id"), str):
                parsed[entity["id"]] = entity
    else:
        raise ValueError("wbgetentities response has no entities object or list")
    return parsed


def fetch_entities(
    qids: list[str],
    *,
    endpoint: str = API_ENDPOINT,
    timeout: int = 90,
    retries: int = 4,
) -> dict[str, dict[str, Any]]:
    ids = sorted(set(qids))
    if not ids or len(ids) > MAX_IDS_PER_REQUEST:
        raise ValueError("each wbgetentities request must contain 1..50 unique QIDs")
    if any(not QID_RE.fullmatch(qid) for qid in ids):
        raise ValueError("invalid QID in wbgetentities request")
    params = urllib.parse.urlencode({
        "action": "wbgetentities", "ids": "|".join(ids),
        "props": "claims|info", "format": "json",
    })
    request = urllib.request.Request(
        f"{endpoint}?{params}", method="GET",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = json.loads(response.read().decode(charset))
                if not isinstance(payload, dict):
                    raise ValueError("wbgetentities returned non-object JSON")
                return parse_entities_payload(payload)
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429, 500, 502, 503, 504)
            try:
                detail = " ".join(exc.read(1000).decode("utf-8", errors="replace").split())[:400]
            except Exception:
                detail = ""
            if not retryable or attempt == retries - 1:
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"Wikidata Action API HTTP {exc.code}{suffix}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(2, min(60, int(retry_after))) if retry_after else min(60, 5 * (2 ** attempt))
            except ValueError:
                delay = min(60, 5 * (2 ** attempt))
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Wikidata Action API request failed: {type(exc).__name__}") from exc
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError("wbgetentities retries exhausted")


def entity_statement_map(entity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return {}
    statements: dict[str, dict[str, Any]] = {}
    for claims_for_property in claims.values():
        if not isinstance(claims_for_property, list):
            continue
        for statement in claims_for_property:
            if isinstance(statement, dict):
                statement_id = statement.get("id")
                if isinstance(statement_id, str):
                    statements[statement_id] = statement
                guid = statement.get("guid")
                if isinstance(guid, str):
                    statements.setdefault(guid, statement)
    return statements


def statement_guid_suffix(statement_uri: str) -> str:
    tail = statement_uri.rsplit("/", 1)[-1]
    if "$" in tail:
        tail = tail.split("$", 1)[-1]
    elif "-" in tail:
        tail = tail.split("-", 1)[-1]
    return re.sub(r"[^0-9a-f]", "", tail.lower())


def match_statement(entity: dict[str, Any], statement_uri: str) -> tuple[dict[str, Any] | None, str]:
    statements = entity_statement_map(entity)
    requested_id = statement_api_id(statement_uri)
    exact = statements.get(requested_id)
    if exact is not None:
        return exact, "EXACT_ID"
    wanted_suffix = statement_guid_suffix(statement_uri)
    if not wanted_suffix:
        return None, "NO_GUID_SUFFIX"
    candidates = []
    seen_statement_objects = set()
    for candidate_id, statement in statements.items():
        candidate_suffix = statement_guid_suffix(candidate_id)
        object_identity = id(statement)
        if (
            candidate_suffix
            and candidate_suffix == wanted_suffix
            and object_identity not in seen_statement_objects
        ):
            candidates.append((candidate_id, statement))
            seen_statement_objects.add(object_identity)
    if len(candidates) == 1:
        return candidates[0][1], "UNIQUE_GUID_SUFFIX"
    if len(candidates) > 1:
        return None, "AMBIGUOUS_GUID_SUFFIX"
    return None, "NO_MATCH"


def enrich_rows(
    rows: list[dict[str, Any]],
    *,
    endpoint: str = API_ENDPOINT,
    request_interval: float = 1.25,
    fetch=fetch_entities,
    sleep=time.sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request_interval < 1.0:
        raise ValueError("request interval must be at least 1 second")
    if not rows:
        return [], {
            "api_endpoint": endpoint, "entity_count": 0, "request_count": 0,
            "statements_matched": 0, "statements_unmatched": 0,
        }

    qids = sorted({str(row.get("subject_qid", "")) for row in rows})
    if any(not QID_RE.fullmatch(qid) for qid in qids):
        raise ValueError("input contains invalid subject_qid")
    entities: dict[str, dict[str, Any]] = {}
    request_count = 0
    last_request_started = 0.0
    retrieved = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for start in range(0, len(qids), MAX_IDS_PER_REQUEST):
        batch = qids[start:start + MAX_IDS_PER_REQUEST]
        elapsed = time.monotonic() - last_request_started if last_request_started else request_interval
        if elapsed < request_interval:
            sleep(request_interval - elapsed)
        last_request_started = time.monotonic()
        entities.update(fetch(batch, endpoint=endpoint))
        request_count += 1

    enriched = []
    matched = 0
    unmatched = 0
    entity_revisions = {}
    for original in rows:
        row = dict(original)
        qid = row["subject_qid"]
        entity = entities.get(qid)
        if entity is None:
            row["reference_enrichment"] = {
                "status": "ENTITY_NOT_RETURNED", "api_endpoint": endpoint,
                "retrieved_at_utc": retrieved,
                "match_method": match_method,
            }
            row["wikidata_reference_records"] = []
            row["has_external_citation"] = False
            unmatched += 1
            enriched.append(row)
            continue
        entity_revisions[qid] = entity.get("lastrevid")
        statement_uri = str(row.get("statement_uri", ""))
        statement_id = statement_api_id(statement_uri)
        statement, match_method = match_statement(entity, statement_uri)
        if statement is None:
            row["reference_enrichment"] = {
                "status": "STATEMENT_NOT_MATCHED", "api_endpoint": endpoint,
                "entity_revision_id": entity.get("lastrevid"),
                "retrieved_at_utc": retrieved,
            }
            row["wikidata_reference_records"] = []
            row["has_external_citation"] = False
            unmatched += 1
            enriched.append(row)
            continue
        raw_references = statement.get("references")
        if not isinstance(raw_references, list):
            raw_references = []
        normalized = [
            normalize_reference(reference)
            for reference in raw_references if isinstance(reference, dict)
        ]
        row["wikidata_reference_records"] = normalized
        row["has_external_citation"] = any(
            reference["has_external_citation"] for reference in normalized
        )
        source_references = []
        for reference in normalized:
            for source_url in reference["source_urls"]:
                source_references.append({
                    "kind": "url", **source_url,
                    "reference_hash": reference["reference_hash"],
                })
            for source_item in reference["source_items"]:
                source_references.append({
                    "kind": "wikidata_source_item", **source_item,
                    "url": f"https://www.wikidata.org/wiki/{source_item['qid']}",
                    "reference_hash": reference["reference_hash"],
                })
        row["source_references"] = source_references
        row["has_statement_reference"] = bool(normalized)
        row["reference_enrichment"] = {
            "status": "MATCHED", "api_endpoint": endpoint,
            "entity_revision_id": entity.get("lastrevid"),
            "retrieved_at_utc": retrieved, "statement_id": statement_id,
            "match_method": match_method,
        }
        matched += 1
        enriched.append(row)

    summary = {
        "api_endpoint": endpoint, "retrieved_at_utc": retrieved,
        "entity_count": len(entities), "request_count": request_count,
        "entity_revisions": entity_revisions,
        "statements_matched": matched, "statements_unmatched": unmatched,
    }
    return enriched, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request-interval", type=float, default=1.25)
    parser.add_argument("--endpoint", default=API_ENDPOINT)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every JSONL row must be an object")
    enriched, summary = enrich_rows(
        rows, endpoint=args.endpoint, request_interval=args.request_interval
    )
    output = "".join(canonical_json(row) + "\n" for row in enriched).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "REFERENCE_ENRICHED_REQUIRES_HUMAN_REVIEW",
        "input_sha256": sha256_bytes(raw), "output_sha256": sha256_bytes(output),
        "input_record_count": len(rows), "output_record_count": len(enriched),
        "records_with_any_reference_records": sum(bool(row.get("wikidata_reference_records")) for row in enriched),
        "records_with_external_citations": sum(bool(row.get("has_external_citation")) for row in enriched),
        "records_without_reference_records": sum(not row.get("wikidata_reference_records") for row in enriched),
        "summary": summary, "training_approved_record_count": 0,
        "training_approval": "NOT_GRANTED_BY_REFERENCE_ENRICHMENT",
        "notice": "Import-source markers (P143) are retained separately and are not counted as external citations. Factual accuracy and privacy clearance are not assessed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "records": len(enriched),
        "with_any_reference_records": manifest["records_with_any_reference_records"],
        "with_external_citations": manifest["records_with_external_citations"],
        "without_reference_records": manifest["records_without_reference_records"],
        "entity_batches": summary["request_count"],
        "statements_matched": summary["statements_matched"],
        "statements_unmatched": summary["statements_unmatched"],
        "training_approved_records": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"reference enrichment failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
