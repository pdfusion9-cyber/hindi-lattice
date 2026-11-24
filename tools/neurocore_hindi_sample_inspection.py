#!/usr/bin/env python3
"""Bounded, metadata-first Hindi corpus sample inspection.

Never downloads a full dataset shard and never writes or prints sample text.
Only aggregate text-quality/privacy heuristics and one-way hashes are persisted.
This script is not a legal review and never changes dataset approvals.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "hindi-corpus-sample-inspection.json"
INDICCORP_REVISION = "2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f"
SANGRAHA_REVISION = "0c9510b4d6d908d32b8d76da877e64fd6c7a6086"
MAX_TEXT_BYTES = 256 * 1024
MAX_API_BYTES = 5 * 1024 * 1024
SAMPLE_ROWS = 5

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR_LIKE_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")


def get_json(url: str, timeout: int = 35) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "NeuroCore-Hindi-Sample-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(MAX_API_BYTES + 1)
    if len(raw) > MAX_API_BYTES:
        raise RuntimeError(f"API response exceeded {MAX_API_BYTES} bytes; response discarded")
    return json.loads(raw.decode("utf-8"))


def inspect_text(text: str, source_row: str) -> dict:
    normalized = unicodedata.normalize("NFC", text)
    alpha = devanagari = latin = digits = 0
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("L"):
            alpha += 1
            name = unicodedata.name(char, "")
            if "DEVANAGARI" in name:
                devanagari += 1
            elif "LATIN" in name:
                latin += 1
        elif category == "Nd":
            digits += 1
    safe_alpha = max(alpha, 1)
    return {
        "source_row": source_row,
        "text_chars": len(normalized),
        "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "devanagari_letter_fraction": round(devanagari / safe_alpha, 4),
        "latin_letter_fraction": round(latin / safe_alpha, 4),
        "digit_fraction_of_all_chars": round(digits / max(len(normalized), 1), 4),
        "email_pattern_count": len(EMAIL_RE.findall(normalized)),
        "url_pattern_count": len(URL_RE.findall(normalized)),
        "phone_pattern_count": len(PHONE_RE.findall(normalized)),
        "aadhaar_like_pattern_count": len(AADHAAR_LIKE_RE.findall(normalized)),
        "empty_or_whitespace": not bool(normalized.strip()),
    }


def aggregate(rows: list[tuple[str, str]]) -> dict:
    items = [inspect_text(text, row_id) for row_id, text in rows]
    hashes = [item["text_sha256"] for item in items if item["text_chars"]]
    unique_hashes = len(set(hashes))
    return {
        "sample_count": len(items),
        "unique_nonempty_texts": unique_hashes,
        "duplicate_exact_text_count": len(hashes) - unique_hashes,
        "mean_text_chars": round(sum(x["text_chars"] for x in items) / max(len(items), 1), 2),
        "min_text_chars": min((x["text_chars"] for x in items), default=0),
        "max_text_chars": max((x["text_chars"] for x in items), default=0),
        "mean_devanagari_letter_fraction": round(
            sum(x["devanagari_letter_fraction"] for x in items) / max(len(items), 1), 4
        ),
        "mean_latin_letter_fraction": round(
            sum(x["latin_letter_fraction"] for x in items) / max(len(items), 1), 4
        ),
        "rows_with_email_patterns": sum(x["email_pattern_count"] > 0 for x in items),
        "rows_with_url_patterns": sum(x["url_pattern_count"] > 0 for x in items),
        "rows_with_phone_patterns": sum(x["phone_pattern_count"] > 0 for x in items),
        "rows_with_aadhaar_like_patterns": sum(x["aadhaar_like_pattern_count"] > 0 for x in items),
        "heuristic_limit": (
            "Pattern counts are triage indicators only. They do not identify all personal data, "
            "do not establish that a match is real PII, and are not a privacy clearance."
        ),
        "row_metrics": items,
        "raw_text_persisted": False,
    }


def fetch_sangraha_rows() -> dict:
    """Use HF dataset viewer rows endpoint; only Parquet exports are supported."""
    dataset = "ai4bharat/sangraha"
    splits_url = "https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": dataset})
    splits = get_json(splits_url)
    exact = [
        item for item in splits.get("splits", [])
        if item.get("config") == "verified" and item.get("split") == "hin"
    ]
    if not exact:
        return {
            "status": "VIEWER_SPLIT_UNAVAILABLE",
            "requested_config": "verified",
            "requested_split": "hin",
            "viewer_splits_found": [
                {"config": x.get("config"), "split": x.get("split")}
                for x in splits.get("splits", [])
                if x.get("config") == "verified"
            ],
        }
    rows_url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({
        "dataset": dataset, "config": "verified", "split": "hin",
        "offset": 0, "length": SAMPLE_ROWS,
    })
    payload = get_json(rows_url)
    features = payload.get("features", [])
    names = [x.get("name") for x in features]
    if not {"doc_id", "type", "text"}.issubset(set(names)):
        raise ValueError(f"Unexpected Sangraha verified schema: {names}")
    samples = []
    for item in payload.get("rows", [])[:SAMPLE_ROWS]:
        row = item.get("row", {})
        value = row.get("text")
        if isinstance(value, str):
            samples.append((str(item.get("row_idx")), value))
    if not samples:
        raise ValueError("Sangraha viewer returned zero non-empty text samples")
    return {
        "status": "SAMPLE_INSPECTED",
        "dataset": dataset,
        "requested_revision": SANGRAHA_REVISION,
        "sampling_note": (
            "The dataset-viewer rows API does not expose a revision selector. "
            "This sample is from the viewer's current indexed version, not proven to match "
            "the pinned 0c9510b4 revision. Pin verification is therefore pending."
        ),
        "config": "verified",
        "split": "hin",
        "features": features,
        "viewer_rows_returned": len(payload.get("rows", [])),
        "sample_metrics": aggregate(samples),
        "partial": payload.get("partial"),
    }


def fetch_range(url: str, start: int, length: int = MAX_TEXT_BYTES) -> tuple[bytes, dict]:
    end = start + length - 1
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NeuroCore-Hindi-Sample-Audit/1.0",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        status = getattr(response, "status", response.getcode())
        data = response.read(length + 1)
        headers = dict(response.headers.items())
    if status != 206:
        raise RuntimeError(
            f"Refusing text sample because server did not honor Range (HTTP {status}); "
            "read was capped and response closed"
        )
    if len(data) > length:
        data = data[:length]
    content_range = headers.get("Content-Range") or headers.get("content-range")
    if not content_range:
        raise RuntimeError("Range response has no Content-Range header")
    match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
    if not match:
        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
    return data, {
        "http_status": status,
        "content_range": content_range,
        "total_bytes": None if match.group(3) == "*" else int(match.group(3)),
        "bytes_read": len(data),
    }


def fetch_indiccorp_range_samples() -> dict:
    dataset = "ai4bharat/IndicCorpV2"
    api_url = f"https://huggingface.co/api/datasets/{dataset}"
    meta = get_json(api_url)
    current_revision = meta.get("sha")
    if current_revision != INDICCORP_REVISION:
        raise RuntimeError(
            f"IndicCorpV2 current Hub revision {current_revision} differs from pinned "
            f"candidate {INDICCORP_REVISION}; sample request stopped."
        )
    file_path = "data/hi-1.txt"
    url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/"
        f"{INDICCORP_REVISION}/{file_path}"
    )
    first, first_meta = fetch_range(url, 0)
    total = first_meta["total_bytes"]
    windows = [(0, first, first_meta)]
    # Three deterministic byte windows: beginning, middle and tail of the same pinned shard.
    if total and total > 3 * MAX_TEXT_BYTES:
        for offset in (total // 2, max(total - MAX_TEXT_BYTES, 0)):
            chunk, meta_range = fetch_range(url, offset)
            windows.append((offset, chunk, meta_range))
    samples: list[tuple[str, str]] = []
    for offset, chunk, _ in windows:
        decoded = chunk.decode("utf-8", errors="replace")
        lines = decoded.splitlines()
        if offset > 0 and lines:
            lines = lines[1:]  # first line may start mid-record
        if len(lines) > 1:
            lines = lines[:-1]  # last line may end mid-record
        for local_index, line in enumerate(lines):
            value = line.strip()
            if len(value) >= 40:
                samples.append((f"byte_{offset}_line_{local_index}", value))
            if len(samples) >= SAMPLE_ROWS * 3:
                break
        if len(samples) >= SAMPLE_ROWS * 3:
            break
    if not samples:
        raise ValueError("IndicCorpV2 range sample returned no usable text lines")
    return {
        "status": "SAMPLE_INSPECTED",
        "dataset": dataset,
        "revision": INDICCORP_REVISION,
        "config": "indiccorp_v2",
        "split": "hin_Deva",
        "file_path": file_path,
        "sampling_method": "HTTP Range requests capped at 256 KiB per window",
        "range_windows": [meta for _, _, meta in windows],
        "sample_metrics": aggregate(samples[:SAMPLE_ROWS * 3]),
        "full_file_downloaded": False,
    }


def main() -> None:
    report = {
        "report_version": "1.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "metadata and bounded text-only sample inspection",
        "raw_sample_text_stored": False,
        "canonical_approval_registry_changed": False,
        "production_approval_granted": False,
        "sources": {},
        "overall_status": "PARTIAL",
        "limitations": [
            "No full corpus shards were downloaded.",
            "Text-pattern checks are heuristic and cannot certify privacy or safety.",
            "Sangraha dataset-viewer API does not offer a revision parameter; its sample is not treated as pinned evidence.",
            "Neither source is approved for production training by this report.",
        ],
    }
    for name, function in (
        ("indiccorp_v2", fetch_indiccorp_range_samples),
        ("sangraha_verified_hin", fetch_sangraha_rows),
    ):
        try:
            report["sources"][name] = function()
        except Exception as exc:
            report["sources"][name] = {
                "status": "INSPECTION_BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
    statuses = [source.get("status") for source in report["sources"].values()]
    if statuses and all(status == "SAMPLE_INSPECTED" for status in statuses):
        report["overall_status"] = "BOTH_SAMPLES_INSPECTED"
    elif any(status == "SAMPLE_INSPECTED" for status in statuses):
        report["overall_status"] = "ONE_SAMPLE_INSPECTED"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall_status": report["overall_status"],
        "source_statuses": {key: value.get("status") for key, value in report["sources"].items()},
        "raw_sample_text_stored": False,
        "canonical_approval_registry_changed": False,
        "report_path": str(OUT.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
