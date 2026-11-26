#!/usr/bin/env python3
"""Bounded, aggregate-only IndicCorpV2 Hindi sample audit; never authorizes training."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from typing import Any

DEVANAGARI_RE=re.compile(r"[\u0900-\u097F]")
LATIN_RE=re.compile(r"[A-Za-z]")
PHONE_RE=re.compile(r"(?<!\w)(?:\+?91[- .]?)?[6-9]\d{9}(?!\w)")
EMAIL_RE=re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
AADHAAR_RE=re.compile(r"(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)")
PAN_RE=re.compile(r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b")
URL_RE=re.compile(r"(?i)\bhttps?://")

def audit_rows(rows, max_rows: int) -> dict[str, Any]:
    counts=collections.Counter()
    seen_hashes=set()
    duplicate_rows=0
    for row in rows:
        if counts["rows_seen"] >= max_rows:
            break
        counts["rows_seen"] += 1
        if isinstance(row, str):
            text=row
        elif isinstance(row, dict):
            text=row.get("text")
        else:
            counts["non_text_rows"] += 1
            continue
        if not isinstance(text,str):
            counts["missing_or_non_string_text"] += 1
            continue
        text=text.strip()
        if not text:
            counts["empty_text"] += 1
            continue
        counts["nonempty_text"] += 1
        counts["chars_total"] += len(text)
        if DEVANAGARI_RE.search(text):
            counts["contains_devanagari"] += 1
        if LATIN_RE.search(text):
            counts["contains_latin"] += 1
        if PHONE_RE.search(text):
            counts["phone_like_rows"] += 1
        if EMAIL_RE.search(text):
            counts["email_like_rows"] += 1
        if AADHAAR_RE.search(text):
            counts["aadhaar_like_rows"] += 1
        if PAN_RE.search(text):
            counts["pan_like_rows"] += 1
        if URL_RE.search(text):
            counts["url_like_rows"] += 1
        digest=hashlib.sha256(text.encode("utf-8")).digest()
        if digest in seen_hashes:
            duplicate_rows += 1
        else:
            seen_hashes.add(digest)
        if len(text) <= 40:
            counts["short_rows_le_40_chars"] += 1
        if len(text) >= 5000:
            counts["long_rows_ge_5000_chars"] += 1
    n=counts["nonempty_text"]
    return {
        "audit_type":"BOUNDED_REVIEW_ONLY_NO_TRAINING",
        "dataset_id":"ai4bharat/IndicCorpV2",
        "config_name":"indiccorp_v2",
        "source_file":"data/hi-1.txt",
        "rows_seen":counts["rows_seen"],
        "nonempty_text":n,
        "empty_text":counts["empty_text"],
        "missing_or_non_string_text":counts["missing_or_non_string_text"],
        "non_text_rows":counts["non_text_rows"],
        "duplicate_text_rows":duplicate_rows,
        "contains_devanagari":counts["contains_devanagari"],
        "contains_latin":counts["contains_latin"],
        "short_rows_le_40_chars":counts["short_rows_le_40_chars"],
        "long_rows_ge_5000_chars":counts["long_rows_ge_5000_chars"],
        "phone_like_rows":counts["phone_like_rows"],
        "email_like_rows":counts["email_like_rows"],
        "aadhaar_like_rows":counts["aadhaar_like_rows"],
        "pan_like_rows":counts["pan_like_rows"],
        "url_like_rows":counts["url_like_rows"],
        "average_chars_nonempty":round(counts["chars_total"]/n,2) if n else 0,
        "sample_text_persisted":False,
        "approval_status":"PENDING_REVIEW",
        "warning":"Heuristic counts do not establish source-level copyright, consent, or privacy clearance."
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--revision",required=True)
    parser.add_argument("--max-rows",type=int,default=500)
    parser.add_argument("--confirm-review-only",action="store_true")
    args=parser.parse_args()
    if not args.confirm_review_only:
        raise SystemExit("Refusing to access corpus: pass --confirm-review-only; this is not a training path")
    if len(args.revision)!=40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise SystemExit("--revision must be an immutable 40-character SHA")
    if not 1<=args.max_rows<=1000:
        raise SystemExit("--max-rows must be between 1 and 1000")
    from datasets import load_dataset
    url=f"https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/{args.revision}/data/hi-1.txt"
    ds=load_dataset("text",data_files={"train":url},split="train",streaming=True)
    report=audit_rows(ds,args.max_rows)
    report["revision"]=args.revision
    report["training_performed"]=False
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
