#!/usr/bin/env python3
"""Bounded, review-only Sangraha Hindi sample audit. Never authorizes training.

Only aggregate statistics are emitted; sample text is never printed or saved.
"""
from __future__ import annotations
import argparse, collections, json, re
from typing import Any

PHONE_RE=re.compile(r"(?<!\w)(?:\+?91[- .]?)?[6-9]\d{9}(?!\w)")
EMAIL_RE=re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_RE=re.compile(r"(?i)\bhttps?://")
AADHAAR_RE=re.compile(r"(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)")
DEVANAGARI_RE=re.compile(r"[\u0900-\u097F]")
LATIN_RE=re.compile(r"[A-Za-z]")

def audit_rows(rows, max_rows: int) -> dict[str, Any]:
    counts=collections.Counter()
    doc_ids=set()
    duplicate_ids=0
    for row in rows:
        if counts["rows_seen"] >= max_rows:
            break
        counts["rows_seen"] += 1
        if not isinstance(row,dict):
            counts["non_object_rows"] += 1
            continue
        text=row.get("text")
        if not isinstance(text,str):
            counts["missing_or_non_string_text"] += 1
            continue
        if not text.strip():
            counts["empty_text"] += 1
            continue
        counts["nonempty_text"] += 1
        counts["chars_total"] += len(text)
        if DEVANAGARI_RE.search(text):
            counts["contains_devanagari"] += 1
        if LATIN_RE.search(text):
            counts["contains_latin"] += 1
        doc_id=row.get("doc_id")
        if doc_id is not None:
            key=str(doc_id)
            if key in doc_ids:
                duplicate_ids += 1
            else:
                doc_ids.add(key)
        if PHONE_RE.search(text):
            counts["phone_like_rows"] += 1
        if EMAIL_RE.search(text):
            counts["email_like_rows"] += 1
        if URL_RE.search(text):
            counts["url_like_rows"] += 1
        if AADHAAR_RE.search(text):
            counts["aadhaar_like_rows"] += 1
    n=counts["nonempty_text"]
    return {
        "audit_type":"BOUNDED_REVIEW_ONLY_NO_TRAINING",
        "rows_seen":counts["rows_seen"],
        "nonempty_text":n,
        "empty_text":counts["empty_text"],
        "missing_or_non_string_text":counts["missing_or_non_string_text"],
        "non_object_rows":counts["non_object_rows"],
        "duplicate_doc_id_rows":duplicate_ids,
        "contains_devanagari":counts["contains_devanagari"],
        "contains_latin":counts["contains_latin"],
        "phone_like_rows":counts["phone_like_rows"],
        "email_like_rows":counts["email_like_rows"],
        "url_like_rows":counts["url_like_rows"],
        "aadhaar_like_rows":counts["aadhaar_like_rows"],
        "average_chars_nonempty":round(counts["chars_total"]/n,2) if n else 0,
        "sample_text_persisted":False,
        "approval_status":"PENDING_REVIEW",
        "warning":"Heuristic PII counts are not a privacy clearance or legal review."
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--revision",required=True)
    parser.add_argument("--config",default="verified")
    parser.add_argument("--split",default="hin")
    parser.add_argument("--max-rows",type=int,default=500)
    parser.add_argument("--confirm-review-only",action="store_true")
    args=parser.parse_args()
    if not args.confirm_review_only:
        raise SystemExit("Refusing to access dataset: pass --confirm-review-only; this script is not a training path")
    if len(args.revision)!=40 or any(ch not in "0123456789abcdefABCDEF" for ch in args.revision):
        raise SystemExit("--revision must be an immutable 40-character commit SHA")
    if not 1 <= args.max_rows <= 1000:
        raise SystemExit("--max-rows must be between 1 and 1000")
    from datasets import load_dataset
    ds=load_dataset("ai4bharat/sangraha",name=args.config,split=args.split,revision=args.revision,streaming=True)
    print(json.dumps(audit_rows(ds,args.max_rows),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
