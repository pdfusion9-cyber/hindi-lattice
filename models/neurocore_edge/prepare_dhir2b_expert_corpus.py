"""Prepare a private, PII-filtered token-ID corpus for fixed Dhīr Expert training.

This utility is deliberately fail-closed:
- requires explicit human dataset and exact tokenizer SHA-256 approvals;
- uses one immutable IndicCorpV2 revision and a bounded streaming read;
- drops common PII-pattern matches and duplicate text;
- stores only token IDs in a local JSONL file;
- refuses output under the Git checkout and never uploads corpus contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "ai4bharat/IndicCorpV2"
DATASET_REVISION = "2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f"
DATASET_SHARD = "data/hi-1.txt"
BASE_MODEL_ID = "zenteiq-ai/Dhir-2B"
BASE_REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"
VOCAB_SIZE = 131072

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
PAN_RE = re.compile(r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("tokenizer directory contains no files")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def find_git_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def require_private_output(path: str | Path, *, repo_root: Path | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    root = repo_root or find_git_root()
    if root is not None:
        root = root.resolve()
        if target == root or root in target.parents:
            raise ValueError("refusing to write corpus inside Git checkout/public repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def require_human_approvals(
    approval_path: str | Path,
    *,
    expert_id: int,
    dataset_id: str,
    dataset_revision: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    doc = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    if doc.get("status") != "ALL_DATASETS_REVIEWED":
        raise PermissionError("training corpus preparation blocked: global dataset review is not complete")

    dataset = next((
        row for row in doc.get("datasets", [])
        if row.get("expert_id") == expert_id
        and row.get("dataset_id") == dataset_id
        and row.get("revision") == dataset_revision
    ), None)
    if dataset is None:
        raise PermissionError("exact dataset and immutable revision are not registered")

    required_fields = (
        "approved",
        "commercial_training_allowed",
        "pii_reviewed",
        "consent_reviewed",
        "schema_reviewed",
        "split_reviewed",
    )
    missing = [field for field in required_fields if dataset.get(field) is not True]
    if missing:
        raise PermissionError("dataset review incomplete: " + ", ".join(missing))
    if not dataset.get("license_id") or not dataset.get("license_terms_url"):
        raise PermissionError("dataset license evidence missing")
    if not all(dataset.get(field) for field in ("approval_id", "reviewer", "reviewed_at")):
        raise PermissionError("dataset approval requires human approval_id, reviewer and reviewed_at")

    tokenizer_ok = any(
        row.get("sha256") == tokenizer_sha256
        and row.get("approved") is True
        and all(row.get(field) for field in ("approval_id", "reviewer", "reviewed_at"))
        for row in doc.get("approved_tokenizers", [])
    )
    if not tokenizer_ok:
        raise PermissionError("exact tokenizer SHA-256 human approval missing")
    return doc


def classify_pii(text: str) -> set[str]:
    flags = set()
    if EMAIL_RE.search(text):
        flags.add("email")
    if PHONE_RE.search(text):
        flags.add("phone")
    if AADHAAR_RE.search(text):
        flags.add("aadhaar_like")
    if PAN_RE.search(text):
        flags.add("pan_like")
    return flags


def _encode(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = result["input_ids"] if isinstance(result, dict) else result.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned more than one sequence")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def prepare_records(
    records: Iterable[Any],
    tokenizer: Any,
    output_path: str | Path,
    *,
    max_records: int = 5000,
    max_scan_rows: int = 100000,
    max_sequence_length: int = 256,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create token-ID-only JSONL in private storage; return aggregate metrics."""
    if not 1 <= max_records <= 50000:
        raise ValueError("max_records must be between 1 and 50000")
    if not max_records <= max_scan_rows <= 1000000:
        raise ValueError("max_scan_rows must be >= max_records and <= 1000000")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")

    target = require_private_output(output_path, repo_root=repo_root)
    counts = Counter()
    seen_text_hashes: set[bytes] = set()
    pii_union = 0
    sequences_written = 0
    token_count = 0
    total_chars = 0
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            for record in records:
                if counts["raw_rows_seen"] >= max_scan_rows:
                    break
                if counts["eligible_records"] >= max_records:
                    break
                counts["raw_rows_seen"] += 1

                if isinstance(record, str):
                    raw_text = record
                elif isinstance(record, dict):
                    raw_text = record.get("text")
                else:
                    counts["non_text_rows"] += 1
                    continue
                if not isinstance(raw_text, str):
                    counts["invalid_text_rows"] += 1
                    continue
                text = unicodedata.normalize("NFC", raw_text.strip())
                if not text:
                    counts["empty_rows"] += 1
                    continue
                flags = classify_pii(text)
                if flags:
                    counts["pii_rows_excluded"] += 1
                    for flag in flags:
                        counts[f"pii_{flag}_rows"] += 1
                    pii_union += 1
                    continue
                if not DEVANAGARI_RE.search(text):
                    counts["non_devanagari_rows_excluded"] += 1
                    continue
                fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
                if fingerprint in seen_text_hashes:
                    counts["duplicate_rows_excluded"] += 1
                    continue
                seen_text_hashes.add(fingerprint)

                ids = _encode(tokenizer, text)
                if not ids:
                    counts["empty_token_sequences"] += 1
                    continue
                if min(ids) < 0 or max(ids) >= VOCAB_SIZE:
                    raise ValueError("tokenizer returned IDs outside pinned Dhīr vocabulary")

                # Split long source records into bounded, non-overlapping sequences.
                chunks = [
                    ids[start:start + max_sequence_length]
                    for start in range(0, len(ids), max_sequence_length)
                ]
                chunks = [chunk for chunk in chunks if len(chunk) >= 2]
                if not chunks:
                    counts["too_short_after_tokenization"] += 1
                    continue
                for chunk in chunks:
                    out.write(json.dumps({"input_ids": chunk}, separators=(",", ":")) + "\n")
                    sequences_written += 1
                    token_count += len(chunk)
                counts["eligible_records"] += 1
                total_chars += len(text)
                counts["source_records_tokenized"] += 1
            out.flush()
            os.fsync(out.fileno())
        if counts["eligible_records"] < max_records:
            raise ValueError(
                f"only {counts['eligible_records']} eligible Hindi records found; "
                f"required {max_records}; output was not committed"
            )
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    return {
        "status": "PRIVATE_TOKEN_ID_CORPUS_PREPARED_PENDING_HASH_APPROVAL",
        "training_performed": False,
        "raw_text_persisted": False,
        "token_ids_persisted": True,
        "output_path": str(target),
        "corpus_sha256": sha256_file(target),
        "max_records": max_records,
        "max_scan_rows": max_scan_rows,
        "max_sequence_length": max_sequence_length,
        "raw_rows_seen": counts["raw_rows_seen"],
        "eligible_records": counts["eligible_records"],
        "sequences_written": sequences_written,
        "tokens_written": token_count,
        "average_chars_per_eligible_record": (
            round(total_chars / counts["eligible_records"], 2)
            if counts["eligible_records"] else 0
        ),
        "pii_rows_excluded": pii_union,
        "email_rows_excluded": counts["pii_email"],
        "phone_rows_excluded": counts["pii_phone"],
        "aadhaar_like_rows_excluded": counts["pii_aadhaar_like"],
        "pan_like_rows_excluded": counts["pii_pan_like"],
        "duplicates_excluded": counts["duplicate_rows_excluded"],
        "non_devanagari_rows_excluded": counts["non_devanagari_rows_excluded"],
        "approval_status": "PENDING_EXACT_CORPUS_AND_TOKENIZER_HASH_APPROVAL",
        "public_upload": False,
    }


def load_pinned_records():
    from datasets import load_dataset
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{DATASET_SHARD}"
    )
    dataset = load_dataset(
        "text",
        data_files={"train": url},
        split="train",
        streaming=True,
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--approval", type=Path, default=Path("dataset_approval.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-id", type=int, default=11)
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--max-scan-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--confirm-private-output", action="store_true")
    parser.add_argument("--confirm-reviewed-source", action="store_true")
    args = parser.parse_args()

    if args.expert_id != 11:
        parser.error("this first-stage corpus builder is restricted to Expert 11")
    if not args.confirm_private_output or not args.confirm_reviewed_source:
        parser.error(
            "explicit private-output and human-reviewed-source confirmations are required"
        )

    from neurocore_tokenizer import NeuroCoreTokenizer

    tokenizer = NeuroCoreTokenizer.from_pretrained(
        args.tokenizer_dir,
        base_model_id=BASE_MODEL_ID,
        base_revision=BASE_REVISION,
        expected_vocab_size=VOCAB_SIZE,
    )
    tokenizer_sha = sha256_directory(args.tokenizer_dir)
    require_human_approvals(
        args.approval,
        expert_id=args.expert_id,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        tokenizer_sha256=tokenizer_sha,
    )
    records = load_pinned_records()
    report = prepare_records(
        records,
        tokenizer,
        args.output,
        max_records=args.max_records,
        max_scan_rows=args.max_scan_rows,
        max_sequence_length=args.max_sequence_length,
    )
    report.update({
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_shard": DATASET_SHARD,
        "tokenizer_sha256": tokenizer_sha,
        "expert_id": args.expert_id,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
