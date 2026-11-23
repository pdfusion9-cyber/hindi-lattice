#!/usr/bin/env python3
"""Approval-gated streaming corpus preparation for one NeuroCore expert.

No source is downloaded until dataset rights, immutable revision, split and
schema approvals pass. This program prepares local corpora only; it never
trains, writes checkpoints, or publishes raw source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dataset_schema_adapters import (
    ADAPTER_VERSION,
    SchemaAdapterError,
    adapt_row,
    adapter_family,
)

REQUIRED_TEXT_FIELDS = (
    "license_id", "license_terms_url", "reviewer", "review_date",
    "revision", "train_split", "schema_fingerprint",
)
REQUIRED_TRUE_FLAGS = (
    "approved", "commercial_training_allowed", "public_redistribution_allowed",
    "pii_reviewed", "consent_reviewed", "schema_reviewed", "split_reviewed",
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_experts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("experts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise SystemExit("REFUSED: expert manifest must be a list or contain an experts list")
    ids = [row.get("expert_id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise SystemExit("REFUSED: expert manifest has invalid or duplicate expert_id")
    if set(ids) != set(range(20)):
        raise SystemExit("REFUSED: manifest must define exactly expert_id 0..19")
    return rows

def load_approved_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "APPROVED_FOR_COMMERCIAL_TRAINING":
        raise SystemExit("REFUSED: approval status is not APPROVED_FOR_COMMERCIAL_TRAINING")
    rows = payload.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("REFUSED: approval file has no dataset rows")
    approved: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("REFUSED: approval row is not an object")
        dataset_id = str(row.get("dataset_id") or "").strip()
        expert_id = row.get("expert_id")
        if not dataset_id or not isinstance(expert_id, int):
            raise SystemExit("REFUSED: approval row needs integer expert_id and dataset_id")
        key = (expert_id, dataset_id)
        if key in approved:
            raise SystemExit(f"REFUSED: duplicate approval entry for expert={expert_id} dataset={dataset_id}")
        missing = [name for name in REQUIRED_TEXT_FIELDS if not str(row.get(name) or "").strip()]
        missing.extend(name for name in REQUIRED_TRUE_FLAGS if row.get(name) is not True)
        if row.get("text_model_compatible") is not True:
            missing.append("text_model_compatible")
        if row.get("schema_adapter_version") != ADAPTER_VERSION:
            missing.append("schema_adapter_version")
        if not str(row.get("license_terms_url") or "").startswith("https://"):
            missing.append("https_license_terms_url")
        if missing:
            continue
        if adapter_family(dataset_id) in {"sidecar_required", "unregistered"}:
            continue
        approved[key] = row
    return approved

def datasets_for_expert(experts: list[dict[str, Any]], expert_id: int) -> list[str]:
    expert_rows = [row for row in experts if row.get("expert_id") == expert_id]
    if len(expert_rows) != 1:
        raise SystemExit(f"REFUSED: expert_id {expert_id} missing or duplicated")
    requested = list(expert_rows[0].get("datasets") or [])
    if expert_id == 18:
        requested = sorted({
            dataset_id
            for row in experts
            for dataset_id in (row.get("datasets") or [])
            if dataset_id != "curated_mixture_from_all_domains"
            and adapter_family(dataset_id) not in {"sidecar_required", "unregistered"}
        })
    if not requested:
        raise SystemExit(f"REFUSED: expert_id {expert_id} has no datasets")
    if len(set(requested)) != len(requested):
        raise SystemExit(f"REFUSED: expert_id {expert_id} lists duplicate datasets")
    return requested

def deterministic_holdout(dataset_id: str, index: int, examples: list[str]) -> bool:
    key = (dataset_id + "\0" + str(index) + "\0" + "\0".join(examples)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 100 < 2

def open_stream(dataset_id: str, approval: dict[str, Any], split: str):
    from datasets import load_dataset
    return load_dataset(
        dataset_id,
        name=str(approval.get("config_name") or "").strip() or None,
        split=split,
        revision=str(approval["revision"]),
        streaming=True,
    )

def write_split(dataset_id: str, expert_id: int, stream, train, valid,
                *, route_hash_to_validation: bool, force_validation: bool = False,
                max_examples: int) -> tuple[int, int, int]:
    rows_seen = train_count = valid_count = 0
    for index, item in enumerate(stream):
        try:
            examples = [
                value.strip().replace("\x00", " ")
                for value in adapt_row(dataset_id, item, expert_id)
                if isinstance(value, str) and value.strip()
            ]
        except SchemaAdapterError as exc:
            raise SystemExit(f"SCHEMA_REFUSED {dataset_id} row {index}: {exc}") from exc
        if not examples:
            raise SystemExit(f"SCHEMA_REFUSED {dataset_id} row {index}: empty adapter output")
        to_valid = force_validation or (route_hash_to_validation and deterministic_holdout(dataset_id, index, examples))
        target = valid if to_valid else train
        target.write("\n\n".join(examples) + "\n\n")
        if to_valid:
            valid_count += len(examples)
        else:
            train_count += len(examples)
        rows_seen += 1
        if rows_seen >= max_examples:
            break
    return rows_seen, train_count, valid_count

def prepare_corpus(manifest_path: Path, approval_path: Path, expert_id: int,
                   out_dir: Path, max_examples: int = 50_000) -> dict[str, Any]:
    if max_examples < 50:
        raise ValueError("max_examples must be at least 50 to retain a useful holdout")
    experts = load_experts(manifest_path)
    all_approvals = load_approved_rows(approval_path)
    dataset_ids = datasets_for_expert(experts, expert_id)
    missing = [
        dataset_id for dataset_id in dataset_ids
        if (expert_id, dataset_id) not in all_approvals
    ]
    if missing:
        raise SystemExit(
            f"REFUSED: expert {expert_id} has unapproved, incomplete, or sidecar-only datasets: "
            + ", ".join(missing)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fd_train, name_train = tempfile.mkstemp(prefix=".train.", dir=out_dir)
    fd_valid, name_valid = tempfile.mkstemp(prefix=".valid.", dir=out_dir)
    train_tmp, valid_tmp = Path(name_train), Path(name_valid)
    total_train = total_valid = 0
    details = []
    try:
        with os.fdopen(fd_train, "w", encoding="utf-8") as train, \
             os.fdopen(fd_valid, "w", encoding="utf-8") as valid:
            for dataset_id in dataset_ids:
                approval = all_approvals[(expert_id, dataset_id)]
                train_split = str(approval["train_split"])
                validation_split = str(approval.get("validation_split") or "").strip()
                train_stream = open_stream(dataset_id, approval, train_split)
                if validation_split:
                    rows, train_count, _ = write_split(
                        dataset_id, expert_id, train_stream, train, valid,
                        route_hash_to_validation=False, max_examples=max_examples,
                    )
                    valid_stream = open_stream(dataset_id, approval, validation_split)
                    valid_rows, _, valid_count = write_split(
                        dataset_id, expert_id, valid_stream, train, valid,
                        route_hash_to_validation=False, force_validation=True,
                        max_examples=max_examples,
                    )
                    if rows < 1 or valid_rows < 1:
                        raise SystemExit(
                            f"SCHEMA_REFUSED {dataset_id}: train/validation split returned no accepted rows"
                        )
                    validation_source = validation_split
                else:
                    rows, train_count, valid_count = write_split(
                        dataset_id, expert_id, train_stream, train, valid,
                        route_hash_to_validation=True, max_examples=max_examples,
                    )
                    if rows < 2 or train_count < 1 or valid_count < 1:
                        raise SystemExit(
                            f"SCHEMA_REFUSED {dataset_id}: deterministic 2% holdout was empty; "
                            "pin a real validation_split in the approval manifest"
                        )
                    validation_source = "deterministic_2_percent_holdout_from_train_split"
                total_train += train_count
                total_valid += valid_count
                details.append({
                    "dataset_id": dataset_id,
                    "config_name": approval.get("config_name"),
                    "revision": approval["revision"],
                    "train_split": train_split,
                    "validation_source": validation_source,
                    "adapter_family": adapter_family(dataset_id),
                    "adapter_version": ADAPTER_VERSION,
                    "schema_fingerprint": approval["schema_fingerprint"],
                    "train_examples": train_count,
                    "validation_examples": valid_count,
                    "license_id": approval["license_id"],
                    "reviewer": approval["reviewer"],
                    "review_date": approval["review_date"],
                })
        if total_train < 1 or total_valid < 1:
            raise SystemExit("REFUSED: final train or validation corpus is empty")
        os.replace(train_tmp, out_dir / "train.txt")
        os.replace(valid_tmp, out_dir / "valid.txt")
        report = {
            "status": "REVIEWED_EXPERT_CORPUS_READY",
            "expert_id": expert_id,
            "expert_name": next(row["name"] for row in experts if row["expert_id"] == expert_id),
            "adapter_version": ADAPTER_VERSION,
            "holdout_policy": "pinned validation split when available; otherwise deterministic row-level 2%",
            "train_examples": total_train,
            "validation_examples": total_valid,
            "datasets": details,
            "train_sha256": sha256(out_dir / "train.txt"),
            "valid_sha256": sha256(out_dir / "valid.txt"),
            "note": "Corpus preparation only; no training or checkpoint writes are performed.",
        }
        (out_dir / "corpus_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    finally:
        train_tmp.unlink(missing_ok=True)
        valid_tmp.unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--approval-file", required=True)
    parser.add_argument("--expert-id", required=True, type=int, choices=range(20))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-examples-per-split", type=int, default=50_000)
    args = parser.parse_args()
    report = prepare_corpus(
        Path(args.manifest), Path(args.approval_file), args.expert_id,
        Path(args.out_dir), args.max_examples_per_split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
