#!/usr/bin/env python3
"""Reproducible 65,536-token BPE training with fail-closed provenance gates.

This script intentionally refuses to train from the current default-deny
manifest. Each dataset source must be individually approved and pinned.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"] + [
    f"<|control_{i:03d}|>" for i in range(127)
]
EXPECTED_VOCAB_SIZE = 65536

ROUND_TRIP_SAMPLES = [
    "नमस्ते। NeuroCore हिंदी और भारतीय भाषाओं के लिए बनाया जा रहा है।",
    "The quick brown fox tests English byte-level tokenization.",
    "संविधान, अनुच्छेद 21, ₹1,23,456.78 — मिश्रित चिह्न और अंक।",
    "தமிழ் తెలుగు বাংলা ગુજરાતી ಕನ್ನಡ മലയാളം ਪੰਜਾਬੀ ଓଡ଼ିଆ मराठी",
    "Email: user@example.org; URL: https://example.org/a?b=1&c=2",
]

def validate_tokenizer_object(tokenizer) -> dict[str, Any]:
    actual_vocab=tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"tokenizer vocabulary is {actual_vocab}, expected {EXPECTED_VOCAB_SIZE}")
    special_ids={}
    for token in SPECIAL_TOKENS:
        token_id=tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"missing required special token: {token}")
        special_ids[token]=int(token_id)
    if len(set(special_ids.values())) != len(SPECIAL_TOKENS):
        raise RuntimeError("special token IDs must all be unique")
    for token in SPECIAL_TOKENS:
        ids=tokenizer.encode(token,add_special_tokens=False).ids
        if ids != [special_ids[token]]:
            raise RuntimeError(f"special token must encode to exactly one dedicated ID: {token}")
    for sample in ROUND_TRIP_SAMPLES:
        encoded=tokenizer.encode(sample)
        decoded=tokenizer.decode(encoded.ids,skip_special_tokens=True)
        if decoded != sample:
            raise RuntimeError(f"Unicode encode/decode round trip failed for sample: {sample[:40]!r}")
    return {
        "vocab_size":actual_vocab,
        "special_token_count":len(SPECIAL_TOKENS),
        "control_token_count":127,
        "round_trip_samples":len(ROUND_TRIP_SAMPLES),
        "unicode_round_trip":"PASS",
        "special_token_ids":"UNIQUE_AND_SINGLE_TOKEN",
    }

def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_approved_sources(manifest_path: Path) -> list[dict[str,Any]]:
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    approval=manifest.get("tokenizer_source_approval",{})
    sources=approval.get("approved_sources",[])
    if approval.get("status") != "APPROVED" or not sources:
        raise RuntimeError("tokenizer training blocked: no explicitly approved tokenizer sources")
    seen=set()
    for row in sources:
        required=("dataset_id","revision","config_name","split","license_id",
                  "license_terms_url","approval_id","reviewer","reviewed_at","text_columns","source_provenance_reviewed","public_redistribution_allowed","derived_artifact_distribution_allowed")
        missing=[key for key in required if not row.get(key)]
        if missing:
            raise RuntimeError(f"incomplete tokenizer source approval: {row.get('dataset_id')}: {missing}")
        revision=row.get("revision","")
        if len(revision)!=40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
            raise RuntimeError(f"tokenizer source revision must be an immutable 40-char commit SHA: {row.get('dataset_id')}")
        if not str(row.get("license_terms_url","")).startswith("https://"):
            raise RuntimeError(f"license_terms_url must use HTTPS: {row.get('dataset_id')}")
        if not isinstance(row.get("text_columns"),list) or not row["text_columns"] or not all(isinstance(x,str) and x for x in row["text_columns"]):
            raise RuntimeError(f"approved source must specify exact text_columns: {row.get('dataset_id')}")
        if row.get("approved") is not True:
            raise RuntimeError(f"tokenizer source not approved: {row.get('dataset_id')}")
        for flag in ("commercial_training_allowed","public_redistribution_allowed","derived_artifact_distribution_allowed","source_provenance_reviewed","pii_reviewed","consent_reviewed","schema_reviewed","split_reviewed"):
            if row.get(flag) is not True:
                raise RuntimeError(f"{row.get('dataset_id')} missing positive approval: {flag}")
        key=(row["dataset_id"],row["revision"],row["config_name"],row["split"])
        if key in seen:
            raise RuntimeError(f"duplicate tokenizer source approval: {key}")
        seen.add(key)
    return sources

def iter_approved_text(sources: list[dict[str,Any]]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets before tokenizer training") from exc
    for source in sources:
        ds=load_dataset(
            source["dataset_id"],
            name=source["config_name"],
            split=source["split"],
            revision=source["revision"],
            streaming=True,
        )
        columns=source["text_columns"]
        for row in ds:
            pieces=[]
            for column in columns:
                value=row.get(column)
                if isinstance(value,str) and value.strip():
                    pieces.append(value.strip())
            if pieces:
                yield "\n".join(pieces)

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--approval-manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--vocab-size",type=int,default=EXPECTED_VOCAB_SIZE)
    args=parser.parse_args()
    if args.vocab_size != EXPECTED_VOCAB_SIZE:
        raise SystemExit(f"canonical vocab_size is fixed at {EXPECTED_VOCAB_SIZE}")
    sources=load_approved_sources(args.approval_manifest)
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError as exc:
        raise SystemExit("install tokenizers before running this script") from exc
    args.output_dir.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_approved_text(sources),trainer=trainer)
    tokenizer.post_processor=processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[("<|bos|>",tokenizer.token_to_id("<|bos|>")),
                        ("<|eos|>",tokenizer.token_to_id("<|eos|>"))],
    )
    validation=validate_tokenizer_object(tokenizer)
    actual_vocab=validation["vocab_size"]
    out=args.output_dir/"tokenizer.json"
    tokenizer.save(str(out))
    metadata={
      "schema_version":1,
      "vocab_size":actual_vocab,
      "tokenizer_type":"byte_level_bpe",
      "special_tokens":SPECIAL_TOKENS,
      "control_token_semantics":"reserved IDs only; semantic mapping not yet approved",
      "sources":[{k:v for k,v in s.items() if k!="text_columns"} for s in sources],
      "source_manifest_sha256":sha256_file(args.approval_manifest),
      "tokenizer_sha256":sha256_file(out),
      "validation":validation,
      "training_parameters":{"min_frequency":2,"initial_alphabet":"ByteLevel"},
      "status":"BUILT_PENDING_TOKENIZER_SHA_APPROVAL",
    }
    (args.output_dir/"tokenizer_manifest.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
