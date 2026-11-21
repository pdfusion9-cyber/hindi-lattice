#!/usr/bin/env python3
"""Package one CHAKRA domain checkpoint as an independent, lossless FP32 expert.

The original checkpoint is read-only. Optimizer/training-only fields are omitted;
model tensors, vocabulary, patterns, graph maps and model config are preserved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    if not src.is_file() or src.stat().st_size == 0:
        raise SystemExit(f"missing/empty checkpoint: {src}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    required = ("m", "V", "dim", "layers", "heads", "vocab", "pats")
    missing = [k for k in required if k not in ck]
    if missing:
        raise SystemExit(f"{args.domain}: missing fields: {missing}")
    if not isinstance(ck["m"], dict) or not ck["m"]:
        raise SystemExit(f"{args.domain}: invalid model state")
    if not isinstance(ck["vocab"], list) or len(ck["vocab"]) != int(ck["V"]):
        raise SystemExit(f"{args.domain}: vocabulary metadata mismatch")
    if "tok.weight" not in ck["m"] or "pat.weight" not in ck["m"]:
        raise SystemExit(f"{args.domain}: embedding tensors missing")
    if ck["m"]["tok.weight"].shape[0] != int(ck["V"]):
        raise SystemExit(f"{args.domain}: token embedding row count mismatch")
    if ck["m"]["pat.weight"].shape[0] != max(len(ck["pats"]), 1):
        raise SystemExit(f"{args.domain}: pattern embedding row count mismatch")

    model = {}
    dtype_counts = {}
    for name, tensor in ck["m"].items():
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"{args.domain}: non-tensor state entry {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise SystemExit(f"{args.domain}: non-finite tensor {name}")
        model[name] = tensor.detach().cpu().contiguous()
        key = str(tensor.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + tensor.numel()

    parameters = sum(t.numel() for t in model.values())
    payload = {
        "format": "chakra-independent-expert-v1",
        "domain": args.domain,
        "source_run_id": str(args.source_run_id),
        "m": model,
        "V": int(ck["V"]),
        "dim": int(ck["dim"]),
        "layers": int(ck["layers"]),
        "heads": int(ck["heads"]),
        "vocab": ck["vocab"],
        "vocab_counts": ck.get("vocab_counts"),
        "pats": ck["pats"],
        "pat_counts": ck.get("pat_counts"),
        "graph": ck.get("graph"),
        "config": ck.get("config", {}),
        "step": int(ck.get("step", 0)),
        "parameter_count": int(parameters),
        "parameter_dtypes_numel": dtype_counts,
        "provenance": {
            "source_checkpoint": str(src.name),
            "source_checkpoint_sha256": sha256(src),
            "packaging": "lossless FP32/state-dtype preserving; no averaging; no quantization",
            "optimizer_state_included": False,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)

    manifest = {
        "format": payload["format"],
        "domain": args.domain,
        "source_run_id": str(args.source_run_id),
        "parameter_count": int(parameters),
        "parameter_bytes": sum(t.numel() * t.element_size() for t in model.values()),
        "vocab_size": int(ck["V"]),
        "pattern_count": len(ck["pats"]),
        "graph_fields": sorted(ck["graph"].keys()) if isinstance(ck.get("graph"), dict) else [],
        "model_config": {
            "dim": int(ck["dim"]), "layers": int(ck["layers"]), "heads": int(ck["heads"])
        },
        "output_bytes": out.stat().st_size,
        "output_sha256": sha256(out),
        "dtype_numel": dtype_counts,
    }
    manifest_path = out.with_suffix(out.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
