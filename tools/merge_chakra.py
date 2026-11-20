#!/usr/bin/env python3
"""Streaming, identity-aligned experimental merge for CHAKRA checkpoints.

Non-embedding weights use equal-weight parameter averaging. Token and pattern
embedding rows are aligned by their actual string/pattern identity. This is an
experimental merge, not a claim that averaging independently fine-tuned models
is always beneficial.
"""
import argparse
import gc
import json
import os
from collections import Counter

import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def normalize_pattern(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def row_alignment(reference_values, source_values):
    source_index = {value: i for i, value in enumerate(source_values)}
    ref_rows, src_rows = [], []
    for ref_i, value in enumerate(reference_values):
        src_i = source_index.get(value)
        if src_i is not None:
            ref_rows.append(ref_i)
            src_rows.append(src_i)
    return ref_rows, src_rows


def add_aligned_rows(acc, counts, source, ref_values, source_values, label):
    if source.ndim != 2 or acc.ndim != 2 or source.shape[1] != acc.shape[1]:
        raise ValueError(f"{label} embedding width mismatch: {tuple(source.shape)} vs {tuple(acc.shape)}")
    ref_rows, src_rows = row_alignment(ref_values, source_values)
    chunk = 1024
    for offset in range(0, len(ref_rows), chunk):
        rr = torch.tensor(ref_rows[offset:offset + chunk], dtype=torch.long)
        sr = torch.tensor(src_rows[offset:offset + chunk], dtype=torch.long)
        acc.index_add_(0, rr, source.index_select(0, sr).to(dtype=acc.dtype))
        counts.index_add_(0, rr, torch.ones(len(rr), dtype=counts.dtype))
    return len(ref_rows)


def atomic_save(payload, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--state", required=True)
    ap.add_argument("--domain", default="")
    ap.add_argument("--final-output", default="")
    ap.add_argument("--report", default="out/merge_report.json")
    args = ap.parse_args()

    if args.finalize_only:
        ck = None
        model = None
        state = load_checkpoint(args.state)
        token_hits = pattern_hits = 0
    else:
        if not args.input or not args.domain:
            raise ValueError("--input and --domain are required unless --finalize-only is used")
        ck = load_checkpoint(args.input)
        if not isinstance(ck, dict) or not isinstance(ck.get("m"), dict):
        raise ValueError(f"{args.domain}: checkpoint does not contain a model state dict")
        model = ck["m"]
        vocab = ck.get("vocab")
        pats = ck.get("pats")
    if not args.finalize_only and not isinstance(vocab, list) or not isinstance(pats, list):
        raise ValueError(f"{args.domain}: checkpoint must contain vocab and pats metadata")
    if not args.finalize_only:
        dim, layers, heads = (int(ck.get(k, -1)) for k in ("dim", "layers", "heads"))
    if not args.finalize_only and min(dim, layers, heads) <= 0:
        raise ValueError(f"{args.domain}: invalid architecture metadata")
    if not args.finalize_only and ("tok.weight" not in model or "pat.weight" not in model):
        raise ValueError(f"{args.domain}: expected tok.weight and pat.weight")

    if args.finalize_only:
        pass
    elif not os.path.exists(args.state):
        sums = {k: v.detach().cpu().clone() for k, v in model.items()}
        if sums["tok.weight"].shape[0] != len(vocab):
            raise ValueError("base token embedding rows do not match vocabulary")
        if sums["pat.weight"].shape[0] != max(len(pats), 1):
            raise ValueError("base pattern embedding rows do not match pattern metadata")
        state = {
            "sums": sums,
            "token_counts": torch.ones(len(vocab), dtype=torch.int32),
            "pattern_counts": torch.ones(max(len(pats), 1), dtype=torch.int32),
            "vocab": vocab,
            "pats": [normalize_pattern(p) for p in pats],
            "vocab_counts": ck.get("vocab_counts"),
            "pat_counts": ck.get("pat_counts"),
            "dim": dim,
            "layers": layers,
            "heads": heads,
            "base_graph": ck.get("graph"),
            "base_config": ck.get("config", {}),
            "domains": [],
            "architecture": {k: list(v.shape) for k, v in model.items()},
        }
        token_hits = len(vocab)
        pattern_hits = len(pats)
    elif os.path.exists(args.state):
        state = load_checkpoint(args.state)
        if (state["dim"], state["layers"], state["heads"]) != (dim, layers, heads):
            raise ValueError(f"{args.domain}: architecture metadata differs from base model")
        if set(model) != set(state["sums"]):
            missing = sorted(set(state["sums"]) - set(model))
            extra = sorted(set(model) - set(state["sums"]))
            raise ValueError(f"{args.domain}: model keys differ; missing={missing[:5]} extra={extra[:5]}")
        if model["tok.weight"].shape[1] != state["sums"]["tok.weight"].shape[1]:
            raise ValueError(f"{args.domain}: token embedding width mismatch")
        if model["pat.weight"].shape[1] != state["sums"]["pat.weight"].shape[1]:
            raise ValueError(f"{args.domain}: pattern embedding width mismatch")

        source_vocab = vocab
        source_pats = [normalize_pattern(p) for p in pats]
        ref_vocab = state["vocab"]
        ref_pats = state["pats"]
        token_hits = add_aligned_rows(
            state["sums"]["tok.weight"], state["token_counts"],
            model["tok.weight"], ref_vocab, source_vocab, "token"
        )
        pattern_hits = add_aligned_rows(
            state["sums"]["pat.weight"], state["pattern_counts"],
            model["pat.weight"], ref_pats, source_pats, "pattern"
        )
        for key, target in state["sums"].items():
            if key in ("tok.weight", "pat.weight"):
                continue
            source = model[key]
            if list(source.shape) != list(target.shape):
                raise ValueError(f"{args.domain}: shape mismatch for {key}: {tuple(source.shape)} vs {tuple(target.shape)}")
            if not torch.is_floating_point(target):
                if not torch.equal(target, source):
                    raise ValueError(f"{args.domain}: non-floating model buffer differs: {key}")
            else:
                target.add_(source.to(dtype=target.dtype))

    if not args.finalize_only:
      state["domains"].append({
        "domain": args.domain,
        "token_rows_matched_to_base": int(token_hits),
        "token_rows_in_source": int(model["tok.weight"].shape[0]),
        "pattern_rows_matched_to_base": int(pattern_hits),
        "pattern_rows_in_source": int(model["pat.weight"].shape[0]),
    })
      atomic_save(state, args.state)
    if not args.finalize_only:
      print(
        f"{args.domain}: added | token rows matched={token_hits:,} "
        f"| pattern rows matched={pattern_hits:,} | checkpoints={len(state['domains'])}",
        flush=True,
    )
      del ck, model, state
      gc.collect()

    if args.final_output:
        state = load_checkpoint(args.state)
        count = len(state["domains"])
        if count != 10:
            raise ValueError(f"refusing final merge: expected 10 domain checkpoints, got {count}")
        merged_weights = {}
        for key, total in state["sums"].items():
            if key == "tok.weight":
                denom = state["token_counts"].clamp_min(1).to(total.dtype)[:, None]
                merged_weights[key] = total / denom
            elif key == "pat.weight":
                denom = state["pattern_counts"].clamp_min(1).to(total.dtype)[:, None]
                merged_weights[key] = total / denom
            else:
                merged_weights[key] = total / float(count)

        for key, value in merged_weights.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"merged tensor has non-finite values: {key}")

        config = dict(state.get("base_config") or {})
        config.setdefault("nbr", 8)
        config.setdefault("cand", 50)
        config.setdefault("pat_cand", 20)
        config.setdefault("res_scale", 0.5)
        config.setdefault("sparse_win", 6)
        merged = {
            "m": merged_weights,
            "step": 0,
            "V": len(state["vocab"]),
            "dim": state["dim"],
            "layers": state["layers"],
            "heads": state["heads"],
            "vocab": state["vocab"],
            "vocab_counts": state.get("vocab_counts"),
            "pats": state["pats"],
            "pat_counts": state.get("pat_counts"),
            "graph": state.get("base_graph"),
            "config": config,
            "merge": {
                "method": "equal-weight parameter mean; token/pattern rows aligned by identity",
                "source_run_id": state.get("source_run_id", "triggering CHAKRA run"),
                "domains": [x["domain"] for x in state["domains"]],
                "base_domain": state["domains"][0]["domain"],
                "note": "Experimental model merge; base vocabulary/pattern inventory retained.",
            },
        }
        atomic_save(merged, args.final_output)
        parameter_count = sum(v.numel() for v in merged_weights.values())
        report = {
            "status": "merged",
            "source_run_id": state.get("source_run_id", "triggering CHAKRA run"),
            "checkpoint_count": count,
            "domains": state["domains"],
            "base_domain": state["domains"][0]["domain"],
            "dim": state["dim"],
            "layers": state["layers"],
            "heads": state["heads"],
            "vocab_size": len(state["vocab"]),
            "pattern_count": len(state["pats"]),
            "parameters": int(parameter_count),
            "token_rows_with_multiple_sources": int((state["token_counts"] > 1).sum()),
            "token_rows_in_base": int(len(state["vocab"])),
            "pattern_rows_with_multiple_sources": int((state["pattern_counts"] > 1).sum()),
            "merge_policy": merged["merge"]["method"],
            "quality_claim": "No accuracy claim; forward-pass smoke test is reported separately.",
        }
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"FINAL MERGE SAVED: {args.final_output}", flush=True)
        print(f"parameters={parameter_count:,} | vocab={len(state['vocab']):,} | patterns={len(state['pats']):,}", flush=True)


if __name__ == "__main__":
    main()
