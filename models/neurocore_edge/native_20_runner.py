#!/usr/bin/env python3
"""One-runner-per-expert native NeuroCore shard orchestration.

This runner utility supports three independent GitHub Actions stages:
  1. shared-base: create one versioned native shared-base artifact
  2. expert: train exactly one native expert ID on its own runner
  3. merge: strictly merge all 20 expert artifacts

Current execution mode is a tiny 20-expert synthetic engineering test. It does
not launch the 2.87B production model. Production readiness is reported
separately and remains fail-closed until the full architecture, approved
datasets, and suitable runner hardware are verified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM, count_parameters
from native_staged_training import (
    config_fingerprint,
    merge_expert_shards,
    production_gate,
    state_fingerprint,
    tiny_native_config,
    train_expert_shard,
    train_shared_backbone,
)
from training_micro_agents import NATIVE_MODEL_EXPERT_COUNT, NATIVE_MODEL_ID


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a mapping: {path}")
    return value


def build_shared_base(output: Path, *, steps: int, seed: int) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("shared-base steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = tiny_native_config()
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    losses = train_shared_backbone(model, cfg, steps=steps, seed=seed)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    cfg_hash = config_fingerprint(cfg)
    base_hash = state_fingerprint(state)
    report = {
        "status": "NATIVE_SHARED_BASE_READY",
        "model_id": NATIVE_MODEL_ID,
        "training_scale": "tiny_20_expert_reference",
        "num_experts": NATIVE_MODEL_EXPERT_COUNT,
        "config": vars(cfg),
        "config_fingerprint": cfg_hash,
        "base_fingerprint": base_hash,
        "shared_base_steps": steps,
        "shared_base_losses": losses,
        "parameter_count": count_parameters(model),
        "provenance": {
            "source_kind": "generated_in_memory_synthetic_only",
            "external_data_used": False,
            "user_data_used": False,
            "external_weights_used": False,
            "open_source_model_integrated": False,
        },
        "note": "Tiny native reference shared-base for 20-runner orchestration test only.",
    }
    atomic_save({
        **report,
        "shared_state": state,
        "seed": seed,
    }, output)
    return report


def run_one_expert(
    base_path: Path,
    expert_id: int,
    output: Path,
    *,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    if expert_id not in range(NATIVE_MODEL_EXPERT_COUNT):
        raise ValueError("expert_id must be in range 0..19")
    if steps < 1:
        raise ValueError("expert steps must be positive")
    base = load_payload(base_path)
    if base.get("status") != "NATIVE_SHARED_BASE_READY":
        raise ValueError("shared-base artifact status is invalid")
    if base.get("model_id") != NATIVE_MODEL_ID:
        raise ValueError("shared-base artifact model ID mismatch")
    if base.get("provenance", {}).get("external_weights_used") is not False:
        raise ValueError("external model weights are not permitted")
    if base.get("provenance", {}).get("open_source_model_integrated") is not False:
        raise ValueError("open-source model integration is not permitted")
    cfg = NeuroCoreEdgeConfig(**base["config"])
    cfg_hash = config_fingerprint(cfg)
    if cfg_hash != base.get("config_fingerprint"):
        raise ValueError("shared-base config fingerprint mismatch")
    base_hash = state_fingerprint(base["shared_state"])
    if base_hash != base.get("base_fingerprint"):
        raise ValueError("shared-base state fingerprint mismatch")
    shard = train_expert_shard(
        base["shared_state"],
        cfg,
        expert_id=expert_id,
        steps=steps,
        base_hash=base_hash,
        cfg_hash=cfg_hash,
        seed=seed,
    )
    report = {
        "status": "NATIVE_EXPERT_SHARD_PASS",
        "model_id": NATIVE_MODEL_ID,
        "expert_id": expert_id,
        "runner_expert_id": expert_id,
        "training_scale": "tiny_20_expert_reference",
        "base_fingerprint": base_hash,
        "config_fingerprint": cfg_hash,
        "steps": steps,
        "losses": shard["losses"],
        "tensor_count": len(shard["expert_state"]),
        "tensor_numel": sum(t.numel() for t in shard["expert_state"].values()),
        "provenance": {
            "source_kind": "generated_in_memory_synthetic_only",
            "external_data_used": False,
            "user_data_used": False,
            "external_weights_used": False,
            "open_source_model_integrated": False,
        },
    }
    atomic_save({**report, **shard}, output)
    return report


def merge_all(
    base_path: Path,
    shard_dir: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    base = load_payload(base_path)
    cfg = NeuroCoreEdgeConfig(**base["config"])
    cfg_hash = config_fingerprint(cfg)
    base_hash = state_fingerprint(base["shared_state"])
    if base_hash != base.get("base_fingerprint"):
        raise ValueError("shared-base fingerprint changed before merge")
    shard_files = sorted(shard_dir.glob("expert-*.pt"))
    shards = [load_payload(path) for path in shard_files]
    merged = merge_expert_shards(
        base["shared_state"],
        shards,
        cfg,
        expected_base_hash=base_hash,
        expected_cfg_hash=cfg_hash,
    )
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    model.load_state_dict(merged, strict=True)
    before = count_parameters(NeuroCoreEdgeForCausalLM(cfg))
    after = count_parameters(model)
    if before != after:
        raise RuntimeError("merged model parameter count changed")
    merged_hash = state_fingerprint(merged)
    payload = {
        "status": "NATIVE_20_RUNNER_MERGE_PASS",
        "model_id": NATIVE_MODEL_ID,
        "training_scale": "tiny_20_expert_reference",
        "config": vars(cfg),
        "config_fingerprint": cfg_hash,
        "shared_base_fingerprint": base_hash,
        "merged_state_fingerprint": merged_hash,
        "expert_ids": sorted(item["expert_id"] for item in shards),
        "expert_count": len(shards),
        "parameter_count_before_merge": before,
        "parameter_count_after_merge": after,
        "parameter_count_preserved": before == after,
        "provenance": {
            "source_kind": "generated_in_memory_synthetic_only",
            "external_data_used": False,
            "user_data_used": False,
            "external_weights_used": False,
            "open_source_model_integrated": False,
        },
        "note": "20 separate runner shards strictly merged; not a production 2.87B checkpoint.",
    }
    atomic_save({**payload, "model_state_dict": merged}, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def inspect_production_readiness(config_path: Path, approvals_path: Path) -> dict[str, Any]:
    gate = production_gate(config_path, approvals_path)
    try:
        available_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        available_memory = None
    try:
        free_disk = shutil.disk_usage(".").free
    except OSError:
        free_disk = None
    try:
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        vram = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
    except Exception:
        cuda_available, gpu_name, vram = False, None, None
    report = {
        **gate,
        "runner_profile": {
            "cpu_count": os.cpu_count(),
            "available_memory_bytes": available_memory,
            "free_disk_bytes": free_disk,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_vram_bytes": vram,
        },
    }
    if not cuda_available:
        report["blockers"].append("current runner has no CUDA GPU; full-scale production training is not enabled on CPU-only standard runners")
    if not available_memory or available_memory < 24 * 1024**3:
        report["blockers"].append("current runner has less than 24 GiB available host RAM for full-scale training preflight")
    report["status"] = "OPEN" if not report["blockers"] else "CLOSED"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("shared-base", "expert", "merge", "production-gate"), required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--expert-id", type=int, choices=range(20))
    parser.add_argument("--expert-checkpoint", type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--approvals", type=Path, default=Path("dataset_approval.json"))
    args = parser.parse_args()

    if args.stage == "shared-base":
        if not args.output:
            parser.error("--output is required for shared-base")
        result = build_shared_base(args.output, steps=args.steps, seed=args.seed)
    elif args.stage == "expert":
        if not args.base_checkpoint or args.expert_id is None or not args.expert_checkpoint:
            parser.error("--base-checkpoint, --expert-id and --expert-checkpoint are required for expert stage")
        result = run_one_expert(
            args.base_checkpoint, args.expert_id, args.expert_checkpoint,
            steps=args.steps, seed=args.seed,
        )
    elif args.stage == "merge":
        if not args.base_checkpoint or not args.shard_dir or not args.output or not args.report:
            parser.error("--base-checkpoint, --shard-dir, --output and --report are required for merge")
        result = merge_all(args.base_checkpoint, args.shard_dir, args.output, args.report)
    else:
        result = inspect_production_readiness(args.config, args.approvals)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.stage == "production-gate" and result["status"] != "OPEN":
        raise SystemExit("PRODUCTION_TRAINING_BLOCKED: " + "; ".join(result["blockers"]))


if __name__ == "__main__":
    main()
