#!/usr/bin/env python3
"""Native NeuroCore shared-backbone + 20-expert staged training smoke.

This exercises the real NeuroCoreEdgeForCausalLM implementation with a small
20-expert configuration, then trains one expert at a time from an identical
shared-backbone snapshot and strictly merges the expert shards. Data is
generated in memory. No external model, tokenizer, weights, data or API is used.

This is an integration/engineering run, not a 2.87B-parameter production run.
Production dispatch remains closed until the canonical architecture, compute
profile and dataset-rights gates have all been independently cleared.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM, count_parameters
from training_micro_agents import MicroAgentTrainingController, NATIVE_MODEL_EXPERT_COUNT, NATIVE_MODEL_ID


def tiny_native_config() -> NeuroCoreEdgeConfig:
    """Small instantiation of the project's own model with all 20 experts."""
    return NeuroCoreEdgeConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_experts=NATIVE_MODEL_EXPERT_COUNT,
        experts_per_token=2,
        expert_intermediate_size=32,
        attention_heads=4,
        kv_heads=2,
        sliding_window=16,
        recurrent_state_size=8,
        router_aux_loss_coefficient=0.01,
    )


def config_fingerprint(cfg: NeuroCoreEdgeConfig) -> str:
    payload = json.dumps(vars(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Hash a deterministic serialization of a CPU state_dict."""
    stream = io.BytesIO()
    cpu_state = {key: value.detach().cpu().contiguous() for key, value in sorted(state.items())}
    torch.save(cpu_state, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def synthetic_batch(domain_id: int, *, batch_size: int = 64, sequence_length: int = 16):
    """Create deterministic byte-token sequences; never accesses the network or disk."""
    if not 0 <= domain_id < NATIVE_MODEL_EXPERT_COUNT:
        raise ValueError("domain_id must be in [0, 19]")
    label = f"neurocore-native-domain-{domain_id:02d}-"
    payload = (label + " own-model synthetic language sequence; shared base and expert shard. ").encode("utf-8")
    stream = list(payload) * 8
    rows = []
    for row_id in range(batch_size):
        start = (row_id * 7 + domain_id * 13) % (len(stream) - sequence_length - 1)
        rows.append(stream[start : start + sequence_length + 1])
    tokens = torch.tensor(rows, dtype=torch.long)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def training_loss(model, x: torch.Tensor, y: torch.Tensor, cfg: NeuroCoreEdgeConfig):
    output = model(x)
    token_loss = F.cross_entropy(
        output["logits"].reshape(-1, cfg.vocab_size), y.reshape(-1)
    )
    return token_loss + cfg.router_aux_loss_coefficient * output["router_aux_loss"]


def train_shared_backbone(
    model: NeuroCoreEdgeForCausalLM,
    cfg: NeuroCoreEdgeConfig,
    *,
    steps: int,
    seed: int,
) -> list[float]:
    """Initial native shared-base stage; all NeuroCore parameters are trainable."""
    if steps < 1:
        raise ValueError("base_steps must be at least one")
    torch.manual_seed(seed)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    x, y = synthetic_batch(0, batch_size=64, sequence_length=16)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, x, y, cfg)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite shared-backbone loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("missing or non-finite shared-backbone gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses


def expert_parameter_keys(state: dict[str, torch.Tensor], expert_id: int) -> set[str]:
    marker = f".experts.{expert_id}."
    return {key for key in state if marker in key}


def train_expert_shard(
    shared_state: dict[str, torch.Tensor],
    cfg: NeuroCoreEdgeConfig,
    *,
    expert_id: int,
    steps: int,
    base_hash: str,
    cfg_hash: str,
    seed: int,
) -> dict[str, Any]:
    """Train one expert only; shared backbone, router and other experts stay frozen."""
    if not 0 <= expert_id < cfg.num_experts:
        raise ValueError("expert_id outside configured expert range")
    if steps < 1:
        raise ValueError("expert_steps must be at least one")
    torch.manual_seed(seed + expert_id + 1)
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    model.load_state_dict(shared_state, strict=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_parameters = []
    marker = f".experts.{expert_id}."
    for name, parameter in model.named_parameters():
        if marker in name:
            parameter.requires_grad_(True)
            target_parameters.append(parameter)
    if not target_parameters:
        raise RuntimeError(f"no native model parameters found for expert {expert_id}")

    optimizer = torch.optim.AdamW(target_parameters, lr=1e-3, weight_decay=0.0)
    x, _ = synthetic_batch(expert_id, batch_size=64, sequence_length=16)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Train the selected expert directly on the native layer activations.
        # This isolates its shard from stochastic top-k router assignment while
        # keeping the shared backbone, router and all other experts frozen.
        hidden = model.embed(x)
        expert_losses = []
        for layer in model.layers:
            memory_out, _ = layer.memory(layer.norm1(hidden))
            expert_input = layer.norm2(hidden + memory_out)
            expert_output = layer.experts[expert_id](expert_input)
            expert_losses.append(expert_output.float().pow(2).mean())
            hidden = hidden + memory_out
        loss = torch.stack(expert_losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss for expert {expert_id}")
        loss.backward()
        gradients = [p.grad for p in target_parameters if p.grad is not None]
        if not gradients or not any(bool(torch.count_nonzero(g).item()) for g in gradients):
            raise RuntimeError(f"expert {expert_id} received no gradient; refusing empty shard")
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"non-finite gradient in expert shard {expert_id}")
        torch.nn.utils.clip_grad_norm_(target_parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))

    trained_state = model.state_dict()
    updated = {
        key: trained_state[key].detach().cpu().clone()
        for key in expert_parameter_keys(trained_state, expert_id)
    }
    if not updated:
        raise RuntimeError(f"expert shard {expert_id} has no parameter tensors")
    return {
        "expert_id": expert_id,
        "base_fingerprint": base_hash,
        "config_fingerprint": cfg_hash,
        "steps": steps,
        "losses": losses,
        "expert_state": updated,
    }


def merge_expert_shards(
    shared_state: dict[str, torch.Tensor],
    shards: list[dict[str, Any]],
    cfg: NeuroCoreEdgeConfig,
    *,
    expected_base_hash: str,
    expected_cfg_hash: str,
) -> dict[str, torch.Tensor]:
    """Fail-closed merge: exactly one compatible shard for each expert ID 0..19."""
    expected_ids = set(range(cfg.num_experts))
    observed_ids = [shard.get("expert_id") for shard in shards]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise ValueError("merge requires exactly one shard for every expert ID 0..19")
    if state_fingerprint(shared_state) != expected_base_hash:
        raise ValueError("shared-backbone fingerprint changed before merge")

    merged = {key: value.detach().cpu().clone() for key, value in shared_state.items()}
    for shard in sorted(shards, key=lambda row: row["expert_id"]):
        expert_id = shard["expert_id"]
        if shard.get("base_fingerprint") != expected_base_hash:
            raise ValueError(f"base fingerprint mismatch in expert shard {expert_id}")
        if shard.get("config_fingerprint") != expected_cfg_hash:
            raise ValueError(f"config fingerprint mismatch in expert shard {expert_id}")
        incoming = shard.get("expert_state")
        if not isinstance(incoming, dict):
            raise ValueError(f"missing expert state in shard {expert_id}")
        expected_keys = expert_parameter_keys(merged, expert_id)
        if set(incoming) != expected_keys:
            raise ValueError(f"unexpected/missing tensor keys in expert shard {expert_id}")
        for key, tensor in incoming.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"non-tensor value in shard {expert_id}: {key}")
            if tensor.shape != merged[key].shape or tensor.dtype != merged[key].dtype:
                raise ValueError(f"tensor shape/dtype mismatch in expert shard {expert_id}: {key}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite tensor in expert shard {expert_id}: {key}")
            merged[key] = tensor.detach().cpu().clone()

    if sum(t.numel() for t in merged.values()) != sum(t.numel() for t in shared_state.values()):
        raise RuntimeError("merge changed parameter count")
    return merged


def production_gate(config_path: Path, approvals_path: Path) -> dict[str, Any]:
    """Report blockers without dispatching full-size training or changing approvals."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    approvals = json.loads(approvals_path.read_text(encoding="utf-8"))
    pending_rows = [
        row for row in approvals.get("datasets", [])
        if not isinstance(row, dict) or row.get("approved") is not True
    ]
    training_enabled = config.get("training", {}).get("enabled") is True
    return {
        "status": "OPEN" if training_enabled and not pending_rows else "CLOSED",
        "training_enabled_in_config": training_enabled,
        "dataset_approval_status": approvals.get("status"),
        "pending_assignment_rows": len(pending_rows),
        "architecture_parameters": config.get("parameter_budget_estimate", {}).get("total_parameters"),
        "blockers": [
            reason for condition, reason in (
                (not training_enabled, "config.json training.enabled is false"),
                (bool(pending_rows), "one or more expert dataset assignments are not approved"),
                (config.get("status") == "UNTRAINED_INITIALIZATION_ONLY",
                 "canonical full-model source is still marked untrained initialization only"),
            ) if condition
        ],
    }



def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so interrupted writes cannot replace a good one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_resume_checkpoint(
    path: Path, *, expected_config_fingerprint: str, expected_seed: int,
) -> dict[str, Any]:
    """Load only a native checkpoint with matching config, seed and provenance."""
    if not path.exists():
        return {"completed_stage": "none", "expert_shards": []}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a mapping")
    if payload.get("model_id") != NATIVE_MODEL_ID:
        raise ValueError("resume checkpoint belongs to a different model")
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("resume checkpoint config fingerprint mismatch")
    if payload.get("seed") != expected_seed:
        raise ValueError("resume checkpoint seed mismatch")
    provenance = payload.get("provenance", {})
    if provenance.get("external_weights_used") is not False or provenance.get("open_source_model_integrated") is not False:
        raise ValueError("resume checkpoint provenance is not native-only")
    if provenance.get("external_data_used") is not False or provenance.get("user_data_used") is not False:
        raise ValueError("synthetic-only resume checkpoint has unapproved data provenance")
    stage = payload.get("completed_stage")
    if stage not in {"none", "shared_backbone", "expert_shards", "merged"}:
        raise ValueError("resume checkpoint has unknown stage")
    shards = payload.get("expert_shards", [])
    if not isinstance(shards, list):
        raise ValueError("resume checkpoint expert_shards must be a list")
    ids = [item.get("expert_id") for item in shards if isinstance(item, dict)]
    if len(ids) != len(shards) or len(ids) != len(set(ids)) or any(i not in range(20) for i in ids):
        raise ValueError("resume checkpoint has invalid or duplicate expert shard IDs")
    return payload

def run_staged_training(
    report_path: Path,
    checkpoint_path: Path,
    *,
    base_steps: int = 2,
    expert_steps: int = 1,
    seed: int = 7,
    config_path: Path = Path("config.json"),
    approvals_path: Path = Path("dataset_approval.json"),
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Run native shared-base + 20 expert-shard synthetic integration."""
    if base_steps < 1 or expert_steps < 1:
        raise ValueError("base_steps and expert_steps must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        cfg = tiny_native_config()
        cfg_hash = config_fingerprint(cfg)
        torch.manual_seed(seed)
        initial_model = NeuroCoreEdgeForCausalLM(cfg).cpu()
        parameter_count = count_parameters(initial_model)
        try:
            import os
            available_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            available_memory = None
        controller = MicroAgentTrainingController()
        plan = controller.create_plan(
            requested_steps=base_steps + cfg.num_experts * expert_steps,
            max_steps=base_steps + cfg.num_experts * expert_steps,
            source_kind="synthetic_only",
            external_data_used=False,
            user_data_used=False,
            dataset_ids=[],
            external_weights_used=False,
            checkpoint_mode="ephemeral_synthetic_only",
            model_parameter_count=parameter_count,
            available_memory_bytes=available_memory,
        )
        if plan["model_under_test"] != NATIVE_MODEL_ID or plan["open_source_model_in_training_path"]:
            raise PermissionError("micro-agent rejected native-only training provenance")

        resume = load_resume_checkpoint(
            resume_path, expected_config_fingerprint=cfg_hash, expected_seed=seed
        ) if resume_path else {"completed_stage": "none", "expert_shards": []}
        if resume["completed_stage"] in {"shared_backbone", "expert_shards", "merged"}:
            shared_state = resume["shared_state"]
            base_hash = resume["shared_backbone_fingerprint"]
            if state_fingerprint(shared_state) != base_hash:
                raise ValueError("resume shared-backbone tensor fingerprint mismatch")
            shared_losses = resume.get("shared_losses", [])
            shared_model = initial_model
            shared_model.load_state_dict(shared_state, strict=True)
        else:
            torch.manual_seed(seed)
            shared_model = initial_model
            shared_losses = train_shared_backbone(
                shared_model, cfg, steps=base_steps, seed=seed
            )
            shared_state = {
                key: value.detach().cpu().clone()
                for key, value in shared_model.state_dict().items()
            }
            base_hash = state_fingerprint(shared_state)
            if resume_path:
                atomic_torch_save({
                    "model_id": NATIVE_MODEL_ID,
                    "config_fingerprint": cfg_hash,
                    "seed": seed,
                    "completed_stage": "shared_backbone",
                    "shared_state": shared_state,
                    "shared_backbone_fingerprint": base_hash,
                    "shared_losses": shared_losses,
                    "expert_shards": [],
                    "provenance": {
                        "external_data_used": False, "user_data_used": False,
                        "external_weights_used": False, "open_source_model_integrated": False,
                    },
                }, resume_path)
        parameter_count_before = count_parameters(shared_model)

        shards = list(resume.get("expert_shards", []))
        completed_ids = {item["expert_id"] for item in shards}
        for expert_id in range(cfg.num_experts):
            if expert_id in completed_ids:
                continue
            shard = train_expert_shard(
                shared_state,
                cfg,
                expert_id=expert_id,
                steps=expert_steps,
                base_hash=base_hash,
                cfg_hash=cfg_hash,
                seed=seed,
            )
            shards.append(shard)
            if resume_path:
                atomic_torch_save({
                    "model_id": NATIVE_MODEL_ID,
                    "config_fingerprint": cfg_hash,
                    "seed": seed,
                    "completed_stage": "expert_shards",
                    "shared_state": shared_state,
                    "shared_backbone_fingerprint": base_hash,
                    "shared_losses": shared_losses,
                    "expert_shards": shards,
                    "provenance": {
                        "external_data_used": False, "user_data_used": False,
                        "external_weights_used": False, "open_source_model_integrated": False,
                    },
                }, resume_path)

        merged_state = merge_expert_shards(
            shared_state,
            shards,
            cfg,
            expected_base_hash=base_hash,
            expected_cfg_hash=cfg_hash,
        )
        merged_model = NeuroCoreEdgeForCausalLM(cfg).cpu()
        merged_model.load_state_dict(merged_state, strict=True)
        parameter_count_after = count_parameters(merged_model)
        merged_hash = state_fingerprint(merged_state)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "format_version": 1,
            "model_id": NATIVE_MODEL_ID,
            "training_scale": "tiny_20_expert_reference",
            "config": vars(cfg),
            "config_fingerprint": cfg_hash,
            "seed": seed,
            "completed_stage": "merged",
            "shared_backbone_fingerprint": base_hash,
            "merged_state_fingerprint": merged_hash,
            "model_state_dict": merged_state,
            "micro_agent_training_control": plan,
            "provenance": {
                "source_kind": "synthetic_only",
                "external_data_used": False,
                "user_data_used": False,
                "external_weights_used": False,
                "open_source_model_integrated": False,
            },
        }, checkpoint_path)

        gate = production_gate(config_path, approvals_path)
        report = {
            "status": "NATIVE_20_EXPERT_STAGED_SYNTHETIC_PASS",
            "model_under_test": NATIVE_MODEL_ID,
            "training_target": "native_neurocore_edge_architecture_family",
            "training_scale": "tiny_20_expert_reference",
            "open_source_model_integrated": False,
            "external_weights_used": False,
            "shared_backbone": {
                "status": "TRAINED",
                "steps": base_steps,
                "losses": shared_losses,
                "fingerprint": base_hash,
                "all_model_parameters_trainable": True,
            },
            "expert_shards": {
                "status": "TRAINED_AND_MERGED",
                "count": len(shards),
                "expert_ids": [row["expert_id"] for row in shards],
                "steps_per_expert": expert_steps,
                "base_fingerprint_match": all(row["base_fingerprint"] == base_hash for row in shards),
                "config_fingerprint_match": all(row["config_fingerprint"] == cfg_hash for row in shards),
                "losses": {str(row["expert_id"]): row["losses"] for row in shards},
                "merge_policy": "strict_tensor_key_shape_dtype_fingerprint_and_exact_20_ids",
            },
            "parameter_count_before_merge": parameter_count_before,
            "parameter_count_after_merge": parameter_count_after,
            "parameter_count_preserved": parameter_count_before == parameter_count_after,
            "merged_state_fingerprint": merged_hash,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "micro_agent_training_control": plan,
            "production_training_gate": gate,
            "production_2_87b_training_started": False,
            "production_blockers": gate["blockers"],
            "provenance": {
                "source_kind": "generated_in_memory_synthetic_only",
                "external_data_used": False,
                "user_data_used": False,
                "external_weights_used": False,
                "open_source_model_integrated": False,
                "dataset_approvals_changed": False,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "note": (
                "This validates the native NeuroCore shared-backbone, 20 expert-shard, "
                "strict merge and micro-agent control path at tiny dimensions. It is "
                "not a 2.87B-parameter production training run or quality/convergence claim."
            ),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-steps", type=int, default=2)
    parser.add_argument("--expert-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--approvals", type=Path, default=Path("dataset_approval.json"))
    args = parser.parse_args()
    report = run_staged_training(
        args.report_path,
        args.checkpoint_path,
        base_steps=args.base_steps,
        expert_steps=args.expert_steps,
        seed=args.seed,
        config_path=args.config,
        approvals_path=args.approvals,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "NATIVE_20_EXPERT_STAGED_SYNTHETIC_PASS":
        raise SystemExit("native staged training did not pass")
    if not report["parameter_count_preserved"]:
        raise SystemExit("expert merge changed parameter count")


if __name__ == "__main__":
    main()
