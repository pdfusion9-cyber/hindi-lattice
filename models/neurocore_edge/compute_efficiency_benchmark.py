#!/usr/bin/env python3
"""Controlled synthetic compute benchmark for Micro-Agent compute policies."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from expert_domain_synthetic_benchmark import (
    SEED, SEQ_LEN, VOCAB_SIZE, build_corpora, build_model, encode_records,
    sample_batch, count_parameters, _state_max_abs_difference,
)
from compute_efficiency_agents import ComputeEfficiencyController

# FP32 gradient summation order differs slightly between one full batch and
# sequential micro-batch accumulation, even for the same token-additive loss.
PARAMETER_PARITY_ATOL = 1e-5


def profile_routing(model, streams: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
    counts = {domain: [0] * 20 for domain in range(20)}
    handles = []
    for layer in model.layers:
        def collect(_module, _inputs, output):
            indices = output[0].detach().reshape(-1, 2).cpu()
            for expert in range(20):
                counts[_profile_domain[0]][expert] += int((indices == expert).sum().item())
        handles.append(layer.router.register_forward_hook(collect))
    global _profile_domain
    try:
        model.eval()
        with torch.no_grad():
            for domain, stream in streams.items():
                _profile_domain = [domain]
                x = stream[: min(stream.numel() - 1, SEQ_LEN)].reshape(1, -1)
                if x.shape[1] < 2:
                    raise ValueError("validation stream too short to profile routing")
                model(x[:, :-1])
    finally:
        for handle in handles:
            handle.remove()
    return [{"domain_id": d, "route_counts": counts[d]} for d in range(20)]


def _loss(model, x, y):
    # Keep parity checking on a token-additive objective. The router auxiliary
    # loss is batch-coupled, so it changes when an effective batch is partitioned.
    output = model(x)
    return F.cross_entropy(output["logits"].reshape(-1, VOCAB_SIZE), y.reshape(-1))


def _batch(stream, size: int, generator: torch.Generator):
    max_start = stream.numel() - SEQ_LEN - 1
    if max_start < 1:
        raise ValueError("encoded expert corpus too short")
    starts = torch.randint(0, max_start + 1, (size,), generator=generator)
    values = torch.stack([stream[int(s):int(s) + SEQ_LEN + 1] for s in starts])
    return values[:, :-1].contiguous(), values[:, 1:].contiguous()


def _split_effective_batch(x, y, *, micro_batch: int, accumulation: int):
    expected = micro_batch * accumulation
    if micro_batch < 1 or accumulation < 1 or x.shape[0] != expected or y.shape[0] != expected:
        raise ValueError("effective batch shape must equal micro_batch * accumulation")
    return [
        (x[offset:offset + micro_batch], y[offset:offset + micro_batch])
        for offset in range(0, expected, micro_batch)
    ]


def _prepare_effective_batches(corpora, schedule, *, seed, effective_batch_size):
    streams = {eid: encode_records(row["train_records"]) for eid, row in corpora.items()}
    generator = torch.Generator().manual_seed(seed + 1000)
    batches = []
    for domain in schedule:
        x, y = _batch(streams[domain], effective_batch_size, generator)
        batches.append((domain, x, y))
    return batches


def _run_trial(batches, *, seed, micro_batch, accumulation, checkpoint_interval):
    torch.manual_seed(seed)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    train_started = time.perf_counter()
    checkpoint_seconds = 0.0
    checkpoint_count = 0
    sparse_steps = 0
    last_loss = 0.0
    with tempfile.TemporaryDirectory(prefix="neurocore-compute-") as tmp:
        latest = Path(tmp) / "latest.pt"
        for step, (_domain, effective_x, effective_y) in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            micro_batches = _split_effective_batch(
                effective_x, effective_y, micro_batch=micro_batch,
                accumulation=accumulation,
            )
            for x, y in micro_batches:
                loss = _loss(model, x, y) / accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                last_loss = float(loss.item()) * accumulation
            present = [p.grad for p in model.parameters() if p.grad is not None]
            if not present or not all(bool(torch.isfinite(g).all()) for g in present):
                raise RuntimeError("missing-all or non-finite gradients")
            if len(present) < len(list(model.parameters())):
                sparse_steps += 1
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            optimizer.step()
            if step % checkpoint_interval == 0 or step == len(batches):
                started = time.perf_counter()
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed_steps": step,
                    "rng_state": torch.get_rng_state(),
                }, latest)
                checkpoint_seconds += time.perf_counter() - started
                checkpoint_count += 1
        training_elapsed = time.perf_counter() - train_started
        checkpoint_bytes = latest.stat().st_size
    return {
        "model": model,
        "training_elapsed_seconds": training_elapsed,
        "checkpoint_elapsed_seconds": checkpoint_seconds,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "last_training_loss": last_loss,
        "sparse_inactive_gradient_steps": sparse_steps,
        "steps": len(batches),
        "examples_seen": len(batches) * micro_batch * accumulation,
        "tokens_seen": len(batches) * micro_batch * accumulation * SEQ_LEN,
        "tokens_per_second": (len(batches) * micro_batch * accumulation * SEQ_LEN) / training_elapsed,
    }


def run_compute_benchmark(report_path: Path, manifest_path: Path, *, steps=40, seed=SEED):
    if steps < 20 or steps > 200 or steps % 20:
        raise ValueError("steps must be a multiple of 20 from 20 through 200")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        corpora = build_corpora(manifest_path)
        streams = {eid: encode_records(row["validation_records"]) for eid, row in corpora.items()}
        torch.manual_seed(seed)
        probe = build_model()
        routing_profile = profile_routing(probe, streams)
        plan = ComputeEfficiencyController().create_plan(
            routing_profile=routing_profile,
            steps=steps,
            effective_batch_size=4,
            sequence_length=SEQ_LEN,
            hidden_size=16,
            activation_budget_bytes=65536,
        )
        schedule = plan["domain_schedule"]
        # Pre-sample once so both arms consume identical examples in identical order.
        effective_batches = _prepare_effective_batches(
            corpora, schedule, seed=seed,
            effective_batch_size=plan["effective_batch_size"],
        )
        # Router auxiliary loss is batch-coupled; parity is scoped explicitly to CE.
        baseline = _run_trial(
            effective_batches, seed=seed, micro_batch=4, accumulation=1,
            checkpoint_interval=steps + 1,
        )
        managed = _run_trial(
            effective_batches, seed=seed,
            micro_batch=plan["micro_batch_size"],
            accumulation=plan["gradient_accumulation_steps"],
            checkpoint_interval=plan["checkpoint_interval"],
        )
        parity = _state_max_abs_difference(baseline["model"], managed["model"])
        if parity > PARAMETER_PARITY_ATOL:
            raise RuntimeError(
                "effective-batch parity failed: max parameter difference "
                f"{parity} exceeds FP32 accumulation tolerance {PARAMETER_PARITY_ATOL}"
            )
        total_steps = len(schedule)
        report = {
            "status": "COMPUTE_EFFICIENCY_BENCHMARK_PASS",
            "experiment": "native_neurocore_micro_agent_compute_efficiency_v1",
            "model": "native_neurocore_reference",
            "parameter_count": count_parameters(managed["model"]),
            "expert_count": 20,
            "seed": seed,
            "steps": total_steps,
            "same_seed_schedule_and_effective_batch": True,
            "same_pre_sampled_examples_and_order": True,
            "training_objective": "token_additive_causal_lm_cross_entropy",
            "router_auxiliary_loss_included": False,
            "router_auxiliary_loss_exclusion_reason": "Router auxiliary loss uses batch-wide importance and assignment means; its gradient is not invariant to physical micro-batch partitioning.",
            "effective_batch_size": plan["effective_batch_size"],
            "micro_agent_plan": plan,
            "routing_profile": routing_profile,
            "baseline": {k:v for k,v in baseline.items() if k != "model"},
            "micro_agent_managed": {k:v for k,v in managed.items() if k != "model"},
            "comparison": {
                "max_abs_parameter_difference": parity,
                "parameter_parity_absolute_tolerance": PARAMETER_PARITY_ATOL,
                "training_time_ratio_baseline_over_managed": baseline["training_elapsed_seconds"] / managed["training_elapsed_seconds"],
                "tokens_per_second_ratio_managed_over_baseline": managed["tokens_per_second"] / baseline["tokens_per_second"],
                "checkpoint_time_seconds": managed["checkpoint_elapsed_seconds"],
                "checkpoint_overhead_fraction": managed["checkpoint_elapsed_seconds"] / managed["training_elapsed_seconds"],
                "interpretation": "Single-run CPU timing is noisy; compare repeated runs before claiming speedup. Parameter parity is scoped to the token-additive cross-entropy objective; router auxiliary loss is excluded because it is batch-coupled and needs a separate global-statistics accumulation design. Activation memory is an analytical estimate, not measured peak RSS/VRAM.",
            },
            "resource_estimates": {
                "activation_memory_estimate_bytes": plan["decisions"]["adaptive_batch_planner"]["activation_bytes_per_micro_batch_estimate"],
                "activation_memory_estimate_only": True,
                "peak_memory_measured": False,
            },
            "provenance": {
                "external_data_used": False,
                "user_data_used": False,
                "external_model_weights_used": False,
                "open_source_model_integrated": False,
            },
            "production_training_gate": "CLOSED",
            "dataset_approval_status": "ALL_DATASETS_PENDING_REVIEW",
            "checkpoint_persisted": False,
            "note": "Small synthetic CPU engineering benchmark only; not full 2.87B model training, real-domain quality evidence, or production readiness.",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("expert_manifest.json"))
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_compute_benchmark(args.report_path, args.manifest, steps=args.steps, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
