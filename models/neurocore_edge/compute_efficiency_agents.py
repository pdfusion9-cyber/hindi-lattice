#!/usr/bin/env python3
"""Deterministic CPU-side compute policy agents for native NeuroCore pilots.

The planner produces a bounded domain schedule, an activation-budgeted physical
micro-batch, and a checkpoint interval. It does not train weights itself.
"""
from __future__ import annotations

from typing import Any


class ExpertLoadBalancer:
    """Create equal-visit domain scheduling while preferring lower measured route load."""

    def plan(self, routing_profile: list[dict[str, Any]], steps: int) -> dict[str, Any]:
        if len(routing_profile) != 20:
            raise ValueError("routing profile must contain exactly 20 expert-domain rows")
        by_domain: dict[int, list[int]] = {}
        for row in routing_profile:
            domain = row.get("domain_id")
            counts = row.get("route_counts")
            if not isinstance(domain, int) or domain in by_domain or domain not in range(20):
                raise ValueError("routing profile must contain unique domain IDs 0..19")
            if not isinstance(counts, list) or len(counts) != 20:
                raise ValueError("each routing row must contain 20 route counts")
            if any(not isinstance(value, int) or value < 0 for value in counts):
                raise ValueError("route counts must be non-negative integers")
            by_domain[domain] = counts
        if set(by_domain) != set(range(20)):
            raise ValueError("routing profile must contain exactly 20 expert-domain rows")
        if not isinstance(steps, int) or steps < 20 or steps % 20:
            raise ValueError("steps must be a positive multiple of 20")
        quotas = {domain: steps // 20 for domain in range(20)}
        # Average each domain's observed top-2 assignment load. This score is
        # used only to deterministically order domains within equal visit quotas.
        load = {
            domain: sum(by_domain[domain]) / max(1, len(by_domain[domain]))
            for domain in range(20)
        }
        ordered = sorted(range(20), key=lambda domain: (load[domain], domain))
        schedule = [ordered[index % 20] for index in range(steps)]
        visits = {str(domain): schedule.count(domain) for domain in range(20)}
        return {
            "domain_schedule": schedule,
            "domain_visits": visits,
            "domain_load_score": {str(k): load[k] for k in range(20)},
            "policy": "equal_domain_quotas_then_lowest_observed_route_load",
        }


class AdaptiveBatchPlanner:
    """Choose a memory-bounded micro-batch without changing effective batch size."""

    def plan(
        self,
        *,
        effective_batch_size: int,
        sequence_length: int,
        hidden_size: int,
        activation_budget_bytes: int,
    ) -> dict[str, Any]:
        values = (effective_batch_size, sequence_length, hidden_size, activation_budget_bytes)
        if any(not isinstance(v, int) or v < 1 for v in values):
            raise ValueError("batch, sequence, hidden size and activation budget must be positive integers")
        # Small conservative FP32 proxy: four hidden-state-sized tensors per token.
        per_example = sequence_length * hidden_size * 4 * 4
        capacity = min(effective_batch_size, activation_budget_bytes // per_example)
        if capacity < 1:
            raise MemoryError("activation budget cannot fit one sequence")
        micro_batch = max(
            (candidate for candidate in range(1, capacity + 1)
             if effective_batch_size % candidate == 0),
            default=0,
        )
        if not micro_batch:
            raise MemoryError("no exact micro-batch divisor fits activation budget")
        accumulation = effective_batch_size // micro_batch
        return {
            "effective_batch_size": effective_batch_size,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "activation_bytes_per_micro_batch_estimate": per_example * micro_batch,
            "effective_batch_preserved": micro_batch * accumulation == effective_batch_size,
        }


class CheckpointScheduler:
    """Select a conservative bounded checkpoint interval for a short pilot."""

    def plan(self, steps: int) -> int:
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        return max(5, min(200, steps // 4))


class ComputeEfficiencyController:
    """Orchestrate the deterministic CPU compute-policy agents."""

    def create_plan(
        self,
        *,
        routing_profile: list[dict[str, Any]],
        steps: int,
        effective_batch_size: int,
        sequence_length: int,
        hidden_size: int,
        activation_budget_bytes: int,
    ) -> dict[str, Any]:
        load = ExpertLoadBalancer().plan(routing_profile, steps)
        batch = AdaptiveBatchPlanner().plan(
            effective_batch_size=effective_batch_size,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            activation_budget_bytes=activation_budget_bytes,
        )
        checkpoint = CheckpointScheduler().plan(steps)
        return {
            "status": "COMPUTE_POLICY_APPROVED",
            "domain_schedule": load["domain_schedule"],
            "checkpoint_interval": checkpoint,
            "effective_batch_size": batch["effective_batch_size"],
            "micro_batch_size": batch["micro_batch_size"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "decisions": {
                "expert_load_balancer": load,
                "adaptive_batch_planner": batch,
                "checkpoint_scheduler": {"checkpoint_interval": checkpoint},
            },
            "provenance": {
                "external_model_weights_used": False,
                "open_source_model_integrated": False,
            },
        }
