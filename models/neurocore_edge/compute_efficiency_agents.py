#!/usr/bin/env python3
"""Micro-agent policies for expert-load balancing and bounded training compute."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from micro_agent_runtime import (
    AgentRegistry, AgentResult, AgentSpec, Difficulty,
    MicroAgentOrchestrator, TaskStatus, make_task,
)

COMPUTE_AGENT_SPECS = (
    AgentSpec(
        "expert_load_balancer",
        frozenset({"router_load_balancing"}),
        "Builds a deterministic domain schedule that compensates measured top-2 router load.",
    ),
    AgentSpec(
        "adaptive_batch_planner",
        frozenset({"adaptive_batch_sizing", "gradient_accumulation"}),
        "Selects a physical micro-batch and accumulation count under an activation-memory budget.",
    ),
    AgentSpec(
        "checkpoint_scheduler",
        frozenset({"checkpoint_scheduling"}),
        "Sets a bounded periodic checkpoint cadence for optimizer-step recovery.",
    ),
)


def _result(task, agent_id: str, payload: Mapping[str, Any]) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        agent_id=agent_id,
        status=TaskStatus.COMPLETED,
        result=json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        confidence=1.0,
        requires_review=False,
    )


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _load_balancer(task, execution_context) -> AgentResult:
    execution_context.raise_if_cancelled()
    ctx = task.context
    profiles = ctx.get("routing_profile")
    steps = ctx.get("steps")
    if not isinstance(profiles, list) or len(profiles) != 20:
        raise ValueError("routing_profile must contain 20 expert-domain rows")
    if not isinstance(steps, int) or steps < 20 or steps % 20:
        raise ValueError("steps must be a multiple of 20")
    profile_by_domain: dict[int, list[float]] = {}
    for row in profiles:
        if not isinstance(row, dict):
            raise ValueError("routing profile row must be an object")
        domain_id = row.get("domain_id")
        counts = row.get("route_counts")
        if not isinstance(domain_id, int) or domain_id in profile_by_domain:
            raise ValueError("domain IDs must be unique integers")
        if not isinstance(counts, list) or len(counts) != 20:
            raise ValueError("each domain route profile must contain 20 expert counts")
        if any(not isinstance(value, (int, float)) or value < 0 for value in counts):
            raise ValueError("route counts must be non-negative numbers")
        profile_by_domain[domain_id] = [float(value) for value in counts]
    if set(profile_by_domain) != set(range(20)):
        raise ValueError("route profile must cover domain IDs 0..19")

    aggregate = [sum(profile_by_domain[d][e] for d in range(20)) for e in range(20)]
    initial_cv = _coefficient_of_variation(aggregate)
    repeats = steps // 20
    remaining = {domain_id: repeats for domain_id in range(20)}
    scheduled_load = [0.0] * 20
    schedule: list[int] = []
    for _ in range(steps):
        execution_context.raise_if_cancelled()
        candidates = [domain_id for domain_id, count in remaining.items() if count > 0]
        scored = []
        for domain_id in candidates:
            projected = [
                scheduled_load[e] + profile_by_domain[domain_id][e]
                for e in range(20)
            ]
            scored.append((_coefficient_of_variation(projected), domain_id, projected))
        _score, selected, projected = min(scored, key=lambda item: (item[0], item[1]))
        schedule.append(selected)
        scheduled_load = projected
        remaining[selected] -= 1

    final_cv = _coefficient_of_variation(scheduled_load)
    return _result(task, "expert_load_balancer", {
        "decision": "ALLOW",
        "domain_schedule": schedule,
        "domain_visits": {str(d): schedule.count(d) for d in range(20)},
        "router_load_cv_before": initial_cv,
        "router_load_cv_after_planned_schedule": final_cv,
        "load_balance_policy": "greedy_minimum_projected_router_load_cv_with_equal_domain_quotas",
        "router_expert_count": 20,
    })


def _adaptive_batch_planner(task, execution_context) -> AgentResult:
    execution_context.raise_if_cancelled()
    ctx = task.context
    effective = ctx.get("effective_batch_size")
    seq_len = ctx.get("sequence_length")
    hidden = ctx.get("hidden_size")
    budget = ctx.get("activation_budget_bytes")
    skew = ctx.get("router_load_skew")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (effective, seq_len, hidden, budget)):
        raise ValueError("batch, sequence, hidden size and activation budget must be positive integers")
    if effective not in (4, 8, 16):
        raise ValueError("effective_batch_size must be 4, 8 or 16")
    if not isinstance(skew, (int, float)) or skew < 0:
        raise ValueError("router_load_skew must be a non-negative number")

    # Conservative FP32 activation proxy for this reference block. This is a
    # planning estimate, not a hardware memory measurement.
    bytes_per_sample = seq_len * hidden * 4 * 6
    candidates = [size for size in (4, 2, 1) if size <= effective and effective % size == 0]
    feasible = [
        size for size in candidates
        if size * bytes_per_sample <= budget
    ]
    if not feasible:
        raise ValueError("activation budget cannot fit even a one-sample micro-batch")
    selected = max(feasible)
    # High router skew deliberately narrows each physical batch so the scheduler
    # can interleave more gradient-accumulation windows without changing effective batch.
    if float(skew) >= 1.25:
        selected = min(selected, 2 if effective >= 8 else 1)
    accumulation = effective // selected
    return _result(task, "adaptive_batch_planner", {
        "decision": "ALLOW",
        "effective_batch_size": effective,
        "micro_batch_size": selected,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_preserved": selected * accumulation == effective,
        "router_load_skew": float(skew),
        "activation_bytes_per_sample_estimate": bytes_per_sample,
        "activation_bytes_per_micro_batch_estimate": bytes_per_sample * selected,
        "activation_budget_bytes": budget,
        "policy": "largest_feasible_micro_batch_with_router_skew_cap",
    })


def _checkpoint_scheduler(task, execution_context) -> AgentResult:
    execution_context.raise_if_cancelled()
    steps = task.context.get("steps")
    if not isinstance(steps, int) or steps < 20:
        raise ValueError("steps must be an integer >= 20")
    max_checkpoints = 4
    interval = max(5, math.ceil(steps / max_checkpoints))
    return _result(task, "checkpoint_scheduler", {
        "decision": "ALLOW",
        "save_every_optimizer_steps": interval,
        "maximum_periodic_checkpoints": steps // interval,
        "retain_latest_only": True,
        "checkpoint_storage": "ephemeral_runner_local",
        "resume_state": ["model_state_dict", "optimizer_state_dict", "completed_steps", "rng_state"],
    })


class ComputeEfficiencyController:
    """Runs three bounded project-owned micro-agents and validates their policy output."""

    def __init__(self) -> None:
        handlers = {
            "expert_load_balancer": _load_balancer,
            "adaptive_batch_planner": _adaptive_batch_planner,
            "checkpoint_scheduler": _checkpoint_scheduler,
        }
        self.orchestrator = MicroAgentOrchestrator(
            registry=AgentRegistry(COMPUTE_AGENT_SPECS),
            handlers=handlers,
        )

    def create_plan(
        self,
        *,
        routing_profile: list[dict[str, Any]],
        steps: int,
        effective_batch_size: int = 8,
        sequence_length: int = 48,
        hidden_size: int = 16,
        activation_budget_bytes: int = 65536,
    ) -> dict[str, Any]:
        aggregate = [
            sum(float(row["route_counts"][expert]) for row in routing_profile)
            for expert in range(20)
        ]
        mean = sum(aggregate) / len(aggregate)
        skew = max(aggregate) / mean if mean > 0 else 1.0
        context = {
            "routing_profile": routing_profile,
            "steps": steps,
            "effective_batch_size": effective_batch_size,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "activation_budget_bytes": activation_budget_bytes,
            "router_load_skew": skew,
        }
        task = make_task(
            "Create a bounded compute-efficiency policy for the native NeuroCore 20-expert synthetic benchmark.",
            language="en",
            difficulty=Difficulty.HARD,
            required_capabilities=(
                "router_load_balancing",
                "adaptive_batch_sizing",
                "gradient_accumulation",
                "checkpoint_scheduling",
            ),
            context=context,
        )
        state = self.orchestrator.execute_parallel(
            self.orchestrator.plan(task),
            max_workers=3,
            timeout_seconds=5.0,
            max_input_chars_total=16000,
            max_output_chars_per_agent=8000,
        )
        expected = {spec.agent_id for spec in COMPUTE_AGENT_SPECS}
        observed = {item.agent_id for item in state.findings}
        if observed != expected or state.status != TaskStatus.COMPLETED:
            detail = "; ".join(f"{x.agent_id}:{x.result}" for x in state.findings)
            raise PermissionError("compute-efficiency micro-agent plan refused: " + detail)
        decisions = {item.agent_id: json.loads(item.result) for item in state.findings}
        schedule = decisions["expert_load_balancer"]["domain_schedule"]
        if len(schedule) != steps or set(schedule) != set(range(20)):
            raise RuntimeError("load-balancer schedule does not cover all 20 domains")
        visits = decisions["expert_load_balancer"]["domain_visits"]
        if any(value != steps // 20 for value in visits.values()):
            raise RuntimeError("load-balancer schedule violated equal per-domain quotas")
        batch = decisions["adaptive_batch_planner"]
        if not batch["effective_batch_preserved"]:
            raise RuntimeError("micro-batch and accumulation do not preserve effective batch")
        return {
            "status": "COMPUTE_POLICY_APPROVED",
            "orchestrator": "MicroAgentOrchestrator",
            "selected_agents": state.selected_agents,
            "decisions": decisions,
            "domain_schedule": schedule,
            "micro_batch_size": batch["micro_batch_size"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "effective_batch_size": effective_batch_size,
            "checkpoint_interval": decisions["checkpoint_scheduler"]["save_every_optimizer_steps"],
            "agent_resource_usage": {
                "agent_calls": state.resource_usage.agent_calls,
                "completed_agents": state.resource_usage.completed_agents,
                "failed_agents": state.resource_usage.failed_agents,
                "elapsed_seconds": state.resource_usage.elapsed_seconds,
                "input_chars_charged": state.resource_usage.input_chars_charged,
                "output_chars": state.resource_usage.output_chars,
            },
        }
