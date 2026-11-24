#!/usr/bin/env python3
"""Micro-agent governed, synthetic-only NeuroCore training pilot.

No external foundation model is loaded, imported, distilled, or merged.
The three deterministic micro-agents gate provenance, allocate a bounded
training budget, and validate checkpoint policy before the native NeuroCore
reference model is trained on generated in-memory tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from micro_agent_runtime import (
    AgentRegistry,
    AgentResult,
    AgentSpec,
    Difficulty,
    MicroAgentOrchestrator,
    TaskStatus,
    make_task,
)
from synthetic_training_smoke import run_experiment


MAX_STEPS_HARD_CAP = 200
NATIVE_MODEL_ID = "native_neurocore_edge_scaled_reference"
NATIVE_MODEL_EXPERT_COUNT = 20

TRAINING_AGENT_SPECS = (
    AgentSpec(
        "training_data_gate",
        frozenset({"training_data_governance"}),
        "Rejects unreviewed external/user data; permits generated synthetic pilot data.",
    ),
    AgentSpec(
        "training_budget",
        frozenset({"training_resource_planning"}),
        "Applies a deterministic step and CPU-thread budget.",
    ),
    AgentSpec(
        "checkpoint_guard",
        frozenset({"training_checkpoint_integrity"}),
        "Requires native NeuroCore weights and an ephemeral synthetic-only checkpoint.",
    ),
)


def _result(task, agent_id: str, status: TaskStatus, payload: Mapping[str, Any], *,
            limitations: Sequence[str] = ()) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        agent_id=agent_id,
        status=status,
        result=json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        limitations=tuple(limitations),
        confidence=1.0 if status == TaskStatus.COMPLETED else 0.0,
        requires_review=status != TaskStatus.COMPLETED,
    )


def _data_gate(task, _context) -> AgentResult:
    ctx = task.context
    safe = (
        ctx.get("source_kind") == "synthetic_only"
        and ctx.get("external_data_used") is False
        and ctx.get("user_data_used") is False
        and ctx.get("dataset_ids") == []
    )
    if not safe:
        return _result(
            task, "training_data_gate", TaskStatus.NEEDS_REVIEW,
            {"decision": "REFUSE", "reason": "pilot accepts generated in-memory synthetic data only"},
            limitations=("No external dataset or user data may enter this pilot.",),
        )
    return _result(
        task, "training_data_gate", TaskStatus.COMPLETED,
        {"decision": "ALLOW", "source_kind": "synthetic_only", "dataset_ids": []},
    )


def _budget_agent(task, _context) -> AgentResult:
    ctx = task.context
    requested = ctx.get("requested_steps")
    ceiling = ctx.get("max_steps")
    parameter_count = ctx.get("model_parameter_count")
    available_memory = ctx.get("available_memory_bytes")
    bytes_per_parameter = ctx.get("bytes_per_parameter", 16)
    memory_fraction = ctx.get("memory_safety_fraction", 0.70)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 10:
        return _result(
            task, "training_budget", TaskStatus.NEEDS_REVIEW,
            {"decision": "REFUSE", "reason": "requested_steps must be an integer >= 10"},
        )
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 10:
        return _result(
            task, "training_budget", TaskStatus.NEEDS_REVIEW,
            {"decision": "REFUSE", "reason": "max_steps must be an integer >= 10"},
        )
    # Reserve memory before model allocation. A full FP32 AdamW path is
    # conservatively budgeted at 16 bytes/parameter before activations.
    estimated_training_bytes = 0
    if parameter_count is not None or available_memory is not None:
        valid = (
            isinstance(parameter_count, int) and not isinstance(parameter_count, bool)
            and parameter_count > 0
            and isinstance(available_memory, int) and not isinstance(available_memory, bool)
            and available_memory > 0
            and isinstance(bytes_per_parameter, int) and bytes_per_parameter >= 8
            and isinstance(memory_fraction, (int, float)) and 0.0 < memory_fraction <= 0.9
        )
        if not valid:
            return _result(
                task, "training_budget", TaskStatus.NEEDS_REVIEW,
                {"decision": "REFUSE", "reason": "compute capacity profile is incomplete or invalid"},
            )
        estimated_training_bytes = parameter_count * bytes_per_parameter
        safe_memory_bytes = int(available_memory * memory_fraction)
        if estimated_training_bytes > safe_memory_bytes:
            return _result(
                task, "training_budget", TaskStatus.NEEDS_REVIEW,
                {
                    "decision": "REFUSE",
                    "reason": "model training-state estimate exceeds the safe available-memory budget",
                    "estimated_training_bytes": estimated_training_bytes,
                    "safe_memory_bytes": safe_memory_bytes,
                    "model_parameter_count": parameter_count,
                },
            )
    steps = min(requested, ceiling, MAX_STEPS_HARD_CAP)
    return _result(
        task, "training_budget", TaskStatus.COMPLETED,
        {
            "decision": "ALLOW",
            "requested_steps": requested,
            "allocated_steps": steps,
            "hard_step_cap": MAX_STEPS_HARD_CAP,
            "cpu_threads": 1,
            "budget_profile": "balanced_synthetic_pilot",
            "estimated_training_state_bytes": estimated_training_bytes,
            "safe_memory_bytes": int(available_memory * memory_fraction) if available_memory else None,
            "model_parameter_count": parameter_count,
        },
    )


def _checkpoint_guard(task, _context) -> AgentResult:
    ctx = task.context
    safe = (
        ctx.get("model_under_test") == NATIVE_MODEL_ID
        and ctx.get("external_weights_used") is False
        and ctx.get("checkpoint_mode") == "ephemeral_synthetic_only"
    )
    if not safe:
        return _result(
            task, "checkpoint_guard", TaskStatus.NEEDS_REVIEW,
            {"decision": "REFUSE", "reason": "checkpoint/model provenance policy failed"},
        )
    return _result(
        task, "checkpoint_guard", TaskStatus.COMPLETED,
        {
            "decision": "ALLOW",
            "model_under_test": NATIVE_MODEL_ID,
            "external_weights_used": False,
            "num_experts": NATIVE_MODEL_EXPERT_COUNT,
            "training_scale": "tiny_20_expert_reference",
            "checkpoint_mode": "ephemeral_synthetic_only",
        },
    )


class MicroAgentTrainingController:
    """Training lifecycle coordinator built on the project's micro-agent runtime."""

    def __init__(self) -> None:
        registry = AgentRegistry(TRAINING_AGENT_SPECS)
        handlers = {
            "training_data_gate": _data_gate,
            "training_budget": _budget_agent,
            "checkpoint_guard": _checkpoint_guard,
        }
        self.orchestrator = MicroAgentOrchestrator(registry=registry, handlers=handlers)

    def create_plan(
        self,
        *,
        requested_steps: int = 40,
        max_steps: int = 80,
        source_kind: str = "synthetic_only",
        external_data_used: bool = False,
        user_data_used: bool = False,
        dataset_ids: Sequence[str] = (),
        external_weights_used: bool = False,
        checkpoint_mode: str = "ephemeral_synthetic_only",
        model_parameter_count: int | None = None,
        available_memory_bytes: int | None = None,
        bytes_per_parameter: int = 16,
        memory_safety_fraction: float = 0.70,
    ) -> dict[str, Any]:
        context = {
            "source_kind": source_kind,
            "external_data_used": external_data_used,
            "user_data_used": user_data_used,
            "dataset_ids": list(dataset_ids),
            "requested_steps": requested_steps,
            "max_steps": max_steps,
            "model_under_test": NATIVE_MODEL_ID,
            "external_weights_used": external_weights_used,
            "checkpoint_mode": checkpoint_mode,
            "model_parameter_count": model_parameter_count,
            "available_memory_bytes": available_memory_bytes,
            "bytes_per_parameter": bytes_per_parameter,
            "memory_safety_fraction": memory_safety_fraction,
        }
        task = make_task(
            "Prepare a bounded NeuroCore training session under governance and resource limits.",
            language="en",
            difficulty=Difficulty.HARD,
            required_capabilities=(
                "training_data_governance",
                "training_resource_planning",
                "training_checkpoint_integrity",
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
        expected = {spec.agent_id for spec in TRAINING_AGENT_SPECS}
        observed = {item.agent_id for item in state.findings}
        if observed != expected:
            raise RuntimeError(f"training micro-agent coverage mismatch: {observed}")
        failed = [
            item for item in state.findings
            if item.status != TaskStatus.COMPLETED or item.requires_review
        ]
        if failed or state.status != TaskStatus.COMPLETED:
            reasons = "; ".join(f"{item.agent_id}:{item.result}" for item in failed)
            raise PermissionError("Micro-agent training gate refused the run: " + reasons)

        decisions = {
            item.agent_id: json.loads(item.result)
            for item in state.findings
        }
        budget = decisions["training_budget"]
        return {
            "status": "TRAINING_PLAN_APPROVED",
            "orchestrator": "MicroAgentOrchestrator",
            "selected_agents": state.selected_agents,
            "agent_decisions": decisions,
            "allocated_steps": budget["allocated_steps"],
            "cpu_threads": budget["cpu_threads"],
            "budget_profile": budget["budget_profile"],
            "resource_usage": {
                "agent_calls": state.resource_usage.agent_calls,
                "completed_agents": state.resource_usage.completed_agents,
                "failed_agents": state.resource_usage.failed_agents,
                "timed_out_agents": state.resource_usage.timed_out_agents,
                "input_chars_charged": state.resource_usage.input_chars_charged,
                "output_chars": state.resource_usage.output_chars,
                "elapsed_seconds": state.resource_usage.elapsed_seconds,
            },
            "open_source_model_in_training_path": False,
            "external_weights_used": False,
            "model_under_test": NATIVE_MODEL_ID,
            "num_experts": NATIVE_MODEL_EXPERT_COUNT,
            "training_scale": "tiny_20_expert_reference",
            "production_2_87b_training_started": False,
            "training_data": "generated_in_memory_synthetic_only",
        }


def run_managed_training(
    report_path: Path,
    *,
    requested_steps: int = 40,
    max_steps: int = 80,
    seed: int = 7,
) -> dict[str, Any]:
    controller = MicroAgentTrainingController()
    plan = controller.create_plan(requested_steps=requested_steps, max_steps=max_steps)
    report = run_experiment(
        report_path,
        steps=plan["allocated_steps"],
        seed=seed,
        num_experts=NATIVE_MODEL_EXPERT_COUNT,
    )
    report["micro_agent_training_control"] = plan
    report["experiment"] = "neurocore_micro_agent_managed_synthetic_pilot_v1"
    report["training_started"] = True
    report["model_under_test"] = NATIVE_MODEL_ID
    report["training_target"] = "native_neurocore_edge_architecture_family"
    report["training_scale"] = "tiny_20_expert_reference"
    report["production_2_87b_training_started"] = False
    report["open_source_model_integrated"] = False
    report["external_model_weights_used"] = False
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_managed_training(
        args.report_path,
        requested_steps=args.steps,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
