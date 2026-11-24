#!/usr/bin/env python3
"""Governed 20-expert Micro-Agent training planner; fail closed by default."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

from training_micro_agents import MicroAgentTrainingController

EXPECTED_EXPERT_IDS = set(range(20))

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def build_expert_plan(manifest_path: Path, approval_path: Path, config_path: Path,
                      *, synthetic_only: bool = True, requested_steps: int = 40,
                      max_steps: int = 80) -> dict[str, Any]:
    experts = load_json(manifest_path)
    approvals_doc = load_json(approval_path)
    config = load_json(config_path)
    if not isinstance(experts, list):
        raise ValueError("expert manifest must be a JSON list")
    ids = [row.get("expert_id") for row in experts if isinstance(row, dict)]
    if len(ids) != len(experts) or set(ids) != EXPECTED_EXPERT_IDS or len(ids) != 20:
        raise ValueError("manifest must contain exactly expert IDs 0..19")
    if len({row.get("name") for row in experts}) != 20:
        raise ValueError("expert names must be unique")
    approval_rows = approvals_doc.get("datasets", [])
    approved_keys = {
        (row.get("expert_id"), row.get("dataset_id"))
        for row in approval_rows
        if isinstance(row, dict) and row.get("approved") is True
    }
    missing = []
    for expert in experts:
        for dataset_id in expert.get("datasets", []):
            if dataset_id == "curated_mixture_from_all_domains":
                missing.append({"expert_id": expert["expert_id"], "dataset_id": dataset_id,
                                "reason": "synthetic mixture placeholder is not a reviewed corpus"})
            elif (expert["expert_id"], dataset_id) not in approved_keys:
                missing.append({"expert_id": expert["expert_id"], "dataset_id": dataset_id,
                                "reason": "dataset assignment is not approved"})
    approvals_clear = (
        approvals_doc.get("status") == "APPROVED_FOR_COMMERCIAL_TRAINING"
        and not missing
    )
    production_config_clear = config.get("training", {}).get("enabled") is True
    controller = MicroAgentTrainingController()
    if synthetic_only:
        gate = controller.create_plan(
            requested_steps=requested_steps, max_steps=max_steps,
            source_kind="synthetic_only", external_data_used=False,
            user_data_used=False, dataset_ids=[], external_weights_used=False,
            checkpoint_mode="ephemeral_synthetic_only",
        )
        status = "SYNTHETIC_20_EXPERT_SMOKE_ALLOWED"
    else:
        # Report a blocked production decision without dispatching a task or
        # allocating compute. The data and checkpoint agents' policies are
        # represented explicitly; an attempted production task must not run.
        gate = {
            "status": "REFUSED",
            "selected_agents": ["training_data_gate", "checkpoint_guard"],
            "agent_decisions": {
                "training_data_gate": {
                    "decision": "REFUSE",
                    "reason": "production data assignments are not fully approved",
                },
                "checkpoint_guard": {
                    "decision": "REFUSE",
                    "reason": "production checkpoint mode is unavailable while approvals/config gate are closed",
                },
            },
            "compute_dispatched": False,
            "allocated_steps": 0,
        }
        status = "PRODUCTION_TRAINING_BLOCKED"
    return {
        "status": status if synthetic_only else "PRODUCTION_TRAINING_BLOCKED",
        "expert_count": len(experts),
        "expert_assignments": [
            {"expert_id": row["expert_id"], "expert_name": row["name"],
             "assigned_datasets": row.get("datasets", []),
             "training_enabled": False if synthetic_only else row.get("training_enabled", False)}
            for row in sorted(experts, key=lambda x: x["expert_id"])
        ],
        "dataset_approval_status": approvals_doc.get("status"),
        "approved_assignment_count": len(approved_keys),
        "unapproved_assignment_count": len(missing),
        "unapproved_assignments": missing,
        "production_config_training_enabled": production_config_clear,
        "production_dispatch_allowed": bool(not synthetic_only and approvals_clear and production_config_clear),
        "synthetic_smoke_allowed": synthetic_only,
        "micro_agent_gate": gate,
        "resource_policy": {
            "max_parallel_experts": 4 if synthetic_only else 20,
            "per_expert_step_budget": gate["allocated_steps"] if synthetic_only else 0,
            "checkpoint_namespace": "ephemeral/synthetic/expert-{expert_id}" if synthetic_only else "blocked",
            "shared_backbone_updates": "disabled; requires separate coordinated stage",
        },
        "external_foundation_model_weights_used": False,
        "plan_sha256": hashlib.sha256(json.dumps(
            {"experts": experts, "approvals": approvals_doc, "synthetic_only": synthetic_only},
            sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "note": "Synthetic smoke validates the 20-expert topology with generated tokens only; it does not train the full 2.87B model or any approved corpus.",
    }

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--approvals", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--production", action="store_true")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=80)
    a=p.parse_args()
    plan=build_expert_plan(a.manifest,a.approvals,a.config,
                           synthetic_only=not a.production,
                           requested_steps=a.steps,max_steps=a.max_steps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    if a.production and not plan["production_dispatch_allowed"]:
        raise SystemExit("REFUSED: production training gates are not satisfied")

if __name__ == "__main__":
    main()
