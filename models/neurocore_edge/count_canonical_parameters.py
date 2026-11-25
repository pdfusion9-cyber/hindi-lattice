#!/usr/bin/env python3
"""Count the canonical NeuroCore parameters without allocating model weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM


CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_experts",
    "experts_per_token",
    "expert_intermediate_size",
    "attention_heads",
    "kv_heads",
    "sliding_window",
    "recurrent_state_size",
    "router_aux_loss_coefficient",
    "prelude_layers",
    "core_unique_layers",
    "core_repeats",
    "coda_layers",
    "meta_token_count",
    "rope_theta",
)


def load_model_config(raw: dict[str, Any]) -> NeuroCoreEdgeConfig:
    values = {key: raw[key] for key in CONFIG_FIELDS if key in raw}
    values["num_hidden_layers"] = int(raw.get("effective_layers", raw.get("num_hidden_layers", 24)))
    if "global_attention_layers" in raw:
        values["global_attention_layers"] = tuple(raw["global_attention_layers"])
    return NeuroCoreEdgeConfig(**values)


def count_canonical_parameters(repo_root: Path) -> dict[str, Any]:
    model_dir = repo_root / "models" / "neurocore_edge"
    config_path = model_dir / "config.json"
    budget_path = model_dir / "canonical_parameter_budget.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    model_config = load_model_config(config)

    # Meta tensors retain shape/dtype metadata, not the full parameter storage.
    with torch.device("meta"):
        model = NeuroCoreEdgeForCausalLM(model_config)

    total = sum(parameter.numel() for parameter in model.parameters())
    named = list(model.named_parameters())
    expert = sum(
        parameter.numel()
        for name, parameter in named
        if ".experts." in name or name.startswith("experts.")
    )
    shared = total - expert

    counts = {
        "total_parameters": int(total),
        "shared_parameters": int(shared),
        "expert_parameters": int(expert),
        "unique_parameterized_blocks": (
            len(model.prelude_layers) + len(model.core_layers) + len(model.coda_layers)
        ),
        "effective_depth": len(model._execution_plan()),
        "expert_intermediate_size": model_config.expert_intermediate_size,
        "meta_token_count": model_config.meta_token_count,
        "tied_embedding_and_lm_head": (
            model.embed.weight is model.lm_head.weight
        ),
        "storage_allocated_for_weights": False,
        "budget_matches": (
            total == int(budget["expected_total_parameters"])
            and shared == int(budget["expected_shared_parameters"])
            and expert == int(budget["expected_expert_parameters"])
        ),
        "expected": {
            "total_parameters": int(budget["expected_total_parameters"]),
            "shared_parameters": int(budget["expected_shared_parameters"]),
            "expert_parameters": int(budget["expected_expert_parameters"]),
        },
    }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    report = count_canonical_parameters(args.repo_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["budget_matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
