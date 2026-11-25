#!/usr/bin/env python3
"""Audit the canonical NeuroCore contract without allocating the 3B model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".gguf"}
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "spiece.model"
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _files_with_suffix(root: Path, suffixes: set[str]) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def audit_repository(root: Path) -> dict[str, Any]:
    model_dir = root / "models" / "neurocore_edge"
    config_path = model_dir / "config.json"
    budget_path = model_dir / "canonical_parameter_budget.json"
    source_path = model_dir / "modeling_neurocore_edge.py"

    missing_inputs = [
        str(path.relative_to(root))
        for path in (config_path, budget_path, source_path)
        if not path.is_file()
    ]
    if missing_inputs:
        return {
            "status": "BLOCKED",
            "can_start_training": False,
            "blockers": [f"Required audit input missing: {name}" for name in missing_inputs],
        }

    config = _load_json(config_path)
    budget = _load_json(budget_path)
    source = source_path.read_text(encoding="utf-8")

    hidden = int(config["hidden_size"])
    experts = int(config["num_experts"])
    intermediate = int(config["expert_intermediate_size"])
    prelude = int(config.get("prelude_layers", 0))
    core_unique = int(config.get("core_unique_layers", 0))
    repeats = int(config.get("core_repeats", 0))
    coda = int(config.get("coda_layers", 0))
    unique_blocks = prelude + core_unique + coda
    effective_depth = prelude + core_unique * repeats + coda

    expected_unique = int(budget.get("unique_parameterized_blocks", unique_blocks))
    expected_expert_params = int(budget["expected_expert_parameters"])
    expected_shared_params = int(budget["expected_shared_parameters"])
    expected_total = int(budget["expected_total_parameters"])
    denominator = unique_blocks * experts * 3 * hidden
    derived_intermediate = expected_expert_params // denominator if denominator else None
    exact_division = bool(denominator and expected_expert_params % denominator == 0)
    expert_params_from_config = unique_blocks * experts * 3 * hidden * intermediate

    blockers: list[str] = []
    if config.get("architecture_version") != "0.3.2-canonical":
        blockers.append("config architecture_version is not 0.3.2-canonical")
    if (prelude, core_unique, repeats, coda) != (2, 4, 5, 2):
        blockers.append("sandwich topology must be 2 prelude + 4 unique core blocks x 5 repeats + 2 coda")
    if effective_depth != int(config.get("effective_layers", -1)):
        blockers.append("effective depth does not match config.effective_layers")
    if unique_blocks != int(config.get("unique_layers", -1)):
        blockers.append("unique block count does not match config.unique_layers")
    if unique_blocks != expected_unique:
        blockers.append("unique block count does not match the parameter-budget contract")
    if not exact_division:
        blockers.append("audited expert parameter budget does not imply an integer expert width")
    if derived_intermediate != intermediate:
        blockers.append(f"expert width mismatch: config={intermediate}, budget-derived={derived_intermediate}")
    if expert_params_from_config != expected_expert_params:
        blockers.append(
            f"expert parameter mismatch: config implies {expert_params_from_config:,}, "
            f"budget records {expected_expert_params:,}"
        )
    if expected_shared_params + expected_expert_params != expected_total:
        blockers.append("shared + expert parameter budgets do not sum to total budget")
    if expected_total != int(config.get("expected_total_parameters", -1)):
        blockers.append("config total parameter target differs from budget contract")

    has_sandwich_modules = all(marker in source for marker in ("prelude_layers", "core_layers", "coda_layers"))
    has_legacy_independent_stack = (
        "self.layers = nn.ModuleList" in source and "range(cfg.num_hidden_layers)" in source
    )
    has_gqa = "class GQARotaryAttention" in source
    has_meta_bank = "meta_token_bank" in source or "meta_tokens = nn.Parameter" in source

    if not has_sandwich_modules or has_legacy_independent_stack:
        blockers.append("model source does not implement the 8 unique sandwich blocks reused over 24 effective operations")
    if not has_gqa:
        blockers.append("model source has no GQA + RoPE attention module")
    if not has_meta_bank:
        blockers.append("model source has no explicit 128-token meta-token bank")

    tokenizer_assets = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name in TOKENIZER_FILES
    )
    if not tokenizer_assets:
        blockers.append("no NeuroCore tokenizer asset is present in the repository")

    weight_assets = _files_with_suffix(root, WEIGHT_SUFFIXES)
    if not weight_assets:
        blockers.append("no trained NeuroCore base checkpoint is present in the repository")

    if config.get("status") != "TRAINED_BASE_CHECKPOINT_AVAILABLE":
        blockers.append("model status does not certify a completed trained base checkpoint")
    training_enabled = bool(config.get("training", {}).get("enabled", False))
    if training_enabled:
        blockers.append("training.enabled must remain false until architecture, tokenizer, data and hardware gates pass")

    return {
        "status": "READY_FOR_NEXT_REVIEW" if not blockers else "BLOCKED",
        "can_start_training": False,
        "architecture": {
            "version": config.get("architecture_version"),
            "prelude_layers": prelude,
            "core_unique_layers": core_unique,
            "core_repeats": repeats,
            "coda_layers": coda,
            "effective_depth": effective_depth,
            "unique_parameterized_blocks": unique_blocks,
            "sandwich_modules_present": has_sandwich_modules,
            "legacy_independent_stack_present": has_legacy_independent_stack,
            "gqa_rope_present": has_gqa,
            "meta_token_bank_present": has_meta_bank,
        },
        "parameter_budget": {
            "expected_total": expected_total,
            "expected_shared": expected_shared_params,
            "expected_expert": expected_expert_params,
            "config_expert_intermediate_size": intermediate,
            "budget_derived_expert_intermediate_size": derived_intermediate,
            "expert_parameters_implied_by_config": expert_params_from_config,
            "expert_parameter_delta": expert_params_from_config - expected_expert_params,
        },
        "assets": {"tokenizer_assets": tokenizer_assets, "weight_assets": weight_assets},
        "training_enabled_in_config": training_enabled,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--strict", action="store_true", help="return exit code 2 when blockers remain")
    args = parser.parse_args()
    report = audit_repository(args.repo_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report.get("status") == "BLOCKED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
