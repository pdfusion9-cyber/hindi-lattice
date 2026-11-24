#!/usr/bin/env python3
"""Deterministic synthetic-only end-to-end NeuroCore training smoke test.

This is an engineering check, not a pretrained model or a production training
launcher. It generates token sequences in memory, trains a tiny reference model,
and checks checkpoint reload and optimizer-resume equivalence. No external
datasets or user data are downloaded or read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from modeling_neurocore_edge import (
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)

SEED = 7
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0


def build_synthetic_batch(
    *, batch_size: int = 8, sequence_length: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a repeatable next-token cycle; all data stays in memory."""
    if batch_size < 2 or sequence_length < 4:
        raise ValueError("batch_size must be >=2 and sequence_length >=4")
    cycle = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    repeats = (sequence_length + cycle.numel() - 1) // cycle.numel()
    base = cycle.repeat(repeats)[:sequence_length]
    rows = torch.stack(
        [torch.roll(base, shifts=offset) for offset in range(batch_size)]
    )
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def make_model_config(num_experts: int = 4) -> NeuroCoreEdgeConfig:
    """A deliberately tiny config; never reads the 2.87B production config."""
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or not 2 <= num_experts <= 20:
        raise ValueError("num_experts must be an integer from 2 through 20")
    return NeuroCoreEdgeConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_experts=num_experts,
        experts_per_token=2,
        expert_intermediate_size=16,
        attention_heads=4,
        kv_heads=2,
        sliding_window=16,
        recurrent_state_size=8,
        router_aux_loss_coefficient=0.01,
    )


def evaluate_loss(
    model: NeuroCoreEdgeForCausalLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: NeuroCoreEdgeConfig,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        loss = F.cross_entropy(
            output["logits"].reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        ) + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
    value = float(loss.item())
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("evaluation loss is not finite")
    return value


def train_one_step(
    model: NeuroCoreEdgeForCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: NeuroCoreEdgeConfig,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids)
    token_loss = F.cross_entropy(
        output["logits"].reshape(-1, cfg.vocab_size),
        targets.reshape(-1),
    )
    loss = token_loss + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("training loss is not finite")
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("missing or non-finite gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
        raise RuntimeError("gradient norm is not finite and positive")
    optimizer.step()
    return float(loss.item()), float(grad_norm.item())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_experiment(
    report_path: Path,
    *,
    steps: int = 40,
    seed: int = SEED,
    num_experts: int = 4,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Train, checkpoint, reload and verify a tiny model on synthetic tokens."""
    if not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or not 1 <= cpu_threads <= 4:
        raise ValueError("cpu_threads must be an integer from 1 through 4")
    if steps < 10:
        raise ValueError("steps must be >=10 for a meaningful loss-reduction check")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        torch.manual_seed(seed)
        cfg = make_model_config(num_experts=num_experts)
        input_ids, targets = build_synthetic_batch()
        model = NeuroCoreEdgeForCausalLM(cfg).cpu()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        initial_loss = evaluate_loss(model, input_ids, targets, cfg)
        last_train_loss = float("nan")
        last_grad_norm = float("nan")
        training_started = time.perf_counter()
        for _ in range(steps):
            last_train_loss, last_grad_norm = train_one_step(
                model, optimizer, input_ids, targets, cfg
            )
        training_elapsed_seconds = max(time.perf_counter() - training_started, 1e-9)
        training_tokens_per_second = (
            steps * targets.numel() / training_elapsed_seconds
        )
        final_loss = evaluate_loss(model, input_ids, targets, cfg)
        if final_loss >= initial_loss * 0.5:
            raise RuntimeError(
                f"synthetic loss did not fall by at least 50%: "
                f"initial={initial_loss:.6f}, final={final_loss:.6f}"
            )

        model.eval()
        with torch.no_grad():
            reference_logits = model(input_ids)["logits"].clone()

        checkpoint = {
            "format_version": 1,
            "provenance": {
                "kind": "synthetic_only",
                "source": "generated_in_memory",
                "seed": seed,
                "sequence_pattern": "cyclic_token_ids_1_to_8",
                "external_data_used": False,
            },
            "model_config": {
                key: value
                for key, value in vars(cfg).items()
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "completed_steps": steps,
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="neurocore-synthetic-") as tmp:
            checkpoint_path = Path(tmp) / "synthetic_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_hash = sha256_file(checkpoint_path)
            checkpoint_bytes = checkpoint_path.stat().st_size

            loaded = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if loaded.get("provenance", {}).get("external_data_used") is not False:
                raise RuntimeError("checkpoint provenance is not synthetic-only")
            if loaded.get("completed_steps") != steps:
                raise RuntimeError("checkpoint step counter did not round-trip")

            resumed_cfg = NeuroCoreEdgeConfig(**loaded["model_config"])
            resumed_model = NeuroCoreEdgeForCausalLM(resumed_cfg).cpu()
            resumed_model.load_state_dict(loaded["model_state_dict"], strict=True)
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])
            resumed_model.eval()
            with torch.no_grad():
                reloaded_logits = resumed_model(input_ids)["logits"]
            reload_max_abs_diff = float(
                (reference_logits - reloaded_logits).abs().max().item()
            )
            if reload_max_abs_diff > 1e-7:
                raise RuntimeError(
                    f"checkpoint reload mismatch: max_abs_diff={reload_max_abs_diff}"
                )

            # Apply the same next optimizer step to the in-memory original and
            # restored model. Equal outputs/weights prove optimizer-state resume.
            original_next_loss, _ = train_one_step(
                model, optimizer, input_ids, targets, cfg
            )
            resumed_next_loss, _ = train_one_step(
                resumed_model,
                resumed_optimizer,
                input_ids,
                targets,
                resumed_cfg,
            )
            parameter_diffs = [
                (left.detach() - right.detach()).abs().max()
                for left, right in zip(model.parameters(), resumed_model.parameters())
            ]
            resume_max_abs_diff = float(torch.stack(parameter_diffs).max().item())
            resume_loss_diff = abs(original_next_loss - resumed_next_loss)
            if resume_max_abs_diff > 1e-7 or resume_loss_diff > 1e-7:
                raise RuntimeError(
                    "optimizer resume diverged: "
                    f"parameter_diff={resume_max_abs_diff}, "
                    f"loss_diff={resume_loss_diff}"
                )

        report: dict[str, Any] = {
            "status": "SYNTHETIC_SMOKE_PASS",
            "experiment": "neurocore_tiny_end_to_end_training_v1",
            "device": "cpu",
            "cpu_threads": cpu_threads,
            "seed": seed,
            "model_config": checkpoint["model_config"],
            "parameter_count": count_parameters(model),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "training_steps": steps,
            "training_elapsed_seconds": training_elapsed_seconds,
            "training_tokens_per_second": training_tokens_per_second,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": (initial_loss - final_loss) / initial_loss,
            "last_train_loss": last_train_loss,
            "last_gradient_norm": last_grad_norm,
            "checkpoint": {
                "format": "torch_state_dict_plus_optimizer_state",
                "sha256": checkpoint_hash,
                "bytes": checkpoint_bytes,
                "reload_max_abs_logit_difference": reload_max_abs_diff,
                "resume_steps_verified": steps + 1,
                "resume_max_abs_parameter_difference": resume_max_abs_diff,
                "resume_next_loss_difference": resume_loss_diff,
            },
            "provenance": {
                "kind": "synthetic_only",
                "source": "generated_in_memory",
                "external_data_used": False,
                "huggingface_datasets_used": [],
                "user_data_used": False,
            },
            "production_training_gate": "CLOSED",
            "production_dataset_approvals_changed": False,
            "note": (
                "This proves a small reference-model training/checkpoint/resume "
                "path only. It is not a production-model convergence, throughput, "
                "accuracy, deployment, or dataset-rights clearance result."
            ),
        }
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
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    report = run_experiment(
        args.report_path, steps=args.steps, seed=args.seed,
        num_experts=args.num_experts, cpu_threads=args.cpu_threads,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
