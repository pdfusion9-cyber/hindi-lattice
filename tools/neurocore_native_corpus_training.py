#!/usr/bin/env python3
"""Train the native NeuroCore reference backbone on the repository Hindi facts corpus.

This is a bounded, CPU-only, real-corpus experiment. It does not enable or
dispatch 2.87B production training. The corpus rights/PII review remains
separate and must be completed before any production use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "neurocore_edge"
sys.path.insert(0, str(MODEL_DIR))
from modeling_neurocore_edge import (  # noqa: E402
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)

EXPECTED_CORPUS_BLOB = "3018157a78e6a35877e114b9297f8ae9e16a76ae"


def git_blob_sha(path: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.as_posix()}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not value:
        raise RuntimeError("could not resolve corpus Git blob SHA")
    return value


def load_corpus(path: Path) -> tuple[list[str], list[str], dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 500:
        raise ValueError(f"corpus too small for this experiment: {len(lines)} rows")
    relations: dict[str, int] = {}
    train_rows, valid_rows = [], []
    malformed = 0
    for line in lines:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            malformed += 1
            continue
        relations[parts[1]] = relations.get(parts[1], 0) + 1
        key = hashlib.sha256(line.encode("utf-8")).digest()
        bucket = int.from_bytes(key[:4], "big") % 100
        (valid_rows if bucket < 2 else train_rows).append(line)
    if malformed:
        raise ValueError(f"refusing malformed corpus rows: {malformed}")
    if not train_rows or not valid_rows:
        raise ValueError("deterministic train/validation split is empty")
    summary = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "characters": len(text),
        "total_rows": len(lines),
        "valid_triples": len(lines),
        "malformed_rows": malformed,
        "relation_type_count": len(relations),
        "relation_counts": dict(sorted(relations.items())),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
    }
    return train_rows, valid_rows, summary


def encode_rows(rows: list[str]) -> torch.Tensor:
    # UTF-8 byte IDs are the project's native, dependency-free 256-token pilot tokenizer.
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.long)


def make_batch(stream: torch.Tensor, starts: torch.Tensor, seq_len: int):
    offsets = torch.arange(seq_len, dtype=torch.long)
    positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return stream[positions], stream[positions + 1]


@torch.no_grad()
def evaluate(model, stream: torch.Tensor, *, batch_size: int, seq_len: int,
             batches: int, seed: int, cfg: NeuroCoreEdgeConfig) -> float:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upper = stream.numel() - seq_len - 1
    if upper < 1:
        raise ValueError("validation stream is shorter than one sequence")
    losses = []
    for _ in range(batches):
        starts = torch.randint(0, upper, (batch_size,), generator=generator)
        x, y = make_batch(stream, starts, seq_len)
        output = model(x)
        loss = F.cross_entropy(output["logits"].reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss = loss + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--report", type=Path, default=Path("reports/neurocore-corpus-training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/neurocore-corpus-checkpoint.pt"))
    parser.add_argument("--expected-blob-sha", default=EXPECTED_CORPUS_BLOB)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.sequence_length < 4:
        parser.error("steps, batch-size and sequence-length must be positive (sequence >= 4)")
    cpu_count = os.cpu_count() or 1
    if not 1 <= args.cpu_threads <= min(4, cpu_count):
        parser.error(f"cpu-threads must be in 1..{min(4, cpu_count)}")

    corpus_path = (ROOT / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus file not found: {corpus_path}")
    relative_corpus = corpus_path.relative_to(ROOT).as_posix()
    actual_blob = git_blob_sha(Path(relative_corpus))
    if args.expected_blob_sha and actual_blob != args.expected_blob_sha:
        raise RuntimeError(
            f"corpus blob SHA mismatch: expected {args.expected_blob_sha}, got {actual_blob}"
        )

    train_rows, valid_rows, corpus_summary = load_corpus(corpus_path)
    train_stream = encode_rows(train_rows)
    valid_stream = encode_rows(valid_rows)
    if train_stream.numel() <= args.sequence_length + 1:
        raise ValueError("training corpus has insufficient byte tokens")

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    cfg = NeuroCoreEdgeConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=32,
        attention_heads=4,
        kv_heads=2,
        sliding_window=16,
        recurrent_state_size=8,
        router_aux_loss_coefficient=0.01,
    )
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    resume_from_step = 0
    resume_used = False
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        try:
            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        except TypeError:
            saved = torch.load(resume_path, map_location="cpu")
        if not isinstance(saved, dict):
            raise ValueError("resume checkpoint must be a mapping")
        if saved.get("model_id") != "neurocore_edge_native_reference":
            raise ValueError("resume checkpoint model ID mismatch")
        if saved.get("config") != vars(cfg):
            raise ValueError("resume checkpoint model config mismatch")
        if saved.get("corpus_git_blob_sha1") != actual_blob:
            raise ValueError("resume checkpoint corpus Git blob SHA mismatch")
        if saved.get("corpus_sha256") != corpus_summary["sha256"]:
            raise ValueError("resume checkpoint corpus content SHA256 mismatch")
        provenance = saved.get("provenance", {})
        if provenance.get("external_model_weights_used") is not False:
            raise ValueError("resume checkpoint provenance includes external weights")
        if provenance.get("open_source_model_integrated") is not False:
            raise ValueError("resume checkpoint provenance includes an external model")
        if saved.get("production_training_gate") != "CLOSED":
            raise ValueError("resume checkpoint production gate is not closed")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if not isinstance(saved.get("optimizer_state_dict"), dict):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        resume_from_step = int(saved.get("step", 0))
        if resume_from_step < 1:
            raise ValueError("resume checkpoint step must be positive")
        resume_used = True
        print(
            f"resume_checkpoint={resume_path} resumed_from_step={resume_from_step} "
            f"model_parameters={count_parameters(model)}",
            flush=True,
        )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    initial_train = evaluate(
        model, train_stream, batch_size=args.batch_size, seq_len=args.sequence_length,
        batches=20, seed=args.seed + 100, cfg=cfg,
    )
    initial_valid = evaluate(
        model, valid_stream, batch_size=args.batch_size, seq_len=args.sequence_length,
        batches=20, seed=args.seed + 200, cfg=cfg,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 300)
    upper = train_stream.numel() - args.sequence_length - 1
    losses: list[float] = []
    validation_curve = []
    started = time.perf_counter()
    model.train()
    for local_step in range(1, args.steps + 1):
        step = resume_from_step + local_step
        starts = torch.randint(0, upper, (args.batch_size,), generator=generator)
        x, y = make_batch(train_stream, starts, args.sequence_length)
        output = model(x)
        loss = F.cross_entropy(
            output["logits"].reshape(-1, cfg.vocab_size), y.reshape(-1)
        ) + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if local_step % 100 == 0 or local_step == args.steps:
            val = evaluate(
                model, valid_stream, batch_size=args.batch_size,
                seq_len=args.sequence_length, batches=20,
                seed=args.seed + 200, cfg=cfg,
            )
            validation_curve.append({"step": step, "validation_loss": val})
            print(
                f"step={step}/{args.steps} train_loss={losses[-1]:.4f} "
                f"validation_loss={val:.4f}", flush=True,
            )
    elapsed = max(time.perf_counter() - started, 1e-9)
    final_train = evaluate(
        model, train_stream, batch_size=args.batch_size, seq_len=args.sequence_length,
        batches=20, seed=args.seed + 100, cfg=cfg,
    )
    final_valid = evaluate(
        model, valid_stream, batch_size=args.batch_size, seq_len=args.sequence_length,
        batches=20, seed=args.seed + 200, cfg=cfg,
    )

    report = {
        "status": "NEUROCORE_NATIVE_REAL_CORPUS_EXPERIMENT_PASS",
        "model_id": "neurocore_edge_native_reference",
        "training_started": True,
        "training_scale": "tiny_20_expert_reference_shared_backbone",
        "model_config": vars(cfg),
        "model_parameter_count": count_parameters(model),
        "trainable_parameter_count": trainable_parameters,
        "cpu": {"threads": args.cpu_threads, "logical_cpus": cpu_count, "device": "cpu"},
        "corpus": {
            "path": relative_corpus,
            "git_blob_sha1": actual_blob,
            **corpus_summary,
            "tokenization": "raw_utf8_bytes_0_to_255",
            "external_data_used": False,
            "external_model_weights_used": False,
            "open_source_model_integrated": False,
            "rights_review_status": "UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW",
            "pii_review_status": "NOT_INDEPENDENTLY_REVIEWED",
            "production_approved": False,
        },
        "training": {
            "steps": args.steps,
            "resumed_from_step": resume_from_step,
            "total_steps": resume_from_step + args.steps,
            "resume_used": resume_used,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "tokens_processed": args.steps * args.batch_size * args.sequence_length,
            "total_tokens_processed": (resume_from_step + args.steps) * args.batch_size * args.sequence_length,
            "elapsed_seconds": elapsed,
            "training_tokens_per_second": (
                args.steps * args.batch_size * args.sequence_length / elapsed
            ),
            "initial_train_loss": initial_train,
            "final_train_loss": final_train,
            "initial_validation_loss": initial_valid,
            "final_validation_loss": final_valid,
            "step_loss_first": losses[0],
            "step_loss_last": losses[-1],
            "validation_curve": validation_curve,
        },
        "checkpoint_path": str(args.checkpoint),
        "production_training_gate": "CLOSED",
        "production_2_87b_training_started": False,
        "dataset_approval_registry_changed": False,
        "provenance": {
            "external_model_weights_used": False,
            "open_source_model_integrated": False,
            "external_tokenizer_used": False,
            "external_data_used": False,
            "source_kind": "repository_project_corpus",
        },
        "note": (
            "This is the first real-corpus training experiment on the project's own "
            "scaled NeuroCore reference architecture. It is not a 2.87B production "
            "checkpoint, and the corpus is not approved for production until its "
            "rights, provenance and privacy review is completed."
        ),
    }
    checkpoint = {
        "format_version": 1,
        "model_id": report["model_id"],
        "training_scale": report["training_scale"],
        "config": vars(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": resume_from_step + args.steps,
        "corpus_git_blob_sha1": actual_blob,
        "corpus_sha256": corpus_summary["sha256"],
        "seed": args.seed,
        "provenance": report["provenance"],
        "production_training_gate": "CLOSED",
    }
    atomic_save(checkpoint, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "steps_this_run": args.steps,
        "total_steps": resume_from_step + args.steps,
        "resume_used": resume_used,
        "parameters": report["model_parameter_count"],
        "initial_validation_loss": initial_valid,
        "final_validation_loss": final_valid,
        "tokens_per_second": report["training"]["training_tokens_per_second"],
        "checkpoint": str(args.checkpoint),
        "production_gate": "CLOSED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
