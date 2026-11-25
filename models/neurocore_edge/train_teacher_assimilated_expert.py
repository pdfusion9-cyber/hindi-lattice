#!/usr/bin/env python3
"""Train exactly one native NeuroCore expert from a frozen HF causal-LM teacher.

Input JSONL schema: {"prompt": "..."} (one prompt per line).
All source datasets/checkpoints must already be approved and locally staged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM
from teacher_weight_assimilation import (
    FrozenTeacher,
    TeacherSpec,
    build_native_neurocore_expert_distiller,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"line {line_no}: expected non-empty prompt")
            prompts.append(prompt.strip())
    if not prompts:
        raise ValueError("JSONL contains no prompts")
    return prompts


def load_student(base_path: Path, device: torch.device):
    raw = torch.load(base_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
        raise ValueError("base checkpoint must contain a config dictionary")
    cfg = NeuroCoreEdgeConfig(**raw["config"])
    if getattr(cfg, "num_experts", None) != 20:
        raise ValueError("base checkpoint must configure exactly 20 experts")
    model = NeuroCoreEdgeForCausalLM(cfg)
    state = raw.get("model_state_dict", raw.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("base checkpoint is missing model_state_dict/state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--student-tokenizer-revision", required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.expert_id < 20:
        raise ValueError("expert-id must be between 0 and 19")
    if args.steps < 1 or args.save_every < 1:
        raise ValueError("steps and save-every must be positive")
    for path in (args.base_checkpoint, args.prompts_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = read_prompts(args.prompts_jsonl)
    base_hash = sha256_file(args.base_checkpoint)
    dataset_hash = sha256_file(args.prompts_jsonl)

    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_repo,
        revision=args.teacher_revision,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_repo,
        revision=args.teacher_revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    spec = TeacherSpec(
        repo_id=args.teacher_repo,
        revision=args.teacher_revision,
        expert_id=args.expert_id,
    )
    teacher_model.to(device)
    teacher = FrozenTeacher(
        spec=spec,
        model=teacher_model,
        tokenizer=teacher_tokenizer,
    )
    student = load_student(args.base_checkpoint, device)
    if student.config.vocab_size != len(student_tokenizer):
        raise ValueError("student tokenizer vocabulary size must match NeuroCore config.vocab_size")
    distiller = build_native_neurocore_expert_distiller(
        teacher=teacher,
        student_model=student,
        student_tokenizer=student_tokenizer,
        expert_id=args.expert_id,
        learning_rate=args.learning_rate,
    )
    tokenizer_hash = hashlib.sha256(
        (args.student_tokenizer + "@" + args.student_tokenizer_revision).encode()
    ).hexdigest()
    if args.resume:
        distiller.load_checkpoint(
            str(args.output),
            base_fingerprint=base_hash,
            tokenizer_fingerprint=tokenizer_hash,
            dataset_fingerprint=dataset_hash,
        )

    remaining = args.steps
    while remaining > 0:
        batch = min(remaining, args.save_every)
        report = distiller.train(
            prompts,
            steps=batch,
            max_new_tokens=args.max_new_tokens,
        )
        remaining -= batch
        distiller.save_checkpoint(
            str(args.output),
            base_fingerprint=base_hash,
            tokenizer_fingerprint=tokenizer_hash,
            dataset_fingerprint=dataset_hash,
        )
        print(json.dumps({
            "expert_id": args.expert_id,
            "report": report,
            "steps_remaining_this_invocation": remaining,
            "checkpoint": str(args.output),
            "data_cursor": distiller.data_cursor,
            "device": str(device),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
