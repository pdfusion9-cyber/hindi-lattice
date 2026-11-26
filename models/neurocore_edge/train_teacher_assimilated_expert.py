#!/usr/bin/env python3
"""Run one frozen-teacher -> one fixed native NeuroCore expert pilot.

The base checkpoint, tokenizer and prompt file must be staged privately.
This launcher refuses untrained/random NeuroCore bases. Teacher revisions and
student tokenizer revisions must be immutable 40-character commit SHAs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM
from teacher_weight_assimilation import (
    FrozenTeacher,
    TeacherSpec,
    build_native_neurocore_expert_distiller,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


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
                raise ValueError(f"line {line_no}: expected a non-empty prompt field")
            prompts.append(prompt.strip())
    if not prompts:
        raise ValueError("prompt JSONL contains no prompts")
    return prompts


def load_student(base_path: Path, device: torch.device):
    try:
        raw = torch.load(base_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(base_path, map_location="cpu")
    if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
        raise ValueError("base checkpoint must contain a config dictionary")
    if raw.get("status") != "TRAINED_BASE_CHECKPOINT_AVAILABLE":
        raise ValueError("refusing to distil into a random or uncertified NeuroCore base")
    if raw.get("architecture_version") != "0.3.2-canonical":
        raise ValueError("base checkpoint is not canonical NeuroCore v0.3.2")
    cfg = NeuroCoreEdgeConfig(**raw["config"])
    if cfg.num_experts != 20:
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
    parser.add_argument("--teacher-device-map", default="auto")
    parser.add_argument("--teacher-dtype", default="auto")
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
    parser.add_argument(
        "--confirm-authorized-prompts",
        action="store_true",
        help="confirm the prompt set is project-authorized, non-sensitive and rights-reviewed",
    )
    args = parser.parse_args()

    if args.expert_id not in range(20):
        raise ValueError("expert-id must be between 0 and 19")
    if not _COMMIT_RE.fullmatch(args.teacher_revision):
        raise ValueError("teacher-revision must be a pinned 40-character commit SHA")
    if not _COMMIT_RE.fullmatch(args.student_tokenizer_revision):
        raise ValueError("student-tokenizer-revision must be a pinned 40-character commit SHA")
    if args.steps < 1 or args.save_every < 1 or args.max_new_tokens < 1:
        raise ValueError("steps, save-every and max-new-tokens must be positive")
    if not args.confirm_authorized_prompts:
        raise ValueError("pass --confirm-authorized-prompts after reviewing prompt rights/privacy")
    for path in (args.base_checkpoint, args.prompts_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = read_prompts(args.prompts_jsonl)
    base_hash = sha256_file(args.base_checkpoint)
    prompts_hash = sha256_file(args.prompts_jsonl)

    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        revision=args.student_tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    student = load_student(args.base_checkpoint, device)
    if student.config.vocab_size != len(student_tokenizer):
        raise ValueError("student tokenizer vocabulary size must match NeuroCore config.vocab_size")

    spec = TeacherSpec(
        repo_id=args.teacher_repo,
        revision=args.teacher_revision,
        expert_id=args.expert_id,
        trust_remote_code=args.trust_remote_code,
    )
    teacher = FrozenTeacher.from_pretrained(
        spec,
        device_map=args.teacher_device_map,
        torch_dtype=args.teacher_dtype,
    )

    distiller = build_native_neurocore_expert_distiller(
        teacher=teacher,
        student_model=student,
        student_tokenizer=student_tokenizer,
        expert_id=args.expert_id,
        learning_rate=args.learning_rate,
    )
    tokenizer_hash = hashlib.sha256(
        (args.student_tokenizer + "@" + args.student_tokenizer_revision).encode("utf-8")
    ).hexdigest()

    if args.resume:
        distiller.load_checkpoint(
            str(args.output),
            base_fingerprint=base_hash,
            tokenizer_fingerprint=tokenizer_hash,
            dataset_fingerprint=prompts_hash,
        )

    remaining = args.steps
    while remaining > 0:
        chunk_steps = min(remaining, args.save_every)
        report = distiller.train(
            prompts,
            steps=chunk_steps,
            max_new_tokens=args.max_new_tokens,
        )
        remaining -= chunk_steps
        distiller.save_checkpoint(
            str(args.output),
            base_fingerprint=base_hash,
            tokenizer_fingerprint=tokenizer_hash,
            dataset_fingerprint=prompts_hash,
        )
        print(json.dumps({
            "expert_id": args.expert_id,
            "report": report,
            "steps_remaining_this_invocation": remaining,
            "checkpoint": str(args.output),
            "data_cursor": distiller.data_cursor,
            "student_device": str(device),
            "teacher_device_map": args.teacher_device_map,
            "teacher_frozen": True,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
