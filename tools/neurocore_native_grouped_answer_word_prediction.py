#!/usr/bin/env python3
"""Train native NeuroCore to predict complete answer lists for each fact query.

Each unique (subject, relation) is one example. All distinct objects for that
pair are serialized together with <ALT> delimiters. Splitting happens by pair,
so no member of a multi-answer group leaks across train/validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "neurocore_edge"))
from modeling_neurocore_edge import (
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)
from training_micro_agents import MicroAgentTrainingController
from cpu_training_agents import detect_cpu_profile

UNK, BOS, EOS, SEP, PAD, ALT = range(6)
SPECIAL = ["<UNK>", "<BOS>", "<EOS>", "<SEP>", "<PAD>", "<ALT>"]
EXPECTED_CORPUS_BLOB = "3018157a78e6a35877e114b9297f8ae9e16a76ae"


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def load_groups(path: Path):
    raw = path.read_bytes()
    source_rows = []
    grouped = {}
    token_counts = defaultdict(int)
    markup_rows = []

    for line_number, raw_line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [unicodedata.normalize("NFC", part.strip()) for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed triple at line {line_number}: {line!r}")
        subject, relation, obj = parts
        source_rows.append((subject, relation, obj))
        key = subject + "\u0000" + relation
        if key not in grouped:
            grouped[key] = {"subject": subject, "relation": relation, "objects": []}
        if norm(obj) not in {norm(x) for x in grouped[key]["objects"]}:
            grouped[key]["objects"].append(obj)
        if any(marker in line.lower() for marker in ("&lt;", "&gt;", "&amp;", "<ref", "</ref")):
            markup_rows.append({"line_number": line_number, "text": line})
        for field in (subject, relation, obj):
            for token in field.split():
                token_counts[token] += 1

    groups = list(grouped.values())
    for group in groups:
        group["objects"] = sorted(group["objects"], key=lambda x: (norm(x), x))
    if not groups:
        raise ValueError("no valid subject-relation groups")
    return raw, source_rows, groups, token_counts, markup_rows


def audit_groups(source_rows, groups, markup_rows):
    by_relation = defaultdict(lambda: {"groups": 0, "rows": 0, "multi_answer_groups": 0})
    for group in groups:
        rel = group["relation"]
        by_relation[rel]["groups"] += 1
        by_relation[rel]["rows"] += len(group["objects"])
        if len(group["objects"]) > 1:
            by_relation[rel]["multi_answer_groups"] += 1

    multi = [g for g in groups if len(g["objects"]) > 1]
    rows_in_multi = sum(len(g["objects"]) for g in multi)
    return {
        "source_rows": len(source_rows),
        "unique_subject_relation_groups": len(groups),
        "single_answer_groups": len(groups) - len(multi),
        "multi_answer_groups": len(multi),
        "rows_in_multi_answer_groups": rows_in_multi,
        "relation_count": len(by_relation),
        "relation_stats": [
            {
                "relation": rel,
                **values,
                "multi_answer_group_fraction": values["multi_answer_groups"] / values["groups"],
            }
            for rel, values in sorted(by_relation.items(), key=lambda item: (-item[1]["groups"], item[0]))
        ],
        "markup_candidate_rows": markup_rows,
        "largest_multi_answer_groups": [
            {"subject": g["subject"], "relation": g["relation"], "objects": g["objects"]}
            for g in sorted(multi, key=lambda x: (-len(x["objects"]), x["subject"]))[:40]
        ],
        "review_guidance": [
            "Multiple objects are preserved as one answer list; this does not certify every object as factually correct.",
            "Capital names, legal-language lists, dates and locations may be time-dependent or have different definitions.",
            "Markup-like rows are flagged for human review, not silently removed.",
            "Source ownership, license and privacy remain unverified; this corpus is for bounded internal experiments only.",
        ],
    }


def split_by_group(groups, validation_percent=5):
    train, valid = [], []
    for group in groups:
        key = (group["subject"] + "\u0000" + group["relation"]).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 100
        (valid if bucket < validation_percent else train).append(group)
    if not train or not valid:
        raise ValueError("deterministic subject-relation split produced an empty partition")
    train_keys = {(g["subject"], g["relation"]) for g in train}
    valid_keys = {(g["subject"], g["relation"]) for g in valid}
    if train_keys & valid_keys:
        raise RuntimeError("subject-relation leakage across partitions")
    return train, valid


def make_vocab(token_counts):
    words = sorted(token_counts, key=lambda word: (-token_counts[word], word))
    id_to_word = SPECIAL + words
    if len(set(id_to_word)) != len(id_to_word):
        raise ValueError("corpus token collides with a reserved control token")
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


def encode_group(group, word_to_id):
    prompt = [BOS] + [word_to_id.get(w, UNK) for w in group["subject"].split()]
    prompt += [SEP] + [word_to_id.get(w, UNK) for w in group["relation"].split()] + [SEP]
    target = []
    for i, answer in enumerate(group["objects"]):
        if i:
            target.append(ALT)
        target.extend(word_to_id.get(w, UNK) for w in answer.split())
    target.append(EOS)
    return prompt, target


def collate(examples):
    max_x = max(len(prompt) + len(target) - 1 for prompt, target in examples)
    xs, ys, loss_masks, word_masks = [], [], [], []
    for prompt, target in examples:
        sequence = prompt + target
        x, y = sequence[:-1], sequence[1:]
        start = len(prompt) - 1
        loss_mask = [False] * start + [True] * len(target)
        word_mask = [False] * start + [token >= len(SPECIAL) for token in target]
        padding = max_x - len(x)
        xs.append(x + [PAD] * padding)
        ys.append(y + [-100] * padding)
        loss_masks.append(loss_mask + [False] * padding)
        word_masks.append(word_mask + [False] * padding)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(loss_masks, dtype=torch.bool),
        torch.tensor(word_masks, dtype=torch.bool),
    )


@torch.inference_mode()
def generate_answer_list(model, prompt, max_new_tokens=64):
    sequence = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    answers, current = [], []
    for _ in range(max_new_tokens):
        token = int(model(sequence)["logits"][0, -1].argmax().item())
        if token == EOS:
            if current:
                answers.append(" ".join(current))
            break
        if token == ALT:
            if current:
                answers.append(" ".join(current))
                current = []
        elif token in (BOS, SEP, PAD):
            break
        elif token == UNK:
            current.append("<UNK>")
        else:
            # Word IDs start after the reserved control-token range.
            current.append(model._word_vocabulary[token] if hasattr(model, "_word_vocabulary") else str(token))
        sequence = torch.cat([sequence, torch.tensor([[token]], dtype=torch.long)], dim=1)
    else:
        if current:
            answers.append(" ".join(current))
    # A model-local decoder mapping is attached by main() for readable output.
    return sorted(set(answers), key=norm)


def answer_list_for_ids(model, prompt, id_to_word, max_new_tokens=64):
    sequence = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    answers, current = [], []
    for _ in range(max_new_tokens):
        token = int(model(sequence)["logits"][0, -1].argmax().item())
        if token == EOS:
            if current:
                answers.append(" ".join(current))
            break
        if token == ALT:
            if current:
                answers.append(" ".join(current))
                current = []
        elif token in (BOS, SEP, PAD):
            break
        else:
            current.append(id_to_word[token])
        sequence = torch.cat([sequence, torch.tensor([[token]], dtype=torch.long)], dim=1)
    else:
        if current:
            answers.append(" ".join(current))
    return sorted(set(answers), key=norm)


@torch.no_grad()
def evaluate(model, groups, word_to_id, id_to_word, *, batch_size, generation_limit=96):
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    top1_correct = top5_correct = word_tokens = 0

    for start in range(0, len(groups), batch_size):
        batch = groups[start:start + batch_size]
        examples = [encode_group(group, word_to_id) for group in batch]
        x, y, loss_mask, word_mask = collate(examples)
        logits = model(x)["logits"]

        lm_logits, lm_targets = logits[loss_mask], y[loss_mask]
        total_loss += float(F.cross_entropy(lm_logits, lm_targets, reduction="sum").item())
        total_target_tokens += lm_targets.numel()

        w_logits, w_targets = logits[word_mask], y[word_mask]
        if w_targets.numel():
            top1_correct += int((w_logits.argmax(dim=-1) == w_targets).sum().item())
            top5_ids = w_logits.topk(min(5, w_logits.shape[-1]), dim=-1).indices
            top5_correct += int((top5_ids == w_targets.unsqueeze(-1)).any(dim=-1).sum().item())
            word_tokens += w_targets.numel()

    # Whole-list generation is measured on a fixed, deterministic validation subset.
    sample_count = min(generation_limit, len(groups))
    indices = sorted({
        round(i * (len(groups) - 1) / max(sample_count - 1, 1))
        for i in range(sample_count)
    })
    exact = 0
    true_positive = predicted_total = expected_total = 0
    examples_out = []
    for index in indices:
        group = groups[index]
        prompt, _ = encode_group(group, word_to_id)
        predicted = answer_list_for_ids(model, prompt, id_to_word)
        expected = sorted(set(group["objects"]), key=norm)
        predicted_norm = {norm(x) for x in predicted}
        expected_norm = {norm(x) for x in expected}
        overlap = len(predicted_norm & expected_norm)
        true_positive += overlap
        predicted_total += len(predicted_norm)
        expected_total += len(expected_norm)
        exact += int(predicted_norm == expected_norm)
        examples_out.append({
            "subject": group["subject"],
            "relation": group["relation"],
            "expected_answers": expected,
            "predicted_answers": predicted,
            "exact_answer_set_match": predicted_norm == expected_norm,
        })

    model.train()
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / expected_total if expected_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sequence_loss_per_target_token": total_loss / max(total_target_tokens, 1),
        "next_word_top1_accuracy": top1_correct / max(word_tokens, 1),
        "next_word_top5_accuracy": top5_correct / max(word_tokens, 1),
        "next_word_tokens_evaluated": word_tokens,
        "answer_set_exact_match": exact / max(len(indices), 1),
        "answer_set_samples": len(indices),
        "answer_set_precision": precision,
        "answer_set_recall": recall,
        "answer_set_f1": f1,
        "answer_set_true_positives": true_positive,
        "answer_set_predicted_count": predicted_total,
        "answer_set_expected_count": expected_total,
        "generation_examples": examples_out,
    }


def atomic_torch_save(payload, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the last save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def validate_resume_checkpoint(
    checkpoint,
    *,
    model_config: dict,
    training_fingerprint: dict,
    corpus_blob: str,
    corpus_sha256: str,
    tokenizer_sha256: str,
    target_steps: int,
) -> int:
    """Fail closed when a resume checkpoint belongs to another run/corpus/model."""
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must contain a dictionary")
    required = {
        "model_state_dict", "optimizer_state_dict", "step",
        "python_rng_state", "torch_rng_state", "validation_curve",
        "initial_validation", "elapsed_seconds",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError("resume checkpoint is missing fields: " + ", ".join(missing))
    if checkpoint.get("model_id") != "neurocore_edge_native_grouped_answer_word_v1":
        raise ValueError("resume checkpoint model ID mismatch")
    if checkpoint.get("config") != model_config:
        raise ValueError("resume checkpoint model configuration mismatch")
    if checkpoint.get("training_fingerprint") != training_fingerprint:
        raise ValueError("resume checkpoint training configuration mismatch")
    if checkpoint.get("corpus_git_blob_sha1") != corpus_blob:
        raise ValueError("resume checkpoint corpus Git blob mismatch")
    if checkpoint.get("corpus_sha256") != corpus_sha256:
        raise ValueError("resume checkpoint corpus SHA-256 mismatch")
    if checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("resume checkpoint tokenizer fingerprint mismatch")
    if checkpoint.get("production_training_gate") != "CLOSED":
        raise ValueError("resume checkpoint production gate is not closed")
    if checkpoint.get("internal_experiment_gate") != "MICRO_AGENT_APPROVED_BOUNDED_INTERNAL_ONLY":
        raise ValueError("resume checkpoint is not an approved internal-only experiment")
    if checkpoint.get("public_artifacts_created") is not False:
        raise ValueError("resume checkpoint is marked as public")
    step = checkpoint.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("resume checkpoint step is invalid")
    if step >= target_steps:
        raise ValueError("resume checkpoint step must be below the requested total steps")
    if not isinstance(checkpoint.get("python_rng_state"), (tuple, list)):
        raise ValueError("resume checkpoint Python RNG state is invalid")
    if not isinstance(checkpoint.get("torch_rng_state"), torch.Tensor):
        raise ValueError("resume checkpoint Torch RNG state is invalid")
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--generation-limit", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="atomically save resumable state every N optimizer steps")
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="resume locally from a trusted internal checkpoint; --steps is the total target")
    parser.add_argument("--report", type=Path, default=Path("reports/neurocore-grouped-answer-word-prediction-5000.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/neurocore-grouped-answer-word-prediction-5000.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/neurocore-hindi-grouped-word-tokenizer.json"))
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.validation_percent <= 20:
        parser.error("steps/batch-size must be positive and validation-percent must be in 1..20")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be a positive integer")
    if not 1 <= args.cpu_threads <= min(4, os.cpu_count() or 1):
        parser.error("cpu-threads must be between 1 and min(4, available CPUs)")

    corpus = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    relative = corpus.resolve().relative_to(ROOT).as_posix()
    blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{relative}"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if blob != EXPECTED_CORPUS_BLOB:
        raise RuntimeError(f"unexpected corpus Git blob SHA: {blob}")

    raw, source_rows, groups, token_counts, markup_rows = load_groups(corpus)
    train_groups, validation_groups = split_by_group(groups, args.validation_percent)
    word_to_id, id_to_word = make_vocab(token_counts)
    vocab_size = len(id_to_word)
    if vocab_size > 20000:
        raise ValueError(f"unexpected vocabulary size: {vocab_size}")


    tokenizer = {
        "format": "neurocore_whitespace_word_grouped_answer_v1",
        "special_tokens": {
            "unk": UNK, "bos": BOS, "eos": EOS, "field_separator": SEP,
            "pad": PAD, "alternative_answer_separator": ALT,
        },
        "id_to_word": id_to_word,
        "normalization": "Unicode NFC; split on whitespace; object answers sorted deterministically; <ALT> separates distinct answers",
    }
    tokenizer_bytes = json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()

    relation_groups = defaultdict(list)
    for group in train_groups:
        relation_groups[group["relation"]].append(group)
    relation_names = sorted(relation_groups)
    # Square-root weighting tempers dominant lexical relations without
    # massively oversampling relations that have only a handful of examples.
    relation_weights = [len(relation_groups[name]) ** 0.5 for name in relation_names]

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)
    cfg = NeuroCoreEdgeConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=3,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=64,
        attention_heads=4,
        kv_heads=2,
        sliding_window=64,
        recurrent_state_size=16,
        router_aux_loss_coefficient=0.01,
    )
    # Count parameters on the meta device so the resource gate runs before
    # allocating real model weights or optimizer state.
    with torch.device("meta"):
        model_for_budget = NeuroCoreEdgeForCausalLM(cfg)
    model_parameter_count = count_parameters(model_for_budget)
    del model_for_budget

    profile = detect_cpu_profile()
    selected_threads = min(args.cpu_threads, profile.effective_cpus, 4)
    rng_state_before_agents = torch.get_rng_state()
    controller = MicroAgentTrainingController()
    micro_agent_plan = controller.create_plan(
        requested_steps=args.steps,
        max_steps=args.steps,
        source_kind="bounded_internal_corpus",
        external_data_used=False,
        user_data_used=False,
        dataset_ids=(),
        external_weights_used=False,
        checkpoint_mode="ephemeral_internal_only",
        model_under_test="neurocore_edge_native_grouped_answer_word_v1",
        internal_only=True,
        public_artifacts=False,
        corpus_rights_status="UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW",
        privacy_status="NOT_FORMALLY_CLEARED",
        model_parameter_count=model_parameter_count,
        available_memory_bytes=profile.available_memory_bytes,
        cpu_threads=selected_threads,
    )
    # The CPU proxy calibration uses its own seed; restore the model/training
    # random stream so --seed is reproducible across fresh and resumed runs.
    torch.set_rng_state(rng_state_before_agents)
    training_steps = micro_agent_plan["allocated_steps"]
    torch.set_num_threads(micro_agent_plan["cpu_threads"])
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    training_fingerprint = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "validation_percent": args.validation_percent,
        "generation_limit": args.generation_limit,
        "cpu_threads": micro_agent_plan["cpu_threads"],
        "checkpoint_every": args.checkpoint_every,
        "model_id": "neurocore_edge_native_grouped_answer_word_v1",
    }
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    start_step = 0
    elapsed_before_resume = 0.0
    validation_curve = []
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint if args.resume_checkpoint.is_absolute() else ROOT / args.resume_checkpoint
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        # Only load trusted, locally-held checkpoint tensors and primitive metadata.
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        start_step = validate_resume_checkpoint(
            saved,
            model_config=vars(cfg),
            training_fingerprint=training_fingerprint,
            corpus_blob=blob,
            corpus_sha256=hashlib.sha256(raw).hexdigest(),
            tokenizer_sha256=tokenizer_sha,
            target_steps=training_steps,
        )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        rng.setstate(tuple(saved["python_rng_state"]))
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        validation_curve = list(saved["validation_curve"])
        initial = saved["initial_validation"]
        elapsed_before_resume = float(saved["elapsed_seconds"])
    else:
        initial = evaluate(
            model, validation_groups, word_to_id, id_to_word,
            batch_size=args.batch_size, generation_limit=args.generation_limit,
        )

    segment_started = time.perf_counter()
    model.train()

    def save_training_checkpoint(current_step: int, elapsed_seconds: float) -> None:
        payload = {
            "format_version": 2,
            "model_id": "neurocore_edge_native_grouped_answer_word_v1",
            "config": vars(cfg),
            "training_fingerprint": training_fingerprint,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": current_step,
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "validation_curve": validation_curve,
            "initial_validation": initial,
            "elapsed_seconds": elapsed_seconds,
            "corpus_git_blob_sha1": blob,
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_format": tokenizer["format"],
            "provenance": {
                "external_model_weights_used": False,
                "open_source_model_integrated": False,
                "external_tokenizer_used": False,
                "external_data_used": False,
                "initialization": "native_random_initialization",
            },
            "production_training_gate": "CLOSED",
            "internal_experiment_gate": "MICRO_AGENT_APPROVED_BOUNDED_INTERNAL_ONLY",
            "public_artifacts_created": False,
        }
        atomic_torch_save(payload, checkpoint_path)

    for step in range(start_step + 1, training_steps + 1):
        batch = []
        for _ in range(args.batch_size):
            rel = rng.choices(relation_names, weights=relation_weights, k=1)[0]
            batch.append(rng.choice(relation_groups[rel]))
        examples = [encode_group(group, word_to_id) for group in batch]
        x, y, loss_mask, word_mask = collate(examples)
        output = model(x)
        logits = output["logits"]
        selected_logits, selected_targets = logits[loss_mask], y[loss_mask]
        loss = F.cross_entropy(selected_logits, selected_targets)
        loss = loss + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == training_steps:
            metrics = evaluate(
                model, validation_groups, word_to_id, id_to_word,
                batch_size=args.batch_size, generation_limit=args.generation_limit,
            )
            validation_curve.append({
                "step": step,
                "sequence_loss_per_target_token": metrics["sequence_loss_per_target_token"],
                "next_word_top1_accuracy": metrics["next_word_top1_accuracy"],
                "next_word_top5_accuracy": metrics["next_word_top5_accuracy"],
                "answer_set_exact_match": metrics["answer_set_exact_match"],
                "answer_set_f1": metrics["answer_set_f1"],
            })
            print(
                f"grouped_word_step={step}/{training_steps} "
                f"loss={metrics['sequence_loss_per_target_token']:.4f} "
                f"word_top1={metrics['next_word_top1_accuracy']:.4f} "
                f"word_top5={metrics['next_word_top5_accuracy']:.4f} "
                f"answer_set_exact={metrics['answer_set_exact_match']:.4f} "
                f"answer_set_f1={metrics['answer_set_f1']:.4f}",
                flush=True,
            )

        if step % args.checkpoint_every == 0 or step == training_steps:
            elapsed_so_far = elapsed_before_resume + (time.perf_counter() - segment_started)
            save_training_checkpoint(step, elapsed_so_far)

    elapsed = max(elapsed_before_resume + time.perf_counter() - segment_started, 1e-9)
    final = evaluate(
        model, validation_groups, word_to_id, id_to_word,
        batch_size=args.batch_size, generation_limit=args.generation_limit,
    )
    data_audit = audit_groups(source_rows, groups, markup_rows)
    report = {
        "status": "NEUROCORE_NATIVE_GROUPED_ANSWER_WORD_PREDICTION_PASS",
        "objective": "subject_relation_to_complete_grouped_answer_list",
        "model_id": "neurocore_edge_native_grouped_answer_word_v1",
        "model_parameter_count": count_parameters(model),
        "vocabulary": {
            "size": vocab_size,
            "unique_corpus_words": len(token_counts),
            "tokenizer_format": tokenizer["format"],
            "tokenizer_sha256": tokenizer_sha,
        },
        "corpus": {
            "path": relative,
            "git_blob_sha1": blob,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_rights_status": "UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW",
            "privacy_status": "NOT_FORMALLY_CLEARED",
            "production_approved": False,
        },
        "data_audit": data_audit,
        "split": {
            "method": "deterministic SHA-256 bucket of subject plus relation; split at group level",
            "validation_percent": args.validation_percent,
            "training_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "group_leakage": False,
        },
        "training": {
            "steps": micro_agent_plan["allocated_steps"],
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "micro_agent_control": micro_agent_plan,
            "cpu_threads": micro_agent_plan["cpu_threads"],
            "training_scope": "bounded_internal_only_no_public_artifacts",
            "resumed_from_step": start_step,
            "total_target_steps": training_steps,
            "checkpoint_every": args.checkpoint_every,
            "relation_sampling": "relation probability proportional to square root of training group count",
            "target_format": "sorted distinct objects separated by ALT; EOS terminates the complete answer list",
            "initial_validation": {k: v for k, v in initial.items() if k != "generation_examples"},
            "final_validation": {k: v for k, v in final.items() if k != "generation_examples"},
            "validation_curve": validation_curve,
            "elapsed_seconds": elapsed,
            "training_groups_per_second": training_steps * args.batch_size / elapsed,
        },
        "generation_examples": final["generation_examples"],
        "provenance": {
            "external_model_weights_used": False,
            "open_source_model_integrated": False,
            "external_tokenizer_used": False,
            "external_data_used": False,
            "initialization": "native_random_initialization",
        },
        "production_training_gate": "CLOSED",
        "internal_experiment_gate": "MICRO_AGENT_APPROVED_BOUNDED_INTERNAL_ONLY",
        "public_artifacts_created": False,
        "production_2_87b_training_started": False,
        "limitations": [
            "Grouping resolves the training-label collision but does not certify that source facts are current or correct.",
            "Multiple objects are preserved, not automatically judged true; questionable pairs need human/source review.",
            "This small experiment is not a production Hindi language model or general fluency benchmark.",
        ],
    }

    def output_path(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    report_path = output_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = ROOT / "reports" / "gyaan-base-grouped-answer-audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "corpus_git_blob_sha1": blob,
        "generated_by": "neurocore_native_grouped_answer_word_prediction.py",
        **data_audit,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_path(args.tokenizer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.write_text(json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "source_rows": len(source_rows),
        "unique_subject_relation_groups": len(groups),
        "multi_answer_groups": data_audit["multi_answer_groups"],
        "training_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "vocabulary_size": vocab_size,
        "parameters": count_parameters(model),
        "initial_word_top1": initial["next_word_top1_accuracy"],
        "final_word_top1": final["next_word_top1_accuracy"],
        "final_word_top5": final["next_word_top5_accuracy"],
        "final_answer_set_exact": final["answer_set_exact_match"],
        "final_answer_set_f1": final["answer_set_f1"],
        "production_gate": "CLOSED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
