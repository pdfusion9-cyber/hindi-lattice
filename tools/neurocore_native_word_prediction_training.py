#!/usr/bin/env python3
"""Train native NeuroCore with whitespace-delimited Hindi word prediction.

The vocabulary is built deterministically from this repository corpus. The
objective predicts the next whole word token (not the next UTF-8 byte).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
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

UNK, BOS, EOS, SEP = 0, 1, 2, 3
SPECIAL_TOKENS = ["<UNK>", "<BOS>", "<EOS>", "<SEP>"]
EXPECTED_CORPUS_BLOB = "3018157a78e6a35877e114b9297f8ae9e16a76ae"


def git_blob_sha(path: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.as_posix()}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not value:
        raise RuntimeError("could not resolve corpus Git blob SHA")
    return value


def split_corpus(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    train_rows, valid_rows = [], []
    malformed = 0
    all_tokens: dict[str, int] = {}
    parsed = []
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        parts = [unicodedata.normalize("NFC", p.strip()) for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            malformed += 1
            continue
        fields = [field.split() for field in parts]
        if not all(fields):
            malformed += 1
            continue
        parsed.append((line, fields))
        for field in fields:
            for token in field:
                all_tokens[token] = all_tokens.get(token, 0) + 1
        bucket = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest()[:4], "big") % 100
        (valid_rows if bucket < 2 else train_rows).append((line, fields))
    if malformed or not train_rows or not valid_rows:
        raise ValueError(f"invalid deterministic split: malformed={malformed}, train={len(train_rows)}, valid={len(valid_rows)}")
    return raw, train_rows, valid_rows, all_tokens, {
        "total_rows": len(parsed),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "malformed_rows": malformed,
    }


def encode_example(fields, word_to_id):
    subject, relation, obj = fields
    ids = [BOS]
    for token in subject:
        ids.append(word_to_id.get(token, UNK))
    ids.append(SEP)
    for token in relation:
        ids.append(word_to_id.get(token, UNK))
    ids.append(SEP)
    for token in obj:
        ids.append(word_to_id.get(token, UNK))
    ids.append(EOS)
    return ids


def encode_stream(rows, word_to_id):
    stream = []
    for _, fields in rows:
        stream.extend(encode_example(fields, word_to_id))
    if len(stream) < 128:
        raise ValueError("encoded token stream is unexpectedly short")
    return torch.tensor(stream, dtype=torch.long)


def make_batch(stream, starts, seq_len):
    offsets = torch.arange(seq_len, dtype=torch.long)
    positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return stream[positions], stream[positions + 1]


@torch.no_grad()
def evaluate(model, stream, *, batch_size, seq_len, batches, seed, cfg):
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upper = stream.numel() - seq_len - 1
    losses, correct1, correct5, total = [], 0, 0, 0
    for _ in range(batches):
        starts = torch.randint(0, upper, (batch_size,), generator=generator)
        x, y = make_batch(stream, starts, seq_len)
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        losses.append(float(loss.item()))
        flat_logits, flat_targets = logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
        correct1 += int((flat_logits.argmax(dim=-1) == flat_targets).sum().item())
        top5 = flat_logits.topk(min(5, cfg.vocab_size), dim=-1).indices
        correct5 += int((top5 == flat_targets.unsqueeze(-1)).any(dim=-1).sum().item())
        total += flat_targets.numel()
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "top1_accuracy": correct1 / total,
        "top5_accuracy": correct5 / total,
        "predicted_tokens": total,
    }


@torch.inference_mode()
def greedy_words(model, prompt_ids, *, max_new_tokens):
    sequence = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    result = []
    for _ in range(max_new_tokens):
        logits = model(sequence)["logits"][0, -1]
        next_id = int(logits.argmax().item())
        if next_id == EOS:
            break
        if next_id in (BOS, SEP):
            break
        result.append(next_id)
        sequence = torch.cat([sequence, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
    return result


def normalized_words(words):
    return [unicodedata.normalize("NFC", word).casefold() for word in words]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--report", type=Path, default=Path("reports/neurocore-word-prediction-training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/neurocore-word-prediction-checkpoint.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/neurocore-word-tokenizer.json"))
    parser.add_argument("--generation-samples", type=int, default=30)
    parser.add_argument("--max-new-words", type=int, default=12)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.sequence_length < 4:
        parser.error("steps, batch-size and sequence-length must be positive (sequence >= 4)")
    cpu_count = os.cpu_count() or 1
    if not 1 <= args.cpu_threads <= min(4, cpu_count):
        parser.error(f"cpu-threads must be in 1..{min(4, cpu_count)}")

    corpus = (ROOT / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    relative = corpus.relative_to(ROOT).as_posix()
    actual_blob = git_blob_sha(Path(relative))
    if args.corpus and actual_blob != EXPECTED_CORPUS_BLOB:
        raise RuntimeError(f"corpus blob SHA mismatch: {actual_blob}")

    raw, train_rows, valid_rows, counts, split_info = split_corpus(corpus)
    # A full-corpus lexicon is used only to assign stable word IDs; validation
    # sequences are still held out from all gradient updates.
    words = sorted(counts, key=lambda token: (-counts[token], token))
    id_to_word = SPECIAL_TOKENS + words
    word_to_id = {word: index for index, word in enumerate(id_to_word) if index >= len(SPECIAL_TOKENS)}
    if len(word_to_id) != len(words):
        raise RuntimeError("word vocabulary contains a reserved-token collision")
    vocab_size = len(id_to_word)
    if vocab_size < 256 or vocab_size > 20000:
        raise ValueError(f"unexpected word vocabulary size: {vocab_size}")
    tokenizer_payload = {
        "format": "neurocore_whitespace_word_v1",
        "special_tokens": {"unk": UNK, "bos": BOS, "eos": EOS, "field_separator": SEP},
        "id_to_word": id_to_word,
        "normalization": "Unicode NFC; split on whitespace; preserve punctuation inside word tokens",
        "vocabulary_size": vocab_size,
    }
    tokenizer_bytes = json.dumps(tokenizer_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()

    train_stream = encode_stream(train_rows, word_to_id)
    valid_stream = encode_stream(valid_rows, word_to_id)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    cfg = NeuroCoreEdgeConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=3,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=64,
        attention_heads=4,
        kv_heads=2,
        sliding_window=16,
        recurrent_state_size=16,
        router_aux_loss_coefficient=0.01,
    )
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    initial = evaluate(model, valid_stream, batch_size=args.batch_size,
                       seq_len=args.sequence_length, batches=32, seed=args.seed + 100, cfg=cfg)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 300)
    upper = train_stream.numel() - args.sequence_length - 1
    train_losses = []
    validation_curve = []
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, upper, (args.batch_size,), generator=generator)
        x, y = make_batch(train_stream, starts, args.sequence_length)
        logits = model(x)
        loss = F.cross_entropy(logits["logits"].reshape(-1, vocab_size), y.reshape(-1))
        loss = loss + cfg.router_aux_loss_coefficient * logits["router_aux_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().item()))
        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, valid_stream, batch_size=args.batch_size,
                               seq_len=args.sequence_length, batches=32,
                               seed=args.seed + 100, cfg=cfg)
            validation_curve.append({"step": step, **metrics})
            print(
                f"word_step={step}/{args.steps} train_loss={train_losses[-1]:.4f} "
                f"validation_loss={metrics['loss']:.4f} "
                f"top1={metrics['top1_accuracy']:.4f} top5={metrics['top5_accuracy']:.4f}",
                flush=True,
            )
    elapsed = max(time.perf_counter() - started, 1e-9)
    final = evaluate(model, valid_stream, batch_size=args.batch_size,
                     seq_len=args.sequence_length, batches=64,
                     seed=args.seed + 100, cfg=cfg)

    # Whole-word generation on held-out subject+relation prefixes.
    sample_count = min(args.generation_samples, len(valid_rows))
    if sample_count < 1:
        raise ValueError("no held-out examples available for generation")
    sample_indices = sorted({round(i * (len(valid_rows) - 1) / max(sample_count - 1, 1))
                              for i in range(sample_count)})
    examples = []
    exact = 0
    token_similarities = []
    for index in sample_indices:
        _, fields = valid_rows[index]
        subject, relation, expected = fields
        prompt_ids = [BOS] + [word_to_id.get(w, UNK) for w in subject] + [SEP]
        prompt_ids += [word_to_id.get(w, UNK) for w in relation] + [SEP]
        predicted_ids = greedy_words(model, prompt_ids, max_new_tokens=args.max_new_words)
        predicted_words = [id_to_word[token_id] for token_id in predicted_ids]
        expected_norm = normalized_words(expected)
        predicted_norm = normalized_words(predicted_words)
        is_exact = predicted_norm == expected_norm
        exact += int(is_exact)
        token_similarities.append(difflib.SequenceMatcher(a=expected_norm, b=predicted_norm).ratio())
        examples.append({
            "subject": " ".join(subject),
            "relation": " ".join(relation),
            "expected_next_words": expected,
            "predicted_words": predicted_words,
            "exact_object_match": is_exact,
        })

    report = {
        "status": "NEUROCORE_NATIVE_WORD_PREDICTION_EXPERIMENT_PASS",
        "objective": "next_whitespace_delimited_word_token_prediction",
        "model_id": "neurocore_edge_native_reference_word",
        "capacity_profile": "expanded_word_vocabulary",
        "model_config": vars(cfg),
        "model_parameter_count": count_parameters(model),
        "vocabulary": {
            "size": vocab_size,
            "unique_corpus_words": len(words),
            "tokenizer_format": tokenizer_payload["format"],
            "tokenizer_sha256": tokenizer_sha256,
            "tokenization": tokenizer_payload["normalization"],
        },
        "corpus": {
            "path": relative,
            "git_blob_sha1": actual_blob,
            "sha256": hashlib.sha256(raw).hexdigest(),
            **split_info,
            "external_data_used": False,
            "production_approved": False,
            "rights_review_status": "UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW",
            "pii_review_status": "NOT_INDEPENDENTLY_REVIEWED",
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_length_words": args.sequence_length,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "initial_validation_loss": initial["loss"],
            "final_validation_loss": final["loss"],
            "initial_validation_next_word_top1": initial["top1_accuracy"],
            "final_validation_next_word_top1": final["top1_accuracy"],
            "final_validation_next_word_top5": final["top5_accuracy"],
            "validation_predicted_tokens": final["predicted_tokens"],
            "training_words_per_second": (
                args.steps * args.batch_size * args.sequence_length / elapsed
            ),
            "elapsed_seconds": elapsed,
            "validation_curve": validation_curve,
        },
        "generation_probe": {
            "scope": "held-out subject-plus-relation prompts; greedy whole-word object generation",
            "samples": len(examples),
            "exact_object_matches": exact,
            "exact_object_match_accuracy": exact / len(examples),
            "mean_word_sequence_similarity": sum(token_similarities) / len(token_similarities),
            "max_generated_words": args.max_new_words,
            "examples": examples,
        },
        "provenance": {
            "external_model_weights_used": False,
            "open_source_model_integrated": False,
            "external_tokenizer_used": False,
            "external_data_used": False,
            "source_kind": "native_training_from_random_initialization_on_repository_project_corpus",
        },
        "production_training_gate": "CLOSED",
        "production_2_87b_training_started": False,
        "dataset_approval_registry_changed": False,
        "note": "This is a small native word-level prediction experiment, not a production language model or proof of general Hindi fluency.",
    }

    checkpoint = {
        "format_version": 1,
        "model_id": report["model_id"],
        "capacity_profile": report["capacity_profile"],
        "config": vars(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": args.steps,
        "corpus_git_blob_sha1": actual_blob,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_format": tokenizer_payload["format"],
        "provenance": report["provenance"],
        "production_training_gate": "CLOSED",
    }

    def save_json(path: Path, payload):
        path = path if path.is_absolute() else ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ckpt_path.with_name(ckpt_path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(ckpt_path)
    tokenizer_path = save_json(args.tokenizer, tokenizer_payload)
    report_path = save_json(args.report, report)
    print(json.dumps({
        "status": report["status"],
        "steps": args.steps,
        "vocabulary_size": vocab_size,
        "parameters": count_parameters(model),
        "initial_validation_loss": initial["loss"],
        "final_validation_loss": final["loss"],
        "next_word_top1": final["top1_accuracy"],
        "next_word_top5": final["top5_accuracy"],
        "generation_exact_object_match": report["generation_probe"]["exact_object_match_accuracy"],
        "words_per_second": report["training"]["training_words_per_second"],
        "checkpoint": str(ckpt_path),
        "tokenizer": str(tokenizer_path),
        "report": str(report_path),
        "production_gate": "CLOSED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
