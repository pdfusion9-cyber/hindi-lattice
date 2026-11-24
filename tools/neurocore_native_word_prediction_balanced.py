#!/usr/bin/env python3
"""Relation-balanced whole-word next-token training for native NeuroCore."""
from __future__ import annotations

import argparse
import difflib
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
MODEL_DIR = ROOT / "models" / "neurocore_edge"
sys.path.insert(0, str(MODEL_DIR))
from modeling_neurocore_edge import (  # noqa: E402
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)

UNK, BOS, EOS, SEP, PAD = range(5)
SPECIAL = ["<UNK>", "<BOS>", "<EOS>", "<SEP>", "<PAD>"]
EXPECTED_BLOB = "3018157a78e6a35877e114b9297f8ae9e16a76ae"


def load_rows(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    train, valid, counts = [], [], defaultdict(int)
    for line in (x.strip() for x in text.splitlines() if x.strip()):
        parts = [unicodedata.normalize("NFC", x.strip()) for x in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed triple: {line!r}")
        fields = [p.split() for p in parts]
        if not all(fields):
            raise ValueError(f"empty triple field: {line!r}")
        for word in sum(fields, []):
            counts[word] += 1
        bucket = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest()[:4], "big") % 100
        (valid if bucket < 2 else train).append((line, fields))
    if not train or not valid:
        raise ValueError("empty train/validation split")
    return raw, train, valid, counts


def make_vocab(counts):
    words = sorted(counts, key=lambda w: (-counts[w], w))
    id_to_word = SPECIAL + words
    if len(set(id_to_word)) != len(id_to_word):
        raise ValueError("reserved token collision in vocabulary")
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return word_to_id, id_to_word


def encode_row(fields, word_to_id):
    subject, relation, obj = fields
    prompt = [BOS] + [word_to_id.get(w, UNK) for w in subject]
    prompt += [SEP] + [word_to_id.get(w, UNK) for w in relation] + [SEP]
    target = [word_to_id.get(w, UNK) for w in obj] + [EOS]
    return prompt, target


def collate(examples, pad_id=PAD):
    max_len = max(len(p) + len(t) - 1 for p, t in examples)
    xs, ys, masks = [], [], []
    for prompt, target in examples:
        full = prompt + target
        x, y = full[:-1], full[1:]
        mask = [False] * (len(prompt) - 1) + [True] * len(target)
        pad = max_len - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)
        masks.append(mask + [False] * pad)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )


@torch.no_grad()
def evaluate(model, rows, word_to_id, *, batch_size, cfg, measure_generation=False):
    model.eval()
    loss_sum = 0.0
    n_tokens = top1 = top5 = 0
    exact_rows = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        examples = [encode_row(fields, word_to_id) for _, fields in batch]
        x, y, mask = collate(examples)
        logits = model(x)["logits"]
        selected_logits = logits[mask]
        selected_targets = y[mask]
        loss_sum += float(F.cross_entropy(selected_logits, selected_targets, reduction="sum").item())
        pred = selected_logits.argmax(dim=-1)
        top1 += int((pred == selected_targets).sum().item())
        top5_ids = selected_logits.topk(5, dim=-1).indices
        top5 += int((top5_ids == selected_targets.unsqueeze(-1)).any(dim=-1).sum().item())
        n_tokens += selected_targets.numel()

        if measure_generation:
            for (_, fields), (prompt, _) in zip(batch, examples):
                obj = fields[2]
                gen_ids = greedy(model, prompt, max_words=max(12, len(obj) + 2))
                expected = [word_to_id.get(w, UNK) for w in obj]
                exact_rows += int(gen_ids == expected)
    model.train()
    return {
        "object_word_loss": loss_sum / max(n_tokens, 1),
        "object_word_top1": top1 / max(n_tokens, 1),
        "object_word_top5": top5 / max(n_tokens, 1),
        "predicted_object_words": n_tokens,
        "exact_object_match": (exact_rows / max(len(rows), 1)) if measure_generation else None,
        "exact_object_matches": exact_rows if measure_generation else None,
        "evaluated_rows": len(rows) if measure_generation else 0,
    }


@torch.inference_mode()
def greedy(model, prompt, max_words):
    sequence = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    result = []
    for _ in range(max_words):
        next_id = int(model(sequence)["logits"][0, -1].argmax().item())
        if next_id == EOS:
            break
        if next_id in (BOS, SEP, PAD):
            break
        result.append(next_id)
        sequence = torch.cat([sequence, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=0.00015)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--report", type=Path, default=Path("reports/neurocore-word-prediction-balanced-5000.json"))
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/neurocore-word-prediction-balanced-5000.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/neurocore-hindi-word-tokenizer-balanced.json"))
    args = ap.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        ap.error("steps and batch-size must be positive")
    if not 1 <= args.cpu_threads <= min(4, os.cpu_count() or 1):
        ap.error("cpu-threads must be between 1 and min(4, available CPUs)")

    corpus = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    relpath = corpus.resolve().relative_to(ROOT).as_posix()
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{relpath}"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    if blob != EXPECTED_BLOB:
        raise RuntimeError(f"unexpected corpus blob {blob}")
    raw, train_rows, valid_rows, counts = load_rows(corpus)
    word_to_id, id_to_word = make_vocab(counts)
    vocab_size = len(id_to_word)
    if vocab_size > 20000:
        raise ValueError(f"vocabulary too large: {vocab_size}")

    relation_groups = defaultdict(list)
    for row in train_rows:
        relation_groups[" ".join(row[1][1])].append(row)
    relation_names = sorted(relation_groups)
    if len(relation_names) < 2:
        raise ValueError("relation-balanced training requires multiple relation groups")

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    py_rng = random.Random(args.seed + 1)
    cfg = NeuroCoreEdgeConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=3,
        num_experts=20, experts_per_token=2, expert_intermediate_size=64,
        attention_heads=4, kv_heads=2, sliding_window=32,
        recurrent_state_size=16, router_aux_loss_coefficient=0.01,
    )
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    initial = evaluate(model, valid_rows, word_to_id, batch_size=args.batch_size, cfg=cfg)
    curve = []
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        # Equal probability per relation, then random row within relation.
        batch = []
        for _ in range(args.batch_size):
            relation = relation_names[py_rng.randrange(len(relation_names))]
            batch.append(py_rng.choice(relation_groups[relation]))
        examples = [encode_row(fields, word_to_id) for _, fields in batch]
        x, y, mask = collate(examples)
        logits = model(x)
        selected_logits = logits["logits"][mask]
        selected_targets = y[mask]
        loss = F.cross_entropy(selected_logits, selected_targets)
        loss = loss + cfg.router_aux_loss_coefficient * logits["router_aux_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, valid_rows, word_to_id,
                               batch_size=args.batch_size, cfg=cfg)
            curve.append({"step": step, **metrics})
            print(
                f"balanced_word_step={step}/{args.steps} "
                f"loss={metrics['object_word_loss']:.4f} "
                f"top1={metrics['object_word_top1']:.4f} "
                f"top5={metrics['object_word_top5']:.4f}",
                flush=True,
            )

    elapsed = max(time.perf_counter() - started, 1e-9)
    final = evaluate(model, valid_rows, word_to_id, batch_size=args.batch_size, cfg=cfg, measure_generation=True)
    sample_rows = valid_rows[:min(30, len(valid_rows))]
    examples_out = []
    for line, fields in sample_rows:
        subject, relation, obj = fields
        prompt, _ = encode_row(fields, word_to_id)
        ids = greedy(model, prompt, max_words=max(12, len(obj) + 2))
        examples_out.append({
            "subject": " ".join(subject),
            "relation": " ".join(relation),
            "expected_words": obj,
            "predicted_words": [id_to_word[i] for i in ids],
            "exact_match": ids == [word_to_id.get(w, UNK) for w in obj],
        })

    tokenizer = {
        "format": "neurocore_whitespace_word_balanced_v2",
        "special_tokens": {"unk": UNK, "bos": BOS, "eos": EOS, "field_separator": SEP, "pad": PAD},
        "id_to_word": id_to_word,
        "normalization": "Unicode NFC; split on whitespace; punctuation remains attached to word",
    }
    tok_bytes = json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tok_sha = hashlib.sha256(tok_bytes).hexdigest()
    report = {
        "status": "NEUROCORE_NATIVE_RELATION_BALANCED_WORD_PREDICTION_PASS",
        "objective": "relation_balanced_subject_relation_to_next_object_word_prediction",
        "model_id": "neurocore_edge_native_reference_word_balanced_v2",
        "model_parameter_count": count_parameters(model),
        "vocabulary": {"size": vocab_size, "unique_words": len(counts),
                       "tokenizer_format": tokenizer["format"], "sha256": tok_sha},
        "corpus": {"path": relpath, "git_blob_sha1": blob,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "train_rows": len(train_rows), "validation_rows": len(valid_rows),
                   "relation_groups": len(relation_names),
                   "external_data_used": False,
                   "rights_review_status": "UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW"},
        "training": {"steps": args.steps, "batch_size": args.batch_size,
                     "sampling": "uniform_over_relation_groups_then_uniform_row",
                     "objective": "cross_entropy_only_on_object_words_and_eos",
                     "learning_rate": args.learning_rate,
                     "initial_validation": initial, "final_validation": final,
                     "validation_curve": curve,
                     "rows_per_second": args.steps * args.batch_size / elapsed,
                     "elapsed_seconds": elapsed},
        "generation_examples": examples_out,
        "provenance": {"external_model_weights_used": False,
                       "open_source_model_integrated": False,
                       "external_tokenizer_used": False,
                       "external_data_used": False},
        "production_training_gate": "CLOSED",
        "production_2_87b_training_started": False,
        "limitations": [
            "This is a small fact-triple next-object-word experiment, not general Hindi language modeling.",
            "The corpus is compact and its source rights remain under review.",
            "Exact match is sensitive to alternate valid spellings and multiword values.",
        ],
    }
    checkpoint = {
        "format_version": 2, "model_id": report["model_id"], "config": vars(cfg),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "step": args.steps, "capacity_profile": "expanded_word_vocabulary",
        "corpus_git_blob_sha1": blob, "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_sha256": tok_sha, "tokenizer_format": tokenizer["format"],
        "provenance": report["provenance"], "production_training_gate": "CLOSED",
    }
    def save(path, payload):
        path = path if path.is_absolute() else ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    temp = ckpt.with_name(ckpt.name + ".tmp")
    torch.save(checkpoint, temp)
    temp.replace(ckpt)
    save(args.tokenizer, tokenizer)
    save(args.report, report)
    print(json.dumps({
        "vocabulary_size": vocab_size, "parameters": count_parameters(model),
        "initial_validation_top1": initial["object_word_top1"],
        "final_validation_top1": final["object_word_top1"],
        "final_validation_top5": final["object_word_top5"],
        "exact_object_match": final["exact_object_match"],
        "object_word_loss": final["object_word_loss"],
        "rows_per_second": report["training"]["rows_per_second"],
        "production_gate": "CLOSED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
