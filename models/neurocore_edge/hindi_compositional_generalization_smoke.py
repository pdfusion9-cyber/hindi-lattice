#!/usr/bin/env python3
"""Synthetic Hindi compositional-generalization LM smoke test.

A word-level GRU predicts continuations for subject + time-marker combinations
withheld in full from training. All lexical items are present elsewhere in the
training split. This is a controlled toy experiment, not a Hindi benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

EOS = "<eos>"
UNK = "<unk>"
SEQ_LEN = 9
BATCH_SIZE = 32
TRAIN_REPETITIONS = 80
TRAIN_STEPS_DEFAULT = 700

SUBJECT_ACTIVITIES = [
    ("विद्यार्थी", ["पुस्तकें", "पढ़ते"]),
    ("शोधकर्ता", ["तथ्य", "खोजते"]),
    ("कलाकार", ["चित्र", "बनाते"]),
    ("स्वयंसेवक", ["लोगों", "की", "सहायता", "करते"]),
    ("शिक्षक", ["पाठ", "समझाते"]),
    ("खिलाड़ी", ["अभ्यास", "करते"]),
    ("माली", ["पौधे", "लगाते"]),
    ("लेखक", ["कहानियाँ", "लिखते"]),
]
TIME_MARKERS = ["सुबह जल्दी", "दिन में", "दोपहर में", "शाम को", "रात में"]


def build_split():
    train_rows: list[list[str]] = []
    heldout_rows: list[dict] = []
    for subject_index, (subject, activity) in enumerate(SUBJECT_ACTIVITIES):
        withheld_time_index = subject_index % len(TIME_MARKERS)
        for time_index, time_marker in enumerate(TIME_MARKERS):
            prefix = [subject, *time_marker.split()]
            row = [*prefix, *activity, "हैं।", EOS]
            if time_index == withheld_time_index:
                heldout_rows.append({
                    "subject": subject,
                    "time_marker": time_marker,
                    "prefix": prefix,
                    "tokens": row,
                })
            else:
                train_rows.append(row)
    return train_rows, heldout_rows


def render(tokens: list[str]) -> str:
    return " ".join(token for token in tokens if token != EOS)


def encode_rows(rows: list[list[str]], stoi: dict[str, int], repetitions: int, seed: int):
    rng = random.Random(seed)
    stream: list[int] = []
    for _ in range(repetitions):
        order = rows.copy()
        rng.shuffle(order)
        for row in order:
            stream.extend(stoi[token] for token in row)
    return torch.tensor(stream, dtype=torch.long)


class HindiCompositionalWordLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int = 64, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, state=None):
        hidden, state = self.gru(self.embedding(token_ids), state)
        return self.output(hidden), state


def make_batch(data: torch.Tensor, generator: torch.Generator):
    starts = torch.randint(
        0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,), generator=generator
    )
    x = torch.stack([data[int(start):int(start) + SEQ_LEN] for start in starts])
    y = torch.stack([data[int(start) + 1:int(start) + SEQ_LEN + 1] for start in starts])
    return x, y


def evaluate(model, rows: list[list[str]], stoi: dict[str, int], vocab_size: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    sentence_results = []
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([stoi[token] for token in row], dtype=torch.long)
            logits, _ = model(ids[:-1].unsqueeze(0))
            logits = logits.squeeze(0)
            targets = ids[1:]
            losses = F.cross_entropy(logits, targets, reduction="sum")
            predictions = logits.argmax(dim=-1)
            hits = int((predictions == targets).sum())
            token_count = int(targets.numel())
            total_loss += float(losses)
            total_tokens += token_count
            correct += hits
            sentence_results.append({
                "expected": render(row),
                "next_word_correct": hits,
                "next_word_count": token_count,
                "exact_teacher_forced_match": hits == token_count,
            })
    if total_tokens == 0:
        raise RuntimeError("empty evaluation split")
    return {
        "loss": total_loss / total_tokens,
        "next_word_accuracy": correct / total_tokens,
        "correct_next_words": correct,
        "scored_next_words": total_tokens,
        "sentence_results": sentence_results,
    }


def generate(model, prefix: list[str], stoi: dict[str, int], itos: list[str],
             max_new_words: int = 12) -> list[str]:
    model.eval()
    prefix_ids = [stoi[token] for token in prefix]
    generated: list[str] = []
    with torch.no_grad():
        logits, state = model(torch.tensor([prefix_ids], dtype=torch.long))
        next_logits = logits[0, -1].clone()
        for _ in range(max_new_words):
            next_logits[stoi[UNK]] = float("-inf")
            selected = int(next_logits.argmax().item())
            if selected == stoi[EOS]:
                break
            token = itos[selected]
            generated.append(token)
            next_out, state = model(torch.tensor([[selected]], dtype=torch.long), state)
            next_logits = next_out[0, -1].clone()
    return [*prefix, *generated]


def run(report_path: Path, steps: int = TRAIN_STEPS_DEFAULT, seed: int = 11):
    if steps < 300:
        raise ValueError("steps must be >=300 for this small generalization experiment")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    train_rows, heldout = build_split()
    train_contexts = {
        tuple(row[:3]) for row in train_rows
    }
    heldout_contexts = [tuple(row["prefix"]) for row in heldout]
    if len(train_rows) != 32 or len(heldout) != 8:
        raise RuntimeError("unexpected split size")
    if any(context in train_contexts for context in heldout_contexts):
        raise RuntimeError("held-out subject/time context leaked into training")

    # The vocabulary is built only from the training split.
    train_lexicon = sorted({token for row in train_rows for token in row if token != EOS})
    itos = [*train_lexicon, EOS, UNK]
    stoi = {token: index for index, token in enumerate(itos)}
    unknown_heldout = sorted({
        token for row in heldout for token in row["tokens"] if token not in stoi
    })
    if unknown_heldout:
        raise RuntimeError(f"held-out lexical items are OOV: {unknown_heldout}")

    train_stream = encode_rows(train_rows, stoi, TRAIN_REPETITIONS, seed)
    heldout_rows = [row["tokens"] for row in heldout]
    model = HindiCompositionalWordLM(len(itos))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    batch_generator = torch.Generator().manual_seed(seed + 100)

    initial_train = evaluate(model, train_rows, stoi, len(itos))
    initial_heldout = evaluate(model, heldout_rows, stoi, len(itos))
    for _ in range(steps):
        model.train()
        x, y = make_batch(train_stream, batch_generator)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(itos)), y.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all())
                                    for gradient in gradients):
            raise RuntimeError("missing or non-finite gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()

    final_train = evaluate(model, train_rows, stoi, len(itos))
    final_heldout = evaluate(model, heldout_rows, stoi, len(itos))
    if final_train["loss"] >= initial_train["loss"]:
        raise RuntimeError("training loss did not decrease")
    if final_heldout["loss"] >= initial_heldout["loss"]:
        raise RuntimeError("held-out compositional loss did not decrease")
    if final_heldout["next_word_accuracy"] < 0.75:
        raise RuntimeError("held-out next-word accuracy below 75%")

    generated = []
    for row in heldout:
        prediction_tokens = generate(model, row["prefix"], stoi, itos)
        expected = render(row["tokens"])
        predicted = render(prediction_tokens)
        generated.append({
            "subject": row["subject"],
            "withheld_time_marker": row["time_marker"],
            "prompt": " ".join(row["prefix"]),
            "expected": expected,
            "generated": predicted,
            "exact_match": predicted == expected,
            "context_seen_in_training": False,
        })
    exact_matches = sum(item["exact_match"] for item in generated)
    if exact_matches != len(heldout):
        raise RuntimeError(
            f"compositional generation exact match {exact_matches}/{len(heldout)}"
        )

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": steps,
        "seed": seed,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(itos),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-compositional-lm-") as directory:
        checkpoint = Path(directory) / "hindi_compositional_word_lm.pt"
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = HindiCompositionalWordLM(restored["vocab_size"])
        clone.load_state_dict(restored["model"], strict=True)
        clone.eval()
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.003)
        clone_optimizer.load_state_dict(restored["optimizer"])
        restored_generated = []
        for row in heldout:
            tokens = generate(clone, row["prefix"], restored["stoi"], restored["itos"])
            restored_generated.append(render(tokens))
        if restored_generated != [item["generated"] for item in generated]:
            raise RuntimeError("checkpoint reload changed generated text")

    report = {
        "status": "HINDI_COMPOSITIONAL_GENERALIZATION_PASS",
        "experiment": "synthetic_unseen_subject_time_word_lm_v1",
        "task": "word-level autoregressive generation on unseen subject/time combinations",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer": {
            "type": "whitespace-delimited orthographic word",
            "vocabulary_size_including_special_tokens": len(itos),
            "vocabulary_built_from_training_split_only": True,
            "all_heldout_lexical_items_seen_in_training": not unknown_heldout,
        },
        "split": {
            "train_unique_sentences": len(train_rows),
            "heldout_unique_sentences": len(heldout),
            "train_repetitions_for_optimization": TRAIN_REPETITIONS,
            "withheld_subject_time_combinations": [
                {"subject": row["subject"], "time_marker": row["time_marker"]}
                for row in heldout
            ],
            "heldout_contexts_seen_in_training": False,
            "heldout_used_for_gradient_updates": False,
            "heldout_oov_tokens": unknown_heldout,
        },
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_heldout": initial_heldout,
        "final_heldout": final_heldout,
        "generation": {
            "decoder": "greedy",
            "samples": generated,
            "exact_sentence_matches": exact_matches,
            "heldout_sentence_count": len(heldout),
        },
        "checkpoint": {
            "sha256": digest,
            "generated_text_equal_after_reload": True,
            "optimizer_state_restored": True,
        },
        "provenance": {
            "kind": "original_synthetic_compositional_templates_only",
            "external_data_used": False,
            "repository_corpus_used": False,
            "user_data_used": False,
        },
        "production_training_gate": "CLOSED",
        "note": (
            "Controlled toy compositional-generalization test: each held-out subject/time "
            "pair is absent from training, while its words and the subject's activity phrase "
            "occur elsewhere in training. Exact held-out generation shows recombination of "
            "this narrow synthetic pattern only; it is not evidence of general Hindi grammar, "
            "meaning, factual reliability, or production readiness."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.report_path, arguments.steps, arguments.seed),
                     ensure_ascii=False, indent=2))
