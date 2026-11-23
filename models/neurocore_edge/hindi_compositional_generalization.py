#!/usr/bin/env python3
"""Synthetic Hindi subject-place compositional generalization smoke test.

A tiny word-level GRU is trained on a grid of subject/place pairs with a held-out
set of pairings. Every subject and every place appears in training, but the exact
held-out pairings do not. Given a held-out subject+place prefix, the model must
complete the shared Hindi locative sentence frame. This is a controlled template
recombination test, not a broad natural-language understanding benchmark.
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

SUBJECTS = ["किसान", "शिक्षक", "माली", "कुम्हार", "दर्जी", "रसोइया", "चालक", "लेखक"]
# Inflected locative forms are used so each phrase is grammatical with "में".
PLACES = ["खेत", "विद्यालय", "बगीचे", "कारखाने", "बाजार", "रसोईघर", "पुस्तकालय", "अस्पताल"]
FRAME_SUFFIX = ["में", "काम", "करता", "है।"]
EOS = "<eos>"
UNK = "<unk>"
SEQ_LEN = 8
BATCH_SIZE = 32
TRAIN_REPETITIONS = 30


def split_pairs():
    train = []
    heldout = []
    for i, subject in enumerate(SUBJECTS):
        for j, place in enumerate(PLACES):
            sentence = f"{subject} {place} {' '.join(FRAME_SUFFIX)}"
            # Exactly 16 of the 64 pairs are held out. This modular pattern
            # ensures every subject and every place still appears in training.
            (heldout if (i * 3 + j * 5) % 8 >= 6 else train).append(
                {"subject": subject, "place": place, "sentence": sentence}
            )
    return train, heldout


class HindiCompositionalWordLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, state=None):
        hidden, state = self.gru(self.embedding(token_ids), state)
        return self.output(hidden), state


def sentence_ids(sentence: str, stoi: dict[str, int]) -> list[int]:
    words = sentence.split()
    return [stoi[word] for word in words] + [stoi[EOS]]


def make_stream(sentences: list[str], stoi: dict[str, int], repetitions: int, seed: int):
    rng = random.Random(seed)
    stream: list[int] = []
    for _ in range(repetitions):
        order = list(sentences)
        rng.shuffle(order)
        for sentence in order:
            stream.extend(sentence_ids(sentence, stoi))
    return torch.tensor(stream, dtype=torch.long)


def make_batch(data: torch.Tensor):
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[int(i):int(i) + SEQ_LEN] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + SEQ_LEN + 1] for i in starts])
    return x, y


def evaluate(model, sentences: list[str], stoi: dict[str, int]) -> dict:
    model.eval()
    total_loss = 0.0
    token_count = 0
    correct = 0
    with torch.no_grad():
        for sentence in sentences:
            ids = torch.tensor(sentence_ids(sentence, stoi), dtype=torch.long)
            x = ids[:-1].unsqueeze(0)
            y = ids[1:]
            logits, _ = model(x)
            logits = logits.squeeze(0)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            token_count += len(y)
    return {
        "loss": total_loss / token_count,
        "next_token_accuracy": correct / token_count,
        "scored_tokens": token_count,
    }


def generate(model, prompt: str, stoi: dict[str, int], itos: list[str], max_new_tokens: int = 4) -> str:
    prompt_tokens = prompt.split()
    if len(prompt_tokens) != 2 or any(token not in stoi for token in prompt_tokens):
        raise ValueError("prompt must be a known subject/place pair")
    model.eval()
    generated: list[str] = []
    with torch.no_grad():
        input_ids = torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long)
        logits, state = model(input_ids)
        next_logits = logits[0, -1].clone()
        for _ in range(max_new_tokens):
            next_logits[stoi[UNK]] = float("-inf")
            selected = int(next_logits.argmax().item())
            if selected == stoi[EOS]:
                break
            generated.append(itos[selected])
            next_output, state = model(torch.tensor([[selected]], dtype=torch.long), state)
            next_logits = next_output[0, -1].clone()
    return " ".join(prompt_tokens + generated)


def run(report_path: Path, steps: int = 100, seed: int = 11) -> dict:
    if steps < 50:
        raise ValueError("steps must be >=50")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    train_pairs, heldout_pairs = split_pairs()
    train_sentences = [row["sentence"] for row in train_pairs]
    heldout_sentences = [row["sentence"] for row in heldout_pairs]
    train_sentence_set = set(train_sentences)
    heldout_sentence_set = set(heldout_sentences)
    if len(train_pairs) != 48 or len(heldout_pairs) != 16:
        raise RuntimeError("unexpected compositional split dimensions")
    if train_sentence_set & heldout_sentence_set:
        raise RuntimeError("a held-out sentence appears in training")
    if {row["subject"] for row in train_pairs} != set(SUBJECTS):
        raise RuntimeError("some subject is absent from training")
    if {row["place"] for row in train_pairs} != set(PLACES):
        raise RuntimeError("some place is absent from training")
    if any(row["subject"] not in SUBJECTS or row["place"] not in PLACES for row in heldout_pairs):
        raise RuntimeError("held-out pairing contains unknown factors")

    # Build vocabulary from training examples only.
    train_words = sorted({word for sentence in train_sentences for word in sentence.split()})
    itos = train_words + [EOS, UNK]
    stoi = {word: index for index, word in enumerate(itos)}
    if any(word not in stoi for sentence in heldout_sentences for word in sentence.split()):
        raise RuntimeError("held-out set contains out-of-vocabulary words")
    if UNK not in stoi or EOS not in stoi:
        raise RuntimeError("missing special vocabulary tokens")

    train = make_stream(train_sentences, stoi, TRAIN_REPETITIONS, seed + 1)
    model = HindiCompositionalWordLM(len(itos))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    initial_train = evaluate(model, train_sentences, stoi)
    initial_heldout = evaluate(model, heldout_sentences, stoi)

    for _ in range(steps):
        model.train()
        x, y = make_batch(train)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(itos)), y.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("missing or non-finite gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()

    final_train = evaluate(model, train_sentences, stoi)
    final_heldout = evaluate(model, heldout_sentences, stoi)
    if final_train["loss"] >= initial_train["loss"]:
        raise RuntimeError("training loss did not decrease")
    if final_heldout["loss"] >= initial_heldout["loss"]:
        raise RuntimeError("held-out loss did not decrease")
    if final_heldout["next_token_accuracy"] < 0.75:
        raise RuntimeError("held-out next-token accuracy below 75%")

    generations = []
    for row in heldout_pairs:
        prompt = f'{row["subject"]} {row["place"]}'
        sample = generate(model, prompt, stoi, itos)
        exact_match = sample == row["sentence"]
        generations.append({
            "prompt": prompt,
            "generated": sample,
            "expected": row["sentence"],
            "exact_match": exact_match,
            "sentence_absent_from_training": sample not in train_sentence_set,
        })
    exact_count = sum(row["exact_match"] for row in generations)
    novel_count = sum(row["generated"] not in train_sentence_set for row in generations)
    if exact_count != len(heldout_pairs):
        raise RuntimeError(f"only {exact_count}/{len(heldout_pairs)} unseen pairings completed exactly")
    if novel_count != len(heldout_pairs):
        raise RuntimeError("generated outputs overlap a training sentence")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "steps": steps,
        "seed": seed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-composition-") as directory:
        checkpoint_path = Path(directory) / "hindi_compositional_lm.pt"
        torch.save(state, checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        clone = HindiCompositionalWordLM(len(restored["itos"]))
        clone.load_state_dict(restored["model"], strict=True)
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.004)
        clone_optimizer.load_state_dict(restored["optimizer"])
        restored_generations = [
            generate(clone, row["prompt"], restored["stoi"], restored["itos"])
            for row in generations
        ]
        if restored_generations != [row["generated"] for row in generations]:
            raise RuntimeError("checkpoint reload changed generated sentences")

    result = {
        "status": "HINDI_COMPOSITIONAL_GENERALIZATION_PASS",
        "experiment": "synthetic_subject_place_pair_holdout_word_gru_v1",
        "task": "word-level autoregressive completion of unseen subject-place pairings",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "training_pairings": len(train_pairs),
            "heldout_pairings": len(heldout_pairs),
            "all_subjects_seen_in_training": True,
            "all_places_seen_in_training": True,
            "heldout_full_sentences_absent_from_training": True,
            "heldout_pairing_overlap_with_training": False,
        },
        "tokenizer": {
            "type": "whitespace-delimited word",
            "vocabulary_size_including_special_tokens": len(itos),
            "vocabulary_built_from_training_only": True,
            "heldout_oov_words": [],
        },
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_heldout": initial_heldout,
        "final_heldout": final_heldout,
        "heldout_used_for_gradient_updates": False,
        "generation": {
            "total_heldout_combinations": len(generations),
            "exact_completions": exact_count,
            "novel_sentences_not_in_training": novel_count,
            "samples": generations,
        },
        "checkpoint": {
            "sha256": checkpoint_hash,
            "generated_sentences_equal_after_reload": True,
            "optimizer_state_restored": True,
        },
        "provenance": {
            "kind": "original_synthetic_template_sentences_only",
            "external_data_used": False,
            "repository_corpus_used": False,
            "user_data_used": False,
        },
        "production_training_gate": "CLOSED",
        "limitation": (
            "This tests one fixed grammatical sentence frame and unseen subject-place pairings. "
            "The suffix after the supplied pair is shared by every example, so success demonstrates "
            "limited template recombination, not broad Hindi generation, semantic understanding, "
            "factual reliability, or production readiness."
        ),
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.report_path, arguments.steps, arguments.seed), ensure_ascii=False, indent=2))
