#!/usr/bin/env python3
"""Synthetic-only autoregressive Hindi Unicode code-point language-model smoke.

The model predicts the next Unicode code point and can generate continuations.
All training/evaluation strings are original toy examples embedded here. This
is not a capable Hindi assistant and reads no external, repository, or user data.
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

TRAIN_TEXTS = [
    "नमस्ते। आज हम हिंदी भाषा का अभ्यास कर रहे हैं।",
    "भारत विविध भाषाओं और संस्कृतियों वाला देश है।",
    "सूरज सुबह निकलता है और शाम को अस्त होता है।",
    "पानी सभी जीवों के जीवन के लिए आवश्यक है।",
    "किताबें पढ़ने से ज्ञान और कल्पना का विकास होता है।",
    "किसान मेहनत से खेतों में अनाज और सब्जियां उगाते हैं।",
    "विद्यालय में बच्चे भाषा, गणित और विज्ञान सीखते हैं।",
    "विज्ञान में प्रश्न पूछना और प्रमाण खोजना महत्वपूर्ण है।",
    "पेड़ छाया देते हैं और हवा को स्वच्छ रखने में मदद करते हैं।",
    "नदी पहाड़ों से निकलकर मैदानों में बहती है।",
    "समय का सदुपयोग करने से कठिन काम भी पूरे होते हैं।",
    "कंप्यूटर दिए गए निर्देशों के अनुसार जानकारी संसाधित करता है।",
    "भाषा के माध्यम से लोग अपने विचार और अनुभव साझा करते हैं।",
    "संगीत, कला और साहित्य जीवन को सुंदर बनाते हैं।",
    "स्वस्थ रहने के लिए भोजन, नींद और व्यायाम जरूरी हैं।",
    "पर्यावरण की रक्षा करना हम सभी की साझा जिम्मेदारी है।",
    "नई तकनीक सीखने के लिए धैर्य और नियमित अभ्यास चाहिए।",
    "छोटे प्रयास मिलकर बड़े बदलाव ला सकते हैं।",
    "प्रश्न को ध्यान से समझकर उत्तर स्पष्ट लिखना चाहिए।",
    "भारत की राजधानी नई दिल्ली है और यह एक बड़ा नगर है।",
]
HELDOUT_TEXTS = [
    "पुस्तकालय में विद्यार्थी शांत होकर पुस्तकें पढ़ते हैं।",
    "बारिश होने पर धरती की मिट्टी से सुगंध आती है।",
    "अच्छा संवाद सुनने और समझने से शुरू होता है।",
    "पहाड़ों पर ठंडी हवा चलती है और बादल दिखाई देते हैं।",
    "लगातार अभ्यास से कौशल में धीरे-धीरे सुधार होता है।",
]
PROMPT = "अच्छा संवाद"
SEQ_LEN = 48
HELDOUT_REPETITIONS = 4


def encode(text: str, stoi: dict[str, int], unk_id: int) -> list[int]:
    return [stoi.get(char, unk_id) for char in text]


def decode(ids: list[int], itos: list[str]) -> str:
    return "".join(itos[token] for token in ids)


class HindiAutoregressiveCharLM(nn.Module):
    """Character-level GRU; every output token is a complete Unicode code point."""

    def __init__(self, vocab_size: int, embedding_size: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, state=None):
        hidden, state = self.gru(self.embedding(token_ids), state)
        return self.output(hidden), state


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, generator=None):
    if len(data) <= seq_len + 1:
        raise ValueError("token stream is too short for the selected sequence length")
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[int(i):int(i) + seq_len] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + seq_len + 1] for i in starts])
    return x, y


def evaluate(model, data: torch.Tensor, seed: int, unk_id: int, batches: int = 16) -> float:
    """Evaluate only held-out targets representable by the train-only vocabulary."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(data, 16, SEQ_LEN, generator)
            logits, _ = model(x)
            known = y != unk_id
            if bool(known.any()):
                token_losses = F.cross_entropy(logits[known], y[known], reduction="sum")
                total_loss += float(token_losses)
                total_tokens += int(known.sum())
    if total_tokens == 0:
        raise RuntimeError("no in-vocabulary held-out targets available for evaluation")
    return total_loss / total_tokens


def generate(model, prompt: str, stoi: dict[str, int], itos: list[str], unk_id: int,
             max_new_chars: int = 96, seed: int = 19,
             temperature: float = 0.85, top_k: int = 12) -> str:
    if not prompt or max_new_chars < 1 or not 0 < temperature <= 2 or top_k < 1:
        raise ValueError("invalid generation settings")
    model.eval()
    ids = encode(prompt, stoi, unk_id)
    generator = torch.Generator().manual_seed(seed)
    generated: list[int] = []
    with torch.no_grad():
        context = torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long)
        logits, state = model(context)
        next_logits = logits[0, -1].clone()
        # UNK is an input fallback only: never emit it as generated text.
        next_logits[unk_id] = float("-inf")
        for _ in range(max_new_chars):
            scaled = next_logits / temperature
            k = min(top_k, len(itos))
            values, indices = torch.topk(scaled, k=k)
            probs = torch.softmax(values, dim=-1)
            selected = int(indices[torch.multinomial(probs, 1, generator=generator)].item())
            generated.append(selected)
            token = torch.tensor([[selected]], dtype=torch.long)
            next_out, state = model(token, state)
            next_logits = next_out[0, -1].clone()
            next_logits[unk_id] = float("-inf")
    return prompt + decode(generated, itos)


def run(report_path: Path, steps: int = 120, seed: int = 23) -> dict:
    if steps < 80:
        raise ValueError("steps must be >=80")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    train_text = "\n".join(TRAIN_TEXTS * 8)
    heldout_text = "\n".join(HELDOUT_TEXTS * HELDOUT_REPETITIONS)
    train_vocab = sorted(set(train_text))
    stoi = {char: index for index, char in enumerate(train_vocab)}
    itos = list(train_vocab)
    unk_id = len(itos)
    vocab_size = unk_id + 1
    if PROMPT in train_text:
        raise RuntimeError("generation prompt must be absent from training sentences")
    if any(char not in stoi for char in PROMPT):
        raise RuntimeError("generation prompt must be covered by the training vocabulary")
    if decode(encode(train_text, stoi, unk_id), itos) != train_text:
        raise RuntimeError("training Unicode code-point round-trip failed")
    if decode(encode(PROMPT, stoi, unk_id), itos) != PROMPT:
        raise RuntimeError("prompt Unicode code-point round-trip failed")

    train_ids = encode(train_text, stoi, unk_id)
    heldout_ids = encode(heldout_text, stoi, unk_id)
    train = torch.tensor(train_ids, dtype=torch.long)
    heldout = torch.tensor(heldout_ids, dtype=torch.long)
    unknown_chars = sorted(set(heldout_text) - set(train_vocab))
    known_heldout_chars = sum(token != unk_id for token in heldout_ids)
    if known_heldout_chars <= SEQ_LEN + 1:
        raise RuntimeError("held-out corpus has too few known code points")

    model = HindiAutoregressiveCharLM(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    initial_train = evaluate(model, train, seed + 1, unk_id)
    initial_heldout = evaluate(model, heldout, seed + 2, unk_id)

    for _ in range(steps):
        model.train()
        x, y = make_batch(train, 16, SEQ_LEN)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("missing or non-finite gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()

    final_train = evaluate(model, train, seed + 1, unk_id)
    final_heldout = evaluate(model, heldout, seed + 2, unk_id)
    if final_train >= initial_train:
        raise RuntimeError("training loss did not decrease")
    if final_heldout >= initial_heldout:
        raise RuntimeError("held-out known-codepoint loss did not improve")

    sample = generate(model, PROMPT, stoi, itos, unk_id,
                      max_new_chars=96, seed=seed + 3)
    if not sample.startswith(PROMPT) or len(sample) <= len(PROMPT):
        raise RuntimeError("autoregressive generation returned no continuation")
    continuation = sample[len(PROMPT):]
    if sample.encode("utf-8").decode("utf-8") != sample:
        raise RuntimeError("generated sample is not valid Unicode text")
    if "\ufffd" in sample:
        raise RuntimeError("generated sample contains the replacement character")
    if any(char not in stoi for char in continuation):
        raise RuntimeError("generator emitted a character outside the train vocabulary")

    with torch.no_grad():
        reference, _ = model(heldout[:64].unsqueeze(0))
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "steps": steps, "seed": seed, "stoi": stoi, "itos": itos,
             "unk_id": unk_id, "vocab_size": vocab_size}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-char-lm-") as td:
        checkpoint = Path(td) / "hindi_char_lm.pt"
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = HindiAutoregressiveCharLM(vocab_size=restored["vocab_size"])
        clone.load_state_dict(restored["model"], strict=True)
        clone.eval()
        with torch.no_grad():
            reloaded, _ = clone(heldout[:64].unsqueeze(0))
        max_diff = float((reference - reloaded).abs().max())
        if max_diff != 0.0:
            raise RuntimeError("checkpoint reload changed logits")
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.004)
        clone_optimizer.load_state_dict(restored["optimizer"])

    result = {
        "status": "HINDI_AUTOREGRESSIVE_LM_SMOKE_PASS",
        "experiment": "synthetic_unicode_codepoint_gru_next_char_v2",
        "task": "autoregressive next-Unicode-code-point prediction and sampled continuation",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokenizer": {"type": "Unicode code-point", "vocabulary_size_including_unk": vocab_size,
                      "training_char_vocabulary_size": len(itos), "unicode_round_trip": True,
                      "vocabulary_built_from_training_only": True},
        "train_examples": len(TRAIN_TEXTS),
        "heldout_examples": len(HELDOUT_TEXTS),
        "heldout_repetitions_for_scoring": HELDOUT_REPETITIONS,
        "heldout_used_for_gradient_updates": False,
        "heldout_unknown_codepoints": len(unknown_chars),
        "heldout_unknown_characters": unknown_chars,
        "heldout_scored_known_codepoints": known_heldout_chars,
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "initial_heldout_loss": initial_heldout,
        "final_heldout_loss": final_heldout,
        "generation": {"prompt": PROMPT, "prompt_is_absent_from_training_sentences": True,
                       "max_new_characters": 96, "sample": sample,
                       "unicode_valid": True, "replacement_character_present": False,
                       "seed": seed + 3, "sampling": "temperature=0.85, top_k=12"},
        "checkpoint": {"sha256": digest, "reload_max_abs_logit_difference": max_diff,
                       "optimizer_state_restored": True},
        "provenance": {"kind": "original_synthetic_toy_text_only",
                       "external_data_used": False, "repository_corpus_used": False,
                       "user_data_used": False},
        "production_training_gate": "CLOSED",
        "note": ("Tiny CPU character-level language-model smoke test only. Code-point tokenization "
                 "prevents malformed UTF-8 sequences but does not ensure meaningful Hindi. Held-out "
                 "loss on five toy sentences does not establish broad Hindi ability, factual accuracy, "
                 "or production readiness.")
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    print(json.dumps(run(args.report_path, args.steps, args.seed), ensure_ascii=False, indent=2))
