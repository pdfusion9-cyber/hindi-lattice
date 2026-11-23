#!/usr/bin/env python3
"""Synthetic-only autoregressive Hindi byte language-model smoke experiment.

The model predicts the next UTF-8 byte and can generate continuations. All
training/evaluation strings are original toy examples embedded here. This is
not a capable Hindi assistant and does not read external, repository, or user data.
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
VOCAB_SIZE = 256
SEQ_LEN = 64


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    # Generation may stop in the middle of a multibyte code point.
    return bytes(ids).decode("utf-8", errors="replace")


class HindiAutoregressiveByteLM(nn.Module):
    def __init__(self, embedding_size: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, token_ids: torch.Tensor, state=None):
        hidden, state = self.gru(self.embedding(token_ids), state)
        return self.output(hidden), state


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, generator=None):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[int(i):int(i) + seq_len] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + seq_len + 1] for i in starts])
    return x, y


def evaluate(model, data: torch.Tensor, seed: int, batches: int = 16) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(data, 16, SEQ_LEN, generator)
            logits, _ = model(x)
            losses.append(float(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))))
    return sum(losses) / len(losses)


def generate(model, prompt: str, max_new_bytes: int = 96, seed: int = 19,
             temperature: float = 0.85, top_k: int = 12) -> str:
    if max_new_bytes < 1 or not 0 < temperature <= 2 or top_k < 1:
        raise ValueError("invalid generation settings")
    model.eval()
    ids = encode(prompt)
    if not ids:
        ids = [ord(" ")]
    generator = torch.Generator().manual_seed(seed)
    generated = list(ids)
    with torch.no_grad():
        context = torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long)
        logits, state = model(context)
        next_logits = logits[0, -1]
        for _ in range(max_new_bytes):
            scaled = next_logits / temperature
            k = min(top_k, VOCAB_SIZE)
            values, indices = torch.topk(scaled, k=k)
            probs = torch.softmax(values, dim=-1)
            selected = int(indices[torch.multinomial(probs, 1, generator=generator)].item())
            generated.append(selected)
            token = torch.tensor([[selected]], dtype=torch.long)
            next_out, state = model(token, state)
            next_logits = next_out[0, -1]
    return decode(generated)


def run(report_path: Path, steps: int = 240, seed: int = 23) -> dict:
    if steps < 80:
        raise ValueError("steps must be >=80")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    train_text = "\n".join(TRAIN_TEXTS * 8)
    heldout_text = "\n".join(HELDOUT_TEXTS * 4)
    if decode(encode(train_text)) != train_text or decode(encode(heldout_text)) != heldout_text:
        raise RuntimeError("UTF-8 byte round-trip failed")
    train = torch.tensor(encode(train_text), dtype=torch.long)
    heldout = torch.tensor(encode(heldout_text), dtype=torch.long)
    model = HindiAutoregressiveByteLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial_train = evaluate(model, train, seed + 1)
    initial_heldout = evaluate(model, heldout, seed + 2)

    for _ in range(steps):
        model.train()
        x, y = make_batch(train, 16, SEQ_LEN)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
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

    final_train = evaluate(model, train, seed + 1)
    final_heldout = evaluate(model, heldout, seed + 2)
    if final_train >= initial_train:
        raise RuntimeError("training loss did not decrease")
    sample = generate(model, "नमस्ते।", max_new_bytes=96, seed=seed + 3)
    if not sample or len(sample) <= len("नमस्ते।"):
        raise RuntimeError("autoregressive generation returned no continuation")

    with torch.no_grad():
        reference, _ = model(heldout[:64].unsqueeze(0))
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "steps": steps, "seed": seed}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-byte-lm-") as td:
        checkpoint = Path(td) / "hindi_byte_lm.pt"
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = HindiAutoregressiveByteLM()
        clone.load_state_dict(restored["model"], strict=True)
        clone.eval()
        with torch.no_grad():
            reloaded, _ = clone(heldout[:64].unsqueeze(0))
        max_diff = float((reference - reloaded).abs().max())
        if max_diff != 0.0:
            raise RuntimeError("checkpoint reload changed logits")
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.003)
        clone_optimizer.load_state_dict(restored["optimizer"])

    result = {
        "status": "HINDI_AUTOREGRESSIVE_LM_SMOKE_PASS",
        "experiment": "synthetic_utf8_byte_gru_next_token_v1",
        "task": "autoregressive next-byte prediction and sampled continuation",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokenizer": {"type": "UTF-8 byte-level", "vocabulary_size": VOCAB_SIZE,
                      "unicode_round_trip": True},
        "train_examples": len(TRAIN_TEXTS),
        "heldout_examples": len(HELDOUT_TEXTS),
        "heldout_used_for_gradient_updates": False,
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "initial_heldout_loss": initial_heldout,
        "final_heldout_loss": final_heldout,
        "generation": {"prompt": "नमस्ते।", "max_new_bytes": 96,
                       "sample": sample, "seed": seed + 3,
                       "sampling": "temperature=0.85, top_k=12"},
        "checkpoint": {"sha256": digest, "reload_max_abs_logit_difference": max_diff,
                       "optimizer_state_restored": True},
        "provenance": {"kind": "original_synthetic_toy_text_only",
                       "external_data_used": False, "repository_corpus_used": False,
                       "user_data_used": False},
        "production_training_gate": "CLOSED",
        "note": ("Tiny CPU byte-level language-model smoke test only. Generated text may be "
                 "nonsensical; held-out loss on five toy sentences is not evidence of broad "
                 "Hindi ability, factual accuracy, or production readiness.")
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    print(json.dumps(run(args.report_path, args.steps, args.seed), ensure_ascii=False, indent=2))
