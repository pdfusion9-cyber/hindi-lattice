#!/usr/bin/env python3
"""Synthetic Hindi QA/dialogue intent-to-response smoke test.

A byte-level character CNN selects a response class from a Hindi prompt, then
maps that class to a fixed toy reply. This is deliberately *not* free-form
language generation. All examples are original synthetic strings embedded
here; no external, repository, web, or user data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

PAD_ID = 256
VOCAB_SIZE = 257
SEED = 11
MAX_BYTES = 384

RESPONSES = {
    "india_capital": "नई दिल्ली",
    "nepal_capital": "काठमांडू",
    "japan_capital": "टोक्यो",
    "water_formula": "H2O",
    "week_days": "सात दिन",
    "greeting": "नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।",
    "thanks": "आपका स्वागत है।",
    "farewell": "अलविदा! आपका दिन शुभ हो।",
    "help": "ज़रूर, बताइए आपको किस विषय में सहायता चाहिए।",
}

TRAIN_PAIRS = [
    ("प्रश्न: भारत की राजधानी क्या है?", "india_capital"),
    ("प्रश्न: भारत का राजधानी शहर बताइए।", "india_capital"),
    ("प्रश्न: भारत की राजधानी कौन-सी है?", "india_capital"),
    ("प्रश्न: भारत में राजधानी किस शहर को कहते हैं?", "india_capital"),
    ("प्रश्न: नेपाल की राजधानी क्या है?", "nepal_capital"),
    ("प्रश्न: नेपाल का राजधानी शहर बताइए।", "nepal_capital"),
    ("प्रश्न: नेपाल की राजधानी कौन-सी है?", "nepal_capital"),
    ("प्रश्न: नेपाल में राजधानी किस नगर को कहते हैं?", "nepal_capital"),
    ("प्रश्न: जापान की राजधानी क्या है?", "japan_capital"),
    ("प्रश्न: जापान का राजधानी शहर बताइए।", "japan_capital"),
    ("प्रश्न: जापान की राजधानी कौन-सी है?", "japan_capital"),
    ("प्रश्न: जापान में राजधानी किस नगर को कहते हैं?", "japan_capital"),
    ("प्रश्न: पानी का रासायनिक सूत्र क्या है?", "water_formula"),
    ("प्रश्न: पानी का सूत्र बताइए।", "water_formula"),
    ("प्रश्न: जल का रासायनिक सूत्र क्या है?", "water_formula"),
    ("प्रश्न: H2O किसका रासायनिक सूत्र है?", "water_formula"),
    ("प्रश्न: एक सप्ताह में कितने दिन होते हैं?", "week_days"),
    ("प्रश्न: सप्ताह के कुल दिनों की संख्या बताइए।", "week_days"),
    ("प्रश्न: एक हफ्ते में कितने दिन होते हैं?", "week_days"),
    ("प्रश्न: सप्ताह में दिनों की संख्या क्या है?", "week_days"),
    ("संवाद: नमस्ते।", "greeting"),
    ("संवाद: नमस्ते जी।", "greeting"),
    ("संवाद: सुप्रभात।", "greeting"),
    ("संवाद: प्रणाम।", "greeting"),
    ("संवाद: धन्यवाद।", "thanks"),
    ("संवाद: बहुत शुक्रिया।", "thanks"),
    ("संवाद: आभार।", "thanks"),
    ("संवाद: आपका धन्यवाद।", "thanks"),
    ("संवाद: फिर मिलेंगे।", "farewell"),
    ("संवाद: अलविदा।", "farewell"),
    ("संवाद: अब मैं चलता हूँ।", "farewell"),
    ("संवाद: विदा लेते हैं।", "farewell"),
    ("संवाद: कृपया सहायता कीजिए।", "help"),
    ("संवाद: मुझे मदद चाहिए।", "help"),
    ("संवाद: क्या आप मदद कर सकते हैं?", "help"),
    ("संवाद: मेरी सहायता कीजिए।", "help"),
]

# Separately worded examples. They are never included in optimizer batches.
HELDOUT_PAIRS = [
    ("प्रश्न: भारत की राजधानी किस नगर को कहा जाता है?", "india_capital"),
    ("प्रश्न: नेपाल में राजधानी कौन-सा शहर है?", "nepal_capital"),
    ("प्रश्न: जापान का मुख्य राजधानी नगर कौन-सा है?", "japan_capital"),
    ("प्रश्न: जल का रासायनिक सूत्र बताइए।", "water_formula"),
    ("प्रश्न: एक सप्ताह में कुल दिनों की संख्या कितनी है?", "week_days"),
    ("संवाद: नमस्कार जी।", "greeting"),
    ("संवाद: आपका बहुत धन्यवाद।", "thanks"),
    ("संवाद: फिर मिलते हैं।", "farewell"),
    ("संवाद: मुझे सहायता चाहिए, कृपया।", "help"),
]


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def batch_encode(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode(text)[:MAX_BYTES] for text in texts]
    lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    width = max(int(lengths.max().item()), 1)
    ids = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
    for i, row in enumerate(encoded):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return ids, lengths


class HindiIntentSelector(nn.Module):
    """Byte n-gram CNN that selects a response class from a prompt."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 24, padding_idx=PAD_ID)
        self.conv3 = nn.Conv1d(24, 48, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(24, 48, kernel_size=5, padding=2)
        self.hidden = nn.Linear(48 * 4, 96)
        self.output = nn.Linear(96, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        mask = mask < lengths.to(input_ids.device)[:, None]
        mask_f = mask.unsqueeze(1).to(embedded.dtype)

        c3 = F.gelu(self.conv3(embedded))
        c5 = F.gelu(self.conv5(embedded))
        neg = torch.finfo(c3.dtype).min
        max3 = c3.masked_fill(~mask.unsqueeze(1), neg).amax(dim=-1)
        max5 = c5.masked_fill(~mask.unsqueeze(1), neg).amax(dim=-1)
        mean3 = (c3 * mask_f).sum(dim=-1) / lengths.to(c3.device).clamp_min(1).unsqueeze(1)
        mean5 = (c5 * mask_f).sum(dim=-1) / lengths.to(c5.device).clamp_min(1).unsqueeze(1)
        pooled = torch.cat([max3, mean3, max5, mean5], dim=-1)
        return self.output(F.gelu(self.hidden(pooled)))


def score(model, prompts: list[str], labels: list[str], label_to_id: dict[str, int]):
    model.eval()
    ids, lengths = batch_encode(prompts)
    with torch.no_grad():
        logits = model(ids, lengths)
    pred_ids = logits.argmax(dim=-1).tolist()
    predictions = [list(label_to_id.keys())[list(label_to_id.values()).index(i)] for i in pred_ids]
    correct = sum(int(p == y) for p, y in zip(predictions, labels))
    samples = [
        {
            "prompt": prompt,
            "expected_intent": target,
            "predicted_intent": prediction,
            "expected_response": RESPONSES[target],
            "predicted_response": RESPONSES[prediction],
            "exact_match": prediction == target,
        }
        for prompt, target, prediction in zip(prompts, labels, predictions)
    ]
    return {
        "correct": correct,
        "example_count": len(labels),
        "accuracy": correct / max(1, len(labels)),
        "samples": samples,
    }


def run_experiment(report_path: Path, steps: int = 1200, seed: int = SEED) -> dict[str, Any]:
    if steps < 200:
        raise ValueError("steps must be >=200")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    classes = sorted(RESPONSES)
    label_to_id = {label: i for i, label in enumerate(classes)}
    train_prompts = [x[0] for x in TRAIN_PAIRS]
    train_labels = [x[1] for x in TRAIN_PAIRS]
    heldout_prompts = [x[0] for x in HELDOUT_PAIRS]
    heldout_labels = [x[1] for x in HELDOUT_PAIRS]

    # Confirm all Hindi/ASCII content round-trips through UTF-8 bytes.
    for prompt, label in TRAIN_PAIRS + HELDOUT_PAIRS:
        if bytes(encode(prompt)).decode("utf-8") != prompt:
            raise RuntimeError("UTF-8 tokenizer round-trip failed")
        if label not in RESPONSES:
            raise RuntimeError(f"unregistered response class: {label}")

    train_ids, train_lengths = batch_encode(train_prompts)
    train_targets = torch.tensor([label_to_id[x] for x in train_labels], dtype=torch.long)
    model = HindiIntentSelector(num_classes=len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0005)

    initial_train = score(model, train_prompts, train_labels, label_to_id)
    initial_heldout = score(model, heldout_prompts, heldout_labels, label_to_id)
    initial_train_loss = float(F.cross_entropy(model(train_ids, train_lengths), train_targets).item())

    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_ids, train_lengths)
        loss = F.cross_entropy(logits, train_targets)
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

    final_train = score(model, train_prompts, train_labels, label_to_id)
    final_heldout = score(model, heldout_prompts, heldout_labels, label_to_id)
    with torch.no_grad():
        final_train_loss = float(F.cross_entropy(model(train_ids, train_lengths), train_targets).item())

    if final_train_loss >= initial_train_loss:
        raise RuntimeError("training classification loss did not decrease")
    if final_train["accuracy"] < 0.95:
        raise RuntimeError(f"train intent accuracy below 95%: {final_train['accuracy']:.3f}")
    if final_heldout["accuracy"] < 0.70:
        raise RuntimeError(f"held-out intent accuracy below 70%: {final_heldout['accuracy']:.3f}")

    checkpoint_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "classes": classes,
        "responses": RESPONSES,
        "steps": steps,
        "seed": seed,
        "vocab_size": VOCAB_SIZE,
        "pad_id": PAD_ID,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-intent-") as td:
        checkpoint_path = Path(td) / "hindi_intent_selector.pt"
        torch.save(checkpoint_state, checkpoint_path)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        clone = HindiIntentSelector(num_classes=len(restored["classes"]))
        clone.load_state_dict(restored["model_state"], strict=True)
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.008, weight_decay=0.0005)
        clone_optimizer.load_state_dict(restored["optimizer_state"])
        clone_score = score(clone, heldout_prompts, heldout_labels, label_to_id)
        if clone_score["samples"] != final_heldout["samples"]:
            raise RuntimeError("checkpoint reload predictions differ")

    report = {
        "status": "HINDI_QA_DIALOGUE_SMOKE_PASS",
        "experiment": "supervised_hindi_intent_to_response_cnn_v2",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokenizer": {
            "type": "UTF-8 byte-level",
            "vocabulary_size": VOCAB_SIZE,
            "byte_vocabulary_size": 256,
            "pad_id": PAD_ID,
            "unicode_round_trip": True,
        },
        "task": "supervised intent classification with fixed response mapping",
        "classes": classes,
        "train_examples": len(TRAIN_PAIRS),
        "heldout_examples": len(HELDOUT_PAIRS),
        "heldout_used_for_gradient_updates": False,
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_heldout": initial_heldout,
        "final_heldout": final_heldout,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "checkpoint": {
            "sha256": digest,
            "optimizer_state_restored": True,
            "predictions_equal_after_reload": True,
        },
        "provenance": {
            "kind": "synthetic_only",
            "external_data_used": False,
            "repository_corpus_used": False,
            "user_data_used": False,
        },
        "production_training_gate": "CLOSED",
        "note": (
            "This is a tiny intent-to-fixed-response classifier, not free-form text generation. "
            "Held-out results measure only template generalization on toy examples, not factual "
            "knowledge or broad Hindi competence."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run_experiment(args.report_path, steps=args.steps, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
