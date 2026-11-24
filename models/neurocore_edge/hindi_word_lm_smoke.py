#!/usr/bin/env python3
"""Synthetic-only Hindi word-level autoregressive language-model smoke test.

Whitespace-delimited Hindi orthographic words (including attached punctuation)
are tokens. The model predicts the next word and generates continuations.
All text is original toy content embedded below; no external, repository, web,
or user data is read. This is a tiny research smoke test, not a Hindi assistant.
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
    "सूरज सुबह पूर्व दिशा से निकलता है।",
    "चिड़िया पेड़ पर बैठकर मधुर गीत गाती है।",
    "किसान खेत में गेहूँ और चावल उगाते हैं।",
    "विद्यार्थी पुस्तकालय में नई किताबें पढ़ते हैं।",
    "शिक्षक बच्चों को विज्ञान समझाते हैं।",
    "नदी पहाड़ों से निकलकर मैदानों में बहती है।",
    "पेड़ हमें फल छाया और स्वच्छ हवा देते हैं।",
    "पानी सभी जीवों के लिए बहुत आवश्यक है।",
    "परिवार शाम को साथ बैठकर भोजन करता है।",
    "माली बगीचे में सुंदर फूल लगाता है।",
    "मछलियाँ पानी में तैरती हैं।",
    "बादल आकाश में धीरे धीरे चलते हैं।",
    "बारिश से खेतों में हरियाली आती है।",
    "अच्छा मित्र कठिन समय में साथ देता है।",
    "लोग मिलकर गाँव की सफाई करते हैं।",
    "डॉक्टर मरीजों का ध्यान रखते हैं।",
    "खिलाड़ी मैदान में नियमित अभ्यास करते हैं।",
    "लेखक नए विचारों पर कहानी लिखता है।",
    "बच्चे खेल के बाद घर लौटते हैं।",
    "संगीत सुनकर मन प्रसन्न होता है।",
    "वैज्ञानिक प्रयोगों से नए तथ्य खोजते हैं।",
    "यात्री बस से शहर जाते हैं।",
    "माँ बच्चों को रोचक कहानी सुनाती है।",
    "पुस्तकें हमें ज्ञान और कल्पना देती हैं।",
    "सुबह की ठंडी हवा मन को अच्छा लगती है।",
    "छात्र प्रश्न पूछकर अपनी शंका दूर करते हैं।",
    "लंबे पेड़ सड़क के किनारे खड़े हैं।",
    "रात में आकाश में तारे चमकते हैं।",
    "स्वच्छ पानी पीना स्वास्थ्य के लिए जरूरी है।",
    "पर्यावरण की रक्षा करना सबकी जिम्मेदारी है।",
    "मजदूर सड़क बनाने में मेहनत करते हैं।",
    "गाय हरी घास खाती है।",
    "बच्चे मैदान में फुटबॉल खेलते हैं।",
    "माँ सुबह बच्चों के लिए नाश्ता बनाती है।",
    "दादी शाम को पुरानी कहानियाँ सुनाती हैं।",
    "शिक्षक विद्यार्थियों को हिंदी पढ़ाते हैं।",
    "कुम्हार मिट्टी से सुंदर बर्तन बनाता है।",
    "डाकिया घर घर चिट्ठियाँ पहुँचाता है।",
    "कुत्ता घर की रखवाली करता है।",
    "बिल्ली धूप में आराम करती है।",
    "बगीचे में रंग बिरंगे फूल खिलते हैं।",
    "नाव नदी के पानी पर चलती है।",
    "पहाड़ों पर बर्फ गिरती है।",
    "सैनिक देश की सीमा की रक्षा करते हैं।",
    "कलाकार सुंदर चित्र बनाते हैं।",
    "दुकानदार ग्राहकों को सामान देता है।",
    "बच्चे मिलकर पेड़ लगाते हैं।",
    "लोग साफ पानी बचाने का प्रयास करते हैं।",
    "रेल गाड़ी समय पर स्टेशन पहुँचती है।",
    "कक्षा में विद्यार्थी ध्यान से सुनते हैं।",
]
HELDOUT_TEXTS = [
    "विद्यार्थी कक्षा में ध्यान से सुनते हैं।",
    "बारिश से बगीचे में सुंदर फूल खिलते हैं।",
    "दादी शाम को रोचक कहानी सुनाती हैं।",
    "मछलियाँ नदी के पानी में तैरती हैं।",
    "यात्री शाम को शहर जाते हैं।",
    "पेड़ सड़क के किनारे छाया देते हैं।",
    "शिक्षक कक्षा में विद्यार्थियों को हिंदी पढ़ाते हैं।",
]
EOS = "<eos>"
UNK = "<unk>"
SEQ_LEN = 12
BATCH_SIZE = 32
TRAIN_REPETITIONS = 10
HELDOUT_REPETITIONS = 4

# New word contexts with a manually specified plausible continuation and
# distractor. These are small diagnostics, not a general Hindi grammar benchmark.
CONTRAST_PROBES = [
    ("विद्यार्थी कक्षा में", "ध्यान", "चावल"),
    ("बारिश से बगीचे में", "सुंदर", "कहानी"),
    ("दादी शाम को", "रोचक", "उगाते"),
    ("मछलियाँ नदी के पानी में", "तैरती", "पढ़ते"),
    ("यात्री शाम को शहर", "जाते", "खाती"),
    ("शिक्षक कक्षा में विद्यार्थियों को", "हिंदी", "मधुर"),
]


def tokenize(text: str) -> list[str]:
    # str.split keeps each Hindi orthographic word, including its matras,
    # nukta and any punctuation attached to that word, intact.
    return text.split()


class HindiWordLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, state=None):
        hidden, state = self.gru(self.embedding(token_ids), state)
        return self.output(hidden), state


def encode_sentence_stream(
    sentences: list[str], stoi: dict[str, int], repetitions: int
) -> list[int]:
    ids: list[int] = []
    for _ in range(repetitions):
        for sentence in sentences:
            ids.extend(stoi.get(token, stoi[UNK]) for token in tokenize(sentence))
            ids.append(stoi[EOS])
    return ids


def make_batch(data: torch.Tensor, batch_size: int = BATCH_SIZE):
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (batch_size,))
    x = torch.stack([data[int(i):int(i) + SEQ_LEN] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + SEQ_LEN + 1] for i in starts])
    return x, y


def evaluate(model, data: torch.Tensor, vocab_size: int, unk_id: int) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(data) - SEQ_LEN - 1, SEQ_LEN + 1):
            x = data[start:start + SEQ_LEN].unsqueeze(0)
            y = data[start + 1:start + SEQ_LEN + 1].unsqueeze(0)
            logits, _ = model(x)
            logits = logits.reshape(-1, vocab_size)
            targets = y.reshape(-1)
            known = targets != unk_id
            if bool(known.any()):
                selected_logits = logits[known]
                selected_targets = targets[known]
                total_loss += float(F.cross_entropy(
                    selected_logits, selected_targets, reduction="sum"
                ))
                correct += int((selected_logits.argmax(dim=-1) == selected_targets).sum())
                total_tokens += int(known.sum())
    if total_tokens == 0:
        raise RuntimeError("no in-vocabulary tokens available for evaluation")
    return {
        "loss": total_loss / total_tokens,
        "next_word_accuracy": correct / total_tokens,
        "scored_tokens": total_tokens,
    }


def generate(
    model,
    prompt: str,
    stoi: dict[str, int],
    itos: list[str],
    unk_id: int,
    max_new_words: int = 16,
) -> str:
    model.eval()
    prompt_tokens = tokenize(prompt)
    if not prompt_tokens:
        raise ValueError("generation prompt must contain at least one token")
    ids = [stoi.get(token, unk_id) for token in prompt_tokens]
    generated: list[str] = []
    with torch.no_grad():
        logits, state = model(torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long))
        next_logits = logits[0, -1].clone()
        for step in range(max_new_words):
            next_logits[unk_id] = float("-inf")
            if step < 3:
                next_logits[stoi[EOS]] = float("-inf")
            selected = int(next_logits.argmax().item())
            if selected == stoi[EOS]:
                break
            generated.append(itos[selected])
            next_out, state = model(torch.tensor([[selected]], dtype=torch.long), state)
            next_logits = next_out[0, -1].clone()
    return " ".join(prompt_tokens + generated)


def score_contrasts(model, stoi: dict[str, int], unk_id: int) -> list[dict]:
    model.eval()
    results = []
    with torch.no_grad():
        for context, expected, distractor in CONTRAST_PROBES:
            context_ids = [stoi.get(token, unk_id) for token in tokenize(context)]
            if expected not in stoi or distractor not in stoi:
                raise RuntimeError("contrast probe contains a token outside training vocabulary")
            logits, _ = model(torch.tensor([context_ids], dtype=torch.long))
            next_logits = logits[0, -1]
            margin = float(next_logits[stoi[expected]] - next_logits[stoi[distractor]])
            results.append({
                "context": context,
                "expected_next_word": expected,
                "distractor": distractor,
                "logit_margin": margin,
                "expected_preferred": margin > 0,
            })
    return results


def run(report_path: Path, steps: int = 50, seed: int = 11) -> dict:
    if steps < 30:
        raise ValueError("steps must be >=30")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    # The vocabulary is derived from training sentences only.
    training_tokens = sorted({token for sentence in TRAIN_TEXTS for token in tokenize(sentence)})
    itos = training_tokens + [EOS, UNK]
    stoi = {token: index for index, token in enumerate(itos)}
    unk_id = stoi[UNK]
    eos_id = stoi[EOS]
    vocab_size = len(itos)

    for sentence in TRAIN_TEXTS + HELDOUT_TEXTS:
        if " ".join(tokenize(sentence)) != sentence:
            raise RuntimeError("whitespace tokenizer round-trip failed")

    train = torch.tensor(
        encode_sentence_stream(TRAIN_TEXTS, stoi, TRAIN_REPETITIONS), dtype=torch.long
    )
    heldout = torch.tensor(
        encode_sentence_stream(HELDOUT_TEXTS, stoi, HELDOUT_REPETITIONS), dtype=torch.long
    )
    unknown_heldout = sorted({
        token for sentence in HELDOUT_TEXTS for token in tokenize(sentence)
        if token not in stoi
    })
    if unknown_heldout:
        raise RuntimeError(f"held-out corpus contains OOV tokens: {unknown_heldout}")

    model = HindiWordLM(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    initial_train = evaluate(model, train, vocab_size, unk_id)
    initial_heldout = evaluate(model, heldout, vocab_size, unk_id)

    for _ in range(steps):
        model.train()
        x, y = make_batch(train)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or not all(bool(torch.isfinite(gradient).all()) for gradient in grads):
            raise RuntimeError("missing or non-finite gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()

    final_train = evaluate(model, train, vocab_size, unk_id)
    final_heldout = evaluate(model, heldout, vocab_size, unk_id)
    if final_train["loss"] >= initial_train["loss"]:
        raise RuntimeError("training next-word loss did not decrease")
    if final_heldout["loss"] >= initial_heldout["loss"]:
        raise RuntimeError("held-out next-word loss did not decrease")
    if final_heldout["next_word_accuracy"] < 0.30:
        raise RuntimeError("held-out next-word accuracy below smoke-test threshold")

    generated = {
        prompt: generate(model, prompt, stoi, itos, unk_id)
        for prompt in ("किसान", "बच्चे", "शिक्षक")
    }
    for prompt, text in generated.items():
        if not text.startswith(prompt) or len(tokenize(text)) <= 1:
            raise RuntimeError(f"generation failed for prompt: {prompt}")
        if any(token not in stoi for token in tokenize(text)):
            raise RuntimeError("generation emitted a token outside the training vocabulary")

    contrast_results = score_contrasts(model, stoi, unk_id)
    preferred_count = sum(row["expected_preferred"] for row in contrast_results)
    if preferred_count < 5:
        raise RuntimeError(f"too few preferred semantic/grammar continuations: {preferred_count}/6")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": steps,
        "seed": seed,
        "stoi": stoi,
        "itos": itos,
        "unk_id": unk_id,
        "vocab_size": vocab_size,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hindi-word-lm-") as directory:
        checkpoint = Path(directory) / "hindi_word_lm.pt"
        torch.save(state, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = HindiWordLM(restored["vocab_size"])
        clone.load_state_dict(restored["model"], strict=True)
        clone.eval()
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.004)
        clone_optimizer.load_state_dict(restored["optimizer"])
        restored_samples = {
            prompt: generate(clone, prompt, restored["stoi"], restored["itos"], restored["unk_id"])
            for prompt in generated
        }
        if restored_samples != generated:
            raise RuntimeError("checkpoint reload changed generated continuations")

    report = {
        "status": "HINDI_WORD_LM_SMOKE_PASS",
        "experiment": "synthetic_whitespace_word_gru_next_word_v1",
        "task": "autoregressive next-word prediction and greedy word continuation",
        "device": "cpu",
        "seed": seed,
        "training_steps": steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer": {
            "type": "whitespace-delimited orthographic word",
            "vocabulary_size_including_special_tokens": vocab_size,
            "training_word_vocabulary_size": len(training_tokens),
            "punctuation_attached_to_word": True,
            "vocabulary_built_from_training_only": True,
        },
        "train_examples": len(TRAIN_TEXTS),
        "heldout_examples": len(HELDOUT_TEXTS),
        "heldout_repetitions_for_scoring": HELDOUT_REPETITIONS,
        "heldout_used_for_gradient_updates": False,
        "heldout_unknown_words": unknown_heldout,
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_heldout": initial_heldout,
        "final_heldout": final_heldout,
        "generation": {
            "samples": generated,
            "decoder": "greedy; UNK suppressed; EOS suppressed for first 3 generated words",
        },
        "grammar_semantic_probes": {
            "kind": "small manually curated next-word preference diagnostics",
            "preferred_count": preferred_count,
            "probe_count": len(contrast_results),
            "results": contrast_results,
            "note": "Not an independent Hindi grammar benchmark or general semantic evaluation.",
        },
        "checkpoint": {
            "sha256": digest,
            "generated_continuations_equal_after_reload": True,
            "optimizer_state_restored": True,
        },
        "provenance": {
            "kind": "original_synthetic_toy_text_only",
            "external_data_used": False,
            "repository_corpus_used": False,
            "user_data_used": False,
        },
        "production_training_gate": "CLOSED",
        "note": (
            "Tiny CPU word-level language-model smoke test only. Whole-word tokens prevent "
            "partial UTF-8 fragments, and outputs are restricted to words seen in synthetic "
            "training text. This does not guarantee grammar or meaning. Seven held-out toy "
            "sentences and six manually designed probes do not establish broad Hindi ability, "
            "factual accuracy, or production readiness."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.report_path, arguments.steps, arguments.seed), ensure_ascii=False, indent=2))
