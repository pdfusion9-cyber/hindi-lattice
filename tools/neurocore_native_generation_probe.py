#!/usr/bin/env python3
"""Probe native NeuroCore byte-generation on held-out Hindi fact triples.

This is an exploratory corpus-format completion probe, not a broad language
understanding benchmark. It uses only the project's own checkpoint and the
held-out split from data/gyaan_base.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "neurocore_edge"
sys.path.insert(0, str(MODEL_DIR))
from modeling_neurocore_edge import (  # noqa: E402
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)


def held_out_rows(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    train_rows, valid_rows = [], []
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed corpus row: {line!r}")
        bucket = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest()[:4], "big") % 100
        (valid_rows if bucket < 2 else train_rows).append(parts)
    if len(valid_rows) < 2:
        raise ValueError("held-out validation split is too small")
    return raw, valid_rows


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return " ".join(value.split()).casefold()


def edit_similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(a), len(b), 1)


@torch.inference_mode()
def generate(model, prompt: str, max_new_bytes: int) -> tuple[str, bool, int]:
    token_ids = list(prompt.encode("utf-8"))
    sequence = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    generated = bytearray()
    for _ in range(max_new_bytes):
        logits = model(sequence)["logits"][0, -1]
        next_byte = int(torch.argmax(logits).item())
        if next_byte == 10:  # newline ends a triple completion
            break
        generated.append(next_byte)
        sequence = torch.cat(
            [sequence, torch.tensor([[next_byte]], dtype=torch.long)], dim=1
        )
    try:
        decoded = bytes(generated).decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        decoded = bytes(generated).decode("utf-8", errors="replace")
        valid_utf8 = False
    return decoded.strip(), valid_utf8, len(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-new-bytes", type=int, default=96)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()

    if args.samples < 1 or args.max_new_bytes < 1:
        parser.error("samples and max-new-bytes must be positive")
    torch.set_num_threads(args.cpu_threads)
    corpus = (ROOT / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    actual_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{corpus.relative_to(ROOT).as_posix()}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    raw, rows = held_out_rows(corpus)
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("model_id") != "neurocore_edge_native_reference":
        raise ValueError("checkpoint model ID mismatch")
    if saved.get("corpus_git_blob_sha1") != actual_blob:
        raise ValueError("checkpoint corpus Git blob mismatch")
    if saved.get("corpus_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("checkpoint corpus SHA256 mismatch")
    if saved.get("provenance", {}).get("external_model_weights_used") is not False:
        raise ValueError("checkpoint provenance includes external model weights")
    if saved.get("provenance", {}).get("open_source_model_integrated") is not False:
        raise ValueError("checkpoint provenance includes an external model")
    if saved.get("production_training_gate") != "CLOSED":
        raise ValueError("production gate must remain closed")

    cfg = NeuroCoreEdgeConfig(**saved["config"])
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()

    count = min(args.samples, len(rows))
    if count == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})
    examples = []
    exact = 0
    similarities = []
    utf8_valid = 0
    for index in indices:
        subject, relation, target = rows[index]
        prompt = f"{subject} | {relation} |"
        prediction, valid, generated_bytes = generate(model, prompt, args.max_new_bytes)
        target_norm = normalize(target)
        prediction_norm = normalize(prediction)
        is_exact = prediction_norm == target_norm
        exact += int(is_exact)
        similarities.append(edit_similarity(prediction, target))
        utf8_valid += int(valid)
        examples.append({
            "prompt": prompt,
            "expected_object": target,
            "generated_object": prediction,
            "exact_match": is_exact,
            "character_similarity": round(similarities[-1], 6),
            "valid_utf8": valid,
            "generated_bytes": generated_bytes,
        })

    report = {
        "status": "NATIVE_HELDOUT_HINDI_GENERATION_PROBE_COMPLETE",
        "scope": "exploratory next-byte greedy completion of held-out subject-relation prefixes",
        "model_id": saved["model_id"],
        "capacity_profile": saved.get("capacity_profile"),
        "model_parameter_count": count_parameters(model),
        "checkpoint_step": int(saved.get("step", 0)),
        "corpus_git_blob_sha1": actual_blob,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "validation_rows_available": len(rows),
        "samples_evaluated": len(examples),
        "decoding": {"method": "greedy_argmax", "max_new_bytes": args.max_new_bytes},
        "metrics": {
            "object_exact_matches": exact,
            "object_exact_match_accuracy": exact / max(len(examples), 1),
            "mean_normalized_character_similarity": sum(similarities) / max(len(similarities), 1),
            "valid_utf8_outputs": utf8_valid,
            "valid_utf8_fraction": utf8_valid / max(len(examples), 1),
        },
        "provenance": saved.get("provenance"),
        "production_training_gate": saved.get("production_training_gate"),
        "limitations": [
            "This measures only held-out fact-triple object completion, not general Hindi fluency or factual reliability.",
            "The tokenizer is UTF-8 bytes, not a production Hindi tokenizer.",
            "Greedy decoding and a small held-out corpus make this a diagnostic probe, not a human evaluation.",
        ],
        "examples": examples,
    }
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "checkpoint_step": report["checkpoint_step"],
        "samples_evaluated": report["samples_evaluated"],
        **report["metrics"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
