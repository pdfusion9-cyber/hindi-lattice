#!/usr/bin/env python3
"""Resume the native relation-balanced whole-word NeuroCore model."""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, time, unicodedata
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "neurocore_edge"))
from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM, count_parameters

UNK, BOS, EOS, SEP, PAD = range(5)
EXPECTED_BLOB = "3018157a78e6a35877e114b9297f8ae9e16a76ae"


def corpus_rows(path):
    raw = path.read_bytes()
    train, valid, counts = [], [], defaultdict(int)
    for line in (x.strip() for x in raw.decode("utf-8").splitlines() if x.strip()):
        parts = [unicodedata.normalize("NFC", x.strip()) for x in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed triple: {line!r}")
        fields = [p.split() for p in parts]
        if not all(fields):
            raise ValueError(f"empty triple field: {line!r}")
        for token in sum(fields, []):
            counts[token] += 1
        bucket = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest()[:4], "big") % 100
        (valid if bucket < 2 else train).append((line, fields))
    return raw, train, valid, counts


def encode(fields, word_to_id):
    subject, relation, obj = fields
    prompt = [BOS] + [word_to_id.get(w, UNK) for w in subject]
    prompt += [SEP] + [word_to_id.get(w, UNK) for w in relation] + [SEP]
    target = [word_to_id.get(w, UNK) for w in obj] + [EOS]
    return prompt, target


def collate(examples):
    max_len = max(len(p) + len(t) - 1 for p, t in examples)
    xs, ys, masks = [], [], []
    for prompt, target in examples:
        full = prompt + target
        x, y = full[:-1], full[1:]
        mask = [False] * (len(prompt) - 1) + [True] * len(target)
        pad = max_len - len(x)
        xs.append(x + [PAD] * pad)
        ys.append(y + [-100] * pad)
        masks.append(mask + [False] * pad)
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(masks, dtype=torch.bool))


@torch.inference_mode()
def greedy(model, prompt, max_words):
    seq = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    out = []
    for _ in range(max_words):
        token = int(model(seq)["logits"][0, -1].argmax().item())
        if token in (BOS, SEP, PAD, EOS):
            break
        out.append(token)
        seq = torch.cat([seq, torch.tensor([[token]], dtype=torch.long)], dim=1)
    return out


@torch.no_grad()
def evaluate(model, rows, word_to_id, batch_size):
    model.eval()
    total_loss = total = top1 = top5 = 0
    exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        examples = [encode(fields, word_to_id) for _, fields in batch]
        x, y, mask = collate(examples)
        logits = model(x)["logits"][mask]
        targets = y[mask]
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum").item())
        pred = logits.argmax(dim=-1)
        top1 += int((pred == targets).sum().item())
        candidates = logits.topk(5, dim=-1).indices
        top5 += int((candidates == targets.unsqueeze(-1)).any(dim=-1).sum().item())
        total += targets.numel()
        for (_, fields), (prompt, target) in zip(batch, examples):
            predicted = greedy(model, prompt, max_words=max(12, len(target) + 1))
            exact += int(predicted == target[:-1])
    model.train()
    return {
        "object_word_loss": total_loss / max(total, 1),
        "object_word_top1": top1 / max(total, 1),
        "object_word_top5": top5 / max(total, 1),
        "predicted_object_words": total,
        "exact_object_matches": exact,
        "evaluated_rows": len(rows),
        "exact_object_match": exact / max(len(rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/gyaan_base.txt"))
    ap.add_argument("--resume-checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=0.00005)
    ap.add_argument("--seed", type=int, default=173)
    ap.add_argument("--report", type=Path, default=Path("reports/neurocore-word-prediction-balanced-8000.json"))
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/neurocore-word-prediction-balanced-8000.pt"))
    args = ap.parse_args()

    corpus = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    checkpoint_path = args.resume_checkpoint if args.resume_checkpoint.is_absolute() else ROOT / args.resume_checkpoint
    tokenizer_path = args.tokenizer if args.tokenizer.is_absolute() else ROOT / args.tokenizer
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{corpus.relative_to(ROOT).as_posix()}"],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if blob != EXPECTED_BLOB:
        raise RuntimeError(f"unexpected corpus Git blob {blob}")
    raw, train_rows, valid_rows, counts = corpus_rows(corpus)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_sha = hashlib.sha256(json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if ck.get("model_id") != "neurocore_edge_native_reference_word_balanced_v2":
        raise RuntimeError("checkpoint model ID mismatch")
    if ck.get("corpus_git_blob_sha1") != blob or ck.get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("checkpoint corpus/tokenizer provenance mismatch")
    if ck.get("provenance", {}).get("external_model_weights_used") is not False:
        raise RuntimeError("external model weights detected")
    if ck.get("provenance", {}).get("open_source_model_integrated") is not False:
        raise RuntimeError("external base model integration detected")

    id_to_word = tokenizer["id_to_word"]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    if len(id_to_word) != ck["config"]["vocab_size"] or set(counts) - set(word_to_id):
        raise RuntimeError("vocabulary mismatch with corpus")
    relation_groups = defaultdict(list)
    for row in train_rows:
        relation_groups[" ".join(row[1][1])].append(row)
    relations = sorted(relation_groups)
    if len(relations) < 20:
        raise RuntimeError("expected at least 20 relation groups")

    torch.set_num_threads(args.cpu_threads)
    cfg = NeuroCoreEdgeConfig(**ck["config"])
    model = NeuroCoreEdgeForCausalLM(cfg).cpu()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    start_step = int(ck["step"])
    initial = evaluate(model, valid_rows, word_to_id, args.batch_size)
    py_rng = random.Random(args.seed)
    curve = []
    best_loss = initial["object_word_loss"]
    best_metrics = initial
    best_step = start_step
    began = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(args.batch_size):
            relation = relations[py_rng.randrange(len(relations))]
            batch.append(py_rng.choice(relation_groups[relation]))
        examples = [encode(fields, word_to_id) for _, fields in batch]
        x, y, mask = collate(examples)
        output = model(x)
        logits = output["logits"][mask]
        targets = y[mask]
        loss = F.cross_entropy(logits, targets) + cfg.router_aux_loss_coefficient * output["router_aux_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, valid_rows, word_to_id, args.batch_size)
            entry = {"step": start_step + step, **metrics}
            curve.append(entry)
            if metrics["object_word_loss"] < best_loss:
                best_loss = metrics["object_word_loss"]
                best_metrics = metrics
                best_step = start_step + step
            print(
                f"resume_word_step={start_step + step} loss={metrics['object_word_loss']:.4f} "
                f"top1={metrics['object_word_top1']:.4f} top5={metrics['object_word_top5']:.4f} "
                f"exact={metrics['exact_object_match']:.4f}",
                flush=True,
            )
    elapsed = max(time.perf_counter() - began, 1e-9)
    final = evaluate(model, valid_rows, word_to_id, args.batch_size)
    examples_out = []
    for _, fields in valid_rows[:30]:
        prompt, target = encode(fields, word_to_id)
        ids = greedy(model, prompt, max_words=max(12, len(target) + 1))
        examples_out.append({
            "subject": " ".join(fields[0]),
            "relation": " ".join(fields[1]),
            "expected_words": fields[2],
            "predicted_words": [id_to_word[i] for i in ids],
            "exact_match": ids == target[:-1],
        })

    total_step = start_step + args.steps
    report = {
        "status": "NEUROCORE_NATIVE_RELATION_BALANCED_WORD_RESUME_PASS",
        "objective": "relation_balanced_subject_relation_to_next_object_word_prediction",
        "model_id": ck["model_id"],
        "model_parameter_count": count_parameters(model),
        "vocabulary": {"size": len(id_to_word), "tokenizer_format": tokenizer["format"], "sha256": tokenizer_sha},
        "corpus": {"git_blob_sha1": blob, "sha256": hashlib.sha256(raw).hexdigest(),
                   "train_rows": len(train_rows), "validation_rows": len(valid_rows),
                   "relation_groups": len(relations),
                   "rights_review_status": "UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW"},
        "training": {
            "resumed_from_step": start_step, "additional_steps": args.steps, "total_steps": total_step,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "sampling": "uniform_over_relation_groups_then_uniform_row",
            "initial_validation": initial, "best_validation": best_metrics, "best_validation_step": best_step, "final_validation": final,
            "validation_curve": curve, "elapsed_seconds": elapsed,
            "training_rows_per_second": args.steps * args.batch_size / elapsed,
        },
        "generation_examples": examples_out,
        "provenance": {"external_model_weights_used": False, "open_source_model_integrated": False,
                       "external_tokenizer_used": False, "external_data_used": False},
        "production_training_gate": "CLOSED",
        "production_2_87b_training_started": False,
        "note": "Small experimental fact-triple word predictor; not a production Hindi language model.",
    }
    out_ck = {
        **ck, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "step": total_step, "best_validation": best_metrics, "best_validation_step": best_step,
        "production_training_gate": "CLOSED",
    }
    def outpath(p):
        return p if p.is_absolute() else ROOT / p
    report_path = outpath(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_out = outpath(args.checkpoint); checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    temp = checkpoint_out.with_name(checkpoint_out.name + ".tmp")
    torch.save(out_ck, temp); temp.replace(checkpoint_out)
    print(json.dumps({
        "initial_object_word_loss": initial["object_word_loss"],
        "best_object_word_loss": best_metrics["object_word_loss"],
        "best_top1": best_metrics["object_word_top1"],
        "best_top5": best_metrics["object_word_top5"],
        "best_exact_match": best_metrics["exact_object_match"],
        "final_object_word_loss": final["object_word_loss"],
        "total_steps": total_step,
        "production_gate": "CLOSED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
