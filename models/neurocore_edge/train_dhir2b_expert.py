"""CPU fixed-expert adapter trainer with strict dataset/provenance approval gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from dhir2b_neurocore import Dhir2BNeuroCore

DEFAULT_DATASET_ID = "ai4bharat/IndicCorpV2"
DEFAULT_DATASET_REVISION = "2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f"

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
PAN_RE = re.compile(r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b")


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_dir(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("tokenizer directory contains no files")
    for file_path in files:
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha_file(file_path)))
    return digest.hexdigest()


def require_approval(
    path: str | Path,
    expert_id: int,
    data_sha: str,
    tokenizer_sha: str,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
) -> dict:
    """Require matching human approvals at dataset, corpus-artifact and tokenizer levels."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))

    dataset_row = next((
        row for row in document.get("datasets", [])
        if row.get("expert_id") == expert_id
        and row.get("dataset_id") == dataset_id
        and row.get("revision") == dataset_revision
    ), None)
    if dataset_row is None:
        raise PermissionError("training blocked: exact dataset ID/revision is not registered")
    required_dataset_checks = (
        "commercial_training_allowed",
        "pii_reviewed",
        "consent_reviewed",
        "schema_reviewed",
        "split_reviewed",
        "approved",
    )
    missing_checks = [field for field in required_dataset_checks
                      if dataset_row.get(field) is not True]
    if missing_checks:
        raise PermissionError(
            "training blocked: dataset approval/review fields are incomplete: "
            + ", ".join(missing_checks)
        )
    if not dataset_row.get("license_id") or not dataset_row.get("license_terms_url"):
        raise PermissionError("training blocked: dataset license evidence is missing")
    if not all(dataset_row.get(field) for field in ("approval_id", "reviewer", "reviewed_at")):
        raise PermissionError("training blocked: dataset approval ID/reviewer/date is missing")

    artifact_ok = any(
        row.get("stage") == "dhir2b_expert"
        and row.get("expert_id") == expert_id
        and row.get("dataset_id") == dataset_id
        and row.get("dataset_revision") == dataset_revision
        and row.get("sha256") == data_sha
        and row.get("approved") is True
        and all(row.get(field) for field in ("approval_id", "reviewer", "reviewed_at"))
        for row in document.get("approved_training_artifacts", [])
    )
    if not artifact_ok:
        raise PermissionError(
            "training blocked: exact dataset-bound corpus SHA-256 approval is missing"
        )

    tokenizer_ok = any(
        row.get("sha256") == tokenizer_sha
        and row.get("approved") is True
        and all(row.get(field) for field in ("approval_id", "reviewer", "reviewed_at"))
        for row in document.get("approved_tokenizers", [])
    )
    if not tokenizer_ok:
        raise PermissionError("training blocked: exact tokenizer SHA-256 approval is missing")

    return document


def contains_pii_pattern(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or AADHAAR_RE.search(text)
        or PAN_RE.search(text)
    )


def _as_token_ids(encoded) -> list[int]:
    """Normalize tokenizer outputs without changing native Dhīr token IDs."""
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("tokenizer unexpectedly returned multiple sequences")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def read_rows(path: str | Path, tokenizer, max_length: int, max_rows: int = 5000):
    """Read legacy LM rows or native-chat prompt/response SFT rows.

    Structured rows use {"prompt": ..., "response": ...}. Their loss is masked
    over the prompt and applies only to the assistant completion. The prompt
    and full conversation must share an exact native chat-template prefix.
    Legacy {"text": ...} and {"input_ids": ...} rows retain full-token LM loss.
    """
    rows = []
    if max_length < 2 or max_rows < 1:
        raise ValueError("max_length must be >= 2 and max_rows must be positive")
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "prompt" in item or "response" in item:
                prompt, response = item.get("prompt"), item.get("response")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"row {line_number} requires a non-empty prompt")
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(f"row {line_number} requires a non-empty response")
                if contains_pii_pattern(prompt) or contains_pii_pattern(response):
                    continue
                apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
                if (not callable(apply_chat_template)
                        or getattr(tokenizer, "chat_template", None) is None):
                    raise ValueError(
                        "structured prompt/response rows require Dhīr's native chat template"
                    )
                prompt_ids = _as_token_ids(apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                ))
                full_ids = _as_token_ids(apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                ))
                if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(
                        f"row {line_number} does not preserve the native assistant-generation prefix"
                    )
                if len(prompt_ids) >= max_length:
                    continue
                ids = full_ids[:max_length]
                labels = ([-100] * len(prompt_ids)) + ids[len(prompt_ids):]
                supervision = "assistant_completion_only"
            elif "input_ids" in item:
                ids = [int(value) for value in item["input_ids"]][:max_length]
                labels = list(ids)
                supervision = "all_tokens"
            elif isinstance(item.get("text"), str):
                text = item["text"]
                if contains_pii_pattern(text):
                    continue
                encoded = tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                ids = _as_token_ids(encoded)[:max_length]
                labels = list(ids)
                supervision = "all_tokens"
            else:
                raise ValueError(
                    f"row {line_number} requires prompt/response, text or input_ids"
                )

            if len(ids) < 2 or not any(value != -100 for value in labels[1:]):
                continue
            if min(ids) < 0 or max(ids) >= len(tokenizer):
                raise ValueError(f"token ID outside vocabulary at row {line_number}")
            rows.append({
                "input_ids": ids,
                "labels": labels,
                "supervision": supervision,
            })
            if len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError("no usable training sequences after filtering")
    return rows


def batch_rows(rows, indices, pad_id):
    length = max(len(rows[index]["input_ids"]) for index in indices)
    input_ids = torch.full((len(indices), length), pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attention_mask = torch.zeros_like(input_ids)
    for batch_index, row_index in enumerate(indices):
        row = rows[row_index]
        sequence = row["input_ids"]
        targets = row["labels"]
        if len(sequence) != len(targets):
            raise ValueError("training input_ids and labels must have equal lengths")
        count = len(sequence)
        input_ids[batch_index, :count] = torch.tensor(sequence)
        labels[batch_index, :count] = torch.tensor(targets)
        attention_mask[batch_index, :count] = 1
    labels[attention_mask == 0] = -100
    if not bool((labels[:, 1:] != -100).any()):
        raise ValueError("batch contains no supervised next-token targets")
    return input_ids, labels, attention_mask


def expert_state(model, expert_id: int):
    return {
        f"layer_{layer_index}.{key}": value.detach().cpu().clone()
        for layer_index, layer in enumerate(model.adapter_layers)
        for key, value in layer.experts[expert_id].state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    expert_id,
    optimizer,
    step,
    data_cursor,
    data_sha,
    tokenizer_sha,
    *,
    batch_size,
    max_length,
    learning_rate,
    training_objective,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "training_objective": training_objective,
        "status": "EXPERT_ADAPTER_TRAINED_UNVALIDATED",
        "architecture_version": "0.4.0-dhir2b-adapters",
        "expert_id": expert_id,
        "base_model_id": model.base_model_id,
        "base_revision": model.base_revision,
        "vocab_size": model.vocab_size,
        "num_hidden_layers": 28,
        "adapter_rank": model.rank,
        "data_sha256": data_sha,
        "tokenizer_sha256": tokenizer_sha,
        "steps_completed": step,
        "data_cursor": data_cursor,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "expert_state_dict": expert_state(model, expert_id),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--approval", default="dataset_approval.json")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if not 0 <= args.expert_id < 20:
        parser.error("--expert-id must be 0..19")
    if min(args.steps, args.batch_size, args.max_length, args.max_rows, args.threads) < 1:
        parser.error("steps, batch-size, max-length, max-rows and threads must be positive")
    if args.max_length < 2 or args.lr <= 0:
        parser.error("max-length must be >= 2 and learning rate positive")

    torch.set_num_threads(args.threads)
    model, tokenizer, manifest = Dhir2BNeuroCore.load_checkpoint(args.model_dir)
    data_sha = sha_file(args.data)
    tokenizer_sha = sha_dir(Path(args.model_dir) / "tokenizer")

    # Fail closed before tokenizing or training if any human approval is absent.
    require_approval(
        args.approval,
        args.expert_id,
        data_sha,
        tokenizer_sha,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
    )

    rows = read_rows(args.data, tokenizer, args.max_length, args.max_rows)
    supervision_modes = sorted({row["supervision"] for row in rows})
    training_objective = "+".join(supervision_modes)
    trainable = model.prepare_expert_training(args.expert_id)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    start_step = 0

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        expected = {
            "schema_version": 3,
            "training_objective": training_objective,
            "status": "EXPERT_ADAPTER_TRAINED_UNVALIDATED",
            "architecture_version": "0.4.0-dhir2b-adapters",
            "expert_id": args.expert_id,
            "base_model_id": model.base_model_id,
            "base_revision": model.base_revision,
            "vocab_size": model.vocab_size,
            "num_hidden_layers": 28,
            "adapter_rank": model.rank,
            "data_sha256": data_sha,
            "tokenizer_sha256": tokenizer_sha,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.lr,
        }
        for key, expected_value in expected.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError("resume checkpoint provenance/config mismatch: " + key)
        model.load_expert_state_dict(args.expert_id, checkpoint["expert_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        start_step = int(checkpoint["steps_completed"])
        if start_step < 0 or checkpoint.get("data_cursor") != start_step * args.batch_size:
            raise ValueError("resume checkpoint data cursor is invalid")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad or EOS token")

    final_step = start_step + args.steps
    for step in range(start_step + 1, final_step + 1):
        start_index = ((step - 1) * args.batch_size) % len(rows)
        indices = [
            (start_index + batch_index) % len(rows)
            for batch_index in range(args.batch_size)
        ]
        input_ids, labels, attention_mask = batch_rows(rows, indices, int(pad_id))
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expert_id=args.expert_id,
            use_cache=False,
        )
        logits = output.logits
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, model.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        cursor = step * args.batch_size
        save_checkpoint(
            args.output,
            model,
            args.expert_id,
            optimizer,
            step,
            cursor,
            data_sha,
            tokenizer_sha,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.lr,
            training_objective=training_objective,
        )
        if step == 1 or step % 10 == 0 or step == final_step:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach()),
                "expert_id": args.expert_id,
                "training_objective": training_objective,
                "device": "cpu",
                "status": "EXPERT_ADAPTER_TRAINING",
            }, ensure_ascii=False))

    print(json.dumps({
        "status": "EXPERT_ADAPTER_TRAINED_UNVALIDATED",
        "expert_id": args.expert_id,
        "training_objective": training_objective,
        "steps_completed": final_step,
        "checkpoint": args.output,
        "base_revision": manifest["base_revision"],
        "gpu_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
