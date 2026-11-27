"""Read-only baseline/Expert-16 evaluation on the held-out validation split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dhir2b_neurocore import Dhir2BNeuroCore
from train_dhir2b_expert import batch_rows, read_rows, sha_dir, sha_file

MODEL_ID = "zenteiq-ai/Dhir-2B"
MODEL_REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"
ARCHITECTURE_VERSION = "0.4.0-dhir2b-adapters"
DATASET_ID = "local/neurocore_synthetic_sft/expert16_self_correction"
DATASET_REVISION = "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9"
TRAIN_SHA256 = "ce148fd31b2850b18f0b398dd55cda12f30a33844b315c98dc8a16847ed23e66"
EXPERT_ID = 16
MAX_LENGTH = 256


def completion_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """Return token-weighted assistant-completion loss and supervised target count."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected [batch, sequence, vocab] logits and [batch, sequence] labels")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and label batch/sequence dimensions do not match")
    targets = labels[:, 1:].reshape(-1)
    count = int((targets != -100).sum().item())
    if count < 1:
        raise ValueError("validation row has no supervised assistant-completion targets")
    total = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        targets,
        ignore_index=-100,
        reduction="sum",
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite validation completion loss")
    return float((total / count).detach().cpu()), count


def load_trained_expert(model, checkpoint_path: str | Path, *, tokenizer_sha: str) -> dict:
    """Load only a trainer-produced Expert 16 state after checking exact provenance."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 3,
        "status": "EXPERT_ADAPTER_TRAINED_UNVALIDATED",
        "architecture_version": ARCHITECTURE_VERSION,
        "expert_id": EXPERT_ID,
        "base_model_id": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "vocab_size": 131072,
        "num_hidden_layers": 28,
        "data_sha256": TRAIN_SHA256,
        "tokenizer_sha256": tokenizer_sha,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError("trained Expert 16 checkpoint provenance mismatch: " + json.dumps(mismatches))
    if int(payload.get("steps_completed", 0)) < 1:
        raise ValueError("trained Expert 16 checkpoint must contain at least one optimizer step")
    model.load_expert_state_dict(EXPERT_ID, payload["expert_state_dict"])
    return {
        "checkpoint_sha256": sha_file(path),
        "steps_completed": int(payload["steps_completed"]),
        "training_objective": payload.get("training_objective"),
        "data_sha256": payload["data_sha256"],
        "tokenizer_sha256": payload["tokenizer_sha256"],
    }


def evaluate(model, tokenizer, validation_path: Path) -> dict:
    """Evaluate each held-out row once against pure Dhīr and fixed Expert 16."""
    raw_rows = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenized = read_rows(validation_path, tokenizer, MAX_LENGTH, max_rows=100)
    if len(raw_rows) != 4 or len(tokenized) != 4:
        raise ValueError(
            f"expected exactly four held-out validation rows, found raw={len(raw_rows)}, "
            f"usable={len(tokenized)}"
        )

    model.eval()
    expert_parameters = [
        parameter for layer in model.adapter_layers
        for parameter in layer.experts[EXPERT_ID].parameters()
    ]
    expert_snapshots = [parameter.detach().clone() for parameter in expert_parameters]
    rows = []
    total_base_loss = 0.0
    total_expert_loss = 0.0
    total_targets = 0
    exact_logit_matches = 0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("checkpoint tokenizer has neither pad nor EOS token")

    with torch.inference_mode():
        for index, (raw, row) in enumerate(zip(raw_rows, tokenized, strict=True)):
            input_ids, labels, attention_mask = batch_rows([row], [0], int(pad_id))
            base_output = model.forward_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            expert_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                expert_id=EXPERT_ID,
            )
            base_loss, target_count = completion_loss(base_output.logits, labels)
            expert_loss, expert_target_count = completion_loss(expert_output.logits, labels)
            if target_count != expert_target_count:
                raise RuntimeError("base and Expert 16 evaluated different target counts")
            exact_match = bool(torch.equal(base_output.logits, expert_output.logits))
            exact_logit_matches += int(exact_match)
            total_base_loss += base_loss * target_count
            total_expert_loss += expert_loss * target_count
            total_targets += target_count
            rows.append({
                "validation_row": index + 1,
                "prompt": raw["prompt"],
                "reference_response": raw["response"],
                "supervised_assistant_tokens": target_count,
                "base_dhir_completion_loss": base_loss,
                "expert16_completion_loss": expert_loss,
                "logits_exactly_equal": exact_match,
            })

    unchanged = all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(expert_parameters, expert_snapshots, strict=True)
    )
    if not unchanged:
        raise RuntimeError("Expert 16 parameters changed during read-only evaluation")
    base_mean = total_base_loss / total_targets
    expert_mean = total_expert_loss / total_targets
    return {
        "validation_rows": len(rows),
        "validation_targets": total_targets,
        "base_dhir_token_weighted_completion_loss": base_mean,
        "expert16_token_weighted_completion_loss": expert_mean,
        "expert16_minus_base_loss": expert_mean - base_mean,
        "exact_logit_match_rows": exact_logit_matches,
        "adapter_weights_unchanged_during_evaluation": unchanged,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    model_dir = Path(args.model_dir)
    validation_path = Path(args.validation_data)
    model, tokenizer, manifest = Dhir2BNeuroCore.load_checkpoint(str(model_dir))
    if manifest.get("status") != "IMPORTED_BASE_UNTRAINED_EXPERTS":
        raise ValueError("baseline must start from the verified imported Dhīr base checkpoint")
    if manifest.get("base_model_id") != MODEL_ID or manifest.get("base_revision") != MODEL_REVISION:
        raise ValueError("baseline checkpoint Dhīr identity/revision mismatch")
    if manifest.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("baseline checkpoint adapter architecture mismatch")

    tokenizer_sha = sha_dir(model_dir / "tokenizer")
    trained_adapter = None
    if args.expert_checkpoint:
        trained_adapter = load_trained_expert(
            model, args.expert_checkpoint, tokenizer_sha=tokenizer_sha
        )

    metrics = evaluate(model, tokenizer, validation_path)
    report = {
        "status": "DHIR2B_EXPERT16_HELDOUT_BASELINE_EVALUATION_COMPLETE",
        "evaluation_mode": "trained_expert16_vs_base_dhir" if trained_adapter else "untrained_expert16_initialization_vs_base_dhir",
        "base_model_id": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "architecture_version": ARCHITECTURE_VERSION,
        "expert_id": EXPERT_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "training_dataset_sha256_expected": TRAIN_SHA256,
        "validation_source_sha256": sha_file(validation_path),
        "validation_source_rows": 4,
        "tokenizer_directory_sha256": tokenizer_sha,
        "tokenizer_vocab_sha256": tokenizer.neurocore_manifest["tokenizer_vocab_sha256"],
        "trained_adapter_provenance": trained_adapter,
        "metrics": metrics,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "validation_used_for_training": False,
        "interpretation_limit": (
            "Four held-out examples are a small diagnostic set, not a reliable generalization estimate. "
            "Loss measures token prediction, not factual or semantic correctness; human review of responses remains required."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "evaluation_mode": report["evaluation_mode"],
        "validation_rows": metrics["validation_rows"],
        "base_loss": metrics["base_dhir_token_weighted_completion_loss"],
        "expert16_loss": metrics["expert16_token_weighted_completion_loss"],
        "exact_logit_match_rows": metrics["exact_logit_match_rows"],
        "optimizer_steps": 0,
        "report": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
