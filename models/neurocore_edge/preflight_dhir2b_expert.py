"""No-update gradient preflight for one fixed Dhīr-2B NeuroCore expert.

This program deliberately does not instantiate an optimizer or call optimizer.step().
It refuses to run unless the dataset/tokenizer registry remains default-deny.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dhir2b_neurocore import Dhir2BNeuroCore
from train_dhir2b_expert import batch_rows, read_rows, sha_dir, sha_file
from neurocore_tokenizer import MANIFEST_NAME, NeuroCoreTokenizer, tokenizer_files_sha256

MODEL_ID = "zenteiq-ai/Dhir-2B"
MODEL_REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"
DATASET_ID = "local/neurocore_synthetic_sft/expert16_self_correction"
DATASET_REVISION = "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9"
EXPECTED_DATA_SHA256 = "ce148fd31b2850b18f0b398dd55cda12f30a33844b315c98dc8a16847ed23e66"
EXPECTED_TOKENIZER_FILES_SHA256 = "cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056"
EXPECTED_TOKENIZER_VOCAB_SHA256 = "cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c"
EXPERT_ID = 16
MAX_LENGTH = 256


def verify_pending_registry(path: str | Path, expert_id: int, dataset_id: str, revision: str) -> dict:
    """Require the reviewed dataset to remain explicitly pending, with no approvals."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("status") != "ALL_DATASETS_PENDING_REVIEW":
        raise PermissionError("preflight requires the repository default-deny registry")
    if document.get("approved_training_artifacts") or document.get("approved_tokenizers"):
        raise PermissionError("preflight expected zero approved corpus artifacts and tokenizers")
    row = next((
        item for item in document.get("datasets", [])
        if item.get("expert_id") == expert_id
        and item.get("dataset_id") == dataset_id
        and item.get("revision") == revision
    ), None)
    if row is None:
        raise PermissionError("preflight dataset ID/revision is not registered exactly")
    for key in (
        "approved",
        "commercial_training_allowed",
        "public_redistribution_allowed",
        "pii_reviewed",
        "consent_reviewed",
        "schema_reviewed",
        "split_reviewed",
    ):
        if row.get(key) is not False:
            raise PermissionError(f"preflight expects {key}=false; registry state changed")
    return {"registry_status": document["status"], "dataset_id": dataset_id,
            "dataset_revision": revision, "dataset_approved": False}


def verify_fixed_expert_trainability(model, expert_id: int, trainable: list[torch.nn.Parameter]) -> dict:
    """Verify the backbone/router/other experts are frozen and only one expert is trainable."""
    if not 0 <= expert_id < model.num_experts:
        raise ValueError("expert_id outside model expert range")
    expected = [
        parameter
        for layer in model.adapter_layers
        for parameter in layer.experts[expert_id].parameters()
    ]
    if not expected or {id(p) for p in expected} != {id(p) for p in trainable}:
        raise RuntimeError("trainer returned a parameter set other than the selected fixed expert")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("base Dhīr backbone must remain frozen")
    for layer_index, layer in enumerate(model.adapter_layers):
        if any(parameter.requires_grad for parameter in layer.router.parameters()):
            raise RuntimeError(f"router weights must remain frozen at layer {layer_index}")
        for other_id, expert in enumerate(layer.experts):
            should_train = other_id == expert_id
            for parameter in expert.parameters():
                if parameter.requires_grad is not should_train:
                    raise RuntimeError(
                        f"unexpected trainability at layer {layer_index}, expert {other_id}"
                    )
    return {
        "selected_expert_id": expert_id,
        "selected_trainable_parameter_tensors": len(expected),
        "base_backbone_frozen": True,
        "all_routers_frozen": True,
        "all_other_experts_frozen": True,
    }


def ensure_checkpoint_tokenizer_manifest(tokenizer_dir: str | Path, model_id: str, revision: str) -> dict:
    """Rebuild only missing metadata after verifying unchanged raw Dhīr tokenizer files."""
    root = Path(tokenizer_dir)
    manifest_path = root / MANIFEST_NAME
    observed_files_sha = tokenizer_files_sha256(root)
    if observed_files_sha != EXPECTED_TOKENIZER_FILES_SHA256:
        raise ValueError(
            "checkpoint raw tokenizer files SHA-256 mismatch: "
            f"{observed_files_sha} != {EXPECTED_TOKENIZER_FILES_SHA256}"
        )

    rebuilt = False
    if not manifest_path.is_file():
        from transformers import PreTrainedTokenizerFast

        raw_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(root), local_files_only=True
        )
        wrapped = NeuroCoreTokenizer.from_tokenizer(
            raw_tokenizer,
            base_model_id=model_id,
            base_revision=revision,
            expected_vocab_size=131072,
        )
        manifest = wrapped.neurocore_manifest
        manifest["tokenizer_files_sha256"] = observed_files_sha
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        rebuilt = True

    return {
        "manifest_rebuilt_from_verified_local_tokenizer": rebuilt,
        "tokenizer_files_sha256": observed_files_sha,
        "manifest_path": str(manifest_path),
    }


def verify_checkpoint_tokenizer(tokenizer, model_id: str, revision: str) -> dict:
    """Verify matching saved-file, vocabulary and chat-template fingerprints."""
    manifest = tokenizer.neurocore_manifest
    expected = {
        "base_model_id": model_id,
        "base_revision": revision,
        "vocab_size": 131072,
        "tokenizer_vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
        "chat_template_sha256": "4bc9a0aac3faabc9a7399b1df253929f73d0448329f55d6700a73a0fff7c2305",
        "tokenizer_files_sha256": EXPECTED_TOKENIZER_FILES_SHA256,
        "token_id_mapping": "IDENTICAL_TO_PINNED_DHIR_BASE",
        "base_embedding_rows_reassigned": False,
        "vocabulary_expansion": False,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError("checkpoint tokenizer semantic/file fingerprint mismatch: " + json.dumps(mismatches))
    return {
        "files_sha256": manifest["tokenizer_files_sha256"],
        "vocab_sha256": manifest["tokenizer_vocab_sha256"],
        "chat_template_sha256": manifest["chat_template_sha256"],
        "file_integrity_verified_by_checkpoint_loader": True,
    }


def verify_gradients_and_no_update(trainable, snapshots) -> dict:
    """Require finite gradients while proving no parameter value changed."""
    if len(trainable) != len(snapshots):
        raise RuntimeError("trainable parameter snapshot count changed")
    missing = [i for i, parameter in enumerate(trainable) if parameter.grad is None]
    if missing:
        raise RuntimeError(f"selected expert parameters are missing gradients: {missing[:8]}")
    finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in trainable)
    if not finite:
        raise FloatingPointError("selected expert produced non-finite gradients")
    nonzero = sum(bool(torch.count_nonzero(parameter.grad).item()) for parameter in trainable)
    if nonzero < 1:
        raise RuntimeError("selected expert produced no non-zero gradient tensors")
    unchanged = all(torch.equal(parameter.detach(), before) for parameter, before in zip(trainable, snapshots, strict=True))
    if not unchanged:
        raise RuntimeError("a parameter changed during no-update preflight")
    return {
        "gradient_tensors": len(trainable),
        "finite_gradient_tensors": len(trainable),
        "nonzero_gradient_tensors": nonzero,
        "selected_expert_weights_unchanged": unchanged,
        "optimizer_steps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--approval", default="dataset_approval.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    tokenizer_dir = model_dir / "tokenizer"

    approval = verify_pending_registry(
        args.approval, EXPERT_ID, DATASET_ID, DATASET_REVISION
    )
    data_sha = sha_file(data_path)
    if data_sha != EXPECTED_DATA_SHA256:
        raise ValueError("Expert 16 training source SHA-256 differs from the reviewed dry-run")
    tokenizer_manifest_info = ensure_checkpoint_tokenizer_manifest(
        tokenizer_dir, MODEL_ID, MODEL_REVISION
    )
    tokenizer_directory_sha = sha_dir(tokenizer_dir)

    model, tokenizer, manifest = Dhir2BNeuroCore.load_checkpoint(str(model_dir))
    tokenizer_fingerprints = verify_checkpoint_tokenizer(tokenizer, MODEL_ID, MODEL_REVISION)
    if manifest.get("status") != "IMPORTED_BASE_UNTRAINED_EXPERTS":
        raise ValueError("checkpoint is not an imported, untrained Dhīr NeuroCore base")
    if manifest.get("base_model_id") != MODEL_ID or manifest.get("base_revision") != MODEL_REVISION:
        raise ValueError("checkpoint Dhīr model ID/revision mismatch")
    if int(manifest.get("num_hidden_layers", -1)) != 28 or int(manifest.get("num_experts", -1)) != 20:
        raise ValueError("checkpoint layer/expert topology mismatch")
    if int(manifest.get("vocab_size", -1)) != 131072 or len(tokenizer) != 131072:
        raise ValueError("checkpoint tokenizer vocabulary mismatch")
    vocab_sha = tokenizer.neurocore_manifest.get("tokenizer_vocab_sha256")

    rows = read_rows(data_path, tokenizer, max_length=MAX_LENGTH, max_rows=5000)
    if len(rows) != 20:
        raise ValueError(f"expected all 20 Expert 16 train rows, retained {len(rows)}")
    # Use the shortest real training sequence to keep the CPU preflight bounded.
    row_index = min(range(len(rows)), key=lambda index: len(rows[index]["input_ids"]))
    chosen = rows[row_index]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("checkpoint tokenizer has neither pad nor EOS token")
    input_ids, labels, attention_mask = batch_rows(rows, [row_index], int(pad_id))
    if not bool((labels[:, 1:] != -100).any()):
        raise ValueError("selected preflight batch has no supervised next-token targets")

    trainable = model.prepare_expert_training(EXPERT_ID)
    topology = verify_fixed_expert_trainability(model, EXPERT_ID, trainable)
    model.train()  # Dhīr backbone remains eval; only selected fixed-expert adapters train.
    snapshots = [parameter.detach().clone() for parameter in trainable]

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        expert_id=EXPERT_ID,
        use_cache=False,
    )
    logits = output.logits
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, model.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("preflight loss is non-finite")
    loss.backward()

    gradients = verify_gradients_and_no_update(trainable, snapshots)
    nontrainable_with_grad = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if nontrainable_with_grad:
        raise RuntimeError(f"{nontrainable_with_grad} frozen parameters unexpectedly received gradients")

    report = {
        "status": "DHIR2B_EXPERT16_FORWARD_BACKWARD_PREFLIGHT_PASSED_NO_UPDATE",
        "base_model_id": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "architecture_version": manifest["architecture_version"],
        "num_hidden_layers": 28,
        "hidden_size": int(model.hidden_size),
        "vocab_size": int(model.vocab_size),
        "num_experts": int(model.num_experts),
        "adapter_rank": int(model.rank),
        "device": "cpu",
        "expert_id": EXPERT_ID,
        "dataset": {
            **approval,
            "source_sha256": data_sha,
            "source_rows": 20,
            "validation_rows_loaded": 0,
            "validation_used_for_training": False,
        },
        "tokenizer": {
            "directory_sha256_including_manifest": tokenizer_directory_sha,
            **tokenizer_manifest_info,
            **tokenizer_fingerprints,
            "native_chat_template_present": bool(tokenizer.chat_template),
        },
        "batch": {
            "selected_source_row_index_zero_based": row_index,
            "sequence_length": int(input_ids.shape[-1]),
            "supervised_next_token_targets": int((labels[:, 1:] != -100).sum().item()),
        },
        "forward": {
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "loss_finite": True,
            "loss_value": float(loss.detach().cpu()),
            "backward_performed": True,
        },
        "trainability": topology,
        "gradients": gradients,
        "frozen_parameters_with_gradients": nontrainable_with_grad,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "weights_changed": False,
        "approval_gate_bypassed": False,
        "interpretation": (
            "One real Dhīr-2B forward/backward pass verified the fixed Expert 16 gradient path. "
            "No optimizer was created, no parameter was updated, and the training approval gate remains closed."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "expert_id": EXPERT_ID,
        "loss_finite": report["forward"]["loss_finite"],
        "gradient_tensors": gradients["gradient_tensors"],
        "nonzero_gradient_tensors": gradients["nonzero_gradient_tensors"],
        "weights_changed": False,
        "optimizer_steps": 0,
        "approval_gate_bypassed": False,
        "report": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
