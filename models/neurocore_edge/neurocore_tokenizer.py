"""NeuroCore-owned compatibility layer around the pinned Dhīr tokenizer.

The base vocabulary and every token ID remain byte-for-byte compatible with
Dhīr-2B embeddings. NeuroCore control IDs are side-channel metadata, not newly
inserted vocabulary rows, so pretrained embedding/output-head indices never
shift.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "neurocore_tokenizer_manifest.json"
TOKENIZER_NAME = "neurocore-dhir-compatible"
CONTROL_ID_COUNT = 127
CONTROL_ID_POLICY = "side_channel_metadata_no_vocab_expansion"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    pairs = sorted(((str(token), int(token_id)) for token, token_id in vocab.items()),
                   key=lambda pair: (pair[1], pair[0]))
    return hashlib.sha256(_canonical_json(pairs)).hexdigest()


def tokenizer_files_sha256(directory: Path) -> str:
    """Hash every tokenizer file except this manifest to avoid self-reference."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    if not files:
        raise ValueError("tokenizer directory contains no saved tokenizer files")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _chat_template_value(tokenizer: Any) -> Any:
    value = getattr(tokenizer, "chat_template", None)
    if value is None or isinstance(value, (str, list, dict)):
        return value
    return str(value)


def _special_token_ids(tokenizer: Any) -> dict[str, int]:
    result = {}
    for name in (
        "bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id",
        "sep_token_id", "cls_token_id", "mask_token_id",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None:
            result[name] = int(value)
    for name, token_id in result.items():
        if token_id < 0 or token_id >= len(tokenizer):
            raise ValueError(f"{name} is outside the Dhīr tokenizer vocabulary")
    return result


def validate_neurocore_tokenizer(
    tokenizer: Any,
    *,
    expected_vocab_size: int,
    base_model_id: str,
    base_revision: str,
) -> dict[str, Any]:
    if not base_model_id or not base_revision:
        raise ValueError("pinned Dhīr model ID and revision are required")
    actual_size = len(tokenizer)
    if actual_size != int(expected_vocab_size):
        raise ValueError(
            f"NeuroCore tokenizer vocabulary mismatch: {actual_size} != {expected_vocab_size}"
        )
    vocab = tokenizer.get_vocab()
    token_ids = [int(value) for value in vocab.values()]
    if len(vocab) != actual_size or len(set(token_ids)) != actual_size:
        raise ValueError("token strings and token IDs must be one-to-one")
    if set(token_ids) != set(range(actual_size)):
        raise ValueError("Dhīr token IDs must remain contiguous and unchanged")
    if getattr(tokenizer, "is_fast", True) is False:
        raise ValueError("NeuroCore Dhīr-compatible tokenizer must be a fast tokenizer")

    chat_template = _chat_template_value(tokenizer)
    special_ids = _special_token_ids(tokenizer)
    return {
        "name": TOKENIZER_NAME,
        "schema_version": 1,
        "tokenizer_type": "dhir_native_fast_tokenizer_with_neurocore_metadata",
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "vocab_size": actual_size,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(tokenizer),
        "special_token_ids": special_ids,
        "chat_template_sha256": (
            hashlib.sha256(_canonical_json(chat_template)).hexdigest()
            if chat_template is not None else None
        ),
        "chat_template_preserved": chat_template is not None,
        "token_id_mapping": "IDENTICAL_TO_PINNED_DHIR_BASE",
        "base_embedding_rows_reassigned": False,
        "vocabulary_expansion": False,
        "control_ids": {
            "count": CONTROL_ID_COUNT,
            "minimum": 0,
            "maximum": CONTROL_ID_COUNT - 1,
            "representation": "out_of_band_integer_metadata",
            "consumes_vocabulary_ids": False,
            "policy": CONTROL_ID_POLICY,
            "semantic_mapping_status": "UNASSIGNED_PENDING_EXPERT_ROUTING_REVIEW",
        },
        "language_metadata": {
            "field": "lang",
            "representation": "out_of_band_metadata",
            "changes_token_ids": False,
        },
    }


class NeuroCoreTokenizer:
    """NeuroCore facade that delegates lexical tokenization to Dhīr unchanged."""

    def __init__(self, tokenizer: Any, manifest: dict[str, Any]):
        self._tokenizer = tokenizer
        self._manifest = dict(manifest)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        base_model_id: str,
        base_revision: str,
        expected_vocab_size: int,
    ) -> "NeuroCoreTokenizer":
        manifest = validate_neurocore_tokenizer(
            tokenizer,
            expected_vocab_size=expected_vocab_size,
            base_model_id=base_model_id,
            base_revision=base_revision,
        )
        return cls(tokenizer, manifest)

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        base_model_id: str,
        base_revision: str,
        expected_vocab_size: int,
    ) -> "NeuroCoreTokenizer":
        from transformers import PreTrainedTokenizerFast

        root = Path(directory)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"NeuroCore tokenizer manifest missing: {manifest_path}"
            )
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(root), local_files_only=True
        )
        current = validate_neurocore_tokenizer(
            tokenizer,
            expected_vocab_size=expected_vocab_size,
            base_model_id=base_model_id,
            base_revision=base_revision,
        )
        if saved_manifest.get("base_model_id") != base_model_id:
            raise ValueError("NeuroCore tokenizer base model ID mismatch")
        if saved_manifest.get("base_revision") != base_revision:
            raise ValueError("NeuroCore tokenizer pinned revision mismatch")
        if saved_manifest.get("vocab_size") != expected_vocab_size:
            raise ValueError("NeuroCore tokenizer manifest vocabulary mismatch")
        if saved_manifest.get("tokenizer_vocab_sha256") != current["tokenizer_vocab_sha256"]:
            raise ValueError("NeuroCore tokenizer token-ID mapping hash mismatch")
        if saved_manifest.get("chat_template_sha256") != current["chat_template_sha256"]:
            raise ValueError("NeuroCore tokenizer chat-template hash mismatch")
        if saved_manifest.get("token_id_mapping") != "IDENTICAL_TO_PINNED_DHIR_BASE":
            raise ValueError("unsupported NeuroCore token-ID mapping policy")
        if saved_manifest.get("control_ids") != current["control_ids"]:
            raise ValueError("NeuroCore control-ID policy mismatch")
        observed_files_sha = tokenizer_files_sha256(root)
        if saved_manifest.get("tokenizer_files_sha256") != observed_files_sha:
            raise ValueError("NeuroCore tokenizer files SHA-256 mismatch")
        current["tokenizer_files_sha256"] = observed_files_sha
        return cls(tokenizer, current)

    def save_pretrained(self, directory: str | Path, **kwargs):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        saved = self._tokenizer.save_pretrained(str(root), **kwargs)
        manifest = validate_neurocore_tokenizer(
            self._tokenizer,
            expected_vocab_size=int(self._manifest["vocab_size"]),
            base_model_id=self._manifest["base_model_id"],
            base_revision=self._manifest["base_revision"],
        )
        manifest["tokenizer_files_sha256"] = tokenizer_files_sha256(root)
        manifest_path = root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._manifest = manifest
        return saved

    @property
    def neurocore_manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str):
        # Delegate encode/decode, chat-template and tokenizer metadata APIs.
        return getattr(self._tokenizer, name)
