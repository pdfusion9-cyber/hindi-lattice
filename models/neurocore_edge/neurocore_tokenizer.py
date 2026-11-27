"""NeuroCore tokenizer layer compatible with pinned Dhīr-2B embeddings.

Lexical tokenization is delegated to the native Dhīr tokenizer. Language,
script, task and expert metadata is carried out-of-band: no token is inserted,
no token ID is remapped, and no embedding row is reassigned.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

MANIFEST_NAME = "neurocore_tokenizer_manifest.json"
TOKENIZER_NAME = "neurocore-dhir-compatible"
MANIFEST_SCHEMA_VERSION = 2
REVIEWED_DHIR2B_TOKENIZER_FILES_SHA256 = "cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056"
CONTROL_ID_COUNT = 127
CONTROL_ID_POLICY = "out_of_band_metadata_no_vocabulary_expansion"

# Stable control metadata IDs. These integers are not tokenizer token IDs.
ROLE_CONTROL_IDS = {"system": 0, "user": 1, "assistant": 2, "tool": 3}
TASK_CONTROL_IDS = {
    "chat": 8,
    "generate": 9,
    "doc_editing": 10,
    "doc_creation": 11,
    "summarization": 12,
    "reading_qa": 13,
    "read_aloud": 14,
    "user_guidance": 15,
    "legal_contracts": 16,
    "legal_compliance": 17,
    "professional_comms": 18,
    "tables_ledgers": 19,
    "ocr_pdf_layout": 20,
    "hindi_indic_docs": 21,
    "translation": 22,
    "citations_dola": 23,
    "technical_academic": 24,
    "hippocampal_d2l": 25,
    "self_correction_mnl": 26,
    "coconut_planning": 27,
    "bg_gate_early_exit": 28,
    "fabric_swarm_tools": 29,
}
EXPERT_NAMES = {
    0: "doc_editing",
    1: "doc_creation",
    2: "summarization",
    3: "reading_qa",
    4: "read_aloud",
    5: "user_guidance",
    6: "legal_contracts",
    7: "legal_compliance",
    8: "professional_comms",
    9: "tables_ledgers",
    10: "ocr_pdf_layout",
    11: "hindi_indic_docs",
    12: "translation",
    13: "citations_dola",
    14: "technical_academic",
    15: "hippocampal_d2l",
    16: "self_correction_mnl",
    17: "coconut_planning",
    18: "bg_gate_early_exit",
    19: "fabric_swarm_tools",
}
EXPERT_CONTROL_OFFSET = 32
SCRIPT_CONTROL_IDS = {
    "deva": 64,
    "beng": 65,
    "guru": 66,
    "gujr": 67,
    "orya": 68,
    "taml": 69,
    "telu": 70,
    "knda": 71,
    "mlym": 72,
    "latn": 73,
    "arab": 74,
    "cyrl": 75,
}
TASK_TO_EXPERT = {name: expert_id for expert_id, name in EXPERT_NAMES.items()}

# Script detection is script classification, not language identification.
_SCRIPT_RANGES = (
    ("deva", "Devanagari", 0x0900, 0x097F),
    ("beng", "Bengali", 0x0980, 0x09FF),
    ("guru", "Gurmukhi", 0x0A00, 0x0A7F),
    ("gujr", "Gujarati", 0x0A80, 0x0AFF),
    ("orya", "Odia", 0x0B00, 0x0B7F),
    ("taml", "Tamil", 0x0B80, 0x0BFF),
    ("telu", "Telugu", 0x0C00, 0x0C7F),
    ("knda", "Kannada", 0x0C80, 0x0CFF),
    ("mlym", "Malayalam", 0x0D00, 0x0D7F),
    ("sinh", "Sinhala", 0x0D80, 0x0DFF),
    ("thai", "Thai", 0x0E00, 0x0E7F),
    ("laoo", "Lao", 0x0E80, 0x0EFF),
    ("mymr", "Myanmar", 0x1000, 0x109F),
    ("arab", "Arabic", 0x0600, 0x06FF),
    ("cyrl", "Cyrillic", 0x0400, 0x052F),
    ("latn", "Latin", 0x0041, 0x024F),
    ("hani", "Han", 0x3400, 0x9FFF),
    ("hang", "Hangul", 0xAC00, 0xD7AF),
)
_SCRIPT_BY_CODE = {code: name for code, name, _, _ in _SCRIPT_RANGES}
_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_DEVANAGARI_VIRAMA = "\u094d"


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
    pairs = sorted(
        ((str(token), int(token_id)) for token, token_id in vocab.items()),
        key=lambda pair: (pair[1], pair[0]),
    )
    return hashlib.sha256(_canonical_json(pairs)).hexdigest()


def tokenizer_files_sha256(directory: Path) -> str:
    """Hash saved tokenizer files except this manifest to avoid self-reference."""
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


def normalize_language_tag(language_tag: str | None) -> str | None:
    """Normalize a caller-supplied BCP-47-like language tag; never infer it."""
    if language_tag is None:
        return None
    if not isinstance(language_tag, str):
        raise TypeError("language tag must be a string or None")
    value = language_tag.strip()
    if not value or not _LANGUAGE_TAG_RE.fullmatch(value):
        raise ValueError("language tag must be a valid BCP-47-like tag, e.g. hi or hi-IN")
    parts = value.split("-")
    normalized = [parts[0].lower()]
    for part in parts[1:]:
        normalized.append(part.upper() if len(part) == 2 and part.isalpha() else part)
    return "-".join(normalized)


def _script_for_codepoint(codepoint: int) -> tuple[str, str] | None:
    for code, name, start, end in _SCRIPT_RANGES:
        if start <= codepoint <= end:
            return code, name
    return None


def _devanagari_orthography(text: str) -> dict[str, Any]:
    """Return conservative orthographic counts, not a morphological analysis."""
    words = []
    for raw_word in text.split():
        if any(0x0900 <= ord(char) <= 0x097F for char in raw_word):
            words.append(raw_word)
    akshara_count = 0
    for word in words:
        current = ""
        previous = ""
        for char in word:
            code = ord(char)
            is_devanagari = 0x0900 <= code <= 0x097F
            category = unicodedata.category(char)
            if not is_devanagari:
                if current:
                    akshara_count += 1
                    current = ""
                previous = char
                continue
            if not current:
                current = char
            elif category.startswith("M") or previous == _DEVANAGARI_VIRAMA or char == "\u200d":
                current += char
            else:
                akshara_count += 1
                current = char
            previous = char
        if current:
            akshara_count += 1
    return {
        "word_count": len(words),
        "approx_akshara_count": akshara_count,
        "analysis_level": "orthographic_heuristic_not_full_morphology",
    }


def analyze_text(text: str, *, language_tag: str | None = None) -> dict[str, Any]:
    """Analyze script mix and supplied language hint without changing input text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_for_analysis = unicodedata.normalize("NFC", text)
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for char in normalized_for_analysis:
        if not unicodedata.category(char).startswith("L"):
            continue
        match = _script_for_codepoint(ord(char))
        code, name = match if match else ("other", "Other")
        counts[code] = counts.get(code, 0) + 1
        names[code] = name
    total = sum(counts.values())
    ranked = sorted(counts, key=lambda code: (-counts[code], code))
    dominant = ranked[0] if ranked else None
    return {
        "language_tag": normalize_language_tag(language_tag),
        "language_tag_source": (
            "caller_supplied_hint" if language_tag is not None
            else "unspecified_not_inferred_from_script"
        ),
        "script_counts": counts,
        "script_names": names,
        "dominant_script": dominant,
        "dominant_script_name": names.get(dominant) if dominant else None,
        "script_mix": "mixed" if len(counts) > 1 else ("single" if counts else "none"),
        "script_proportions": {
            code: round(count / total, 4) for code, count in counts.items()
        } if total else {},
        "letter_count": total,
        "approx_whitespace_word_count": len(normalized_for_analysis.split()),
        "devanagari_orthography": _devanagari_orthography(normalized_for_analysis),
        "normalization_policy": "NFC_FOR_METADATA_ONLY_INPUT_TEXT_PRESERVED",
    }


def _control_registry() -> dict[str, Any]:
    return {
        "registry_version": 1,
        "count": CONTROL_ID_COUNT,
        "minimum": 0,
        "maximum": CONTROL_ID_COUNT - 1,
        "representation": "out_of_band_integer_metadata",
        "consumes_vocabulary_ids": False,
        "policy": CONTROL_ID_POLICY,
        "roles": dict(ROLE_CONTROL_IDS),
        "tasks": dict(TASK_CONTROL_IDS),
        "expert_control_offset": EXPERT_CONTROL_OFFSET,
        "expert_names": {str(k): v for k, v in EXPERT_NAMES.items()},
        "scripts": dict(SCRIPT_CONTROL_IDS),
        "unassigned_range": [76, 126],
    }


def build_control_metadata(
    *,
    role: str | None = None,
    task: str | None = None,
    expert_id: int | None = None,
    script_code: str | None = None,
) -> dict[str, Any]:
    """Build stable control/routing hints; this does not execute model routing."""
    if role is not None and role not in ROLE_CONTROL_IDS:
        raise ValueError(f"unknown role: {role}")
    if task is not None and task not in TASK_CONTROL_IDS:
        raise ValueError(f"unknown task: {task}")
    if expert_id is not None:
        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id not in EXPERT_NAMES:
            raise ValueError("expert_id must be an integer in [0, 19]")
    task_expert = TASK_TO_EXPERT.get(task) if task else None
    if expert_id is not None and task_expert is not None and expert_id != task_expert:
        raise ValueError("explicit expert_id conflicts with task's fixed expert mapping")
    suggested_expert = expert_id if expert_id is not None else task_expert
    if script_code is not None and script_code not in _SCRIPT_BY_CODE:
        raise ValueError(f"unsupported script code: {script_code}")

    ids = []
    if role is not None:
        ids.append(ROLE_CONTROL_IDS[role])
    if task is not None:
        ids.append(TASK_CONTROL_IDS[task])
    if suggested_expert is not None:
        ids.append(EXPERT_CONTROL_OFFSET + suggested_expert)
    if script_code in SCRIPT_CONTROL_IDS:
        ids.append(SCRIPT_CONTROL_IDS[script_code])

    return {
        "role": role,
        "task": task,
        "explicit_expert_id": expert_id,
        "suggested_expert_id": suggested_expert,
        "suggested_expert_name": EXPERT_NAMES.get(suggested_expert),
        "expert_selection_policy": "hint_only_no_automatic_dispatch",
        "script_code": script_code,
        "control_ids": ids,
        "control_ids_are_vocabulary_tokens": False,
        "vocabulary_ids_changed": False,
    }


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
        "schema_version": MANIFEST_SCHEMA_VERSION,
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
        "language_metadata": {
            "field": "lang",
            "script_detection": "unicode_script_ranges_v1",
            "language_inference": "explicit_hint_only_no_script_to_language_guess",
            "unicode_normalization": "NFC_FOR_METADATA_ONLY",
            "input_text_rewritten": False,
            "devanagari_analysis": "orthographic_akshara_heuristic_not_full_morphology",
        },
        "control_ids": _control_registry(),
        "routing_metadata": {
            "fixed_expert_names": {str(k): v for k, v in EXPERT_NAMES.items()},
            "task_to_expert": dict(TASK_TO_EXPERT),
            "automatic_dispatch": False,
            "explicit_expert_training_required": True,
        },
    }


class NeuroCoreTokenizer:
    """NeuroCore facade over Dhīr's unchanged lexical tokenizer."""

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
        expected_tokenizer_files_sha256: str | None = None,
    ) -> "NeuroCoreTokenizer":
        from transformers import PreTrainedTokenizerFast

        root = Path(directory)
        manifest_path = root / MANIFEST_NAME
        observed_files_sha = tokenizer_files_sha256(root)
        if expected_tokenizer_files_sha256 is not None and observed_files_sha != expected_tokenizer_files_sha256:
            raise ValueError("NeuroCore tokenizer files do not match the required pinned SHA-256")

        if manifest_path.is_file():
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            if expected_tokenizer_files_sha256 is None:
                raise FileNotFoundError(f"NeuroCore tokenizer manifest missing: {manifest_path}")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(root), local_files_only=True
            )
            saved_manifest = validate_neurocore_tokenizer(
                tokenizer,
                expected_vocab_size=expected_vocab_size,
                base_model_id=base_model_id,
                base_revision=base_revision,
            )
            saved_manifest["tokenizer_files_sha256"] = observed_files_sha
            manifest_path.write_text(
                json.dumps(saved_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        saved_schema = saved_manifest.get("schema_version")
        if saved_schema not in (1, MANIFEST_SCHEMA_VERSION):
            raise ValueError("unsupported NeuroCore tokenizer manifest schema")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(root), local_files_only=True
        )
        current = validate_neurocore_tokenizer(
            tokenizer,
            expected_vocab_size=expected_vocab_size,
            base_model_id=base_model_id,
            base_revision=base_revision,
        )
        common_keys = (
            "base_model_id", "base_revision", "vocab_size",
            "tokenizer_vocab_sha256", "chat_template_sha256",
            "token_id_mapping",
        )
        for key in common_keys:
            if saved_manifest.get(key) != current[key]:
                raise ValueError(f"NeuroCore tokenizer manifest mismatch: {key}")
        # Schema 1 checkpoints predate NeuroCore language/routing metadata.
        # Validate their original Dhīr lexical identity and file hash, then
        # expose the current schema-2 metadata facade without changing IDs.
        if saved_schema == MANIFEST_SCHEMA_VERSION:
            for key in ("control_ids", "language_metadata", "routing_metadata"):
                if saved_manifest.get(key) != current[key]:
                    raise ValueError(f"NeuroCore tokenizer manifest mismatch: {key}")
        if saved_manifest.get("tokenizer_files_sha256") != observed_files_sha:
            raise ValueError("NeuroCore tokenizer files SHA-256 mismatch")
        if expected_tokenizer_files_sha256 is not None and current["tokenizer_files_sha256"] != expected_tokenizer_files_sha256:
            raise ValueError("loaded NeuroCore tokenizer file hash differs from required pinned SHA-256")
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

    def analyze_text(self, text: str, *, language_tag: str | None = None) -> dict[str, Any]:
        return analyze_text(text, language_tag=language_tag)

    def build_control_metadata(
        self,
        *,
        role: str | None = None,
        task: str | None = None,
        expert_id: int | None = None,
        script_code: str | None = None,
    ) -> dict[str, Any]:
        return build_control_metadata(
            role=role, task=task, expert_id=expert_id, script_code=script_code
        )

    def encode_chat_with_metadata(
        self,
        text: str,
        *,
        language_tag: str | None = None,
        role: str = "user",
        task: str | None = None,
        expert_id: int | None = None,
        add_generation_prompt: bool = True,
        **tokenizer_kwargs,
    ) -> dict[str, Any]:
        """Encode a single native chat message with Dhīr's preserved template.

        If the upstream tokenizer has no chat template, retain the explicit raw
        text path and report that no template was applied. Never invent a format.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        chat_template = _chat_template_value(self._tokenizer)
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if chat_template is None or not callable(apply_chat_template):
            encoded = self.encode_with_metadata(
                text,
                language_tag=language_tag,
                role=role,
                task=task,
                expert_id=expert_id,
                **tokenizer_kwargs,
            )
            encoded["chat_template_applied"] = False
            return encoded

        return_tensors = tokenizer_kwargs.pop("return_tensors", None)
        encoding = apply_chat_template(
            [{"role": role, "content": text}],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
            return_dict=True,
            **tokenizer_kwargs,
        )
        text_metadata = analyze_text(text, language_tag=language_tag)
        controls = build_control_metadata(
            role=role,
            task=task,
            expert_id=expert_id,
            script_code=text_metadata["dominant_script"],
        )
        return {
            "encoding": encoding,
            "text_metadata": text_metadata,
            "control_metadata": controls,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_vocab_sha256": self._manifest["tokenizer_vocab_sha256"],
            "token_ids_unchanged": True,
            "chat_template_applied": True,
        }

    def encode_with_metadata(
        self,
        text: str,
        *,
        language_tag: str | None = None,
        role: str | None = None,
        task: str | None = None,
        expert_id: int | None = None,
        **tokenizer_kwargs,
    ) -> dict[str, Any]:
        """Return native Dhīr token IDs plus separate non-token metadata."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoding = self._tokenizer(text, **tokenizer_kwargs)
        text_metadata = analyze_text(text, language_tag=language_tag)
        controls = build_control_metadata(
            role=role,
            task=task,
            expert_id=expert_id,
            script_code=text_metadata["dominant_script"],
        )
        return {
            "encoding": encoding,
            "text_metadata": text_metadata,
            "control_metadata": controls,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_vocab_sha256": self._manifest["tokenizer_vocab_sha256"],
            "token_ids_unchanged": True,
        }

    @property
    def neurocore_manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)
