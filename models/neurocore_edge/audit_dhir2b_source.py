"""Read-only Hub metadata audit for the official Dhīr-2B checkpoint.

Downloads only config/tokenizer metadata and an optional safetensors index.
It deliberately does not fetch multi-gigabyte model weight shards.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

REPO_ID = os.environ.get("DHIR_MODEL_ID", "zenteiq-ai/Dhir-2B")
REPORT_PATH = Path(os.environ.get("DHIR_REPORT_PATH", "dhir2b_source_audit.json"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    api = HfApi()
    info = api.model_info(REPO_ID, files_metadata=True)
    revision = info.sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Hugging Face did not return a 40-character immutable commit SHA")

    config = AutoConfig.from_pretrained(
        REPO_ID, revision=revision, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID, revision=revision, trust_remote_code=False
    )

    facts = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "hidden_act": getattr(config, "hidden_act", None),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
    }
    checks = {
        "hidden_size_2048": facts["hidden_size"] == 2048,
        "layers_28": facts["num_hidden_layers"] == 28,
        "query_heads_16": facts["num_attention_heads"] == 16,
        "kv_heads_8": facts["num_key_value_heads"] == 8,
        "context_at_least_16384": (
            isinstance(facts["max_position_embeddings"], int)
            and facts["max_position_embeddings"] >= 16384
        ),
        "rope_theta_1e6": (
            facts["rope_theta"] is not None
            and abs(float(facts["rope_theta"]) - 1_000_000.0) < 1e-3
        ),
        "silu_activation": facts["hidden_act"] == "silu",
        "tied_embeddings": facts["tie_word_embeddings"] is True,
        "intermediate_size_present": (
            isinstance(facts["intermediate_size"], int)
            and facts["intermediate_size"] > 0
        ),
    }

    tokenizer_vocab = tokenizer.get_vocab()
    token_ids = [int(value) for value in tokenizer_vocab.values()]
    tokenizer_len = len(tokenizer)
    checks["tokenizer_len_matches_config"] = tokenizer_len == facts["vocab_size"]
    checks["tokenizer_ids_are_contiguous"] = (
        len(token_ids) == tokenizer_len
        and set(token_ids) == set(range(tokenizer_len))
    )

    files = []
    for sibling in info.siblings:
        files.append({
            "name": sibling.rfilename,
            "size": getattr(sibling, "size", None),
            "lfs_oid": getattr(getattr(sibling, "lfs", None), "oid", None),
        })

    metadata_hashes = {}
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if any(item["name"] == filename for item in files):
            local = Path(hf_hub_download(REPO_ID, filename=filename, revision=revision))
            metadata_hashes[filename] = sha256(local)

    index_candidates = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    index_file = next((name for name in index_candidates
                       if any(item["name"] == name for item in files)), None)
    weight_index_checks = {"index_present": index_file is not None}
    if index_file:
        index_path = Path(hf_hub_download(REPO_ID, filename=index_file, revision=revision))
        index = json.loads(index_path.read_text())
        names = list(index.get("weight_map", {}))
        for component in ("q_norm", "k_norm"):
            per_layer = [
                any(f"model.layers.{i}.self_attn.{component}." in name for name in names)
                for i in range(28)
            ]
            weight_index_checks[f"{component}_present_all_28_layers"] = all(per_layer)
        weight_index_checks["embedding_tensor_present"] = any(
            name.endswith("embed_tokens.weight") for name in names
        )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "repo_id": REPO_ID,
        "revision": revision,
        "license": getattr(info.card_data, "license", None) if info.card_data else None,
        "config": facts,
        "tokenizer_length": tokenizer_len,
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer_vocab, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "metadata_file_sha256": metadata_hashes,
        "files": files,
        "architecture_checks": checks,
        "weight_index_checks": weight_index_checks,
        "weight_shards_downloaded": False,
        "note": "This audit reads metadata/index only; it does not download or import model weight shards.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("Dhīr-2B architecture/tokenizer checks failed")


if __name__ == "__main__":
    main()
