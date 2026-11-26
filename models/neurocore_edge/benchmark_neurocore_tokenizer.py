"""Privacy-conscious, measurement-only benchmark for NeuroCore's Dhīr tokenizer."""
from __future__ import annotations
import argparse, json, re, statistics
from collections import defaultdict
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from neurocore_tokenizer import NeuroCoreTokenizer

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")

def contains_pii_pattern(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or AADHAAR_RE.search(text))

def run_benchmark(tokenizer_dir: Path, input_path: Path, *, base_model_id: str,
                  base_revision: str, expected_vocab_size: int, max_samples: int) -> dict:
    wrapped = NeuroCoreTokenizer.from_pretrained(
        tokenizer_dir, base_model_id=base_model_id, base_revision=base_revision,
        expected_vocab_size=expected_vocab_size)
    baseline = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    unk_id = getattr(baseline, "unk_token_id", None)
    token_counts, word_counts, char_counts, unknown_counts = [], [], [], []
    by_script = defaultdict(lambda: {"samples": 0, "tokens": 0, "words": 0})
    by_mix = defaultdict(int)
    exact_id_matches = roundtrip_matches = metadata_ok = used = skipped_pii = skipped_empty = 0

    with input_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                skipped_empty += 1
                continue
            if contains_pii_pattern(text):
                skipped_pii += 1
                continue
            if used >= max_samples:
                break
            baseline_ids = baseline(text, add_special_tokens=False)["input_ids"]
            result = wrapped.encode_with_metadata(text, add_special_tokens=False)
            wrapped_ids = result["encoding"]["input_ids"]
            decoded = baseline.decode(
                wrapped_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            words = max(1, len(text.split()))
            script = result["text_metadata"].get("dominant_script") or "none"
            mix = result["text_metadata"].get("script_mix") or "none"
            exact_id_matches += int(baseline_ids == wrapped_ids)
            roundtrip_matches += int(decoded == text)
            metadata_ok += int(isinstance(result["text_metadata"], dict)
                               and isinstance(result["control_metadata"], dict))
            token_counts.append(len(wrapped_ids))
            word_counts.append(words)
            char_counts.append(len(text))
            bucket = by_script[script]
            bucket["samples"] += 1
            bucket["tokens"] += len(wrapped_ids)
            bucket["words"] += words
            by_mix[mix] += 1
            if unk_id is not None:
                unknown_counts.append(sum(token_id == unk_id for token_id in wrapped_ids))
            used += 1
            del text, baseline_ids, wrapped_ids, decoded, result

    if not used:
        raise ValueError("no eligible records found after empty/PII filtering")
    total_tokens, total_words = sum(token_counts), sum(word_counts)
    total_unknown = sum(unknown_counts) if unk_id is not None else None
    script_metrics = {
        key: {
            **values,
            "tokens_per_word": values["tokens"] / values["words"] if values["words"] else None,
        }
        for key, values in sorted(by_script.items())
    }
    return {
        "benchmark": "neurocore_dhir_tokenizer_measurement_only_v2",
        "training_performed": False,
        "vocabulary_modified": False,
        "raw_text_persisted": False,
        "tokenized_records_persisted": False,
        "tokenizer_name": wrapped.neurocore_manifest["name"],
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "vocab_size": len(wrapped),
        "tokenizer_vocab_sha256": wrapped.neurocore_manifest["tokenizer_vocab_sha256"],
        "samples_used": used,
        "records_skipped_for_pii_patterns": skipped_pii,
        "records_skipped_empty": skipped_empty,
        "baseline_exact_token_id_match_rate": exact_id_matches / used,
        "decode_roundtrip_exact_rate": roundtrip_matches / used,
        "metadata_success_rate": metadata_ok / used,
        "total_tokens": total_tokens,
        "total_whitespace_words": total_words,
        "tokens_per_word": total_tokens / total_words,
        "mean_tokens_per_record": statistics.fmean(token_counts),
        "median_tokens_per_record": statistics.median(token_counts),
        "mean_characters_per_record": statistics.fmean(char_counts),
        "script_bucket_metrics": script_metrics,
        "script_mix_sample_counts": dict(sorted(by_mix.items())),
        "unknown_token_count": total_unknown,
        "unknown_token_rate": (
            total_unknown / total_tokens if total_unknown is not None and total_tokens else None),
        "privacy_filter_note": (
            "Simple email/India-mobile/Aadhaar-like pattern filter; "
            "not a complete PII detector or privacy certification."),
        "benchmark_scope": "self-authored multilingual fixture; not a representative public corpus",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        args.tokenizer_dir, args.input, base_model_id=args.base_model_id,
        base_revision=args.base_revision, expected_vocab_size=args.vocab_size,
        max_samples=args.max_samples)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")

if __name__ == "__main__":
    main()
