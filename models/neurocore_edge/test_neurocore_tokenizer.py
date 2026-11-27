import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from neurocore_tokenizer import (
    CONTROL_ID_COUNT,
    EXPERT_CONTROL_OFFSET,
    EXPERT_NAMES,
    MANIFEST_NAME,
    NeuroCoreTokenizer,
    analyze_text,
    build_control_metadata,
    normalize_language_tag,
    tokenizer_vocab_sha256,
    validate_neurocore_tokenizer,
)


class FakeTokenizer:
    is_fast = True
    chat_template = "{{ messages }}"

    def __init__(self):
        self.vocab = {f"tok_{i}": i for i in range(32)}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 0
        self.pad_token_id = 3

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 7, 2], "attention_mask": [1, 1, 1]}

    def save_pretrained(self, directory, **kwargs):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text(
            json.dumps({"vocab": self.vocab}, sort_keys=True),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": self.chat_template}),
            encoding="utf-8",
        )
        return (str(root / "tokenizer.json"),)


class NeuroCoreTokenizerTests(unittest.TestCase):
    def wrap(self):
        raw = FakeTokenizer()
        return raw, NeuroCoreTokenizer.from_tokenizer(
            raw,
            base_model_id="zenteiq-ai/Dhir-2B",
            base_revision="a" * 40,
            expected_vocab_size=32,
        )

    def test_vocab_ids_are_preserved_exactly(self):
        raw, tokenizer = self.wrap()
        self.assertEqual(tokenizer.get_vocab(), raw.get_vocab())
        self.assertEqual(
            tokenizer_vocab_sha256(tokenizer),
            tokenizer.neurocore_manifest["tokenizer_vocab_sha256"],
        )
        self.assertEqual(len(tokenizer), 32)
        self.assertFalse(tokenizer.neurocore_manifest["vocabulary_expansion"])
        self.assertFalse(tokenizer.neurocore_manifest["base_embedding_rows_reassigned"])

    def test_127_control_ids_are_side_channel_not_vocab_rows(self):
        _, tokenizer = self.wrap()
        controls = tokenizer.neurocore_manifest["control_ids"]
        self.assertEqual(CONTROL_ID_COUNT, 127)
        self.assertEqual(controls["count"], 127)
        self.assertEqual(controls["minimum"], 0)
        self.assertEqual(controls["maximum"], 126)
        self.assertFalse(controls["consumes_vocabulary_ids"])
        self.assertEqual(len(tokenizer), 32)
        self.assertEqual(controls["expert_control_offset"], EXPERT_CONTROL_OFFSET)

    def test_unicode_script_profile_for_hindi_and_english_mixture(self):
        report = analyze_text("नमस्ते दुनिया भारत AI")
        self.assertEqual(report["dominant_script"], "deva")
        self.assertEqual(report["script_mix"], "mixed")
        self.assertEqual(report["language_tag"], None)
        self.assertEqual(
            report["language_tag_source"],
            "unspecified_not_inferred_from_script",
        )
        self.assertGreater(report["devanagari_orthography"]["word_count"], 0)
        self.assertEqual(
            report["normalization_policy"],
            "NFC_FOR_METADATA_ONLY_INPUT_TEXT_PRESERVED",
        )

    def test_multilingual_script_profile(self):
        report = analyze_text("नमस्ते বাংলা தமிழ் ગુજરાતી")
        for code in ("deva", "beng", "taml", "gujr"):
            self.assertIn(code, report["script_counts"])
        self.assertEqual(report["script_mix"], "mixed")

    def test_language_tag_is_explicit_and_normalized(self):
        self.assertEqual(normalize_language_tag("hi-in"), "hi-IN")
        report = analyze_text("नमस्ते", language_tag="hi-in")
        self.assertEqual(report["language_tag"], "hi-IN")
        self.assertEqual(report["language_tag_source"], "caller_supplied_hint")
        with self.assertRaises(ValueError):
            normalize_language_tag("not a language tag")

    def test_control_registry_matches_fixed_expert_manifest(self):
        control = build_control_metadata(
            role="user", task="hindi_indic_docs", script_code="deva"
        )
        self.assertEqual(control["suggested_expert_id"], 11)
        self.assertEqual(control["suggested_expert_name"], EXPERT_NAMES[11])
        self.assertEqual(control["control_ids"], [1, 21, 32 + 11, 64])
        self.assertEqual(control["expert_selection_policy"], "hint_only_no_automatic_dispatch")
        self.assertFalse(control["control_ids_are_vocabulary_tokens"])

    def test_explicit_expert_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            build_control_metadata(task="hindi_indic_docs", expert_id=10)

    def test_invalid_control_values_are_rejected(self):
        with self.assertRaises(ValueError):
            build_control_metadata(role="administrator")
        with self.assertRaises(ValueError):
            build_control_metadata(expert_id=20)
        with self.assertRaises(ValueError):
            build_control_metadata(task="unregistered_task")

    def test_encode_with_metadata_keeps_ids_separate(self):
        _, tokenizer = self.wrap()
        result = tokenizer.encode_with_metadata(
            "हिंदी दस्तावेज़",
            language_tag="hi",
            role="user",
            task="hindi_indic_docs",
        )
        self.assertEqual(result["encoding"]["input_ids"], [1, 7, 2])
        self.assertEqual(result["text_metadata"]["language_tag"], "hi")
        self.assertEqual(result["control_metadata"]["suggested_expert_id"], 11)
        self.assertTrue(result["token_ids_unchanged"])
        self.assertEqual(len(tokenizer), 32)

    def test_save_writes_versioned_manifest_and_hashes(self):
        _, tokenizer = self.wrap()
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer.save_pretrained(tmp)
            path = Path(tmp) / MANIFEST_NAME
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 2)
            self.assertEqual(
                data["tokenizer_vocab_sha256"],
                tokenizer_vocab_sha256(tokenizer),
            )
            self.assertTrue(data["tokenizer_files_sha256"])
            self.assertEqual(data["base_revision"], "a" * 40)
            self.assertIn("language_metadata", data)
            self.assertIn("routing_metadata", data)

    def test_missing_manifest_rebuilt_only_with_pinned_file_hash(self):
        from unittest.mock import patch
        expected_files_sha = "cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056"
        with tempfile.TemporaryDirectory() as tmp:
            with patch("neurocore_tokenizer.tokenizer_files_sha256", return_value=expected_files_sha):
                with patch("transformers.PreTrainedTokenizerFast.from_pretrained", return_value=FakeTokenizer()):
                    tokenizer = NeuroCoreTokenizer.from_pretrained(
                        tmp,
                        base_model_id="zenteiq-ai/Dhir-2B",
                        base_revision="a" * 40,
                        expected_vocab_size=32,
                        expected_tokenizer_files_sha256=expected_files_sha,
                    )
            manifest_path = Path(tmp) / MANIFEST_NAME
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokenizer_files_sha256"], expected_files_sha)
            self.assertEqual(tokenizer.get_vocab(), FakeTokenizer().get_vocab())

    def test_missing_manifest_without_pinned_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("neurocore_tokenizer.tokenizer_files_sha256", return_value="unreviewed"):
                with self.assertRaises(FileNotFoundError):
                    NeuroCoreTokenizer.from_pretrained(
                        tmp,
                        base_model_id="zenteiq-ai/Dhir-2B",
                        base_revision="a" * 40,
                        expected_vocab_size=32,
                    )

    def test_token_id_gaps_are_rejected(self):
        raw = FakeTokenizer()
        raw.vocab["tok_31"] = 32
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_neurocore_tokenizer(
                raw, expected_vocab_size=32,
                base_model_id="zenteiq-ai/Dhir-2B",
                base_revision="a" * 40,
            )

    def test_vocab_growth_is_rejected(self):
        raw = FakeTokenizer()
        raw.vocab["<|new_control|>"] = 32
        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            validate_neurocore_tokenizer(
                raw, expected_vocab_size=32,
                base_model_id="zenteiq-ai/Dhir-2B",
                base_revision="a" * 40,
            )

    def test_special_token_outside_vocab_is_rejected(self):
        raw = FakeTokenizer()
        raw.pad_token_id = 100
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_neurocore_tokenizer(
                raw, expected_vocab_size=32,
                base_model_id="zenteiq-ai/Dhir-2B",
                base_revision="a" * 40,
            )

    def test_chat_template_is_hashed_and_preserved(self):
        _, tokenizer = self.wrap()
        self.assertTrue(tokenizer.neurocore_manifest["chat_template_preserved"])
        self.assertTrue(tokenizer.neurocore_manifest["chat_template_sha256"])


if __name__ == "__main__":
    unittest.main()
