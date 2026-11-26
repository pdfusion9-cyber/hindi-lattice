import json
import tempfile
import unittest
from pathlib import Path

from neurocore_tokenizer import (
    CONTROL_ID_COUNT,
    MANIFEST_NAME,
    NeuroCoreTokenizer,
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
        self.assertEqual(tokenizer_vocab_sha256(tokenizer), tokenizer.neurocore_manifest["tokenizer_vocab_sha256"])
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

    def test_save_writes_neurocore_manifest(self):
        _, tokenizer = self.wrap()
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer.save_pretrained(tmp)
            path = Path(tmp) / MANIFEST_NAME
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["tokenizer_vocab_sha256"], tokenizer_vocab_sha256(tokenizer))
            self.assertTrue(data["tokenizer_files_sha256"])
            self.assertEqual(data["base_revision"], "a" * 40)

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
