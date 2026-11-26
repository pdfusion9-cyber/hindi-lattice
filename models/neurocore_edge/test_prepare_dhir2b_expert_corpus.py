import json
import tempfile
import unittest
from pathlib import Path

from prepare_dhir2b_expert_corpus import (
    DATASET_ID,
    DATASET_REVISION,
    classify_pii,
    prepare_records,
    require_human_approvals,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        # Test double: stable IDs and no raw text persisted to corpus.
        return {"input_ids": [1] + [10 + (ord(ch) % 50) for ch in text[:20]] + [2]}


def approval_doc(tokenizer_sha):
    return {
        "status": "ALL_DATASETS_REVIEWED",
        "datasets": [{
            "expert_id": 11,
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license_id": "CC0-1.0",
            "license_terms_url": "https://example.org/license",
            "approved": True,
            "commercial_training_allowed": True,
            "pii_reviewed": True,
            "consent_reviewed": True,
            "schema_reviewed": True,
            "split_reviewed": True,
            "approval_id": "DATASET-001",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-11-26T00:00:00Z",
        }],
        "approved_tokenizers": [{
            "sha256": tokenizer_sha,
            "approved": True,
            "approval_id": "TOKENIZER-001",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-11-26T00:00:00Z",
        }],
    }


class PrepareDhīrExpertCorpusTests(unittest.TestCase):
    def test_pii_classification(self):
        self.assertEqual(classify_pii("फोन 9876543210"), {"phone"})
        self.assertEqual(classify_pii("ईमेल demo@example.org"), {"email"})
        self.assertEqual(classify_pii("आधार 1234 5678 9012"), {"aadhaar_like"})
        self.assertEqual(classify_pii("PAN ABCDE1234F"), {"pan_like"})
        self.assertEqual(classify_pii("साफ हिंदी वाक्य"), set())

    def test_only_clean_devanagari_token_ids_are_written(self):
        rows = [
            {"text": "भारत की भाषाई विविधता"},
            {"text": "फोन 9876543210"},
            {"text": "English only sample"},
            {"text": "भारत की भाषाई विविधता"},  # duplicate
            {"text": "हिंदी विज्ञान और शिक्षा"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "private" / "expert11.jsonl"
            report = prepare_records(
                rows,
                FakeTokenizer(),
                output,
                max_records=2,
                max_scan_rows=20,
                max_sequence_length=8,
                repo_root=Path(tmp) / "checkout-not-used",
            )
            stored = output.read_text(encoding="utf-8")
            self.assertNotIn("भारत", stored)
            self.assertNotIn("9876543210", stored)
            self.assertEqual(report["eligible_records"], 2)
            self.assertEqual(report["pii_rows_excluded"], 1)
            self.assertEqual(report["non_devanagari_rows_excluded"], 1)
            self.assertEqual(report["duplicates_excluded"], 1)
            self.assertTrue(report["corpus_sha256"])
            self.assertFalse(report["raw_text_persisted"])
            self.assertFalse(report["training_performed"])
            for line in stored.splitlines():
                ids = json.loads(line)["input_ids"]
                self.assertTrue(all(0 <= token_id < 131072 for token_id in ids))

    def test_insufficient_clean_records_does_not_commit_output(self):
        rows = [{"text": "फोन 9876543210"}, {"text": "English only"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "private" / "expert11.jsonl"
            with self.assertRaisesRegex(ValueError, "eligible Hindi records"):
                prepare_records(
                    rows,
                    FakeTokenizer(),
                    output,
                    max_records=1,
                    max_scan_rows=10,
                    repo_root=Path(tmp) / "checkout-not-used",
                )
            self.assertFalse(output.exists())

    def test_output_inside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / ".git").mkdir()
            output = root / "data" / "corpus.jsonl"
            with self.assertRaisesRegex(ValueError, "inside Git checkout"):
                prepare_records(
                    [{"text": "हिंदी भाषा"}],
                    FakeTokenizer(),
                    output,
                    max_records=1,
                    max_scan_rows=10,
                    repo_root=root,
                )

    def test_current_default_deny_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            path.write_text(json.dumps({
                "status": "ALL_DATASETS_PENDING_REVIEW",
                "datasets": [{
                    "expert_id": 11,
                    "dataset_id": DATASET_ID,
                    "revision": DATASET_REVISION,
                    "license_id": "CC0-1.0",
                    "license_terms_url": "https://example.org/license",
                    "approved": False,
                    "commercial_training_allowed": False,
                    "pii_reviewed": False,
                    "consent_reviewed": False,
                    "schema_reviewed": True,
                    "split_reviewed": True,
                }],
                "approved_tokenizers": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "dataset review incomplete"):
                require_human_approvals(
                    path, expert_id=11, dataset_id=DATASET_ID,
                    dataset_revision=DATASET_REVISION, tokenizer_sha256="a" * 64,
                )

    def test_unrelated_experts_pending_do_not_block_fully_approved_expert11(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            document = approval_doc("a" * 64)
            document["status"] = "ALL_DATASETS_PENDING_REVIEW"
            document["datasets"].append({
                "expert_id": 1,
                "dataset_id": "unrelated/other-dataset",
                "revision": "f" * 40,
                "approved": False,
            })
            self.write_doc(path, document)
            result = require_human_approvals(
                path, expert_id=11, dataset_id=DATASET_ID,
                dataset_revision=DATASET_REVISION, tokenizer_sha256="a" * 64,
            )
            self.assertEqual(result["status"], "ALL_DATASETS_PENDING_REVIEW")

    def test_missing_tokenizer_hash_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            path.write_text(json.dumps(approval_doc("b" * 64)), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "tokenizer SHA-256"):
                require_human_approvals(
                    path, expert_id=11, dataset_id=DATASET_ID,
                    dataset_revision=DATASET_REVISION, tokenizer_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
