import json
import tempfile
import unittest
import sys
from unittest.mock import patch

import train_dhir2b_expert as trainer
from pathlib import Path

from train_dhir2b_expert import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    contains_pii_pattern,
    read_rows,
    require_approval,
)


def approved_document(data_sha, tokenizer_sha):
    return {
        "datasets": [{
            "expert_id": 11,
            "dataset_id": DEFAULT_DATASET_ID,
            "revision": DEFAULT_DATASET_REVISION,
            "license_id": "CC0-1.0",
            "license_terms_url": "https://example.org/license",
            "commercial_training_allowed": True,
            "pii_reviewed": True,
            "consent_reviewed": True,
            "schema_reviewed": True,
            "split_reviewed": True,
            "approved": True,
            "approval_id": "DATASET-APPROVAL-001",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-11-26T00:00:00Z",
        }],
        "approved_training_artifacts": [{
            "stage": "dhir2b_expert",
            "expert_id": 11,
            "dataset_id": DEFAULT_DATASET_ID,
            "dataset_revision": DEFAULT_DATASET_REVISION,
            "sha256": data_sha,
            "approved": True,
            "approval_id": "CORPUS-APPROVAL-001",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-11-26T00:00:00Z",
        }],
        "approved_tokenizers": [{
            "sha256": tokenizer_sha,
            "approved": True,
            "approval_id": "TOKENIZER-APPROVAL-001",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-11-26T00:00:00Z",
        }],
    }


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __len__(self):
        return 32

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 5, 2]}


class TrainingApprovalGateTests(unittest.TestCase):
    def write_doc(self, path, document):
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_pending_approval_stops_before_checkpoint_load_and_optimizer_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            (model_dir / "tokenizer").mkdir(parents=True)
            data_path = root / "expert16.jsonl"
            data_path.write_text('{"prompt":"test","response":"test"}\\n', encoding="utf-8")
            approval_path = root / "approval.json"
            self.write_doc(approval_path, {
                "status": "ALL_DATASETS_PENDING_REVIEW",
                "datasets": [],
                "approved_training_artifacts": [],
                "approved_tokenizers": [],
            })
            argv = [
                "train_dhir2b_expert.py",
                "--model-dir", str(model_dir),
                "--data", str(data_path),
                "--approval", str(approval_path),
                "--dataset-id", "local/neurocore_synthetic_sft/expert16_self_correction",
                "--dataset-revision", "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
                "--expert-id", "16",
                "--steps", "10",
                "--output", str(root / "adapter.pt"),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(trainer, "sha_dir", return_value="b" * 64):
                    with patch.object(trainer.torch, "set_num_threads"):
                        with patch.object(trainer.Dhir2BNeuroCore, "load_checkpoint") as load_model:
                            with patch.object(trainer.torch.optim, "AdamW") as make_optimizer:
                                with self.assertRaisesRegex(PermissionError, "dataset ID/revision"):
                                    trainer.main()
                load_model.assert_not_called()
            make_optimizer.assert_not_called()

    def test_current_repository_default_deny_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            self.write_doc(path, {
                "status": "ALL_DATASETS_PENDING_REVIEW",
                "datasets": [],
                "approved_training_artifacts": [],
                "approved_tokenizers": [],
            })
            with self.assertRaisesRegex(PermissionError, "dataset ID/revision"):
                require_approval(path, 11, "a" * 64, "b" * 64)

    def test_exact_all_level_approvals_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            self.write_doc(path, approved_document("a" * 64, "b" * 64))
            result = require_approval(path, 11, "a" * 64, "b" * 64)
            self.assertIn("datasets", result)

    def test_wrong_dataset_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            document = approved_document("a" * 64, "b" * 64)
            document["datasets"][0]["revision"] = "f" * 40
            self.write_doc(path, document)
            with self.assertRaisesRegex(PermissionError, "dataset ID/revision"):
                require_approval(path, 11, "a" * 64, "b" * 64)

    def test_incomplete_privacy_or_license_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            document = approved_document("a" * 64, "b" * 64)
            document["datasets"][0]["pii_reviewed"] = False
            self.write_doc(path, document)
            with self.assertRaisesRegex(PermissionError, "pii_reviewed"):
                require_approval(path, 11, "a" * 64, "b" * 64)

    def test_unapproved_or_mismatched_corpus_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            self.write_doc(path, approved_document("a" * 64, "b" * 64))
            with self.assertRaisesRegex(PermissionError, "corpus SHA-256"):
                require_approval(path, 11, "c" * 64, "b" * 64)

    def test_unapproved_or_mismatched_tokenizer_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            self.write_doc(path, approved_document("a" * 64, "b" * 64))
            with self.assertRaisesRegex(PermissionError, "tokenizer SHA-256"):
                require_approval(path, 11, "a" * 64, "c" * 64)

    def test_training_corpus_pii_filter(self):
        self.assertTrue(contains_pii_pattern("फोन 9876543210"))
        self.assertTrue(contains_pii_pattern("test@example.org"))
        self.assertTrue(contains_pii_pattern("आधार 1234 5678 9012"))
        self.assertTrue(contains_pii_pattern("PAN ABCDE1234F"))
        self.assertFalse(contains_pii_pattern("भारत की भाषाई विविधता"))

    def test_training_rows_are_bounded_and_pii_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            rows = [
                {"text": "भारत की भाषाई विविधता"},
                {"text": "फोन 9876543210"},
                {"text": "हिंदी भाषा का अध्ययन"},
                {"text": "गणित और विज्ञान"},
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = read_rows(path, FakeTokenizer(), max_length=8, max_rows=2)
            self.assertEqual(
                [row["input_ids"] for row in result],
                [[1, 5, 2], [1, 5, 2]],
            )
            self.assertTrue(all(row["labels"] == row["input_ids"] for row in result))
            self.assertTrue(all(row["supervision"] == "all_tokens" for row in result))

    def test_prompt_response_uses_native_template_and_masks_prompt_loss(self):
        class ChatTokenizer(FakeTokenizer):
            chat_template = "test-native-dhir-template"

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                if add_generation_prompt:
                    return [1, 4, 5]
                return [1, 4, 5, 8, 9, 2]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt": "प्रश्न", "response": "उत्तर"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows = read_rows(path, ChatTokenizer(), max_length=8, max_rows=5)
            self.assertEqual(rows[0]["input_ids"], [1, 4, 5, 8, 9, 2])
            self.assertEqual(rows[0]["labels"], [-100, -100, -100, 8, 9, 2])
            self.assertEqual(rows[0]["supervision"], "assistant_completion_only")

    def test_prompt_response_rejects_non_prefix_chat_template(self):
        class BrokenChatTokenizer(FakeTokenizer):
            chat_template = "test-native-dhir-template"

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return [1, 4, 6] if add_generation_prompt else [1, 4, 5, 8, 2]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt": "प्रश्न", "response": "उत्तर"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "native assistant-generation prefix"):
                read_rows(path, BrokenChatTokenizer(), max_length=8, max_rows=5)

    def test_prompt_response_requires_native_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt": "प्रश्न", "response": "उत्तर"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "native chat template"):
                read_rows(path, FakeTokenizer(), max_length=8, max_rows=5)

    def test_batch_rows_preserves_completion_only_loss_mask(self):
        from train_dhir2b_expert import batch_rows
        rows = [
            {"input_ids": [1, 4, 5, 8, 2], "labels": [-100, -100, -100, 8, 2]},
            {"input_ids": [1, 6, 7, 9, 2], "labels": [-100, -100, -100, 9, 2]},
        ]
        input_ids, labels, attention_mask = batch_rows(rows, [0, 1], pad_id=0)
        self.assertEqual(input_ids.tolist(), [[1, 4, 5, 8, 2], [1, 6, 7, 9, 2]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 8, 2], [-100, -100, -100, 9, 2]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    def test_invalid_row_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.jsonl"
            path.write_text('{"text":"भारत"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_length"):
                read_rows(path, FakeTokenizer(), max_length=1, max_rows=10)

    def test_pending_synthetic_expert_datasets_are_rejected(self):
        approval_path = Path(__file__).parent / "dataset_approval.json"
        document = json.loads(approval_path.read_text(encoding="utf-8"))
        for expert_id, dataset_id in (
            (16, "local/neurocore_synthetic_sft/expert16_self_correction"),
            (17, "local/neurocore_synthetic_sft/expert17_planning"),
        ):
            with self.subTest(expert_id=expert_id):
                registered = next(
                    (
                        row for row in document.get("datasets", [])
                        if row.get("expert_id") == expert_id
                        and row.get("dataset_id") == dataset_id
                    ),
                    None,
                )
                self.assertIsNotNone(
                    registered, f"pending synthetic dataset is not registered: {dataset_id}"
                )
                self.assertTrue(registered.get("revision"))
                for field in (
                    "commercial_training_allowed",
                    "pii_reviewed",
                    "consent_reviewed",
                    "schema_reviewed",
                    "split_reviewed",
                    "approved",
                ):
                    self.assertIs(
                        registered.get(field), False,
                        f"{dataset_id} must keep {field}=false until review",
                    )

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "approval.json"
                    self.write_doc(path, document)
                    with self.assertRaisesRegex(PermissionError, "approval/review fields"):
                        require_approval(
                            path, expert_id, "a" * 64, "b" * 64,
                            dataset_id=dataset_id,
                            dataset_revision=registered["revision"],
                        )


if __name__ == "__main__":
    unittest.main()
