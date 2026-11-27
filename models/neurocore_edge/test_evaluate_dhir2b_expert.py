import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from evaluate_dhir2b_expert import (
    ARCHITECTURE_VERSION,
    EXPERT_ID,
    MODEL_ID,
    MODEL_REVISION,
    TRAIN_SHA256,
    completion_loss,
    load_trained_expert,
)


class HeldoutEvaluatorTests(unittest.TestCase):
    def test_completion_loss_ignores_prompt_and_counts_only_assistant_targets(self):
        # Two next-token positions: first masked as prompt, second supervised.
        logits = torch.tensor([[
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]])
        labels = torch.tensor([[-100, -100, 2]])
        loss, count = completion_loss(logits, labels)
        self.assertEqual(count, 1)
        self.assertLess(loss, 0.02)

    def test_completion_loss_rejects_empty_assistant_targets(self):
        logits = torch.zeros((1, 3, 8))
        labels = torch.tensor([[-100, -100, -100]])
        with self.assertRaisesRegex(ValueError, "no supervised"):
            completion_loss(logits, labels)

    def test_completion_loss_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            completion_loss(torch.zeros((1, 3, 8)), torch.zeros((1, 2), dtype=torch.long))

    def make_payload(self, **overrides):
        payload = {
            "schema_version": 3,
            "status": "EXPERT_ADAPTER_TRAINED_UNVALIDATED",
            "architecture_version": ARCHITECTURE_VERSION,
            "expert_id": EXPERT_ID,
            "base_model_id": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "vocab_size": 131072,
            "num_hidden_layers": 28,
            "data_sha256": TRAIN_SHA256,
            "tokenizer_sha256": "tokenizer-directory-hash",
            "steps_completed": 1,
            "training_objective": "assistant_completion_only",
            "expert_state_dict": {"layer_0.up.weight": torch.ones((2, 2))},
        }
        payload.update(overrides)
        return payload

    def test_load_trained_expert_requires_exact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.pt"
            torch.save(self.make_payload(), path)
            model = Mock()
            result = load_trained_expert(
                model, path, tokenizer_sha="tokenizer-directory-hash"
            )
        model.load_expert_state_dict.assert_called_once_with(
            EXPERT_ID, {"layer_0.up.weight": torch.ones((2, 2))}
        )
        self.assertEqual(result["steps_completed"], 1)
        self.assertEqual(result["data_sha256"], TRAIN_SHA256)

    def test_load_trained_expert_rejects_dataset_or_tokenizer_mismatch(self):
        for overrides, tokenizer_sha in (
            ({"data_sha256": "wrong"}, "tokenizer-directory-hash"),
            ({}, "wrong-tokenizer"),
        ):
            with self.subTest(overrides=overrides, tokenizer_sha=tokenizer_sha):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "expert.pt"
                    torch.save(self.make_payload(**overrides), path)
                    with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                        load_trained_expert(Mock(), path, tokenizer_sha=tokenizer_sha)

    def test_load_trained_expert_rejects_untrained_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.pt"
            torch.save(self.make_payload(steps_completed=0), path)
            with self.assertRaisesRegex(ValueError, "at least one optimizer step"):
                load_trained_expert(Mock(), path, tokenizer_sha="tokenizer-directory-hash")


if __name__ == "__main__":
    unittest.main()
