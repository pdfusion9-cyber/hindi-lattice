import unittest
import torch

from neurocore_inference import run_neurocore_inference


class FakeTokenizer:
    def encode_with_metadata(self, prompt, **kwargs):
        return {
            "encoding": {"input_ids": torch.tensor([[1, 2, 3]])},
            "control_metadata": {
                "suggested_expert_id": 11,
                "suggested_expert_name": "hindi_indic_docs",
            },
            "text_metadata": {"dominant_script": "deva"},
            "tokenizer_name": "neurocore-dhir-compatible",
            "tokenizer_vocab_sha256": "a" * 64,
            "token_ids_unchanged": True,
        }

    def decode(self, ids, **kwargs):
        return "नमस्ते दुनिया"


class FakeModel:
    def __init__(self):
        self.calls = []
        self.backbone = self

    def generate(self, input_ids, max_new_tokens, do_sample, expert_id=None):
        self.calls.append({"expert_id": expert_id, "max_new_tokens": max_new_tokens})
        extra = torch.tensor([[4, 5]])
        return torch.cat([input_ids, extra], dim=1)


class NeuroCoreInferenceRunnerTests(unittest.TestCase):
    def test_task_metadata_dispatches_fixed_expert(self):
        model = FakeModel()
        result = run_neurocore_inference(
            model, FakeTokenizer(), "नमस्ते",
            task="hindi_indic_docs", language_tag="hi",
            mode="task_expert", max_new_tokens=8,
        )
        self.assertEqual(model.calls[-1]["expert_id"], 11)
        self.assertEqual(result["selected_expert_id"], 11)
        self.assertEqual(result["answer"], "नमस्ते दुनिया")
        self.assertEqual(result["generated_tokens"], 2)
        self.assertTrue(result["token_ids_unchanged"])
        self.assertFalse(result["training_performed"])

    def test_top2_mode_uses_native_router_path(self):
        model = FakeModel()
        result = run_neurocore_inference(
            model, FakeTokenizer(), "नमस्ते",
            task="hindi_indic_docs", language_tag="hi",
            mode="top2",
        )
        self.assertIsNone(model.calls[-1]["expert_id"])
        self.assertIsNone(result["selected_expert_id"])
        self.assertEqual(result["router_status"], "UNTRAINED_NOT_QUALITY_VALIDATED")

    def test_base_dhir_mode_bypasses_neurocore_adapters(self):
        model = FakeModel()
        result = run_neurocore_inference(
            model, FakeTokenizer(), "नमस्ते",
            task="hindi_indic_docs", language_tag="hi",
            mode="base_dhir",
        )
        self.assertIsNone(model.calls[-1]["expert_id"])
        self.assertEqual(result["mode"], "base_dhir")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            run_neurocore_inference(
                FakeModel(), FakeTokenizer(), "नमस्ते",
                task="hindi_indic_docs", language_tag="hi", mode="unknown",
            )

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            run_neurocore_inference(
                FakeModel(), FakeTokenizer(), " ",
                task="hindi_indic_docs", language_tag="hi",
            )


if __name__ == "__main__":
    unittest.main()
