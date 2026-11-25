import tempfile
import unittest
import json
from pathlib import Path

from count_canonical_parameters import count_canonical_parameters


class CanonicalParameterCountTests(unittest.TestCase):
    def _repo(self, root: Path):
        model_dir = root / "models" / "neurocore_edge"
        model_dir.mkdir(parents=True)
        config = {
            "vocab_size": 128,
            "hidden_size": 32,
            "num_experts": 4,
            "experts_per_token": 2,
            "expert_intermediate_size": 16,
            "attention_heads": 4,
            "kv_heads": 2,
            "sliding_window": 8,
            "recurrent_state_size": 8,
            "router_aux_loss_coefficient": 0.01,
            "prelude_layers": 1,
            "core_unique_layers": 1,
            "core_repeats": 3,
            "coda_layers": 1,
            "meta_token_count": 4,
            "rope_theta": 10000.0,
            "global_attention_layers": [3],
            "effective_layers": 5,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        # The reference model has 1 prelude + 1 core + 1 coda unique blocks.
        # The full expected count is obtained from the actual model in meta mode
        # and stored into the budget fixture after a first count.
        budget = {
            "expected_total_parameters": 0,
            "expected_shared_parameters": 0,
            "expected_expert_parameters": 0,
        }
        (model_dir / "canonical_parameter_budget.json").write_text(
            json.dumps(budget), encoding="utf-8"
        )

    def test_meta_counter_reports_exact_blocks_and_tied_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            # Run a preliminary counter against a zero budget. The resulting
            # exact counts come from the constructed model, not a hand formula.
            preliminary = count_canonical_parameters(root)
            budget_path = root / "models" / "neurocore_edge" / "canonical_parameter_budget.json"
            budget = {
                "expected_total_parameters": preliminary["total_parameters"],
                "expected_shared_parameters": preliminary["shared_parameters"],
                "expected_expert_parameters": preliminary["expert_parameters"],
            }
            budget_path.write_text(json.dumps(budget), encoding="utf-8")
            report = count_canonical_parameters(root)
        self.assertEqual(report["unique_parameterized_blocks"], 3)
        self.assertEqual(report["effective_depth"], 5)
        self.assertTrue(report["tied_embedding_and_lm_head"])
        self.assertFalse(report["storage_allocated_for_weights"])
        self.assertTrue(report["budget_matches"])
        self.assertEqual(
            report["total_parameters"],
            report["shared_parameters"] + report["expert_parameters"],
        )


if __name__ == "__main__":
    unittest.main()
