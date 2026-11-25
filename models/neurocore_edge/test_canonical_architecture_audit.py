import json
import tempfile
import unittest
from pathlib import Path

from canonical_architecture_audit import audit_repository


class CanonicalArchitectureAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, intermediate: int = 2899) -> None:
        model_dir = root / "models" / "neurocore_edge"
        model_dir.mkdir(parents=True)
        config = {
            "architecture_version": "0.3.2-canonical",
            "status": "UNTRAINED_INITIALIZATION_ONLY",
            "vocab_size": 65536,
            "hidden_size": 2048,
            "num_experts": 20,
            "expert_intermediate_size": intermediate,
            "prelude_layers": 2,
            "core_unique_layers": 4,
            "core_repeats": 5,
            "coda_layers": 2,
            "effective_layers": 24,
            "unique_layers": 8,
            "expected_total_parameters": 3074885656,
            "training": {"enabled": False},
        }
        budget = {
            "expected_total_parameters": 3074885656,
            "expected_shared_parameters": 872876056,
            "expected_expert_parameters": 2202009600,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model_dir / "canonical_parameter_budget.json").write_text(json.dumps(budget), encoding="utf-8")
        (model_dir / "modeling_neurocore_edge.py").write_text(
            "self.layers = nn.ModuleList([\n"
            "    Layer(cfg) for _ in range(cfg.num_hidden_layers)\n"
            "])\n",
            encoding="utf-8",
        )

    def test_budget_implies_2240_not_configured_2899(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, intermediate=2899)
            report = audit_repository(root)
        budget = report["parameter_budget"]
        self.assertEqual(budget["budget_derived_expert_intermediate_size"], 2240)
        self.assertEqual(budget["config_expert_intermediate_size"], 2899)
        self.assertEqual(budget["expert_parameters_implied_by_config"], 2849832960)
        self.assertEqual(budget["expected_expert"], 2202009600)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["can_start_training"])

    def test_detects_missing_sandwich_gqa_meta_tokenizer_and_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            report = audit_repository(root)
        blockers = " ".join(report["blockers"])
        self.assertIn("sandwich blocks", blockers)
        self.assertIn("GQA + RoPE", blockers)
        self.assertIn("128-token meta-token bank", blockers)
        self.assertIn("tokenizer asset", blockers)
        self.assertIn("trained NeuroCore base checkpoint", blockers)
        self.assertFalse(report["can_start_training"])

    def test_detects_an_incorrect_total_budget_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            budget_path = root / "models" / "neurocore_edge" / "canonical_parameter_budget.json"
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            budget["expected_shared_parameters"] -= 1
            budget_path.write_text(json.dumps(budget), encoding="utf-8")
            report = audit_repository(root)
        self.assertTrue(any("do not sum" in x for x in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
