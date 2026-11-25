import json, tempfile, unittest
from pathlib import Path
from canonical_architecture_audit import audit_repository

class TestCanonicalAudit(unittest.TestCase):
    def fixture(self,root,expert=2849832960):
        d=root/"models/neurocore_edge"; d.mkdir(parents=True)
        cfg={"architecture_version":"0.3.2-canonical","status":"UNTRAINED_INITIALIZATION_ONLY",
          "vocab_size":65536,"hidden_size":2048,"num_experts":20,"expert_intermediate_size":2899,
          "prelude_layers":2,"core_unique_layers":4,"core_repeats":5,"coda_layers":2,
          "effective_layers":24,"unique_layers":8,"expected_total_parameters":3074921472,
          "training":{"enabled":False}}
        budget={"expected_total_parameters":3074921472,"expected_shared_parameters":225088512,
          "expected_expert_parameters":expert}
        (d/"config.json").write_text(json.dumps(cfg))
        (d/"canonical_parameter_budget.json").write_text(json.dumps(budget))
        (d/"modeling_neurocore_edge.py").write_text(
          "self.prelude_layers = nn.ModuleList([])\nself.core_layers = nn.ModuleList([])\n"
          "self.coda_layers = nn.ModuleList([])\nclass GQARotaryAttention: pass\n"
          "self.meta_token_bank = nn.Parameter(torch.empty(128, 2048))\n")
    def test_config_width_2899_matches_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root); r=audit_repository(root)
        self.assertEqual(r["parameter_budget"]["budget_derived_expert_intermediate_size"],2899)
        self.assertEqual(r["parameter_budget"]["expert_parameters_implied_by_config"],2849832960)
        self.assertEqual(r["status"],"BLOCKED") # no tokenizer/checkpoint in fixture
    def test_stale_budget_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,2202009600); r=audit_repository(root)
        self.assertTrue(any("expert width mismatch" in x for x in r["blockers"]))
    def test_missing_assets_block_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root); r=audit_repository(root)
        self.assertFalse(r["can_start_training"])
        self.assertTrue(any("tokenizer asset" in x for x in r["blockers"]))
    def test_bad_budget_sum_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            p=root/"models/neurocore_edge/canonical_parameter_budget.json"
            b=json.loads(p.read_text()); b["expected_shared_parameters"]-=1; p.write_text(json.dumps(b))
            r=audit_repository(root)
        self.assertTrue(any("do not sum" in x for x in r["blockers"]))

if __name__=="__main__": unittest.main()
