import json, tempfile, unittest
from pathlib import Path
from count_canonical_parameters import count_canonical_parameters

class TestMetaCounter(unittest.TestCase):
    def test_exact_count_without_allocating_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=root/"models/neurocore_edge"; d.mkdir(parents=True)
            config={"vocab_size":128,"hidden_size":32,"num_experts":4,
              "experts_per_token":2,"expert_intermediate_size":16,
              "attention_heads":4,"kv_heads":2,"sliding_window":8,
              "recurrent_state_size":8,"router_aux_loss_coefficient":0.01,
              "prelude_layers":1,"core_unique_layers":1,"core_repeats":3,
              "coda_layers":1,"meta_token_count":4,"rope_theta":10000.0,
              "global_attention_layers":[3],"effective_layers":5}
            (d/"config.json").write_text(json.dumps(config))
            budget={"expected_total_parameters":0,"expected_shared_parameters":0,"expected_expert_parameters":0}
            (d/"canonical_parameter_budget.json").write_text(json.dumps(budget))
            first=count_canonical_parameters(root)
            budget={"expected_total_parameters":first["total_parameters"],
              "expected_shared_parameters":first["shared_parameters"],
              "expected_expert_parameters":first["expert_parameters"]}
            (d/"canonical_parameter_budget.json").write_text(json.dumps(budget))
            result=count_canonical_parameters(root)
        self.assertTrue(result["budget_matches"])
        self.assertEqual(result["unique_parameterized_blocks"],3)
        self.assertEqual(result["effective_depth"],5)
        self.assertTrue(result["tied_embedding_and_lm_head"])
        self.assertFalse(result["weight_storage_allocated"])
        self.assertEqual(result["total_parameters"],result["shared_parameters"]+result["expert_parameters"])

if __name__=="__main__": unittest.main()
