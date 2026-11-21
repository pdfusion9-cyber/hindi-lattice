import tempfile
import unittest
from pathlib import Path
import torch
from tools.expert_inference import ExpertModel, ExpertRunner, pattern_id_for, tokenize

class ExpertInferenceTests(unittest.TestCase):
    def make_checkpoint(self, path):
        vocab=["नमस्ते","भारत","विज्ञान","है"]
        pats=[("*","*","*")]
        model=ExpertModel(4,1,8,1,2,2,0.5)
        package={
            "format":"chakra-independent-expert-v1","domain":"science",
            "m":model.state_dict(),"V":4,"dim":8,"layers":1,"heads":2,
            "vocab":vocab,"vocab_counts":[10,10,10,10],"pats":pats,
            "graph":{
                "nbrs":{"0":[1,2],"1":[0,2],"2":[0,1],"3":[1,2]},
                "cands":{"1":[2,3],"2":[3,0]},
                "slot_cand":[{"pattern":["*","*","*"],"position":2,"ids":[3]}]},
            "config":{"nbr":2,"cand":3,"pat_cand":2,"res_scale":0.5,"sparse_win":6}}
        torch.save(package,path)

    def test_tokenizer_word_boundaries_match_training_regex(self):
        self.assertEqual(tokenize("नमस्ते भारत, विज्ञान है।"),["नमस्ते","भारत","विज्ञान","है"])

    def test_strict_load_and_finite_greedy_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"science.expert.pt"; self.make_checkpoint(path)
            runner=ExpertRunner(path)
            result=runner.generate("नमस्ते भारत विज्ञान",max_new_tokens=3)
            self.assertEqual(result["domain"],"science")
            self.assertEqual(len(result["generated_tokens"]),3)
            self.assertTrue(all(d["candidate_count"]>0 for d in result["diagnostics"]))
            self.assertTrue(all(torch.isfinite(torch.tensor(d["score"])) for d in result["diagnostics"]))

    def test_unmatched_pattern_falls_back_to_zero(self):
        pid,exact=pattern_id_for(["नमस्ते","भारत"],["नमस्ते","भारत"],[10,10],{("x",):0})
        self.assertEqual(pid,0); self.assertFalse(exact)

if __name__=="__main__":
    unittest.main()
