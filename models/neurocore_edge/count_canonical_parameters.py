#!/usr/bin/env python3
"""Count canonical NeuroCore parameters using meta tensors (no weight storage)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM

FIELDS = ("vocab_size","hidden_size","num_experts","experts_per_token",
"expert_intermediate_size","attention_heads","kv_heads","sliding_window",
"recurrent_state_size","router_aux_loss_coefficient","prelude_layers",
"core_unique_layers","core_repeats","coda_layers","meta_token_count","rope_theta")

def count_canonical_parameters(root: Path):
    d = root / "models/neurocore_edge"
    raw = json.loads((d/"config.json").read_text())
    budget = json.loads((d/"canonical_parameter_budget.json").read_text())
    kwargs = {k: raw[k] for k in FIELDS if k in raw}
    kwargs["num_hidden_layers"] = int(raw.get("effective_layers",24))
    kwargs["global_attention_layers"] = tuple(raw.get("global_attention_layers",(7,15,23)))
    cfg = NeuroCoreEdgeConfig(**kwargs)
    with torch.device("meta"):
        model = NeuroCoreEdgeForCausalLM(cfg)
    named = list(model.named_parameters())
    total = sum(p.numel() for _,p in named)
    expert = sum(p.numel() for n,p in named if ".experts." in n or n.startswith("experts."))
    shared = total-expert
    return {
      "total_parameters":int(total),"shared_parameters":int(shared),
      "expert_parameters":int(expert),
      "unique_parameterized_blocks":len(model.prelude_layers)+len(model.core_layers)+len(model.coda_layers),
      "effective_depth":len(model._execution_plan()),
      "expert_intermediate_size":cfg.expert_intermediate_size,
      "tied_embedding_and_lm_head":model.embed.weight is model.lm_head.weight,
      "weight_storage_allocated":False,
      "budget_matches":total==int(budget["expected_total_parameters"]) and shared==int(budget["expected_shared_parameters"]) and expert==int(budget["expected_expert_parameters"]),
      "expected":{"total_parameters":int(budget["expected_total_parameters"]),"shared_parameters":int(budget["expected_shared_parameters"]),"expert_parameters":int(budget["expected_expert_parameters"])}
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args()
    report=count_canonical_parameters(a.repo_root.resolve())
    print(json.dumps(report,indent=2))
    return 0 if report["budget_matches"] else 2
if __name__=="__main__":
    raise SystemExit(main())
