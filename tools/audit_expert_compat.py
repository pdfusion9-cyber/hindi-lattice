#!/usr/bin/env python3
"""Read-only compatibility, strict-loader and graph-backed inference smoke audit."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

DOMAINS = ("science","math","reasoning","news","qa","general","coding","document","history","geography")

class ExpertModel(nn.Module):
    """Architecture mirrored from punah.py; inference only."""
    def __init__(self, V: int, P: int, d: int, L: int, H: int, K: int, res_scale: float):
        super().__init__()
        self.d, self.H, self.K, self.res_scale = d, H, K, res_scale
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(256, d)
        self.pat = nn.Embedding(max(P, 1), d)
        self.blk = nn.ModuleList()
        for _ in range(L):
            self.blk.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "q": nn.Linear(d, d, bias=False),
                "k": nn.Linear(d, d // H, bias=False), "v": nn.Linear(d, d // H, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d),
                "n3": nn.LayerNorm(d), "lq": nn.Linear(d, d, bias=False),
                "lo": nn.Linear(d, d, bias=False)}))
        self.nf = nn.LayerNorm(d)

    def forward(self, x, nb, pid):
        B,T=x.shape
        h=self.tok(x)+self.pos(torch.arange(T,device=x.device))[None]+self.pat(pid)[:,None,:]
        for b in self.blk:
            xx=b["n1"](h)
            q=b["q"](xx).view(B,T,self.H,-1).transpose(1,2)
            k=b["k"](xx).view(B,T,1,-1).transpose(1,2)
            v=b["v"](xx).view(B,T,1,-1).transpose(1,2)
            att=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            h=h+self.res_scale*b["out"](att.transpose(1,2).contiguous().view(B,T,self.d))
            h=h+self.res_scale*b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem=self.tok(nb)
            q2=b["lq"](b["n3"](h)).view(B,T,self.H,-1)
            k2=mem.view(B,T,self.K,self.H,-1).permute(0,1,3,2,4)
            sc=(q2.unsqueeze(3)@k2.transpose(-1,-2))/math.sqrt(self.d//self.H)
            h=h+self.res_scale*b["lo"]((sc.softmax(-1)@k2).squeeze(3).reshape(B,T,self.d))
        return self.nf(h)

def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()

def load_package(path: Path):
    # Artifacts originate from the repository's own verified export workflow.
    # weights_only=True avoids unrestricted pickle object construction.
    return torch.load(path,map_location="cpu",weights_only=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="experts")
    ap.add_argument("--out",default="expert_compatibility_audit.json")
    args=ap.parse_args()
    root=Path(args.input_dir)
    reports=[]; packages={}
    for domain in DOMAINS:
        path=root/f"{domain}.expert.pt"
        if not path.is_file(): raise SystemExit(f"MISSING_PACKAGE:{domain}:{path}")
        ck=load_package(path)
        if ck.get("format")!="chakra-independent-expert-v1" or ck.get("domain")!=domain:
            raise SystemExit(f"PACKAGE_IDENTITY_FAILURE:{domain}")
        required=("m","V","dim","layers","heads","vocab","pats","graph","config")
        missing=[k for k in required if k not in ck]
        if missing: raise SystemExit(f"MISSING_FIELDS:{domain}:{missing}")
        m=ck["m"]; vocab=ck["vocab"]; pats=ck["pats"]; graph=ck["graph"] or {}; cfg=ck["config"] or {}
        if len(vocab)!=int(ck["V"]): raise SystemExit(f"VOCAB_LENGTH_FAILURE:{domain}")
        if "tok.weight" not in m or "pat.weight" not in m: raise SystemExit(f"EMBEDDING_MISSING:{domain}")
        if tuple(m["tok.weight"].shape)!=(len(vocab),int(ck["dim"])):
            raise SystemExit(f"TOKEN_EMBED_SHAPE_FAILURE:{domain}")
        if tuple(m["pat.weight"].shape)!=(max(len(pats),1),int(ck["dim"])):
            raise SystemExit(f"PATTERN_EMBED_SHAPE_FAILURE:{domain}")
        if int(ck["dim"])%int(ck["heads"]): raise SystemExit(f"HEAD_DIM_FAILURE:{domain}")
        if not isinstance(graph,dict): raise SystemExit(f"GRAPH_FORMAT_FAILURE:{domain}")
        V=len(vocab)
        def check_map(name, mapping):
            if not isinstance(mapping,dict): raise SystemExit(f"GRAPH_MAP_FORMAT:{domain}:{name}")
            for k, vals in mapping.items():
                ki=int(k)
                if not 0<=ki<V: raise SystemExit(f"GRAPH_KEY_OOB:{domain}:{name}:{ki}")
                if not isinstance(vals,(list,tuple)): raise SystemExit(f"GRAPH_VALUES_FORMAT:{domain}:{name}:{ki}")
                for x in vals:
                    if not 0<=int(x)<V: raise SystemExit(f"GRAPH_ID_OOB:{domain}:{name}:{ki}:{x}")
        check_map("nbrs",graph.get("nbrs",{})); check_map("cands",graph.get("cands",{}))
        slots=graph.get("slot_cand",[])
        if not isinstance(slots,list): raise SystemExit(f"SLOT_CAND_FORMAT:{domain}")
        for row in slots:
            if not isinstance(row,dict) or not isinstance(row.get("ids"),list):
                raise SystemExit(f"SLOT_ROW_FORMAT:{domain}")
            for x in row["ids"]:
                if not 0<=int(x)<V: raise SystemExit(f"SLOT_ID_OOB:{domain}:{x}")
        arch=(int(ck["V"]),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),int(cfg.get("nbr",0)))
        vocab_hash=sha(vocab); pattern_hash=sha(pats)
        model=ExpertModel(int(ck["V"]),len(pats),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),int(cfg["nbr"]),float(cfg.get("res_scale",0.5)))
        incompat=model.load_state_dict(m,strict=True)
        if incompat.missing_keys or incompat.unexpected_keys: raise SystemExit(f"STRICT_LOAD_KEYS:{domain}")
        model.eval()
        # Real graph IDs are used: no fabricated neighbour IDs or graph-free dummy path.
        nbrs=graph.get("nbrs",{})
        cands=graph.get("cands",{})
        seed=next((int(k) for k,v in nbrs.items() if v),None)
        if seed is None: raise SystemExit(f"NO_USABLE_GRAPH_NEIGHBOURS:{domain}")
        nb_ids=[int(x) for x in nbrs[str(seed)] if 0<=int(x)<V] if str(seed) in nbrs else [int(x) for x in nbrs[seed]]
        if not nb_ids: raise SystemExit(f"EMPTY_GRAPH_NEIGHBOURS:{domain}")
        K=int(cfg["nbr"])
        nb_ids=(nb_ids[:K]+[seed]*K)[:K]
        xids=[seed]
        # Include a second valid token when available; use that expert's own vocab IDs.
        xids=(xids+[nb_ids[0]])[:2]
        x=torch.tensor([xids],dtype=torch.long)
        nb=torch.tensor([[nb_ids[:K] for _ in xids]],dtype=torch.long)
        pid=torch.zeros((1,),dtype=torch.long)
        with torch.inference_mode():
            h=model(x,nb,pid)
            candidate_ids=cands.get(str(seed),cands.get(seed,[]))
            candidate_ids=[int(z) for z in candidate_ids if 0<=int(z)<V][:32]
            if not candidate_ids: candidate_ids=nb_ids[:min(8,len(nb_ids))]
            cand=torch.tensor(candidate_ids,dtype=torch.long)
            scores=(h[:,-1,:] @ model.tok(cand).T).flatten()
        if not torch.isfinite(h).all() or not torch.isfinite(scores).all():
            raise SystemExit(f"NONFINITE_INFERENCE:{domain}")
        packages[domain]={"vocab":vocab,"pats":pats,"arch":arch}
        reports.append({
            "domain":domain,"status":"PASS","parameter_count":sum(t.numel() for t in m.values()),
            "tensor_count":len(m),"tensor_dtypes":sorted({str(t.dtype) for t in m.values()}),
            "dim":int(ck["dim"]),"layers":int(ck["layers"]),"heads":int(ck["heads"]),
            "vocab_size":V,"vocab_sha256":vocab_hash,"pattern_count":len(pats),
            "pattern_sha256":pattern_hash,"token_embedding_shape":list(m["tok.weight"].shape),
            "pattern_embedding_shape":list(m["pat.weight"].shape),
            "graph_fields":sorted(graph.keys()),"graph_nbr_keys":len(nbrs),
            "graph_candidate_keys":len(cands),"graph_slot_rows":len(slots),
            "strict_state_dict_load":"PASS","real_graph_forward":"PASS",
            "forward_shape":list(h.shape),"candidate_score_count":int(scores.numel()),
            "config":cfg
        })
    # Cross-expert ID compatibility is a strict equality test, not just equal sizes.
    base=packages[DOMAINS[0]]
    for domain in DOMAINS[1:]:
        cur=packages[domain]
        reports[DOMAINS.index(domain)]["vocab_exact_match_first_expert"]=(cur["vocab"]==base["vocab"])
        reports[DOMAINS.index(domain)]["patterns_exact_match_first_expert"]=(cur["pats"]==base["pats"])
    all_vocab_equal=all(packages[d]["vocab"]==base["vocab"] for d in DOMAINS)
    all_patterns_equal=all(packages[d]["pats"]==base["pats"] for d in DOMAINS)
    all_arch_equal=all(packages[d]["arch"]==base["arch"] for d in DOMAINS)
    result={
        "status":"PASS" if len(reports)==10 else "FAIL",
        "experts":reports,
        "cross_expert":{
            "all_vocabularies_identical":all_vocab_equal,
            "all_pattern_tables_identical":all_patterns_equal,
            "all_architecture_tuples_identical":all_arch_equal,
            "safe_to_share_token_ids":all_vocab_equal,
            "safe_to_share_pattern_ids":all_patterns_equal
        },
        "interpretation":"Each expert was strictly instantiated and loaded independently. The forward smoke test used its own checkpoint graph and token IDs. This is not a held-out accuracy benchmark and does not establish free-form text generation."
    }
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
