#!/usr/bin/env python3
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

class M(nn.Module):
    def __init__(self,V,P,d,L,H,K,RS):
        super().__init__(); self.d,self.H,self.K,self.RS=d,H,K,RS
        self.tok=nn.Embedding(V,d); self.pos=nn.Embedding(256,d); self.pat=nn.Embedding(max(P,1),d); self.blk=nn.ModuleList()
        for _ in range(L):
            self.blk.append(nn.ModuleDict({"n1":nn.LayerNorm(d),"q":nn.Linear(d,d,bias=False),"k":nn.Linear(d,d//H,bias=False),"v":nn.Linear(d,d//H,bias=False),"out":nn.Linear(d,d,bias=False),"n2":nn.LayerNorm(d),"ff1":nn.Linear(d,4*d),"ff2":nn.Linear(4*d,d),"n3":nn.LayerNorm(d),"lq":nn.Linear(d,d,bias=False),"lo":nn.Linear(d,d,bias=False)}))
        self.nf=nn.LayerNorm(d)
    def forward(self,x,nb,pid):
        B,T=x.shape; h=self.tok(x)+self.pos(torch.arange(T,device=x.device))[None]+self.pat(pid)[:,None,:]
        for b in self.blk:
            xx=b["n1"](h); q=b["q"](xx).view(B,T,self.H,-1).transpose(1,2); k=b["k"](xx).view(B,T,1,-1).transpose(1,2); v=b["v"](xx).view(B,T,1,-1).transpose(1,2)
            att=F.scaled_dot_product_attention(q,k,v,is_causal=True); h=h+self.RS*b["out"](att.transpose(1,2).contiguous().view(B,T,self.d)); h=h+self.RS*b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem=self.tok(nb); q2=b["lq"](b["n3"](h)).view(B,T,self.H,-1); k2=mem.view(B,T,self.K,self.H,-1).permute(0,1,3,2,4); sc=(q2.unsqueeze(3)@k2.transpose(-1,-2))/math.sqrt(self.d//self.H); h=h+self.RS*b["lo"]((sc.softmax(-1)@k2).squeeze(3).reshape(B,T,self.d))
        return self.nf(h)

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",default="out/smoke_report.json"); a=ap.parse_args()
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
domains=ck.get("merge",{}).get("domains",[])
assert len(domains)==10, "merged checkpoint must contain all ten domains"
cfg=ck.get("config",{}); K=int(cfg.get("nbr",8)); RS=float(cfg.get("res_scale",0.5))
vocab=ck["vocab"]; V=int(ck["V"])
assert len(vocab)==V and V>1
m=M(V,len(ck["pats"]),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),K,RS)
m.load_state_dict(ck["m"],strict=True); m.eval()
graph=ck.get("graph") or {}; nbrs=graph.get("nbrs",{}) or {}; cands=graph.get("cands",{}) or {}
context_source="graph" if cands else "vocabulary-fallback"
if cands:
    start=next((int(k) for k,v in cands.items() if v),None)
    assert start is not None and 0<=start<V, "invalid graph start token"
    path=[start]
    while len(path)<8:
        nxt=cands.get(path[-1],cands.get(str(path[-1]),[]))
        if not nxt: break
        next_id=int(nxt[0])
        if not 0<=next_id<V: raise ValueError("graph token ID outside vocabulary")
        path.append(next_id)
    if len(path)<2: raise RuntimeError("could not construct graph-backed inference context")
    ctx=path[-8:]; ctx=([ctx[0]]*(8-len(ctx)))+ctx
else:
    # Older CHAKRA checkpoints did not persist graph metadata. Exercise the
    # actual model with valid non-zero vocabulary IDs rather than failing.
    ctx=[min(i+1,V-1) for i in range(8)]
x=torch.tensor([ctx],dtype=torch.long); nb=torch.zeros((1,8,K),dtype=torch.long)
if nbrs:
    for t,w in enumerate(ctx):
        ns=nbrs.get(w,nbrs.get(str(w),[]))[:K]
        if ns:
            clean=[int(z) for z in ns if 0<=int(z)<V]
            if clean: nb[0,t,:len(clean)]=torch.tensor(clean,dtype=torch.long)
pid=torch.zeros(1,dtype=torch.long)
with torch.no_grad():
    h=m(x,nb,pid)[:,-1,:]
    if cands:
        pool=list(dict.fromkeys(int(z) for z in cands.get(ctx[-1],cands.get(str(ctx[-1]),[])) if 0<=int(z)<V))
    else:
        pool=list(range(min(64,V)))
    if not pool: pool=list(range(min(64,V)))
    ids=torch.tensor(pool,dtype=torch.long)
    scores=(h[:,None,:]*m.tok(ids)[None,:,:]).sum(-1)
    probs=scores.softmax(-1)
    vals,idx=probs.topk(min(5,len(pool)),-1)
    predictions=[{"token":vocab[pool[int(i)]],"score":float(s)} for s,i in zip(vals[0],idx[0])]
assert torch.isfinite(h).all() and all(math.isfinite(p["score"]) for p in predictions)
report={"status":"forward_pass_and_next_token_ranking_passed","context_source":context_source,"domains":domains,"parameters":sum(p.numel() for p in m.parameters()),"vocab":len(vocab),"patterns":len(ck["pats"]),"context":[vocab[i] for i in ctx],"candidate_count":len(pool),"top_predictions":predictions,"finite":True,"accuracy_claim":False,"note":"Forward-pass smoke test with candidate ranking; not a held-out accuracy evaluation."}
os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
with open(a.out,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
