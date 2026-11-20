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
m=M(int(ck["V"]),len(ck["pats"]),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),K,RS)
m.load_state_dict(ck["m"],strict=True); m.eval()
vocab=ck["vocab"]; graph=ck.get("graph") or {}; nbrs=graph.get("nbrs",{}); cands=graph.get("cands",{})
# Build a real graph-backed context, then rank next-token candidates.
start=next((int(k) for k,v in cands.items() if v),None)
assert start is not None, "merged graph has no candidate edges"
path=[start]
while len(path)<8:
    nxt=cands.get(path[-1],[])
    if not nxt: break
    path.append(int(nxt[0]))
if len(path)<2: raise RuntimeError("could not construct a graph-backed inference context")
ctx=path[-8:]; ctx=([ctx[0]]*(8-len(ctx)))+ctx
x=torch.tensor([ctx],dtype=torch.long); nb=torch.zeros((1,8,K),dtype=torch.long)
for t,w in enumerate(ctx):
    ns=nbrs.get(w,[])[:K]
    if ns: nb[0,t,:len(ns)]=torch.tensor(ns,dtype=torch.long)
pid=torch.zeros(1,dtype=torch.long)
with torch.no_grad():
    h=m(x,nb,pid)[:,-1,:]
    pool=list(dict.fromkeys(int(z) for z in cands.get(ctx[-1],[]) if 0<=int(z)<len(vocab)))
    if not pool: pool=list(range(min(32,len(vocab))))
    ids=torch.tensor(pool,dtype=torch.long)
    scores=(h[:,None,:]*m.tok(ids)[None,:,:]).sum(-1)
    probs=scores.softmax(-1)
    vals,idx=probs.topk(min(5,len(pool)),-1)
    predictions=[{"token":vocab[pool[int(i)]],"score":float(s)} for s,i in zip(vals[0],idx[0])]
assert torch.isfinite(h).all() and all(math.isfinite(p["score"]) for p in predictions)
report={"status":"forward_pass_and_next_token_ranking_passed","domains":domains,"parameters":sum(p.numel() for p in m.parameters()),"vocab":len(vocab),"patterns":len(ck["pats"]),"context":[vocab[i] for i in ctx],"candidate_count":len(pool),"top_predictions":predictions,"finite":True,"accuracy_claim":False,"note":"Graph-backed next-token ranking smoke test; not a held-out accuracy evaluation."}
os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
with open(a.out,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
