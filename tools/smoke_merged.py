#!/usr/bin/env python3
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

class M(nn.Module):
    # Mirrors sampurna2.py exactly; checkpoint keys are strict-loaded.
    def __init__(self,V,P,d,L,H,K,use):
        super().__init__(); self.d,self.H,self.K,self.use=d,H,K,use
        self.tok=nn.Embedding(V,d); self.pos=nn.Embedding(256,d)
        self.pat=nn.Embedding(max(P,1),d); self.blk=nn.ModuleList()
        for _ in range(L):
            b=nn.ModuleDict({"n1":nn.LayerNorm(d),"qkv":nn.Linear(d,3*d,bias=False),"out":nn.Linear(d,d,bias=False),"n2":nn.LayerNorm(d),"ff1":nn.Linear(d,4*d),"ff2":nn.Linear(4*d,d)})
            if use:
                b["n3"]=nn.LayerNorm(d); b["lq"]=nn.Linear(d,d,bias=False); b["lo"]=nn.Linear(d,d,bias=False)
            self.blk.append(b)
        self.nf=nn.LayerNorm(d)
    def forward(self,x,nb,pid):
        B,T=x.shape
        h=self.tok(x)+self.pos(torch.arange(T,device=x.device))[None]+self.pat(pid)[:,None,:]
        for b in self.blk:
            q,k,v=b["qkv"](b["n1"](h)).chunk(3,-1)
            q=q.view(B,T,self.H,-1).transpose(1,2); k=k.view(B,T,self.H,-1).transpose(1,2); v=v.view(B,T,self.H,-1).transpose(1,2)
            att=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            h=h+b["out"](att.transpose(1,2).contiguous().view(B,T,self.d))
            h=h+b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            if self.use:
                mem=self.tok(nb); q2=b["lq"](b["n3"](h)).view(B,T,self.H,-1)
                k2=mem.view(B,T,self.K,self.H,-1).permute(0,1,3,2,4)
                sc=(q2.unsqueeze(3)@k2.transpose(-1,-2))/math.sqrt(self.d//self.H)
                h=h+b["lo"]((sc.softmax(-1)@k2).squeeze(3).reshape(B,T,self.d))
        return self.nf(h)

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",default="out/smoke_report.json"); a=ap.parse_args()
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
domains=ck.get("merge",{}).get("domains",[])
assert len(domains)==10, "merged checkpoint must contain all ten domains"
vocab=ck["vocab"]; V=int(ck["V"])
assert len(vocab)==V and V>1
K=8; use=int(ck.get("use",1))
m=M(V,len(ck["pats"]),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),K,use)
m.load_state_dict(ck["m"],strict=True); m.eval()
# Checkpoint stores no graph in this training pipeline; use valid token IDs
# and zero neighbour IDs to test the exact forward path.
ctx=[min(i+1,V-1) for i in range(8)]
x=torch.tensor([ctx],dtype=torch.long); nb=torch.zeros((1,8,K),dtype=torch.long); pid=torch.zeros(1,dtype=torch.long)
with torch.no_grad():
    h=m(x,nb,pid)[:,-1,:]
    # Rank all vocabulary rows in chunks to avoid allocating a V-wide matrix.
    top_scores=torch.full((1,5),float("-inf")); top_ids=torch.zeros((1,5),dtype=torch.long)
    for start in range(0,V,8192):
        ids=torch.arange(start,min(start+8192,V),dtype=torch.long)
        scores=h@m.tok(ids).weight.T
        vals,ix=scores.topk(min(5,len(ids)),-1); ix=ix+start
        merged_scores=torch.cat([top_scores,vals],-1); merged_ids=torch.cat([top_ids,ix],-1)
        top_scores,order=merged_scores.topk(5,-1); top_ids=merged_ids.gather(-1,order)
    predictions=[{"token":vocab[int(i)],"score":float(s)} for s,i in zip(top_scores[0],top_ids[0])]
assert torch.isfinite(h).all() and all(math.isfinite(p["score"]) for p in predictions)
report={"status":"strict_load_forward_pass_and_next_token_ranking_passed","source_run_id":ck["merge"].get("source_run_id"),"context_source":"valid vocabulary token IDs; checkpoint contains no graph metadata","domains":domains,"parameters":sum(p.numel() for p in m.parameters()),"vocab":len(vocab),"patterns":len(ck["pats"]),"context":[vocab[i] for i in ctx],"top_predictions":predictions,"finite":True,"accuracy_claim":False,"note":"Architecture copied from sampurna2.py and strict state_dict loading passed; ranking smoke test is not held-out accuracy."}
os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
with open(a.out,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
