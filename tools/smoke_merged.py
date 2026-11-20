#!/usr/bin/env python3
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self,V,P,d,L,H,K):
        super().__init__(); self.d,self.H,self.K=d,H,K
        self.tok=nn.Embedding(V,d); self.pos=nn.Embedding(256,d); self.pat=nn.Embedding(max(P,1),d); self.blk=nn.ModuleList()
        for _ in range(L):
            self.blk.append(nn.ModuleDict({"n1":nn.LayerNorm(d),"q":nn.Linear(d,d,bias=False),"k":nn.Linear(d,d//H,bias=False),"v":nn.Linear(d,d//H,bias=False),"out":nn.Linear(d,d,bias=False),"n2":nn.LayerNorm(d),"ff1":nn.Linear(d,4*d),"ff2":nn.Linear(4*d,d),"n3":nn.LayerNorm(d),"lq":nn.Linear(d,d,bias=False),"lo":nn.Linear(d,d,bias=False)}))
        self.nf=nn.LayerNorm(d)
    def forward(self,x,nb,pid):
        B,T=x.shape; h=self.tok(x)+self.pos(torch.arange(T,device=x.device))[None]+self.pat(pid)[:,None,:]
        for b in self.blk:
            xx=b["n1"](h); q=b["q"](xx).view(B,T,self.H,-1).transpose(1,2); k=b["k"](xx).view(B,T,1,-1).transpose(1,2); v=b["v"](xx).view(B,T,1,-1).transpose(1,2)
            att=F.scaled_dot_product_attention(q,k,v,is_causal=True); h=h+0.5*b["out"](att.transpose(1,2).contiguous().view(B,T,self.d)); h=h+0.5*b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem=self.tok(nb); q2=b["lq"](b["n3"](h)).view(B,T,self.H,-1); k2=mem.view(B,T,self.K,self.H,-1).permute(0,1,3,2,4); sc=(q2.unsqueeze(3)@k2.transpose(-1,-2))/math.sqrt(self.d//self.H); h=h+0.5*b["lo"]((sc.softmax(-1)@k2).squeeze(3).reshape(B,T,self.d))
        return self.nf(h)
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",default="out/smoke_report.json"); a=ap.parse_args()
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
assert len(ck.get("merge",{}).get("domains",[]))==10
K=int(ck.get("config",{}).get("nbr",8)); m=M(int(ck["V"]),len(ck["pats"]),int(ck["dim"]),int(ck["layers"]),int(ck["heads"]),K)
m.load_state_dict(ck["m"],strict=True); m.eval()
x=torch.zeros((2,8),dtype=torch.long); nb=torch.zeros((2,8,K),dtype=torch.long); pid=torch.zeros(2,dtype=torch.long)
with torch.no_grad(): y=m(x,nb,pid)
assert list(y.shape)==[2,8,int(ck["dim"])] and torch.isfinite(y).all()
r={"status":"forward_pass_passed","domains":ck["merge"]["domains"],"parameters":sum(p.numel() for p in m.parameters()),"vocab":int(ck["V"]),"patterns":len(ck["pats"]),"output_shape":list(y.shape),"finite":True,"accuracy_claim":False}
os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True); json.dump(r,open(a.out,"w"),indent=2); print(json.dumps(r,indent=2))
