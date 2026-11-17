"""quality_up.py -- GUNVATTA BADHAO + ON-THE-FLY GENERATION.

  A) int4 plain                    (baseline)
  B) int4 + OUTLIERS fp16 me       (quality up, thoda bada)
  C) low-rank r + int4 residual    (on-the-fly generation)
"""
import json, math, os, struct
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT=open("sample.txt").read()
HELD="""A machine that learns from data can be asked two questions. The first is how well it predicts.
The second is why it predicts that way. The second question is harder, and it matters more when we
decide whether to rely on the machine. Numbers help us answer both, but only if we are honest about
the error in every measurement we make. Careful people keep asking this question."""

def q4_plain(t):
    hi=7; sc=t.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
    return torch.clamp(torch.round(t/sc),-hi,hi)*sc

def q4_outlier(t, frac):
    """int4 + top `frac` outliers fp16 me"""
    hi=7; sc=t.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
    q=torch.clamp(torch.round(t/sc),-hi,hi)*sc
    if frac<=0: return q, 0
    flat=t.abs().reshape(-1)
    k=max(1,int(len(flat)*frac))
    thr=torch.topk(flat,k).values[-1]
    msk=t.abs()>=thr
    q[msk]=t[msk]
    return q, int(msk.sum())

def lowrank_resid(t, r):
    """W ~= U@V + int4 residual -> on-the-fly rebuild"""
    o,i=t.shape
    rr=min(r,min(o,i))
    U,S,Vt=torch.linalg.svd(t,full_matrices=False)
    A=U[:,:rr]*S[:rr].sqrt(); B=(Vt[:rr].T)*S[:rr].sqrt()
    R=t-A@B
    Rq=q4_plain(R)
    return A,B,Rq

def main():
    m,cfg,miss,_=load_qwen("model.safetensors")
    n=sum(p.numel() for p in m.parameters())
    ids=tokenize(TEXT); hids=tokenize(HELD)
    p0=perplexity(m,ids); t0=top1(m,ids); h0=perplexity(m,hids)
    print(f"=== BASELINE ===",flush=True)
    print(f"  ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f} | {n*2/1e6:.0f} MB",flush=True)
    SD={k:v.clone() for k,v in m.state_dict().items()}

    def chalao(nm, fn, extra_bytes):
        mk,_,_,_=load_qwen("model.safetensors")
        mk.load_state_dict(SD)
        with torch.no_grad():
            for pn,p in mk.named_parameters():
                if p.dim()<2: continue
                fn(pn,p)
        p1=perplexity(mk,ids); t1=top1(mk,ids); h1=perplexity(mk,hids)
        mb=n*0.5/1e6 + extra_bytes/1e6
        print(f"{nm:<30} {mb:>7.1f} {p1:>7.3f} {t1*100:>6.1f}% {t1*100-t0*100:>+7.1f} {h1:>7.3f}",flush=True)
        del mk
        return mb,p1,t1*100

    print(f"\n{'setup':<30} {'MB':>7} {'ppl':>7} {'top-1':>7} {'girawat':>7} {'held':>7}",flush=True)
    print("-"*72,flush=True)

    # A) int4 plain
    chalao("A) int4 plain", lambda pn,p: p.data.copy_(q4_plain(p.data)), 0)

    # B) outliers
    for frac in (0.001, 0.005, 0.02):
        cnt=[0]
        def f(pn,p,frac=frac):
            q,k=q4_outlier(p.data,frac); p.data.copy_(q); cnt[0]+=k
        # pehle count karo
        cnt[0]=0
        tmp,_,_,_=load_qwen("model.safetensors")
        with torch.no_grad():
            for pn,p in tmp.named_parameters():
                if p.dim()<2: continue
                _,k=q4_outlier(p.data,frac); cnt[0]+=k
        del tmp
        extra=cnt[0]*1.5   # fp16 (2B) - int4 (0.5B) = 1.5B extra
        chalao(f"B) int4 + {frac*100:.1f}% outlier fp16", f, extra)

    # C) low-rank + int4 residual
    for r in (32, 64):
        tot=[0]
        def f(pn,p,r=r):
            A,B,Rq=lowrank_resid(p.data,r)
            p.data.copy_(A@B+Rq)
            tot[0]+=A.numel()+B.numel()
        tmp,_,_,_=load_qwen("model.safetensors")
        tot[0]=0
        with torch.no_grad():
            for pn,p in tmp.named_parameters():
                if p.dim()<2: continue
                A,B,_=lowrank_resid(p.data,r); tot[0]+=A.numel()+B.numel()
        del tmp
        # aakar: residual int4 + factors fp16
        extra=tot[0]*2
        chalao(f"C) low-rank r={r} + int4 resid", f, extra)

    print(f"\n(extra = outlier/factor ke liye int4 se zyada bytes)",flush=True)

main()
