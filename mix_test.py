"""mix_test.py -- SHABDKOSH 2-bit codon + BAAKI int4. Asli Qwen par gunvatta."""
import json, math, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT = open("sample.txt").read()
HELD = """A machine that learns from data can be asked two questions. The first is how well it predicts.
The second is why it predicts that way. The second question is harder, and it matters more when we
decide whether to rely on the machine. Numbers help us answer both, but only if we are honest about
the error in every measurement we make. Careful people keep asking this question, and that habit is
what moves knowledge forward. The same habit applies to machines that learn from data."""

def main():
    print("=== LOAD ===", flush=True)
    m,cfg,miss,unexp = load_qwen("model.safetensors")
    n=sum(p.numel() for p in m.parameters())
    print(f"  param {n:,} | miss {len(miss)} | fp16 {n*2/1e6:.0f} MB", flush=True)
    ids=tokenize(TEXT); hids=tokenize(HELD)
    print(f"  eval {ids.shape[1]} tok | held {hids.shape[1]} tok", flush=True)

    p0=perplexity(m,ids); t0=top1(m,ids); h0=perplexity(m,hids)
    print(f"\nBASELINE: ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}", flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}", flush=True)

    # ---------- 1) SHABDKOSH -> 2-bit codon ----------
    print(f"\n=== 1) SHABDKOSH 2-bit codon ===", flush=True)
    E=m.emb.weight.data.clone()
    Vv,Dd=E.shape
    samp=[]
    for i0 in range(0,Vv,20000):
        Ab=E[i0:i0+20000].reshape(-1); nn3=len(Ab)//3*3
        A=Ab[:nn3].reshape(-1,3)
        samp.append(A[torch.randperm(len(A))[:800]])
    S=torch.cat(samp,0)[:25000]
    C=S[torch.randperm(len(S))[:64]].clone()
    for it in range(10):
        d=(S**2).sum(1)[:,None]-2*S@C.T+(C**2).sum(1)[None]
        lab=d.argmin(1)
        for j in range(64):
            mk=lab==j
            if mk.any(): C[j]=S[mk].mean(0)
    Cn=(C*C).sum(1)[None]
    with torch.no_grad():
        for i0 in range(0,Vv,8000):
            Ab=E[i0:i0+8000].reshape(-1); nn3=len(Ab)//3*3
            A=Ab[:nn3].reshape(-1,3)
            out=torch.empty_like(A); B=200000
            for s0 in range(0,len(A),B):
                s1=min(s0+B,len(A)); Q=A[s0:s1]
                d=(Q*Q).sum(1)[:,None]-2*Q@C.T+Cn
                out[s0:s1]=C[d.argmin(1)]
            m.emb.weight.data[i0:i0+8000].copy_(
                torch.cat([out.reshape(-1),Ab[nn3:]]).reshape(-1,Dd))
    p1=perplexity(m,ids); t1=top1(m,ids); h1=perplexity(m,hids)
    print(f"  ppl {p1:.3f} | top-1 {t1*100:.1f}% ({t1*100-t0*100:+.1f}) | held {h1:.3f} ({h1-h0:+.3f})", flush=True)

    # ---------- 2) BAAKI -> int4 ----------
    print(f"\n=== 2) BAAKI (attention+FFN) int4 ===", flush=True)
    with torch.no_grad():
        for nm,p in m.named_parameters():
            if p.dim()<2 or "emb" in nm: continue
            hi=7; sc=p.data.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
            p.data.copy_(torch.clamp(torch.round(p.data/sc),-hi,hi)*sc)
    p2=perplexity(m,ids); t2=top1(m,ids); h2=perplexity(m,hids)
    print(f"  ppl {p2:.3f} | top-1 {t2*100:.1f}% ({t2*100-t0*100:+.1f}) | held {h2:.3f} ({h2-h0:+.3f})", flush=True)

    # ---------- 3) AAKAR ----------
    print(f"\n=== 3) ASLI AAKAR ===", flush=True)
    vocab_mb = Vv*Dd*0.25/1e6
    rest = n - Vv*Dd
    rest_mb = rest*0.5/1e6
    tot = vocab_mb + rest_mb
    print(f"  shabdakosh 2-bit codon : {vocab_mb:.1f} MB", flush=True)
    print(f"  baaki int4             : {rest_mb:.1f} MB", flush=True)
    print(f"  KUL                    : {tot:.1f} MB | fp16 se {n*2/1e6/tot:.1f}x", flush=True)

    print(f"\n=== NATIJA ===", flush=True)
    print(f"  baseline : ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}", flush=True)
    print(f"  +vocab   : ppl {p1:.3f} | top-1 {t1*100:.1f}% | held {h1:.3f}", flush=True)
    print(f"  +int4    : ppl {p2:.3f} | top-1 {t2*100:.1f}% | held {h2:.3f}", flush=True)
    print(f"  aakar    : {n*2/1e6:.0f} MB -> {tot:.1f} MB ({n*2/1e6/tot:.1f}x)", flush=True)
    print(f"  sample   : {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}", flush=True)
    json.dump({"base":{"ppl":p0,"top1":t0,"held":h0},"vocab":{"ppl":p1,"top1":t1,"held":h1},
               "int4":{"ppl":p2,"top1":t2,"held":h2},"size_mb":tot}, open("result.json","w"), indent=2)

main()
