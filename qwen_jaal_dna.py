"""qwen_jaal_dna.py -- ASLI Qwen 0.5B: DNA reference shabdakosh + int4 baaki."""
import json, math, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT=open("sample.txt").read()
HELD="""A machine that learns from data can be asked two questions. The first is how well it predicts.
The second is why it predicts that way. The second question is harder, and it matters more when we
decide whether to rely on the machine. Numbers help us answer both, but only if we are honest about
the error in every measurement we make. Careful people keep asking this question, and that habit is
what moves knowledge forward."""

def main():
    print("=== LOAD ===",flush=True)
    m,cfg,miss,unexp=load_qwen("model.safetensors")
    n=sum(p.numel() for p in m.parameters())
    print(f"  param {n:,} | miss {len(miss)} | fp16 {n*2/1e6:.0f} MB",flush=True)
    ids=tokenize(TEXT); hids=tokenize(HELD)
    p0=perplexity(m,ids); t0=top1(m,ids); h0=perplexity(m,hids)
    print(f"\nBASELINE: ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}",flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)

    # ---------- DNA REFERENCE VOCAB ----------
    print(f"\n=== 1) DNA REFERENCE SHABDAKOSH ===",flush=True)
    E=m.emb.weight.data.numpy().copy()
    Vv,Dd=E.shape
    fl=E.reshape(-1); nn3=len(fl)//3*3
    A=fl[:nn3].reshape(-1,3)
    S=A[np.random.choice(len(A),30000,replace=False)]
    C=S[np.random.permutation(len(S))[:64]].copy()
    for _ in range(10):
        d=(S*S).sum(1)[:,None]-2*S@C.T+(C*C).sum(1)[None]
        lab=d.argmin(1)
        for j in range(64):
            mk=lab==j
            if mk.any(): C[j]=S[mk].mean(0)
    Cn=(C*C).sum(1)[None]
    ids_all=np.empty(len(A),np.uint16)
    for s0 in range(0,len(A),300000):
        s1=min(s0+300000,len(A)); Q=A[s0:s1]
        d=(Q*Q).sum(1)[:,None]-2*Q@C.T+Cn
        ids_all[s0:s1]=d.argmin(1).astype(np.uint16)
    packed=len(ids_all)*6//8
    print(f"  codon {len(ids_all):,} | packed {packed/1e6:.1f} MB",flush=True)
    print(f"  float32 se {Vv*Dd*4/1e6:.0f} MB -> {packed/1e6:.1f} MB = {Vv*Dd*4/packed:.1f}x",flush=True)

    # decode wapas (quality naapne ke liye)
    R=np.empty_like(A)
    for s0 in range(0,len(A),300000):
        s1=min(s0+300000,len(A))
        R[s0:s1]=C[ids_all[s0:s1]]
    newE=np.concatenate([R.reshape(-1),fl[nn3:]]).reshape(Vv,Dd)
    cs=float((E*newE).sum()/(np.linalg.norm(E)*np.linalg.norm(newE)+1e-9))
    m.emb.weight.data=torch.tensor(newE)
    p1=perplexity(m,ids); t1=top1(m,ids); h1=perplexity(m,hids)
    print(f"  cosine {cs:.4f} | ppl {p1:.3f} | top-1 {t1*100:.1f}% ({t1*100-t0*100:+.1f}) | held {h1:.3f}",flush=True)

    # ---------- int4 BAAKI ----------
    print(f"\n=== 2) BAAKI int4 ===",flush=True)
    with torch.no_grad():
        for nm,p in m.named_parameters():
            if p.dim()<2 or "emb" in nm: continue
            hi=7; sc=p.data.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
            p.data.copy_(torch.clamp(torch.round(p.data/sc),-hi,hi)*sc)
    p2=perplexity(m,ids); t2=top1(m,ids); h2=perplexity(m,hids)
    print(f"  ppl {p2:.3f} | top-1 {t2*100:.1f}% ({t2*100-t0*100:+.1f}) | held {h2:.3f}",flush=True)

    # ---------- 3) UMMIDVAR-SEEMA (jaal merge) ----------
    print(f"\n=== 3) JAAL MERGE: ummidvar-seema ===",flush=True)
    # har position par sirf top-K shabd (embedding ke sabse paas)
    En=newE/(np.linalg.norm(newE,axis=1,keepdims=True)+1e-9)
    print(f"  (output layer par lattice candidates -- test ke liye skip)",flush=True)

    print(f"\n=== 4) ASLI AAKAR ===",flush=True)
    vocab_mb=packed/1e6
    rest=n-Vv*Dd
    rest_mb=rest*0.5/1e6
    tot=vocab_mb+rest_mb
    print(f"  shabdakosh DNA reference: {vocab_mb:.1f} MB",flush=True)
    print(f"  baaki int4              : {rest_mb:.1f} MB",flush=True)
    print(f"  KUL                     : {tot:.1f} MB | fp16 se {n*2/1e6/tot:.1f}x",flush=True)

    print(f"\n=== NATIJA ===",flush=True)
    print(f"  baseline : ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}",flush=True)
    print(f"  +DNA ref : ppl {p1:.3f} | top-1 {t1*100:.1f}% | held {h1:.3f}",flush=True)
    print(f"  +int4    : ppl {p2:.3f} | top-1 {t2*100:.1f}% | held {h2:.3f}",flush=True)
    print(f"  aakar    : {n*2/1e6:.0f} MB -> {tot:.1f} MB ({n*2/1e6/tot:.1f}x)",flush=True)
    print(f"  sample   : {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)
    json.dump({"base":{"ppl":p0,"top1":t0,"held":h0},"dna":{"ppl":p1,"top1":t1,"held":h1},
               "int4":{"ppl":p2,"top1":t2,"held":h2},"size_mb":tot,"vocab_mb":vocab_mb},open("result.json","w"),indent=2)

main()
