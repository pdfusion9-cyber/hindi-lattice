"""qwen_int4_all.py -- SAB KUCH int4 (shabdakosh bhi). Asli Qwen par."""
import json, math, os, struct
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT=open("sample.txt").read()
HELD="""A machine that learns from data can be asked two questions. The first is how well it predicts.
The second is why it predicts that way. The second question is harder, and it matters more when we
decide whether to rely on the machine. Numbers help us answer both, but only if we are honest about
the error in every measurement we make. Careful people keep asking this question, and that habit is
what moves knowledge forward."""

def q4(t):
    hi=7; sc=t.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
    return torch.clamp(torch.round(t/sc),-hi,hi)*sc

def main():
    print("=== LOAD ===",flush=True)
    m,cfg,miss,unexp=load_qwen("model.safetensors")
    n=sum(p.numel() for p in m.parameters())
    ids=tokenize(TEXT); hids=tokenize(HELD)
    p0=perplexity(m,ids); t0=top1(m,ids); h0=perplexity(m,hids)
    print(f"  param {n:,} | fp16 {n*2/1e6:.0f} MB",flush=True)
    print(f"BASELINE: ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}",flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)

    # ---- sab int4 ----
    print(f"\n=== SAB KUCH int4 ===",flush=True)
    with torch.no_grad():
        for nm,p in m.named_parameters():
            if p.dim()<2: continue
            p.data.copy_(q4(p.data))
    p1=perplexity(m,ids); t1=top1(m,ids); h1=perplexity(m,hids)
    print(f"  ppl {p1:.3f} | top-1 {t1*100:.1f}% ({t1*100-t0*100:+.1f}) | held {h1:.3f} ({h1-h0:+.3f})",flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)

    # ---- aakar ----
    tot=n*0.5/1e6
    print(f"\n=== AAKAR ===",flush=True)
    print(f"  int4: {tot:.1f} MB | fp16 se {n*2/1e6/tot:.1f}x",flush=True)
    print(f"  +30% runtime = {tot/0.7:.1f} MB",flush=True)

    # ---- int8 shabdakosh bhi dekho (safest) ----
    print(f"\n=== TULNA: shabdakosh int8 + baaki int4 ===",flush=True)
    m2,_,_,_=load_qwen("model.safetensors")
    with torch.no_grad():
        for nm,p in m2.named_parameters():
            if p.dim()<2: continue
            if "emb" in nm:
                hi=127; sc=p.data.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
                p.data.copy_(torch.clamp(torch.round(p.data/sc),-hi,hi)*sc)
            else:
                p.data.copy_(q4(p.data))
    p2=perplexity(m2,ids); t2=top1(m2,ids); h2=perplexity(m2,hids)
    Vv,Dd=m2.emb.weight.shape
    tot2=(Vv*Dd*1.0 + (n-Vv*Dd)*0.5)/1e6
    print(f"  ppl {p2:.3f} | top-1 {t2*100:.1f}% ({t2*100-t0*100:+.1f}) | held {h2:.3f}",flush=True)
    print(f"  aakar {tot2:.1f} MB | fp16 se {n*2/1e6/tot2:.1f}x",flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m2(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)

    print(f"\n=== NATIJA ===",flush=True)
    print(f"  baseline            : ppl {p0:.3f} | top-1 {t0*100:.1f}% | {n*2/1e6:.0f} MB",flush=True)
    print(f"  sab int4            : ppl {p1:.3f} | top-1 {t1*100:.1f}% | {tot:.1f} MB ({n*2/1e6/tot:.1f}x)",flush=True)
    print(f"  vocab int8+int4     : ppl {p2:.3f} | top-1 {t2*100:.1f}% | {tot2:.1f} MB ({n*2/1e6/tot2:.1f}x)",flush=True)
    json.dump({"base":{"ppl":p0,"top1":t0,"held":h0},
               "int4":{"ppl":p1,"top1":t1,"held":h1,"mb":tot},
               "v8":{"ppl":p2,"top1":t2,"held":h2,"mb":tot2}},open("result.json","w"),indent=2)

main()
