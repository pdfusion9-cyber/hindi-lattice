"""codon_only.py -- BINA PRUNING. Sirf 2-bit codon. Asli Qwen par gunvatta naap."""
import json, math, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)

exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT = open("sample.txt").read()          # ek baar, dohraya nahi
HELD = """A machine that learns from data can be asked two questions. The first is how well it predicts.
The second is why it predicts that way. The second question is harder, and it matters more when we
decide whether to rely on the machine. Numbers help us answer both, but only if we are honest about
the error in every measurement we make."""

def main():
    print("=== 1) LOAD QWEN 0.5B (bina pruning) ===", flush=True)
    m,cfg,miss,unexp = load_qwen("model.safetensors")
    n = sum(p.numel() for p in m.parameters())
    print(f"  param {n:,} | missing {len(miss)} | unexpected {len(unexp)}", flush=True)
    print(f"  fp16 {n*2/1e6:.1f} MB", flush=True)

    ids = tokenize(TEXT); hids = tokenize(HELD)
    print(f"  eval tokens {ids.shape[1]} | held-out {hids.shape[1]}", flush=True)

    p0=perplexity(m,ids); t0=top1(m,ids)
    h0=perplexity(m,hids)
    print(f"\n=== BASELINE ===", flush=True)
    print(f"  eval  : ppl {p0:.3f} | top-1 {t0*100:.1f}%", flush=True)
    print(f"  held  : ppl {h0:.3f}", flush=True)
    OUT0={"ppl":p0,"top1":t0,"held":h0}

    print(f"\n=== 2) 2-BIT CODON (64-codon table) ===", flush=True)
    # halka sample: har parat se 1500 codon
    pool=[]
    with torch.no_grad():
        for b in m.blk[:12]:
            fl=b.mlp.gate.weight.data.reshape(-1); nn3=len(fl)//3*3
            A=fl[:nn3].reshape(-1,3)
            ix=torch.randperm(len(A))[:1500]
            pool.append(A[ix])
    P=torch.cat(pool,0).clone()
    print(f"  sample {P.shape[0]:,} codon", flush=True)
    C=P[torch.randperm(len(P))[:64]].clone()
    for it in range(10):
        d=(P**2).sum(1)[:,None]-2*P@C.T+(C**2).sum(1)[None]
        lab=d.argmin(1)
        for j in range(64):
            msk=lab==j
            if msk.any(): C[j]=P[msk].mean(0)
    del P, d, lab
    errs=[]
    Cn=(C*C).sum(1)[None]
    with torch.no_grad():
        for p in m.parameters():
            if p.dim()<2: continue
            fl=p.data.reshape(-1); nn3=len(fl)//3*3
            A=fl[:nn3].reshape(-1,3)
            newA=torch.empty_like(A)
            B=200000
            e_num=0.0; e_den=0.0
            for s0 in range(0,len(A),B):
                s1=min(s0+B,len(A)); Ab=A[s0:s1]
                d=(Ab*Ab).sum(1)[:,None]-2*Ab@C.T+Cn
                ix=d.argmin(1)
                newA[s0:s1]=C[ix]
                e_num+=float((Ab-C[ix]).abs().sum()); e_den+=float(Ab.abs().sum())
            errs.append(e_num/(e_den+1e-9))
            p.data.copy_(torch.cat([newA.reshape(-1),fl[nn3:]]).reshape(p.shape))
            del A, newA
    p1=perplexity(m,ids); t1=top1(m,ids); h1=perplexity(m,hids)
    print(f"  codon galti {sum(errs)/len(errs)*100:.1f}%", flush=True)
    print(f"  eval  : ppl {p1:.3f} | top-1 {t1*100:.1f}%  ({t1*100-t0*100:+.1f})", flush=True)
    print(f"  held  : ppl {h1:.3f}  ({h1-h0:+.3f})", flush=True)

    print(f"\n=== 3) FINE-TUNE ===", flush=True)
    toks=tokenize(TEXT*6)[0]; seq=64; nwin=len(toks)//seq
    opt=torch.optim.AdamW(m.parameters(), lr=1.5e-4, weight_decay=0.01)
    m.train(); t_start=time.time()
    for st in range(1,41):
        i=torch.randint(0,nwin-1,(1,)).item()
        x=toks[i*seq:(i+1)*seq][None]; y=toks[i*seq+1:(i+1)*seq+1][None]
        loss=F.cross_entropy(m(x)[0].float(), y[0])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if st%10==0:
            print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t_start:.0f}s", flush=True)
    p2=perplexity(m,ids); t2=top1(m,ids); h2=perplexity(m,hids)
    print(f"  eval  : ppl {p2:.3f} | top-1 {t2*100:.1f}%  ({t2*100-t0*100:+.1f})", flush=True)
    print(f"  held  : ppl {h2:.3f}  ({h2-h0:+.3f})", flush=True)

    print(f"\n=== 4) ASLI AAKAR ===", flush=True)
    nc=n//3; packed=nc*6//8; parity=packed//8; table=64*3*2
    tot=packed+parity+table
    print(f"  codon {nc:,} | packed {packed/1e6:.1f} MB | parity {parity/1e6:.1f} MB", flush=True)
    print(f"  KUL {tot/1e6:.1f} MB | fp16 se {n*2/tot:.1f}x chhota", flush=True)
    print(f"  + 30% runtime = {tot/0.7/1e6:.1f} MB", flush=True)

    print(f"\n=== NATIJA (bina pruning) ===", flush=True)
    print(f"  baseline : ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}", flush=True)
    print(f"  + codon  : ppl {p1:.3f} | top-1 {t1*100:.1f}% | held {h1:.3f}", flush=True)
    print(f"  + FT     : ppl {p2:.3f} | top-1 {t2*100:.1f}% | held {h2:.3f}", flush=True)
    print(f"  aakar    : {n*2/1e6:.0f} MB -> {tot/1e6:.1f} MB ({n*2/tot:.1f}x)", flush=True)
    json.dump({"base":OUT0,"codon":{"ppl":p1,"top1":t1,"held":h1},
               "ft":{"ppl":p2,"top1":t2,"held":h2},
               "size_mb":tot/1e6,"ratio":n*2/tot}, open("result.json","w"), indent=2)

main()
