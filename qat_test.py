"""qat_test.py -- int4 ke baad FINE-TUNE (QAT). Memorization se bachne ke liye
train / eval / held teen ALAG text."""
import json, math, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

EVAL = open("sample.txt").read()          # baseline isi par
TRAIN = """Water moves through the world in many ways. It rises as vapour, forms clouds, and falls
again as rain or snow. Rivers carry it downhill, and lakes hold it in place for a time. Plants take
it up through their roots and release it from their leaves. Animals drink it and carry it further.
None of this water is new; the same molecules have been moving for a very long time. Understanding
this cycle helps us see why a dry year in one place can be felt somewhere else much later."""
HELD = """A library is an unusual kind of building. It holds knowledge that no single person could
carry in memory, and it lends that knowledge to anyone who asks. The order on the shelves is not
arbitrary; it is a map that lets a reader find a book they have never seen. When the map is good,
a question becomes a short walk. When it is bad, the same question becomes a search through walls
of paper. Every library is therefore an argument about how knowledge should be arranged."""

def q4(t):
    hi=7; sc=t.abs().amax(1,keepdim=True).clamp_min(1e-8)/hi
    return torch.clamp(torch.round(t/sc),-hi,hi)*sc

def main():
    m,cfg,miss,_=load_qwen("model.safetensors")
    n=sum(p.numel() for p in m.parameters())
    ev=tokenize(EVAL); hd=tokenize(HELD); tr=tokenize(TRAIN)
    print(f"=== LOAD === param {n:,} | {n*2/1e6:.0f} MB",flush=True)
    print(f"  train {tr.shape[1]} tok | eval {ev.shape[1]} tok | held {hd.shape[1]} tok",flush=True)

    p0=perplexity(m,ev); t0=top1(m,ev); h0=perplexity(m,hd)
    print(f"\nBASELINE : ppl {p0:.3f} | top-1 {t0*100:.1f}% | held {h0:.3f}",flush=True)

    # ---- int4 ----
    print(f"\n=== int4 ===",flush=True)
    with torch.no_grad():
        for nm,p in m.named_parameters():
            if p.dim()<2: continue
            p.data.copy_(q4(p.data))
    p1=perplexity(m,ev); t1=top1(m,ev); h1=perplexity(m,hd)
    print(f"  ppl {p1:.3f} | top-1 {t1*100:.1f}% | held {h1:.3f}",flush=True)

    # ---- QAT: TRAIN text par fine-tune ----
    print(f"\n=== QAT (fine-tune, sirf TRAIN text par) ===",flush=True)
    seq=64
    toks=tokenize(TRAIN*40)[0]
    nwin=len(toks)//seq
    print(f"  windows {nwin}",flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    m.train(); t_start=time.time()
    for st in range(1,241):
        i=torch.randint(0,nwin-1,(1,)).item()
        x=toks[i*seq:(i+1)*seq][None]; y=toks[i*seq+1:(i+1)*seq+1][None]
        loss=F.cross_entropy(m(x)[0].float(), y[0])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if st%40==0:
            print(f"  kadam {st} | train loss {loss.item():.3f} | {time.time()-t_start:.0f}s",flush=True)
    p2=perplexity(m,ev); t2=top1(m,ev); h2=perplexity(m,hd)
    print(f"  ppl {p2:.3f} | top-1 {t2*100:.1f}% | held {h2:.3f}",flush=True)

    tot=n*0.5/1e6
    print(f"\n=== NATIJA (aakar {tot:.1f} MB, {n*2/1e6/tot:.1f}x) ===",flush=True)
    print(f"  {'':<12} {'ppl':>8} {'top-1':>8} {'held':>8}",flush=True)
    print(f"  {'baseline':<12} {p0:>8.3f} {t0*100:>7.1f}% {h0:>8.3f}",flush=True)
    print(f"  {'int4':<12} {p1:>8.3f} {t1*100:>7.1f}% {h1:>8.3f}",flush=True)
    print(f"  {'int4+QAT':<12} {p2:>8.3f} {t2*100:>7.1f}% {h2:>8.3f}",flush=True)
    print(f"\n  QAT ka faayda: ppl {p1:.3f}->{p2:.3f} ({p2-p1:+.3f}) | held {h1:.3f}->{h2:.3f} ({h2-h1:+.3f})",flush=True)
    print(f"  sample: {[get_tok().decode([i]) for i in m(tokenize('The capital of France is'))[0,-1].topk(5).indices.tolist()]}",flush=True)
    json.dump({"base":{"ppl":p0,"top1":t0,"held":h0},"int4":{"ppl":p1,"top1":t1,"held":h1},
               "qat":{"ppl":p2,"top1":t2,"held":h2},"mb":tot},open("result.json","w"),indent=2)

main()
