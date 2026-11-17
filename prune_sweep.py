"""prune_sweep.py -- ASLI Qwen par pruning ki ASLI SEEMA dhoondo."""
import json, math, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)

exec(open("qwen_dna_test.py").read().split("# ---------------- main ----------------")[0])

TEXT = open("sample.txt").read()*8

def main():
    print("=== LOAD ===", flush=True)
    m,cfg,miss,unexp = load_qwen("model.safetensors")
    print(f"  param {sum(p.numel() for p in m.parameters()):,} | miss {len(miss)}", flush=True)
    ids = tokenize(TEXT)
    print(f"  tokens {ids.shape[1]}", flush=True)

    p0=perplexity(m,ids); t0=top1(m,ids)
    print(f"\nBASELINE: ppl {p0:.3f} | top-1 {t0*100:.1f}%", flush=True)

    import copy
    base_sd={k:v.clone() for k,v in m.state_dict().items()}
    n_full=sum(p.numel() for p in m.parameters())

    print(f"\n{'FFN hate':<10} {'param':>12} {'ppl':>8} {'top-1':>7} {'girawat':>9}", flush=True)
    print("-"*52, flush=True)
    for frac in (0.10, 0.20, 0.30, 0.50, 0.70):
        mm,_,_,_ = load_qwen("model.safetensors")
        mm.load_state_dict(base_sd)
        with torch.no_grad():
            for b in mm.blk:
                imp=b.mlp.gate.weight.norm(dim=1)*b.mlp.up.weight.norm(dim=1)*b.mlp.down.weight.norm(dim=0)
                keep=max(8,int(len(imp)*(1-frac)))
                ix=imp.topk(keep).indices.sort().values
                b.mlp.gate.weight=nn.Parameter(b.mlp.gate.weight.data[ix].clone())
                b.mlp.up.weight=nn.Parameter(b.mlp.up.weight.data[ix].clone())
                b.mlp.down.weight=nn.Parameter(b.mlp.down.weight.data[:,ix].clone())
        n=sum(p.numel() for p in mm.parameters())
        p=perplexity(mm,ids); t=top1(mm,ids)
        print(f"{frac*100:>4.0f}%     {n:>12,} {p:>8.3f} {t*100:>6.1f}% {t*100-t0*100:>+8.1f}", flush=True)
        del mm
    print("\ndone", flush=True)

main()
