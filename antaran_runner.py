#!/usr/bin/env python3
"""antaran_runner.py -- प्रशिक्षित चेकपॉइंट से जाल में ज्ञान का अंतरण नापो।

चेकपॉइंट से:
  1. मॉडल बनाओ (संरचना state-dict से पहचानो)
  2. 50,000 संदर्भों पर उसकी भविष्यवाणी निकालो
  3. उन्हीं से जाल बनाओ (डुप्लीकेट हटाकर)
  4. नापो: कवरेज, शुद्धता, आकार
"""
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--data", default="data/corpus.txt")
ap.add_argument("--vocab", type=int, default=30000)
ap.add_argument("--ctx", type=int, default=3)
ap.add_argument("--seq", type=int, default=16)
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-test", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default="store/antaran.json")
a = ap.parse_args()
torch.set_num_threads(4)

# ---------- पाठ ----------
sents = []
for line in open(a.data, encoding="utf-8"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 8 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= 200000: break
cnt = Counter(w for ws in sents for w in ws)
top = [w for w, _ in cnt.most_common(a.vocab)]
stoi = {w: i for i, w in enumerate(top)}
print(f"वाक्य {len(sents):,} | शब्दकोश {len(stoi):,}", flush=True)

edge = defaultdict(Counter)
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(8)] for x, c in edge.items()}

seqs = [[stoi[w] for w in ws if w in stoi] for ws in sents]
seqs = [s for s in seqs if len(s) >= a.seq + 2]
import random
random.seed(0); random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train, test = seqs[:cut], seqs[cut:]
print(f"अनुक्रम: प्रशिक्षण {len(train):,} | परीक्षा {len(test):,}", flush=True)

# ---------- चेकपॉइंट से संरचना पहचानो ----------
if not os.path.exists(a.ckpt):
    print("चेकपॉइंट नहीं:", a.ckpt, flush=True); sys.exit(1)
d = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
sd = d["m"]
V = sd["tok.weight"].shape[0]
dim = sd["tok.weight"].shape[1]
layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
heads = 16 if dim >= 1024 else (8 if dim >= 512 else 4)
K = 8
print(f"चेकपॉइंट: V={V} dim={dim} layers={layers} heads={heads} step={d.get('step')}", flush=True)


class M(nn.Module):
    def __init__(s, V, d_, L, H, K):
        super().__init__(); s.d, s.H, s.K = d_, H, K
        s.tok = nn.Embedding(V, d_); s.pos = nn.Embedding(256, d_)
        s.blk = nn.ModuleList()
        for _ in range(L):
            s.blk.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d_), "qkv": nn.Linear(d_, 3*d_, bias=False),
                "out": nn.Linear(d_, d_, bias=False), "n2": nn.LayerNorm(d_),
                "ff1": nn.Linear(d_, 4*d_), "ff2": nn.Linear(4*d_, d_),
                "n3": nn.LayerNorm(d_), "lq": nn.Linear(d_, d_, bias=False),
                "lo": nn.Linear(d_, d_, bias=False)}))
        s.nf = nn.LayerNorm(d_)
    def forward(s, x, nb):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None]
        for b in s.blk:
            q, k, v = b["qkv"](b["n1"](h)).chunk(3, -1)
            q = q.view(B,T,s.H,-1).transpose(1,2); k = k.view(B,T,s.H,-1).transpose(1,2)
            v = v.view(B,T,s.H,-1).transpose(1,2)
            att = F.scaled_dot_product_attention(q,k,v,is_causal=True)
            h = h + b["out"](att.transpose(1,2).contiguous().view(B,T,s.d))
            h = h + b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem = s.tok(nb)
            q2 = b["lq"](b["n3"](h)).view(B,T,s.H,-1)
            k2 = mem.view(B,T,s.K,s.H,-1).permute(0,1,3,2,4)
            sc = (q2.unsqueeze(3) @ k2.transpose(-1,-2)) / math.sqrt(s.d//s.H)
            h = h + b["lo"]((sc.softmax(-1) @ k2).squeeze(3).reshape(B,T,s.d))
        return s.nf(h)


m = M(V, dim, layers, heads, K)
missing = m.load_state_dict(sd, strict=False)
print(f"लदा: छूटे {len(missing.missing_keys)} | अतिरिक्त {len(missing.unexpected_keys)}", flush=True)
m.eval()
npar = sum(p.numel() for p in m.parameters())
print(f"पैरामीटर {npar:,} ({npar*4/1e6:.1f} MB fp32)", flush=True)


def batch(rows):
    T = a.seq
    X = torch.zeros(len(rows), T, dtype=torch.long)
    N = torch.zeros(len(rows), T, K, dtype=torch.long)
    for i, r in enumerate(rows):
        seq = r[-T:]
        X[i, :len(seq)] = torch.tensor(seq)
        for t, w in enumerate(seq):
            for j, u in enumerate(nbrs.get(w, [])[:K]): N[i, t, j] = u
    return X, N


def predict(ctxs):
    out = []
    with torch.no_grad():
        for i in range(0, len(ctxs), a.batch):
            b = ctxs[i:i+a.batch]
            if not b: continue
            X, N = batch(b)
            pr = (m(X, N)[:, -1, :] @ m.tok.weight.T).argmax(-1)
            out.extend(pr.tolist())
    return out

# ---------- जाल बनाओ ----------
keys, ctxs = [], []
for r in train[:a.n_train]:
    for i in range(a.ctx, len(r) - 1):
        keys.append(tuple(r[i-a.ctx:i])); ctxs.append(r[:i+1])
print(f"संदर्भ निकाले: {len(ctxs):,}", flush=True)
preds = predict(ctxs)
jaal = {}
for k, p in zip(keys, preds):
    jaal.setdefault(k, Counter())[p] += 1
print(f"जाल: {len(jaal):,} यूनीक संदर्भ", flush=True)

# ---------- नापो ----------
tk, tc, ty = [], [], []
for r in test[:a.n_test]:
    for i in range(a.ctx, len(r) - 1):
        tk.append(tuple(r[i-a.ctx:i])); tc.append(r[:i+1]); ty.append(r[i+1])
mp = predict(tc)
m_hit = sum(1 for p, y in zip(mp, ty) if p == y)
j_hit = cov = 0
for k, y in zip(tk, ty):
    if k in jaal:
        cov += 1
        if jaal[k].most_common(1)[0][0] == y: j_hit += 1
n = max(len(ty), 1)
w_bytes = npar * 4
j_bytes = len(jaal) * (a.ctx * 4 + 4)
rep = {
    "step": d.get("step"), "params": npar, "dim": dim, "layers": layers,
    "V": V, "contexts": len(jaal), "test_positions": n,
    "model_acc": m_hit/n, "lattice_cov": cov/n,
    "lattice_acc_covered": j_hit/max(cov,1), "lattice_acc_total": j_hit/n,
    "weights_bytes": w_bytes, "lattice_bytes": j_bytes,
    "ratio": w_bytes/max(j_bytes,1),
}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=2)
print("\n=== नतीजा ===", flush=True)
print(f"  मॉडल (वज़न):            {m_hit/n*100:5.1f}%", flush=True)
print(f"  जाल कवरेज:              {cov/n*100:5.1f}%", flush=True)
print(f"  जाल (जहाँ कवर):          {j_hit/max(cov,1)*100:5.1f}%", flush=True)
print(f"  जाल (कुल):              {j_hit/n*100:5.1f}%", flush=True)
print(f"  वज़न {w_bytes/1e6:.1f} MB | जाल {j_bytes/1e6:.2f} MB | अनुपात {w_bytes/max(j_bytes,1):.1f}x", flush=True)
print(f"सहेजा: {a.out}", flush=True)
