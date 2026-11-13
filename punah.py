#!/usr/bin/env python3
"""punah.py -- purani copy se training phir shuru (sahi settings: lr 1e-4, vocab saved)."""
import argparse, json, math, os, random, re, time
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="")
ap.add_argument("--data", default="data/corpus.txt")
ap.add_argument("--vocab", type=int, default=30000)
ap.add_argument("--dim", type=int, default=1024)
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--seq", type=int, default=16)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--steps", type=int, default=500000)
ap.add_argument("--budget", type=int, default=12000)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--cand", type=int, default=50)
ap.add_argument("--pat-cand", type=int, default=20)
ap.add_argument("--minpat", type=int, default=5)
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-sent", type=int, default=200000)
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(a.threads)

sents = []
for line in open(a.data, encoding="utf-8", errors="ignore"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 6 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= a.max_sent: break
cnt = Counter(w for ws in sents for w in ws)
vocab = [w for w, _ in cnt.most_common(a.vocab)]
stoi = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"vakya {len(sents):,} | shabdakosh {V:,}", flush=True)
if not sents:
    print("kosh khali", flush=True); raise SystemExit(1)

edge = defaultdict(Counter)
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(a.nbr)] for x, c in edge.items()}
cands = {x: [y for y, _ in c.most_common(a.cand)] for x, c in edge.items()}
print(f"jaal {len(edge):,} sel | kadiya {sum(len(c) for c in edge.values()):,}", flush=True)

STOP = {w for w, c in cnt.items() if c >= 300}
def pattern(q): return tuple(w if w in STOP else "*" for w in q)
pat_slots = defaultdict(lambda: defaultdict(Counter)); pat_count = Counter()
for ws in sents:
    pt = pattern(ws); pat_count[pt] += 1
    for i, w in enumerate(ws):
        if pt[i] == "*": pat_slots[pt][i][w] += 1
pats = [p for p, c in pat_count.items() if c >= a.minpat]
pidx = {p: i for i, p in enumerate(pats)}
slot_cand = {}
for p in pats:
    for i in range(len(p)):
        if p[i] == "*":
            slot_cand[(p, i)] = [stoi[w] for w, _ in pat_slots[p][i].most_common(a.pat_cand) if w in stoi]
print(f"pattern {len(pats):,} | slot {len(slot_cand):,}", flush=True)

seqs = []
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    if len(ids) >= a.seq + 2: seqs.append((ids, pattern(ws)))
random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train, test = seqs[:cut], seqs[cut:]
print(f"anukram: train {len(train):,} | test {len(test):,}", flush=True)
if not train:
    print("anukram nahi", flush=True); raise SystemExit(1)


class M(nn.Module):
    def __init__(s, V, P, d, L, H, K):
        super().__init__(); s.d, s.H, s.K = d, H, K
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(256, d); s.pat = nn.Embedding(max(P, 1), d)
        s.blk = nn.ModuleList()
        for _ in range(L):
            s.blk.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "qkv": nn.Linear(d, 3*d, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d),
                "n3": nn.LayerNorm(d), "lq": nn.Linear(d, d, bias=False),
                "lo": nn.Linear(d, d, bias=False)}))
        s.nf = nn.LayerNorm(d)

    def forward(s, x, nb, pid):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None] + s.pat(pid)[:, None, :]
        for b in s.blk:
            q, k, v = b["qkv"](b["n1"](h)).chunk(3, -1)
            q = q.view(B, T, s.H, -1).transpose(1, 2)
            k = k.view(B, T, s.H, -1).transpose(1, 2)
            v = v.view(B, T, s.H, -1).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + b["out"](att.transpose(1, 2).contiguous().view(B, T, s.d))
            h = h + b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem = s.tok(nb)
            q2 = b["lq"](b["n3"](h)).view(B, T, s.H, -1)
            k2 = mem.view(B, T, s.K, s.H, -1).permute(0, 1, 3, 2, 4)
            sc = (q2.unsqueeze(3) @ k2.transpose(-1, -2)) / math.sqrt(s.d // s.H)
            h = h + b["lo"]((sc.softmax(-1) @ k2).squeeze(3).reshape(B, T, s.d))
        return s.nf(h)


m = M(V, len(pats), a.dim, a.layers, a.heads, a.nbr)
print(f"param {sum(p.numel() for p in m.parameters()):,}", flush=True)

if a.old and os.path.exists(a.old):
    try:
        ck = torch.load(a.old, map_location="cpu", weights_only=False)
        old = ck["m"]
        cur = m.state_dict(); loaded = 0; skipped = 0
        with torch.no_grad():
            for name, t in old.items():
                tgt = name.replace("blocks.", "blk.")
                if tgt in cur and list(cur[tgt].shape) == list(t.shape):
                    cur[tgt].copy_(t.float()); loaded += 1
                else:
                    skipped += 1
        print(f"purani copy: lade {loaded} | chhode {skipped}", flush=True)
    except Exception as e:
        print(f"purani copy nahi ladi: {type(e).__name__}", flush=True)
else:
    print("purani copy nahi mili -- shuru se", flush=True)

opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)


def mk(rows):
    B = len(rows); T = a.seq; K = a.cand + a.pat_cand
    X = torch.zeros(B, T, dtype=torch.long); NB = torch.zeros(B, T, a.nbr, dtype=torch.long)
    PI = torch.zeros(B, dtype=torch.long); CL = torch.zeros(B, K, dtype=torch.long)
    YC = torch.zeros(B, dtype=torch.long); keep = []
    for j, (ids, pt) in enumerate(rows):
        i = random.randrange(1, len(ids) - 1)
        ctx = ids[max(0, i - T):i]
        if len(ctx) < 2: continue
        X[j, :len(ctx)] = torch.tensor(ctx)
        for t in range(len(ctx)):
            for q, u in enumerate(nbrs.get(ctx[t], [])[:a.nbr]): NB[j, t, q] = u
        PI[j] = pidx.get(pt, 0)
        tgt = ids[i]
        pool = list(dict.fromkeys(cands.get(ids[i-1], []) + slot_cand.get((pt, i), [])))
        if tgt not in pool: pool.append(tgt)
        if len(pool) > K:
            pool = pool[:K]
            if tgt not in pool: pool[-1] = tgt
        CL[j, :len(pool)] = torch.tensor(pool)
        YC[j] = pool.index(tgt); keep.append(j)
    if not keep: return None
    ix = torch.tensor(keep)
    return X[ix], NB[ix], PI[ix], CL[ix], YC[ix]


t0 = time.time(); step = 0; last = 0.0
while step < a.steps and (time.time() - t0) < a.budget:
    step += 1
    b = mk(random.sample(train, min(a.batch, len(train))))
    if b is None: continue
    X, NB, PI, CL, YC = b
    h = m(X, NB, PI)[:, -1, :]
    loss = F.cross_entropy((h[:, None, :] * m.tok(CL)).sum(-1), YC)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    if step % 200 == 0:
        print(f"  kadam {step:,} | loss {last:.3f}", flush=True)
    if step % 1000 == 0:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V,
                    "dim": a.dim, "layers": a.layers, "heads": a.heads,
                    "vocab": vocab, "pats": pats}, a.ckpt)

m.eval(); h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(test), 800), 16):
        b = mk(test[i:i+16])
        if b is None: continue
        X, NB, PI, CL, YC = b
        sc = (m(X, NB, PI)[:, -1, :][:, None, :] * m.tok(CL)).sum(-1)
        h1 += (sc.argmax(-1) == YC).sum().item()
        h5 += sum(1 for j in range(len(YC)) if YC[j] in sc.topk(5, -1).indices[j])
        tot += len(YC)
print(f"\ntest ({tot:,}): top1 {h1/max(tot,1)*100:.1f}% | top5 {h5/max(tot,1)*100:.1f}%", flush=True)
os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V,
            "dim": a.dim, "layers": a.layers, "heads": a.heads,
            "vocab": vocab, "pats": pats}, a.ckpt)
print(f"saheja: {a.ckpt} (kadam {step:,}) -- shabdakosh sahit", flush=True)
