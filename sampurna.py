#!/usr/bin/env python3
"""sampurna.py -- पूरा अपग्रेड: जाल + पैटर्न-तालिका + आधुनिक ट्रांसफॉर्मर।

तीनों एक साथ:
  1. जाल      -- सेल-embedding + हर परत में cross-attention (पड़ोसी सेल्स)
  2. पैटर्न   -- ढाँचा + स्लॉट के उम्मीदवार (पूरे शब्दकोश के बजाय K)
  3. ट्रांसफॉर्मर -- causal, कई परतें, कई heads
निर्गम: सिर्फ़ उम्मीदवारों पर अंक (K+1), 30,000 पर नहीं -- 1,500x सस्ता।
"""
import argparse, math, os, random, re, time
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/corpus.txt")
ap.add_argument("--vocab", type=int, default=30000)
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq", type=int, default=24)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=500000)
ap.add_argument("--budget", type=int, default=15000)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--cand", type=int, default=50, help="जाल के उम्मीदवार")
ap.add_argument("--pat-cand", type=int, default=20, help="पैटर्न-स्लॉट के उम्मीदवार")
ap.add_argument("--minpat", type=int, default=5)
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-sent", type=int, default=400000)
ap.add_argument("--no-lattice", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(a.threads)

# ---------- 1. पाठ ----------
sents = []
for line in open(a.data, encoding="utf-8"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 6 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= a.max_sent: break
if not sents:
    print("कोष खाली -- रुक रहे हैं", flush=True); raise SystemExit(1)
cnt = Counter(w for ws in sents for w in ws)
top = [w for w, _ in cnt.most_common(a.vocab)]
stoi = {w: i for i, w in enumerate(top)}
V = len(stoi)
print(f"वाक्य {len(sents):,} | शब्दकोश {V:,} | कवरेज {sum(cnt[w] for w in top)/sum(cnt.values())*100:.1f}%", flush=True)

# ---------- 2. जाल: कड़ियाँ ----------
edge = defaultdict(Counter)
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(a.nbr)] for x, c in edge.items()}
cands = {x: [y for y, _ in c.most_common(a.cand)] for x, c in edge.items()}
print(f"जाल: {len(edge):,} सेल | कड़ियाँ {sum(len(c) for c in edge.values()):,}", flush=True)

# ---------- 3. पैटर्न-तालिका ----------
STOP = {w for w, c in cnt.items() if c >= 300}
def pattern(ws): return tuple(w if w in STOP else "*" for w in ws)
pat_slots = defaultdict(lambda: defaultdict(Counter))
pat_count = Counter()
for ws in sents:
    p = pattern(ws); pat_count[p] += 1
    for i, w in enumerate(ws):
        if p[i] == "*": pat_slots[p][i][w] += 1
pats = [p for p, c in pat_count.items() if c >= a.minpat]
pidx = {p: i for i, p in enumerate(pats)}
slot_cand = {}
for p in pats:
    for i in range(len(p)):
        if p[i] == "*":
            slot_cand[(p, i)] = [stoi[w] for w, _ in pat_slots[p][i].most_common(a.pat_cand) if w in stoi]
cap = sum(max(len(v), 1) for v in slot_cand.values())
print(f"पैटर्न: {len(pats):,} | स्लॉट: {len(slot_cand):,} | क्षमता {cap:,}", flush=True)

# ---------- 4. नमूने ----------
seqs = []
for ws in sents:
    p = pattern(ws)
    ids = [stoi.get(w, -1) for w in ws]
    if any(i < 0 for i in ids): continue
    if len(ids) < a.seq + 2: continue
    seqs.append((ids, p))
seqs = [s for s in seqs if len(s[0]) >= a.seq + 2]
random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train, test = seqs[:cut], seqs[cut:]
print(f"अनुक्रम: प्रशिक्षण {len(train):,} | परीक्षा {len(test):,}", flush=True)
if not train:
    print("प्रशिक्षण-अनुक्रम नहीं -- रुक रहे हैं", flush=True); raise SystemExit(1)


class Sampurna(nn.Module):
    """जाल + पैटर्न + ट्रांसफॉर्मर।"""
    def __init__(s, V, P, d, L, H, K, use_lat):
        super().__init__()
        s.d, s.H, s.K, s.use = d, H, K, use_lat
        s.tok = nn.Embedding(V, d)
        s.pos = nn.Embedding(256, d)
        s.pat = nn.Embedding(max(P, 1), d)
        s.blocks = nn.ModuleList()
        for _ in range(L):
            b = nn.ModuleDict({
                "n1": nn.LayerNorm(d), "qkv": nn.Linear(d, 3*d, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d)})
            if use_lat:
                b["n3"] = nn.LayerNorm(d)
                b["lq"] = nn.Linear(d, d, bias=False)
                b["lo"] = nn.Linear(d, d, bias=False)
            s.blocks.append(b)
        s.nf = nn.LayerNorm(d)

    def forward(s, x, nb, pid):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None] + s.pat(pid)[:, None, :]
        for b in s.blocks:
            q, k, v = b["qkv"](b["n1"](h)).chunk(3, -1)
            q = q.view(B, T, s.H, -1).transpose(1, 2)
            k = k.view(B, T, s.H, -1).transpose(1, 2)
            v = v.view(B, T, s.H, -1).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).contiguous().view(B, T, s.d)
            h = h + b["out"](att)
            h = h + b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            if s.use:
                mem = s.tok(nb)                                   # (B,T,K,d)
                q2 = b["lq"](b["n3"](h)).view(B, T, s.H, -1)
                k2 = mem.view(B, T, s.K, s.H, -1).permute(0, 1, 3, 2, 4)
                sc = (q2.unsqueeze(3) @ k2.transpose(-1, -2)) / math.sqrt(s.d // s.H)
                h = h + b["lo"]((sc.softmax(-1) @ k2).squeeze(3).reshape(B, T, s.d))
        return s.nf(h)


use_lat = 0 if a.no_lattice else 1
m = Sampurna(V, len(pats), a.dim, a.layers, a.heads, a.nbr, use_lat)
npar = sum(p.numel() for p in m.parameters())
print(f"पैरामीटर {npar:,} | जाल {'हाँ' if use_lat else 'नहीं'} | dim {a.dim} | {a.layers} परतें", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
step0 = 0
if os.path.exists(a.ckpt):
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("V") == V and ck.get("dim") == a.dim:
            m.load_state_dict(ck["m"]); opt.load_state_dict(ck["o"]); step0 = ck["step"]
            print(f"चेकपॉइंट से: कदम {step0:,}", flush=True)
    except Exception as e:
        print(f"चेकपॉइंट नहीं लदा ({type(e).__name__})", flush=True)


def make_batch(rows):
    """नमूना: संदर्भ + जाल-उम्मीदवार + पैटर्न-स्लॉट उम्मीदवार।"""
    B = len(rows); T = a.seq
    X = torch.zeros(B, T, dtype=torch.long)
    NB = torch.zeros(B, T, a.nbr, dtype=torch.long)
    PI = torch.zeros(B, dtype=torch.long)
    YC = torch.zeros(B, dtype=torch.long)      # सही उम्मीदवार का सूचकांक
    CL = torch.zeros(B, a.cand + a.pat_cand, dtype=torch.long)  # उम्मीदवार आईडी
    keep = []
    for j, (ids, p) in enumerate(rows):
        i = random.randrange(1, len(ids) - 1)
        ctx = ids[max(0, i - T):i]
        if len(ctx) < 2: continue
        X[j, :len(ctx)] = torch.tensor(ctx)
        for t in range(len(ctx)):
            nb = nbrs.get(ctx[t], [])[:a.nbr]
            for q, w in enumerate(nb): NB[j, t, q] = w
        PI[j] = pidx.get(p, 0)
        tgt = ids[i]
        pool = list(dict.fromkeys(cands.get(ids[i-1], []) + slot_cand.get((p, i), [])))
        if tgt not in pool:
            pool.append(tgt)          # सही उत्तर हमेशा शामिल
        pool = pool[:a.cand + a.pat_cand]
        CL[j, :len(pool)] = torch.tensor(pool)
        YC[j] = pool.index(tgt)
        keep.append(j)
    if not keep: return None
    idx = torch.tensor(keep)
    return X[idx], NB[idx], PI[idx], CL[idx], YC[idx]


t0 = time.time(); last = 0.0; step = step0
while step < a.steps and (time.time() - t0) < a.budget:
    step += 1
    b = make_batch(random.sample(train, min(a.batch, len(train))))
    if b is None: continue
    X, NB, PI, CL, YC = b
    h = m(X, NB, PI)[:, -1, :]                      # आख़िरी स्थान
    logits = (h[:, None, :] * m.tok(CL)).sum(-1)    # उम्मीदवारों पर अंक
    loss = F.cross_entropy(logits, YC)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    if step % 200 == 0:
        print(f"  कदम {step:,} | loss {last:.3f} | {200/(time.time()-t0)*max(1,step-step0)/max(step-step0,1):.1f} कदम/सेकंड", flush=True)
    if step % 2000 == 0:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V, "dim": a.dim}, a.ckpt)

# ---------- मूल्यांकन ----------
m.eval()
h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(test), 2000), 32):
        b = make_batch(test[i:i+32])
        if b is None: continue
        X, NB, PI, CL, YC = b
        sc = (m(X, NB, PI)[:, -1, :][:, None, :] * m.tok(CL)).sum(-1)
        h1 += (sc.argmax(-1) == YC).sum().item()
        h5 += sum(1 for j in range(len(YC)) if YC[j] in sc.topk(5, -1).indices[j])
        tot += len(YC)
print(f"\nपरीक्षा ({tot:,}): पहले 1 में {h1/max(tot,1)*100:.1f}% | पहले 5 में {h5/max(tot,1)*100:.1f}%", flush=True)
os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V, "dim": a.dim}, a.ckpt)
print(f"सहेजा: {a.ckpt} (कदम {step:,})", flush=True)
