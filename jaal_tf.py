#!/usr/bin/env python3
"""jaal_tf.py -- जाल से जुड़ा ट्रांसफॉर्मर: गहराई की जगह जाल।

2 परतें + हर परत में जाल से cross-attention।
गहराई लेयरों से नहीं, जाल से आती है जिसमें सीखा हुआ डेटा है।
"""
import argparse, math, random, re, time, os
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/corpus.txt")
ap.add_argument("--vocab", type=int, default=20000)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq", type=int, default=10)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--steps", type=int, default=280000)
ap.add_argument("--budget", type=int, default=15000)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--use-lattice", type=int, default=1)
ap.add_argument("--ckpt", default="ckpt/jaal.pt")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-sent", type=int, default=400000)
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(a.threads)

# ---------- पाठ ----------
sents = []
for line in open(a.data, encoding="utf-8"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 8 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= a.max_sent: break
cnt = Counter(w for ws in sents for w in ws)
top = [w for w, _ in cnt.most_common(a.vocab)]
stoi = {w: i for i, w in enumerate(top)}
V = len(stoi)
print(f"वाक्य {len(sents):,} | शब्दकोश {V:,} | कवरेज {sum(cnt[w] for w in top)/sum(cnt.values())*100:.1f}%", flush=True)

# ---------- जाल ----------
edge = defaultdict(Counter)
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(a.nbr)] for x, c in edge.items()}
print(f"जाल: {len(nbrs):,} सेल | कड़ियाँ {sum(len(c) for c in edge.values()):,}", flush=True)

seqs = [[stoi[w] for w in ws if w in stoi] for ws in sents]
seqs = [s for s in seqs if len(s) >= a.seq + 2]
random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train, test = seqs[:cut], seqs[cut:]
print(f"अनुक्रम: प्रशिक्षण {len(train):,} | परीक्षा {len(test):,}", flush=True)


class JaalTF(nn.Module):
    def __init__(s, V, d, L, H, K, use_lattice):
        super().__init__()
        s.d, s.H, s.K, s.use = d, H, K, use_lattice
        s.tok = nn.Embedding(V, d)
        s.pos = nn.Embedding(256, d)
        s.blocks = nn.ModuleList()
        for _ in range(L):
            b = nn.ModuleDict({
                "n1": nn.LayerNorm(d), "qkv": nn.Linear(d, 3*d, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d)})
            if use_lattice:
                b["n3"] = nn.LayerNorm(d)
                b["lq"] = nn.Linear(d, d, bias=False)
                b["lo"] = nn.Linear(d, d, bias=False)
            s.blocks.append(b)
        s.nf = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False)

    def forward(s, x, nbr_ids):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None]
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
                mem = s.tok(nbr_ids)                       # (B,T,K,d)
                q2 = b["lq"](b["n3"](h)).view(B, T, s.H, -1)
                k2 = mem.view(B, T, s.K, s.H, -1).permute(0, 1, 3, 2, 4)
                sc = (q2.unsqueeze(3) @ k2.transpose(-1, -2)) / math.sqrt(s.d // s.H)
                got = (sc.softmax(-1) @ k2).squeeze(3).reshape(B, T, s.d)
                h = h + b["lo"](got)
        return s.head(s.nf(h))


m = JaalTF(V, a.dim, a.layers, a.heads, a.nbr, a.use_lattice)
npar = sum(p.numel() for p in m.parameters())
print(f"पैरामीटर {npar:,} | जाल-जोड़ {'हाँ' if a.use_lattice else 'नहीं'}", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
step0 = 0
if os.path.exists(a.ckpt):
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("V") == V:
            m.load_state_dict(ck["m"]); opt.load_state_dict(ck["o"]); step0 = ck["step"]
            print(f"चेकपॉइंट से: कदम {step0:,}", flush=True)
    except Exception as e:
        print(f"चेकपॉइंट नहीं लदा ({type(e).__name__})", flush=True)


def batch(rows):
    X = torch.tensor([r[:a.seq] for r in rows])
    Y = torch.tensor([r[a.seq] for r in rows])
    N = torch.zeros(X.shape[0], a.seq, a.nbr, dtype=torch.long)
    for i, r in enumerate(rows):
        for t in range(a.seq):
            nb = nbrs.get(r[t], [])[:a.nbr]
            for j, w in enumerate(nb): N[i, t, j] = w
    return X, N, Y


t0 = time.time(); last = 0.0
for step in range(step0 + 1, a.steps + 1):
    if time.time() - t0 > a.budget: break
    X, N, Y = batch(random.sample(train, a.batch))
    loss = F.cross_entropy(m(X, N)[:, -1, :], Y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    if step % 500 == 0:
        print(f"  कदम {step:,} | loss {last:.3f} | {step/(time.time()-t0):.1f} कदम/सेकंड", flush=True)
    if step % 2000 == 0:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V}, a.ckpt)

m.eval()
h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(test), 3000), 32):
        rows = test[i:i+32]
        if not rows: continue
        X, N, Y = batch(rows)
        sc = m(X, N)[:, -1, :]
        h1 += (sc.argmax(-1) == Y).sum().item()
        h5 += sum(1 for j in range(len(Y)) if Y[j] in sc.topk(5, -1).indices[j])
        tot += len(Y)
print(f"\nपरीक्षा ({tot:,}):  पहले 1 में {h1/max(tot,1)*100:.1f}% | पहले 5 में {h5/max(tot,1)*100:.1f}%", flush=True)
os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V}, a.ckpt)
print(f"सहेजा: {a.ckpt}", flush=True)
