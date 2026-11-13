#!/usr/bin/env python3
"""sampurna2.py -- असली जाल (gyaan.pack) से जुड़ा ट्रांसफॉर्मर।

अंतर sampurna.py से:
  पहले: कच्चे पाठ से bigram ग्राफ़ बनता था (ट्रांसफॉर्मर जाल नहीं पढ़ता था)
  अब:   gyaan.pack सीधे पढ़ता है -- शब्द, पाथ, कड़ियाँ, रिंगें
"""
import argparse, math, mmap, os, random, struct, time
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="data/gyaan.pack")
ap.add_argument("--facts", default="data/gyaan.txt")
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq", type=int, default=24)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=500000)
ap.add_argument("--budget", type=int, default=15000)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--cand", type=int, default=50)
ap.add_argument("--pat-cand", type=int, default=20)
ap.add_argument("--minpat", type=int, default=5)
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-paths", type=int, default=200000)
ap.add_argument("--no-lattice", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(a.threads)


# ---------------- GPACK5 पाठक ----------------
class Pack:
    def __init__(s, path):
        s.f = open(path, "rb")
        s.mm = mmap.mmap(s.f.fileno(), 0, access=mmap.ACCESS_READ)
        if s.mm[:6] != b"GPACK5":
            raise SystemExit(f"GPACK5 नहीं: {s.mm[:6]}")
        (s.V, s.ns, s.nch, s.cbits, s.missed, s.nblk, s.block, s.nlen) = struct.unpack("<IIIIIIII", s.mm[6:38])
        off = 38
        s.part = {}
        for name in ("chars", "voc", "lat", "latidx", "starts", "body", "blk", "lens"):
            (n,) = struct.unpack("<I", s.mm[off:off+4]); off += 4
            s.part[name] = (off, n); off += n
        co, cn = s.part["chars"]
        s.chars = s.mm[co:co+cn].decode("utf-8")
        s._w = None

    def _vint(s, data, i):
        n = 0; sh = 0
        while True:
            b = data[i]; i += 1
            n |= (b & 0x7F) << sh
            if not (b & 0x80): return n, i
            sh += 7

    def words(s):
        if s._w is not None: return s._w
        vo, vn = s.part["voc"]
        data = s.mm[vo:vo+vn]
        out, cur, acc, n = [], [], 0, 0
        cb, stop = s.cbits, len(s.chars)
        for byte in data:
            for k in range(8):
                acc |= ((byte >> k) & 1) << n; n += 1
                if n == cb:
                    if acc == stop:
                        out.append("".join(cur)); cur = []
                    else:
                        cur.append(s.chars[acc])
                    acc = 0; n = 0
        if cur: out.append("".join(cur))
        s._w = out
        return out

    def links(s, w):
        lo, _ = s.part["lat"]; io, _ = s.part["latidx"]
        (x,) = struct.unpack("<I", s.mm[io+w*4:io+w*4+4])
        (y,) = struct.unpack("<I", s.mm[io+(w+1)*4:io+(w+1)*4+4])
        data = s.mm[lo+x:lo+y]
        cnt, i = s._vint(data, 0)
        out = []
        for _ in range(cnt):
            v, i = s._vint(data, i); out.append(v)
        return out

    def _bits(s, bitpos, nbits):
        bo, _ = s.part["body"]; v = 0
        for i in range(nbits):
            p = bitpos + i
            v |= ((s.mm[bo + (p >> 3)] >> (p & 7)) & 1) << i
        return v

    def path(s, idx):
        if idx < 0 or idx >= s.ns: return None
        bo, _ = s.part["blk"]; so, _ = s.part["starts"]; lo2, _ = s.part["lens"]
        (bp,) = struct.unpack("<I", s.mm[bo+idx*4:bo+idx*4+4])
        (wid,) = struct.unpack("<I", s.mm[so+idx*4:so+idx*4+4])
        L = s.mm[lo2+idx]
        w = s.words()
        out = [wid]
        for _ in range(L):
            lk = s.links(wid)
            b = max(1, (max(len(lk), 2) - 1).bit_length())
            k = s._bits(bp, b); bp += b
            if k >= len(lk): break
            wid = lk[k]; out.append(wid)
        return out


if not os.path.exists(a.pack):
    print("पैक नहीं:", a.pack, flush=True); raise SystemExit(1)
p = Pack(a.pack)
words = p.words()
print(f"पैक {os.path.getsize(a.pack)/1e6:.2f} MB | शब्द {p.V:,} | पाथ {p.ns:,}", flush=True)

# ---------------- पाथ पढ़ो ----------------
paths = []
for i in range(min(p.ns, a.max_paths)):
    q = p.path(i)
    if q and 6 <= len(q) <= 60: paths.append(q)
print(f"पाथ पढ़े: {len(paths):,}", flush=True)
if not paths:
    print("पाथ नहीं मिले", flush=True); raise SystemExit(1)

# पैक की आईडी ही शब्दकोश हैं
cnt = Counter(i for q in paths for i in q)
top = [i for i, _ in cnt.most_common(p.V)]
remap = {old: new for new, old in enumerate(top)}
V = len(top)
paths = [[remap[i] for i in q if i in remap] for q in paths]
paths = [q for q in paths if len(q) >= 6]
print(f"शब्दकोश {V:,} | पाथ {len(paths):,}", flush=True)

# ---------------- जाल: असली कड़ियों से ----------------
nbrs, cands = {}, {}
for old, new in remap.items():
    lk = p.links(old)
    if not lk: continue
    ids = [remap[x] for x in lk if x in remap]
    if ids:
        nbrs[new] = ids[:a.nbr]
        cands[new] = ids[:a.cand]
print(f"जाल: {len(nbrs):,} सेल से कड़ियाँ (पैक से)", flush=True)

# ---------------- पैटर्न-तालिका ----------------
STOP = {w for w, c in cnt.items() if c >= 300}
def pattern(q): return tuple(w if w in STOP else "*" for w in q)
pat_slots = defaultdict(lambda: defaultdict(Counter)); pat_count = Counter()
for q in paths:
    pt = pattern(q); pat_count[pt] += 1
    for i, w in enumerate(q):
        if pt[i] == "*": pat_slots[pt][i][w] += 1
pats = [x for x, c in pat_count.items() if c >= a.minpat]
pidx = {x: i for i, x in enumerate(pats)}
slot_cand = {}
for pt in pats:
    for i in range(len(pt)):
        if pt[i] == "*":
            slot_cand[(pt, i)] = [w for w, _ in pat_slots[pt][i].most_common(a.pat_cand) if w in remap.values() or True]
print(f"पैटर्न {len(pats):,} | स्लॉट {len(slot_cand):,}", flush=True)

# ---------------- अनुक्रम ----------------
seqs = []
for q in paths:
    if len(q) < a.seq + 2: continue
    seqs.append((q, pattern(q)))
random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train, test = seqs[:cut], seqs[cut:]
print(f"अनुक्रम: प्रशिक्षण {len(train):,} | परीक्षा {len(test):,}", flush=True)
if not train:
    print("अनुक्रम नहीं", flush=True); raise SystemExit(1)


class M(nn.Module):
    def __init__(s, V, P, d, L, H, K, use):
        super().__init__(); s.d, s.H, s.K, s.use = d, H, K, use
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(256, d)
        s.pat = nn.Embedding(max(P, 1), d)
        s.blk = nn.ModuleList()
        for _ in range(L):
            b = nn.ModuleDict({
                "n1": nn.LayerNorm(d), "qkv": nn.Linear(d, 3*d, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d)})
            if use:
                b["n3"] = nn.LayerNorm(d); b["lq"] = nn.Linear(d, d, bias=False)
                b["lo"] = nn.Linear(d, d, bias=False)
            s.blk.append(b)
        s.nf = nn.LayerNorm(d)

    def forward(s, x, nb, pid):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None] + s.pat(pid)[:, None, :]
        for b in s.blk:
            q, k, v = b["qkv"](b["n1"](h)).chunk(3, -1)
            q = q.view(B,T,s.H,-1).transpose(1,2); k = k.view(B,T,s.H,-1).transpose(1,2)
            v = v.view(B,T,s.H,-1).transpose(1,2)
            att = F.scaled_dot_product_attention(q,k,v,is_causal=True)
            h = h + b["out"](att.transpose(1,2).contiguous().view(B,T,s.d))
            h = h + b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            if s.use:
                mem = s.tok(nb)
                q2 = b["lq"](b["n3"](h)).view(B,T,s.H,-1)
                k2 = mem.view(B,T,s.K,s.H,-1).permute(0,1,3,2,4)
                sc = (q2.unsqueeze(3) @ k2.transpose(-1,-2)) / math.sqrt(s.d//s.H)
                h = h + b["lo"]((sc.softmax(-1) @ k2).squeeze(3).reshape(B,T,s.d))
        return s.nf(h)


use = 0 if a.no_lattice else 1
m = M(V, len(pats), a.dim, a.layers, a.heads, a.nbr, use)
npar = sum(x.numel() for x in m.parameters())
print(f"पैरामीटर {npar:,} | जाल {'हाँ' if use else 'नहीं'} | dim {a.dim} | {a.layers} परतें", flush=True)
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


def mk(rows):
    B = len(rows); T = a.seq; K = a.cand + a.pat_cand
    X = torch.zeros(B, T, dtype=torch.long); NB = torch.zeros(B, T, a.nbr, dtype=torch.long)
    PI = torch.zeros(B, dtype=torch.long); CL = torch.zeros(B, K, dtype=torch.long)
    YC = torch.zeros(B, dtype=torch.long); keep = []
    for j, (ids, pt) in enumerate(rows):
        i = random.randrange(1, len(ids) - 1)
        ctx = ids[max(0, i-T):i]
        if len(ctx) < 2: continue
        X[j, :len(ctx)] = torch.tensor(ctx)
        for t in range(len(ctx)):
            for q, u in enumerate(nbrs.get(ctx[t], [])[:a.nbr]): NB[j, t, q] = u
        PI[j] = pidx.get(pt, 0)
        tgt = ids[i]
        pool = list(dict.fromkeys(cands.get(ids[i-1], []) + slot_cand.get((pt, i), [])))
        if tgt not in pool:
            pool.append(tgt)
        if len(pool) > K:
            pool = pool[:K]
            if tgt not in pool:          # काटने से लक्ष्य न गिरे
                pool[-1] = tgt
        CL[j, :len(pool)] = torch.tensor(pool)
        YC[j] = pool.index(tgt); keep.append(j)
    if not keep: return None
    ix = torch.tensor(keep)
    return X[ix], NB[ix], PI[ix], CL[ix], YC[ix]


t0 = time.time(); last = 0.0; step = step0
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
        print(f"  कदम {step:,} | loss {last:.3f}", flush=True)
    if step % 2000 == 0:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V, "dim": a.dim}, a.ckpt)

m.eval(); h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(test), 1500), 32):
        b = mk(test[i:i+32])
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
