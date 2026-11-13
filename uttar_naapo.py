#!/usr/bin/env python3
"""uttar_naapo.py -- उत्तर की सटीकता नापो: जाल + पैटर्न + ट्रांसफॉर्मर, तीनों अलग-अलग।

  जाल-अकेला   : पैक की कड़ियों से सबसे आम अगला शब्द
  पैटर्न-अकेला: उस स्लॉट का सबसे आम शब्द
  ट्रांसफॉर्मर: संदर्भ + जाल-पड़ोस + पैटर्न से अंक
"""
import argparse, json, math, mmap, os, random, struct, sys
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="data/gyaan.pack")
ap.add_argument("--facts", default="data/gyaan.txt")
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--cand", type=int, default=50)
ap.add_argument("--pat-cand", type=int, default=20)
ap.add_argument("--minpat", type=int, default=5)
ap.add_argument("--seq", type=int, default=24)
ap.add_argument("--nq", type=int, default=300)
ap.add_argument("--out", default="store/uttar.json")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
random.seed(0)


class Pack:
    def __init__(s, path):
        s.f = open(path, "rb"); s.mm = mmap.mmap(s.f.fileno(), 0, access=mmap.ACCESS_READ)
        assert s.mm[:6] == b"GPACK5", "GPACK5 नहीं"
        (s.V, s.ns, s.nch, s.cbits, s.missed, s.nblk, s.block, s.nlen) = struct.unpack("<IIIIIIII", s.mm[6:38])
        off = 38; s.part = {}
        for name in ("chars","voc","lat","latidx","starts","body","blk","lens"):
            (n,) = struct.unpack("<I", s.mm[off:off+4]); off += 4
            s.part[name] = (off, n); off += n
        co, cn = s.part["chars"]; s.chars = s.mm[co:co+cn].decode("utf-8"); s._w = None
    def _vint(s, d, i):
        n = 0; sh = 0
        while True:
            b = d[i]; i += 1; n |= (b & 0x7F) << sh
            if not (b & 0x80): return n, i
            sh += 7
    def words(s):
        if s._w is not None: return s._w
        vo, vn = s.part["voc"]; data = s.mm[vo:vo+vn]
        out, cur, acc, n = [], [], 0, 0
        cb, stop = s.cbits, len(s.chars)
        for byte in data:
            for k in range(8):
                acc |= ((byte >> k) & 1) << n; n += 1
                if n == cb:
                    if acc == stop: out.append("".join(cur)); cur = []
                    else: cur.append(s.chars[acc])
                    acc = 0; n = 0
        if cur: out.append("".join(cur))
        s._w = out; return out
    def links(s, w):
        lo, _ = s.part["lat"]; io, _ = s.part["latidx"]
        (x,) = struct.unpack("<I", s.mm[io+w*4:io+w*4+4])
        (y,) = struct.unpack("<I", s.mm[io+(w+1)*4:io+(w+1)*4+4])
        d = s.mm[lo+x:lo+y]; cnt, i = s._vint(d, 0); out = []
        for _ in range(cnt):
            v, i = s._vint(d, i); out.append(v)
        return out
    def _bits(s, bp, nb):
        bo, _ = s.part["body"]; v = 0
        for i in range(nb):
            p = bp + i; v |= ((s.mm[bo + (p >> 3)] >> (p & 7)) & 1) << i
        return v
    def path(s, idx):
        if idx < 0 or idx >= s.ns: return None
        bo, _ = s.part["blk"]; so, _ = s.part["starts"]; lo2, _ = s.part["lens"]
        (bp,) = struct.unpack("<I", s.mm[bo+idx*4:bo+idx*4+4])
        (wid,) = struct.unpack("<I", s.mm[so+idx*4:so+idx*4+4])
        L = s.mm[lo2+idx]; out = [wid]
        for _ in range(L):
            lk = s.links(wid)
            b = max(1, (max(len(lk), 2) - 1).bit_length())
            k = s._bits(bp, b); bp += b
            if k >= len(lk): break
            wid = lk[k]; out.append(wid)
        return out


if not os.path.exists(a.pack):
    print("पैक नहीं:", a.pack, flush=True); sys.exit(1)
p = Pack(a.pack); words = p.words()
paths = []
for i in range(min(p.ns, 200000)):
    q = p.path(i)
    if q and 6 <= len(q) <= 60: paths.append(q)
cnt = Counter(i for q in paths for i in q)
top = [i for i, _ in cnt.most_common(p.V)]
remap = {o: n for n, o in enumerate(top)}
V = len(top)
paths = [[remap[i] for i in q if i in remap] for q in paths]
paths = [q for q in paths if len(q) >= 6]
print(f"पैक: पाथ {len(paths):,} | शब्दकोश {V:,}", flush=True)

nbrs, cands = {}, {}
for old, new in remap.items():
    lk = p.links(old)
    ids = [remap[x] for x in lk if x in remap]
    if ids: nbrs[new] = ids[:a.nbr]; cands[new] = ids[:a.cand]

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
            slot_cand[(pt, i)] = [w for w, _ in pat_slots[pt][i].most_common(a.pat_cand)]
print(f"पैटर्न {len(pats):,} | स्लॉट {len(slot_cand):,}", flush=True)


class M(nn.Module):
    def __init__(s, V, P, d, L, H, K):
        super().__init__(); s.d, s.H, s.K = d, H, K
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(256, d); s.pat = nn.Embedding(max(P,1), d)
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


m = M(V, len(pats), a.dim, a.layers, a.heads, a.nbr)
loaded = False
if os.path.exists(a.ckpt):
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
        r = m.load_state_dict(ck["m"], strict=False)
        loaded = True
        print(f"चेकपॉइट लदा (कदम {ck.get('step')}) छूटे {len(r.missing_keys)}", flush=True)
    except Exception as e:
        print("चेकपॉइंट नहीं लदा:", type(e).__name__, flush=True)
m.eval()

# ---------- सवाल बनाओ ----------
qs = []
for q in paths:
    for i in range(2, len(q) - 1):
        qs.append((q[:i+1][-a.seq:], q[i+1], pattern(q), i))
random.shuffle(qs)
qs = qs[:a.nq]
print(f"सवाल: {len(qs):,}", flush=True)

j1 = p1 = t1 = 0
for ctx, y, pt, pos in qs:
    # जाल-अकेला
    if cands.get(ctx[-1]) and cands[ctx[-1]][0] == y: j1 += 1
    # पैटर्न-अकेला
    sc = slot_cand.get((pt, pos), [])
    if sc and sc[0] == y: p1 += 1
    # ट्रांसफॉर्मर
    X = torch.zeros(1, a.seq, dtype=torch.long); NB = torch.zeros(1, a.seq, a.nbr, dtype=torch.long)
    X[0, :len(ctx)] = torch.tensor(ctx)
    for t, w in enumerate(ctx):
        for j, u in enumerate(nbrs.get(w, [])[:a.nbr]): NB[0, t, j] = u
    pool = list(dict.fromkeys(cands.get(ctx[-1], []) + slot_cand.get((pt, pos), []) + [y]))
    CL = torch.zeros(1, len(pool), dtype=torch.long); CL[0] = torch.tensor(pool)
    with torch.no_grad():
        h = m(X, NB, torch.tensor([pidx.get(pt, 0)]))[:, -1, :]
        sc2 = (h[:, None, :] * m.tok(CL)).sum(-1)
    if pool[sc2.argmax(-1).item()] == y: t1 += 1

n = max(len(qs), 1)
rep = {"n": n, "jaal": j1/n, "pattern": p1/n, "transformer": t1/n, "loaded": loaded,
       "paths": len(paths), "V": V, "patterns": len(pats)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=2)
print("\n=== उत्तर-सटीकता ===", flush=True)
print(f"  जाल-अकेला:        {j1/n*100:5.1f}%", flush=True)
print(f"  पैटर्न-अकेला:      {p1/n*100:5.1f}%", flush=True)
print(f"  ट्रांसफॉर्मर:      {t1/n*100:5.1f}%", flush=True)
print(f"सहेजा: {a.out}", flush=True)
