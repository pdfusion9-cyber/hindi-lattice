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
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--fixed-vocab", type=int, default=0)
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
if a.fixed_vocab:
    V = p.V; remap = {i: i for i in range(V)}; top = list(range(V))
    paths = [q for q in paths if len(q) >= 6]
    print(f"साझा शब्दकोश {V:,} | पाथ {len(paths):,}", flush=True)
else:
    top = [i for i, _ in cnt.most_common(p.V)]
    remap = {old: new for new, old in enumerate(top)}
    V = len(top)
    paths = [[remap[i] for i in q if i in remap] for q in paths]
    paths = [q for q in paths if len(q) >= 6]
    print(f"शब्दकोश {V:,} | पाथ {len(paths):,}", flush=True)

vocab_tokens = [words[i] for i in top]
if len(vocab_tokens) != V:
    raise SystemExit("शब्दकोश मैपिंग असंगत है")

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
STOP = {token_id for token_id, count in cnt.items() if count >= 300}
PATTERN_WINDOW = max(1, min(int(a.seq), 8))

def pattern(prefix):
    suffix = prefix[-PATTERN_WINDOW:]
    return tuple(token_id if token_id in STOP else "*" for token_id in suffix)

# Build pattern slots from the prefix available at prediction time, not from
# the complete future sentence. Positions saturate at the model context size.
pat_slots = defaultdict(lambda: defaultdict(Counter)); pat_count = Counter()
for q in paths:
    for i in range(2, len(q)):
        pt = pattern(q[:i])
        pos_key = min(i, a.seq)
        pat_count[pt] += 1
        pat_slots[pt][pos_key][q[i]] += 1
pats = [x for x, count in pat_count.items() if count >= a.minpat]
pidx = {x: i for i, x in enumerate(pats)}
slot_cand = {}
for pt in pats:
    for pos_key, candidates in pat_slots[pt].items():
        slot_cand[(pt, pos_key)] = [
            token_id for token_id, _ in candidates.most_common(a.pat_cand)
            if 0 <= token_id < V
        ]
print(f"पैटर्न {len(pats):,} | स्लॉट {len(slot_cand):,} | prefix-window {PATTERN_WINDOW}", flush=True)

# ---------------- अनुक्रम ----------------
# पैक में पाथ 4-25 शब्दों के होते हैं; seq केवल अधिकतम context-window है।
# हर उदाहरण को पूरे पाथ से sample करें, न कि seq+2 लंबाई वाले पाथ तक सीमित करें।
seqs = [q for q in paths if len(q) >= 3]
if not seqs:
    print("दो-token context वाले पाथ नहीं मिले", flush=True)
    raise SystemExit(1)
lengths = [len(q) for q, _ in seqs]
print(f"अनुक्रम-पाथ: {len(seqs):,} | लंबाई {min(lengths)}-{max(lengths)} | context {a.seq}", flush=True)
random.shuffle(seqs)
cut = max(1, int(len(seqs) * 0.9))
cut = min(cut, len(seqs))
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
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
step0 = 0
if os.path.exists(a.ckpt):
    # केवल वही checkpoint लें जिसका वास्तविक token-map और model shape समान हो।
    fresh_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        compatible = (
            ck.get("V") == V and ck.get("dim") == a.dim
            and ck.get("layers") == a.layers and ck.get("heads") == a.heads
            and ck.get("vocab") == vocab_tokens and ck.get("pats") == pats
            and ck.get("use", use) == use and "m" in ck and "o" in ck
            and "step" in ck
        )
        if compatible:
            opt.load_state_dict(ck["o"])
            m.load_state_dict(ck["m"])
            step0 = int(ck["step"])
            print(f"चेकपॉइंट से: कदम {step0:,}", flush=True)
        else:
            print("पुराना checkpoint असंगत: नया सुरक्षित training शुरू", flush=True)
    except Exception as e:
        m.load_state_dict(fresh_state)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
        step0 = 0
        print(f"चेकपॉइंट नहीं लदा ({type(e).__name__}); नया training शुरू", flush=True)


def mk(rows):
    B = len(rows); T = a.seq; K = a.cand + a.pat_cand
    X = torch.zeros(B, T, dtype=torch.long); NB = torch.zeros(B, T, a.nbr, dtype=torch.long)
    PI = torch.zeros(B, dtype=torch.long); CL = torch.zeros(B, K, dtype=torch.long)
    CM = torch.zeros(B, K, dtype=torch.bool)
    YC = torch.full((B,), -1, dtype=torch.long); keep = []
    for j, ids in enumerate(rows):
        i = random.randrange(2, len(ids))
        ctx = ids[max(0, i-T):i]
        if len(ctx) < 2: continue
        X[j, :len(ctx)] = torch.tensor(ctx)
        for t in range(len(ctx)):
            for q, u in enumerate(nbrs.get(ctx[t], [])[:a.nbr]): NB[j, t, q] = u
        pt = pattern(ids[:i])
        pos_key = min(i, a.seq)
        PI[j] = pidx.get(pt, 0)
        tgt = ids[i]
        pool = list(dict.fromkeys(cands.get(ids[i-1], []) + slot_cand.get((pt, pos_key), [])))
        pool = [v for v in pool if 0 <= v < V][:K]
        if not pool:
            pool = list(range(min(K, V)))
        CL[j, :len(pool)] = torch.tensor(pool)
        CM[j, :len(pool)] = True
        if tgt in pool:
            YC[j] = pool.index(tgt)
        keep.append(j)
    if not keep: return None
    ix = torch.tensor(keep)
    return X[ix], NB[ix], PI[ix], CL[ix], YC[ix], CM[ix]


t0 = time.time(); last = 0.0; step = step0
while step < a.steps and (time.time() - t0) < a.budget:
    step += 1
    b = mk(random.sample(train, min(a.batch, len(train))))
    if b is None: continue
    X, NB, PI, CL, YC, CM = b
    h = m(X, NB, PI)[:, -1, :]
    logits = (h[:, None, :] * m.tok(CL)).sum(-1)
    logits = logits.masked_fill(~CM, torch.finfo(logits.dtype).min)
    valid = YC >= 0
    if not valid.any():
        continue
    loss = F.cross_entropy(logits[valid], YC[valid])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    if step % 200 == 0:
        print(f"  कदम {step:,} | loss {last:.3f}", flush=True)
    if step % 2000 == 0:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V, "dim": a.dim, "layers": a.layers, "heads": a.heads, "vocab": vocab_tokens, "pats": pats, "use": use,
           "seq": a.seq, "pattern_window": PATTERN_WINDOW}, a.ckpt)

m.eval(); h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(test), 1500), 32):
        b = mk(test[i:i+32])
        if b is None: continue
        X, NB, PI, CL, YC, CM = b
        sc = (m(X, NB, PI)[:, -1, :][:, None, :] * m.tok(CL)).sum(-1)
        sc = sc.masked_fill(~CM, torch.finfo(sc.dtype).min)
        valid = YC >= 0
        h1 += ((sc.argmax(-1) == YC) & valid).sum().item()
        topk = sc.topk(min(5, sc.shape[-1]), -1).indices
        h5 += sum(1 for j in range(len(YC)) if valid[j] and YC[j] in topk[j])
        tot += len(YC)
print(f"\nheld-out candidate evaluation ({tot:,}; target-miss counts wrong): Top-1 {h1/max(tot,1)*100:.1f}% | Top-5 {h5/max(tot,1)*100:.1f}%", flush=True)
os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "V": V, "dim": a.dim, "layers": a.layers, "heads": a.heads, "vocab": vocab_tokens, "pats": pats, "use": use,
           "seq": a.seq, "pattern_window": PATTERN_WINDOW}, a.ckpt)
print(f"सहेजा: {a.ckpt} (कदम {step:,})", flush=True)
