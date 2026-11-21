#!/usr/bin/env python3
"""punah.py -- purani copy se training, POORI data par.

SUDHAR (v2):
  1. RESIDUAL SCALING -- har jod 0.5 se. Isse activation dhamaka rukta hai.
     Pehle |h| 46 -> 21,459 (460x) ho jata tha. Ab ~1x par rahega.
  2. MQA -- K,V sab head me saajha (16x kam ganana)
  3. Kram se poora kosh + coverage tracking

Chalane ke liye:
  python3 punah.py --data data/corpus.txt --old old/ckpt/sampurna.pt --budget 12000
"""
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
ap.add_argument("--pattern-window", type=int, default=8)
ap.add_argument("--minpat", type=int, default=5)
ap.add_argument("--res-scale", type=float, default=0.5, help="residual ka bhaar")
ap.add_argument("--sparse-win", type=int, default=6, help="jaal-sparse dhyan ki khidki (0 = band)")
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-sent", type=int, default=400000)
ap.add_argument("--emb", default="", help="Qwen se nigla hua shabd-embedding (.npy)")
ap.add_argument("--emb-vocab", default="", help="us embedding ka shabdakosh (.json)")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(a.threads)

# ---------- data ----------
sents = []
for line in open(a.data, encoding="utf-8", errors="ignore"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 6 <= len(ws) <= 60:
        sents.append(ws)
    if len(sents) >= a.max_sent:
        break
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
    for x, y in zip(ids, ids[1:]):
        edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(a.nbr)] for x, c in edge.items()}
cands = {x: [y for y, _ in c.most_common(a.cand)] for x, c in edge.items()}
print(f"jaal {len(edge):,} sel | kadiya {sum(len(c) for c in edge.values()):,}", flush=True)

STOP = {w for w, c in cnt.items() if c >= 300 and w in stoi}
PATTERN_WINDOW = max(1, min(int(a.pattern_window), a.seq))

def pattern(ids):
    suffix = ids[-PATTERN_WINDOW:]
    return tuple(vocab[token_id] if vocab[token_id] in STOP else "*" for token_id in suffix)

seqs = []
for ws in sents:
    ids = [stoi[w] for w in ws if w in stoi]
    if len(ids) >= a.seq + 2:
        seqs.append(ids)
print(f"anukram (kram me): {len(seqs):,}", flush=True)
if not seqs:
    print("anukram nahi", flush=True); raise SystemExit(1)

# Prefix patterns mirror inference: pattern(prefix), plus a saturated
# position key matching min(prefix_length, seq_len).
pat_slots = defaultdict(lambda: defaultdict(Counter)); pat_count = Counter()
for ids in seqs:
    for i in range(2, len(ids)):
        pt = pattern(ids[:i])
        pos_key = min(i, a.seq)
        pat_count[pt] += 1
        pat_slots[pt][pos_key][vocab[ids[i]]] += 1
pats = [p for p, n in pat_count.items() if n >= a.minpat]
pidx = {p: i for i, p in enumerate(pats)}
slot_cand = {}
for p in pats:
    for pos_key, candidates in pat_slots[p].items():
        slot_cand[(p, pos_key)] = [
            stoi[w] for w, _ in candidates.most_common(a.pat_cand) if w in stoi
        ]
print(f"pattern {len(pats):,} | slot {len(slot_cand):,} | prefix-window {PATTERN_WINDOW}", flush=True)

RS = a.res_scale


class M(nn.Module):
    def __init__(s, V, P, d, L, H, K):
        super().__init__(); s.d, s.H, s.K = d, H, K
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(256, d); s.pat = nn.Embedding(max(P, 1), d)
        s.blk = nn.ModuleList()
        for _ in range(L):
            s.blk.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "q": nn.Linear(d, d, bias=False),
                "k": nn.Linear(d, d // H, bias=False), "v": nn.Linear(d, d // H, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4 * d), "ff2": nn.Linear(4 * d, d),
                "n3": nn.LayerNorm(d), "lq": nn.Linear(d, d, bias=False),
                "lo": nn.Linear(d, d, bias=False)}))
        s.nf = nn.LayerNorm(d)

    def forward(s, x, nb, pid, amask=None):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None] + s.pat(pid)[:, None, :]
        for b in s.blk:
            xx = b["n1"](h)
            q = b["q"](xx).view(B, T, s.H, -1).transpose(1, 2)
            k = b["k"](xx).view(B, T, 1, -1).transpose(1, 2)     # MQA
            v = b["v"](xx).view(B, T, 1, -1).transpose(1, 2)     # MQA
            if amask is None:
                att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                att = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
            h = h + RS * b["out"](att.transpose(1, 2).contiguous().view(B, T, s.d))
            h = h + RS * b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem = s.tok(nb)
            q2 = b["lq"](b["n3"](h)).view(B, T, s.H, -1)
            k2 = mem.view(B, T, s.K, s.H, -1).permute(0, 1, 3, 2, 4)
            sc = (q2.unsqueeze(3) @ k2.transpose(-1, -2)) / math.sqrt(s.d // s.H)
            h = h + RS * b["lo"]((sc.softmax(-1) @ k2).squeeze(3).reshape(B, T, s.d))
        return s.nf(h)


m = M(V, len(pats), a.dim, a.layers, a.heads, a.nbr)

# ---------- QWEN ka gyan bhardo ----------
if a.emb and os.path.exists(a.emb):
    try:
        import numpy as np
        z = np.load(a.emb, allow_pickle=False)
        if "q" in z and "s" in z:                 # int8 + scale (npz)
            E = z["q"].astype(np.float32) * z["s"][:, None]
            if "vocab" in z:
                vv = z["vocab"]
                qw = json.loads(vv.item() if hasattr(vv, "item") else str(vv))
            else:
                qw = None
        else:                                      # saada npy
            E = np.load(a.emb)
            qw = json.load(open(a.emb_vocab)) if a.emb_vocab and os.path.exists(a.emb_vocab) else None
        if qw:
            qi = {w: i for i, w in enumerate(qw)}
            D = min(E.shape[1], a.dim)
            hit = 0
            with torch.no_grad():
                W = m.tok.weight.data
                for i, w in enumerate(vocab):
                    j = qi.get(w)
                    if j is None: continue
                    W[i, :D].copy_(torch.from_numpy(np.array(E[j][:D], dtype=np.float32)))
                    hit += 1
            print(f"QWEN gyan: {hit:,}/{V:,} shabd bhare ({hit/V*100:.1f}%)", flush=True)
        else:
            print("emb-vocab nahi mila -- random shuruat", flush=True)
    except Exception as e:
        print(f"embedding nahi bhari: {type(e).__name__} {str(e)[:80]}", flush=True)
else:
    print("koi embedding nahi -- random shuruat", flush=True)

print(f"param {sum(p.numel() for p in m.parameters()):,}", flush=True)
print(f"residual scale: {RS}", flush=True)

# ---------- purani copy ----------
# Chakra cycles rotate data shards. Transfer learned weights, not the old
# shard's cursor or optimizer state. Token/pattern rows must be mapped by meaning.
pos = 0; epoch = 0
if a.old and os.path.exists(a.old):
    try:
        ck = torch.load(a.old, map_location="cpu", weights_only=False)
        old = ck["m"]
        cur = m.state_dict()
        loaded = skipped = token_rows = pattern_rows = 0
        old_vocab = ck.get("vocab")
        old_pats = ck.get("pats")
        architecture_ok = (
            int(ck.get("dim", -1)) == a.dim
            and int(ck.get("layers", -1)) == a.layers
            and int(ck.get("heads", -1)) == a.heads
        )
        if not architecture_ok:
            print("purani copy ka architecture alag hai; model weights skip", flush=True)
        else:
            old_word_to_id = {w: i for i, w in enumerate(old_vocab)} if isinstance(old_vocab, list) else {}
            old_pattern_to_id = {tuple(p): i for i, p in enumerate(old_pats)} if isinstance(old_pats, list) else {}
            with torch.no_grad():
                for name, tensor in old.items():
                    target = name.replace("blocks.", "blk.")
                    if target not in cur or not isinstance(tensor, torch.Tensor):
                        skipped += 1
                        continue
                    if target == "tok.weight":
                        if not old_word_to_id or tensor.ndim != 2 or tensor.shape[1] != cur[target].shape[1]:
                            skipped += 1
                            continue
                        for word, old_id in old_word_to_id.items():
                            new_id = stoi.get(word)
                            if new_id is not None and old_id < tensor.shape[0] and new_id < cur[target].shape[0]:
                                cur[target][new_id].copy_(tensor[old_id].to(dtype=cur[target].dtype))
                                token_rows += 1
                        loaded += 1
                    elif target == "pat.weight":
                        if not old_pattern_to_id or tensor.ndim != 2 or tensor.shape[1] != cur[target].shape[1]:
                            skipped += 1
                            continue
                        for new_id, pattern_value in enumerate(pats):
                            old_id = old_pattern_to_id.get(tuple(pattern_value))
                            if old_id is not None and old_id < tensor.shape[0] and new_id < cur[target].shape[0]:
                                cur[target][new_id].copy_(tensor[old_id].to(dtype=cur[target].dtype))
                                pattern_rows += 1
                        loaded += 1
                    elif list(cur[target].shape) == list(tensor.shape):
                        cur[target].copy_(tensor.to(dtype=cur[target].dtype))
                        loaded += 1
                    else:
                        skipped += 1
        # Old cursor/epoch belongs to a different rotating shard; start this
        # shard at its beginning. A fresh optimizer avoids stale moment states.
        pos = 0
        epoch = 0
        print(
            f"semantic warm start: {loaded} tensors | {skipped} skipped | "
            f"matched word rows {token_rows:,}/{V:,} | matched pattern rows {pattern_rows:,}/{len(pats):,}; "
            "new shard cursor and optimizer",
            flush=True
        )
    except Exception as e:
        print(f"purani copy nahi ladi: {type(e).__name__}: {str(e)[:160]}", flush=True)
else:
    print("purani copy nahi -- shuru se", flush=True)

opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)


def mk(rows):
    B = len(rows); T = a.seq; K = a.cand + a.pat_cand
    X = torch.zeros(B, T, dtype=torch.long); NB = torch.zeros(B, T, a.nbr, dtype=torch.long)
    PI = torch.zeros(B, dtype=torch.long); CL = torch.zeros(B, K, dtype=torch.long)
    CM = torch.zeros(B, K, dtype=torch.bool)
    YC = torch.zeros(B, dtype=torch.long); keep = []
    for j, ids in enumerate(rows):
        i = (pos + j * 7) % max(1, len(ids) - 1)
        if i < 1: i = 1
        ctx = ids[max(0, i - T):i]
        if len(ctx) < 2: continue
        X[j, :len(ctx)] = torch.tensor(ctx)
        for t in range(len(ctx)):
            for q, u in enumerate(nbrs.get(ctx[t], [])[:a.nbr]): NB[j, t, q] = u
        pt = pattern(ids[:i])
        pos_key = min(i, a.seq)
        PI[j] = pidx.get(pt, 0)
        tgt = ids[i]
        pool = list(dict.fromkeys(cands.get(ids[i-1], []) + slot_cand.get((pt, pos_key), [])))
        if tgt not in pool: pool.append(tgt)
        if len(pool) > K:
            pool = pool[:K]
            if tgt not in pool: pool[-1] = tgt
        CL[j, :len(pool)] = torch.tensor(pool)
        CM[j, :len(pool)] = True
        YC[j] = pool.index(tgt); keep.append(j)
    if not keep: return None
    ix = torch.tensor(keep)
    AM = None
    if a.sparse_win > 0:
        # har token: apni khidki + jaal-padosi (causal)
        AM = torch.zeros(len(ix), T, T, dtype=torch.bool)
        idx = torch.arange(T)
        local = (idx[None, :] >= (idx[:, None] - a.sparse_win)) & (idx[None, :] <= idx[:, None])
        AM[:] = local[None]
        for j, ids in enumerate(rows):
            if j not in keep: continue
            jj = keep.index(j)
            i = (pos + j * 7) % max(1, len(ids) - 1)
            if i < 1: i = 1
            ctx = ids[max(0, i - T):i]
            for t in range(len(ctx)):
                for q in range(t):
                    if ctx[q] in nbrs.get(ctx[t], []): AM[jj, t, q] = True
        AM = AM.unsqueeze(1)
    return X[ix], NB[ix], PI[ix], CL[ix], YC[ix], CM[ix], AM


def save_checkpoint():
    os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
    graph = {
        "nbrs": {int(k): list(v) for k, v in nbrs.items()},
        "cands": {int(k): list(v) for k, v in cands.items()},
        "slot_cand": [
            {"pattern": list(pt), "position": int(i), "ids": list(ids)}
            for (pt, i), ids in slot_cand.items()
        ],
    }
    payload = {
        "m": m.state_dict(),
        "step": step,
        "V": V,
        "dim": a.dim,
        "layers": a.layers,
        "heads": a.heads,
        "vocab": vocab,
        "vocab_counts": [int(cnt[w]) for w in vocab],
        "pats": pats,
        "pat_counts": [int(pat_count[p]) for p in pats],
        "graph": graph,
        "config": {
            "nbr": a.nbr, "cand": a.cand, "pat_cand": a.pat_cand,
            "res_scale": RS, "sparse_win": a.sparse_win,
            "seq": a.seq, "pattern_window": PATTERN_WINDOW,
        },
        "pos": pos + seen,
        "epoch": epoch,
    }
    tmp_path = a.ckpt + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, a.ckpt)
    print(f"checkpoint saved atomically: {a.ckpt}", flush=True)


# ---------- POORI data par (kram me) ----------
t0 = time.time(); step = 0; last = 0.0; seen = 0; N = len(seqs)
h_norm = 0.0
while step < a.steps and (time.time() - t0) < a.budget:
    batch = [seqs[(pos + seen + k) % N] for k in range(a.batch)]
    b = mk(batch)
    seen += a.batch
    if seen >= N:
        seen = 0; epoch += 1; pos = 0
        print(f"  -- poora kosh ek baar dekh liya (epoch {epoch}) --", flush=True)
    if b is None: continue
    step += 1
    X, NB, PI, CL, YC, CM, AM = b
    h = m(X, NB, PI, AM)[:, -1, :]
    logits = (h[:, None, :] * m.tok(CL)).sum(-1)
    logits = logits.masked_fill(~CM, torch.finfo(logits.dtype).min)
    loss = F.cross_entropy(logits, YC)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    h_norm = float(h.norm(dim=-1).mean())
    if step % 200 == 0:
        cov = (seen / N * 100) if N else 0
        print(f"  kadam {step:,} | loss {last:.3f} | |h| {h_norm:.1f} | coverage {cov:.1f}% | epoch {epoch}", flush=True)
    if step % 1000 == 0:
        save_checkpoint()

# ---------- jaanch ----------
m.eval(); h1 = h5 = tot = 0
with torch.no_grad():
    for i in range(0, min(len(seqs), 600), 16):
        b = mk(seqs[i:i+16])
        if b is None: continue
        X, NB, PI, CL, YC, CM, AM = b
        sc = (m(X, NB, PI, AM)[:, -1, :][:, None, :] * m.tok(CL)).sum(-1)
        sc = sc.masked_fill(~CM, torch.finfo(sc.dtype).min)
        h1 += (sc.argmax(-1) == YC).sum().item()
        h5 += sum(1 for j in range(len(YC)) if YC[j] in sc.topk(5, -1).indices[j])
        tot += len(YC)
cov = ((pos + seen) / N * 100) if N else 0
print(f"\ntrain-stream diagnostic ({tot:,}; not held-out): top1 {h1/max(tot,1)*100:.1f}% | top5 {h5/max(tot,1)*100:.1f}%", flush=True)
print(f"coverage: {cov:.1f}% | epoch {epoch} | kadam {step:,} | |h| {h_norm:.1f}", flush=True)
os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
save_checkpoint()
print(f"saheja: {a.ckpt} -- shabdakosh + position sahit", flush=True)
