#!/usr/bin/env python3
"""aakar_naapo.py -- Qwen ka kitna hissa chahiye + jaal + pattern = kitna aakar?

Chaar kaam:
  1. HAR PARAT KA MAHATVA -- kaun si parat kitni zaroori
  2. CHUNA HUA HISA -- sirf zaroori paratein
  3. JAAL + PATTERN -- gyan kadiyon me
  4. AAKAR -- kul kitna bacha

Naap: perplexity (kam = behtar)
"""
import json, math, os, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6

with open("m.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n

def load(name):
    v = hdr[name]
    with open("m.safetensors", "rb") as f:
        f.seek(base + v["data_offsets"][0])
        raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
    ne = len(raw) // 2
    out = np.empty(ne, dtype=np.float32)
    CH = 2_000_000
    for st in range(0, ne, CH):
        en = min(st + CH, ne)
        u = np.frombuffer(raw[st*2:en*2], dtype=np.uint16)
        out[st:en] = (u.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(out).reshape(v["shape"])

print("wazan laa rahe...", flush=True)
W = {}
need = ["model.embed_tokens.weight", "model.norm.weight"]
for i in range(L):
    p = f"model.layers.{i}."
    need += [p+"input_layernorm.weight", p+"post_attention_layernorm.weight",
             p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
             p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
             p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
             p+"self_attn.o_proj.weight",
             p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight"]
for nm in need: W[nm] = load(nm)
NP_ALL = sum(t.numel() for t in W.values())
print(f"  {len(W)} tensors | {NP_ALL:,} param | fp32 {NP_ALL*4/1e6:.0f} MB", flush=True)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    half = hd // 2
    x1 = x[..., :half]; x2 = x[..., half:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

def attn(x, i, T):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rope(q.view(T, H, hd), T).transpose(0, 1)
    k = rope(k.view(T, KVH, hd), T).transpose(0, 1)
    v = v.view(T, KVH, hd).transpose(0, 1)
    rep = H // KVH
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask[None], -1e9)
    o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, D)
    return o @ W[p+"o_proj.weight"].T

def mlp(x, i):
    p = f"model.layers.{i}.mlp."
    g = torch.nn.functional.silu(x @ W[p+"gate_proj.weight"].T)
    u = x @ W[p+"up_proj.weight"].T
    return (g * u) @ W[p+"down_proj.weight"].T

def forward(ids, layers):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in layers:
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

# ---------- jaanch ka text ----------
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text)[:600]
print(f"jaanch: {len(ids_all)} token", flush=True)

def ppl(layers):
    tot = 0.0; cnt = 0
    with torch.no_grad():
        for st in range(0, len(ids_all) - 32, 16):
            seg = ids_all[st:st+32]
            if len(seg) < 8: break
            lg = forward(seg[:-1], layers)
            tgt = torch.tensor(seg[1:])
            tot += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="sum"))
            cnt += len(tgt)
    return math.exp(tot / max(cnt, 1))

# ---------- 1. HAR PARAT KA MAHATVA ----------
print("\n=== 1) HAR PARAT KA MAHATVA (akela hata kar dekho) ===", flush=True)
base_layers = list(range(L))
p_full = ppl(base_layers)
print(f"  poora ({L} parat): ppl {p_full:.2f}", flush=True)
drop_effect = []
for i in range(L):
    ls = [j for j in range(L) if j != i]
    t0 = time.time()
    try:
        p = ppl(ls)
        drop_effect.append((i, p, p - p_full))
        print(f"  parat {i:>2} hataya: ppl {p:>9.2f} ({p-p_full:+.2f}) [{time.time()-t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"  parat {i:>2}: fail {type(e).__name__}", flush=True)
drop_effect.sort(key=lambda x: x[2])
print(f"\n  SABSE KAM ZAROORI paratein: {[d[0] for d in drop_effect[:8]]}", flush=True)

# ---------- 2. CHUNA HUA HISA ----------
print("\n=== 2) CHUNA HUA HISA -- kitni paratein chahiye? ===", flush=True)
order = [d[0] for d in drop_effect]          # kam zaroori pehle
print(f"{'parat':<8} {'param':>10} {'int4':>8} {'ppl':>10}", flush=True)
res = []
for keep in (24, 20, 16, 12, 8):
    drop = set(order[:L-keep])
    ls = [i for i in range(L) if i not in drop]
    t0 = time.time()
    try:
        p = ppl(ls)
        npar = W["model.embed_tokens.weight"].numel()
        for i in ls:
            for x in ("self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
                      "self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight",
                      "mlp.down_proj.weight"):
                npar += W[f"model.layers.{i}.{x}"].numel()
        res.append((keep, npar, p))
        print(f"{keep:<8} {npar/1e6:>9.0f}M {npar*0.5/1e6:>7.0f}MB {p:>10.2f} [{time.time()-t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"{keep}: fail {type(e).__name__}", flush=True)

# ---------- 3. JAAL + PATTERN ----------
print("\n=== 3) JAAL + PATTERN (gyan kadiyon me) ===", flush=True)
sents = []
for line in text.split("\n"):
    ws = line.split()
    if 4 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= 20000: break
edge = defaultdict(Counter)
for ws in sents:
    ids = tok.encode(" ".join(ws)).ids
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nkad = sum(len(c) for c in edge.values())
jaal_mb = nkad * 5 / 1e6
STOP = {w for w, c in Counter(w for ws in sents for w in ws).items() if c >= 20}
def pat(ws): return tuple(w if w in STOP else "*" for w in ws)
pc = Counter()
for ws in sents: pc[pat(ws)] += 1
npats = len([p for p, c in pc.items() if c >= 2])
pat_mb = npats * D * 0.5 / 1e6        # int4 pattern table
print(f"  jaal: {len(edge):,} sel | {nkad:,} kadiya | {jaal_mb:.2f} MB", flush=True)
print(f"  pattern: {npats:,} | {pat_mb:.2f} MB (int4)", flush=True)

# ---------- 4. KUL AAKAR ----------
print("\n=== 4) KUL AAKAR ===", flush=True)
print(f"{'setup':<26} {'param':>10} {'int4':>9} {'+jaal':>9} {'ppl':>9}", flush=True)
print("-" * 68)
for keep, npar, p in res:
    int4 = npar * 0.5 / 1e6
    kul = int4 + jaal_mb + pat_mb
    print(f"{str(keep)+' parat':<26} {npar/1e6:>9.0f}M {int4:>8.0f}MB {kul:>8.0f}MB {p:>9.2f}", flush=True)

print(f"\n  aaj (fp32, 24 parat): {NP_ALL*4/1e6:.0f} MB", flush=True)
best = min(res, key=lambda r: r[1]*0.5 + jaal_mb + pat_mb if r[2] < p_full * 2 else 1e18) if res else None
if best:
    keep, npar, p = best
    kul = npar*0.5/1e6 + jaal_mb + pat_mb
    print(f"  SABSE BEHTAR: {keep} parat | {kul:.0f} MB | ppl {p:.2f}", flush=True)
    print(f"  ghatav: {NP_ALL*4/1e6/kul:.1f}x", flush=True)
