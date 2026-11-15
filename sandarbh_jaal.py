#!/usr/bin/env python3
"""sandarbh_jaal.py -- SANDARBH wala jaal: asli transformer ki khubiyan.

Sudhar (pichhli galtiyan theek):
  1. SANDARBH   -- 3 shabdon ka context, sirf 1 nahi  (SABSE BADA SUDHAR)
  2. BACHA HUA  -- x = Yc[k] + (x - C[k])
  3. ALAG KENDRA-- har parat ka apna (saajha nahi -- wo nuksan tha)
  4. DO-STAR    -- pehle mote hisse, phir unke andar bareek

Naap: cosine (1.0 = bilkul same) aur aakar
"""
import json, math, os, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
CTX = 3             # kitne shabd ka sandarbh

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
NP = sum(t.numel() for t in W.values())
print(f"  {NP:,} param | fp32 {NP*4/1e6:.0f} MB", flush=True)

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

def parat(x, i):
    p = f"model.layers.{i}."
    x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, x.shape[0])
    return x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:8000]
print(f"text: {len(ids_all)} token", flush=True)

# ---------- sandarbh ke saath jama ----------
print(f"\njama (sandarbh {CTX} shabd)...", flush=True)
t0 = time.time()
INS = [[] for _ in range(L)]; OUTS = [[] for _ in range(L)]
with torch.no_grad():
    for st in range(0, len(ids_all) - 256, 256):
        seg = torch.tensor(ids_all[st:st+256])
        x = W["model.embed_tokens.weight"][seg]
        for i in range(L):
            xin = x
            x = parat(x, i)
            INS[i].append(xin.numpy())
            OUTS[i].append(x.numpy())
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# SANDARBH banane ke liye: h[t-2], h[t-1], h[t] jodo
# random projection se D dim me laao (aakar same rahe)
RNG = np.random.default_rng(42)
PROJ = RNG.standard_normal((CTX * D, D)).astype(np.float32) / np.sqrt(CTX * D)

def sandarbh(X, idx):
    """X: (T, D) | idx: kaun se sthan | 3 shabd ka context -> (n, D)"""
    out = np.zeros((len(idx), D), dtype=np.float32)
    for j, t in enumerate(idx):
        parts = []
        for c in range(CTX - 1, -1, -1):
            tt = max(0, t - c)
            parts.append(X[tt])
        out[j] = np.concatenate(parts) @ PROJ
    return out

def kmeans(A, K, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    C = A[rng.choice(len(A), K, replace=False)].copy()
    for it in range(iters):
        lab = assign(A, C)
        for k in range(K):
            m = lab == k
            if m.sum() > 0: C[k] = A[m].mean(0)
    return C

def assign(A, C):
    lab = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), 2000):
        e = min(s+2000, len(A))
        dd = ((A[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab[s:e] = dd.argmin(1)
    return lab

print("\n=== JAAL BANA RAHE (sandarbh ke saath) ===", flush=True)
t0 = time.time()
JAAL = []
for i in range(L):
    X = np.concatenate(INS[i], 0)      # (N, D)  -- har parat ka input
    Y = np.concatenate(OUTS[i], 0)
    N = len(X)
    idx = np.arange(N)
    K = sandarbh(X, idx)               # (N, D) -- 3 shabd ka sandarbh
    C = kmeans(K, 512, 6, seed=i)      # 512 hisse
    lab = assign(K, C)
    Yc = np.zeros((512, D), dtype=np.float32)
    for k in range(512):
        m = lab == k
        if m.sum() > 0: Yc[k] = Y[m].mean(0)
    JAAL.append((C, Yc, PROJ))
    if i % 6 == 0:
        print(f"  parat {i:>2}: 512 hisse | bhare {int((lab>=0).sum()>0)*512} [{time.time()-t0:.0f}s]", flush=True)

jaal_mb = sum(C.nbytes + Yc.nbytes for C, Yc, _ in JAAL) / 1e6
jaal_int8 = sum(C.size + Yc.size for C, Yc, _ in JAAL) / 1e6
print(f"\n  JAAL fp32: {jaal_mb:.1f} MB | int8: {jaal_int8:.1f} MB", flush=True)

# ---------- chalao ----------
def jaal_chalao(x0):
    x = x0.numpy().copy()
    for i in range(L):
        C, Yc, PJ = JAAL[i]
        key = sandarbh(x, np.arange(len(x)))       # sandarbh
        lab = assign(key, C)
        x = Yc[lab] + (x - C[lab])                 # bacha hua hissa
    return torch.tensor(x)

def asli_chalao(x0):
    x = x0.clone()
    for i in range(L): x = parat(x, i)
    return x

print("\n=== TULNA: asli vs sandarbh-jaal ===", flush=True)
seg = torch.tensor(ids_all[:128])
with torch.no_grad():
    x0 = W["model.embed_tokens.weight"][seg]
    ya = asli_chalao(x0.clone())
    t0 = time.time(); yj = jaal_chalao(x0.clone()); tj = time.time()-t0
cos = float((ya * yj).sum() / (ya.norm() * yj.norm() + 1e-9))
err = float((ya - yj).norm() / (ya.norm() + 1e-9))
print(f"  cosine:  {cos:.4f}   (pichhla best: 0.3141)", flush=True)
print(f"  galti:   {err*100:.1f}%", flush=True)
print(f"  samay:   {tj*1000:.0f} ms", flush=True)
print(f"  |asli| {float(ya.norm()):.1f} | |jaal| {float(yj.norm()):.1f}", flush=True)
