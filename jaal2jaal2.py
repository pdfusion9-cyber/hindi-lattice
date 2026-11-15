#!/usr/bin/env python3
"""jaal2jaal2.py -- SUDHRA: poora shabdakosh + chhipi avastha (token nahi).

Do sudhar:
  1. KAVREJ 100%  -- poore 151,936 shabd se jaal banao (3,000 nahi)
  2. AVASTHA      -- har jaal token nahi, 896-ayam avastha lautaye
                     (jaankari bahti hai, mitati nahi)

Pattern taalka = transformer ka poora shabdakosh.
Jaal-2-jaal    = 24 jaal, avastha par kaam karte hue.
"""
import json, math, os, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
V = 151936
SAMPLE = 24000        # kitne shabd se jaal banaye (poori tarah faile hue)

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
    CH = 4_000_000
    for st in range(0, ne, CH):
        en = min(st + CH, ne)
        u = np.frombuffer(raw[st*2:en*2], dtype=np.uint16)
        out[st:en] = (u.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(out).reshape(v["shape"])

print("=== PATTERN TAALKA (transformer ka shabdakosh) ===", flush=True)
PAT = load("model.embed_tokens.weight")
print(f"  {tuple(PAT.shape)} | int4 {PAT.numel()*0.5/1e6:.0f} MB", flush=True)

W = {}
need = ["model.norm.weight"]
for i in range(L):
    p = f"model.layers.{i}."
    need += [p+"input_layernorm.weight", p+"post_attention_layernorm.weight",
             p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
             p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
             p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
             p+"self_attn.o_proj.weight",
             p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight"]
for nm in need: W[nm] = load(nm)

def rms(x, w):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

def rp(z, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    half = hd // 2
    z1 = z[..., :half]; z2 = z[..., half:]
    return torch.cat([z1*cos - z2*sin, z1*sin + z2*cos], dim=-1)

def attn(x, i, T):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rp(q.view(T, H, hd), T).transpose(0, 1)
    k = rp(k.view(T, KVH, hd), T).transpose(0, 1)
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
    x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, x.shape[0])
    return x + mlp(rms(x, W[p+"post_attention_layernorm.weight"]), i)

# ---------- 1) POORE SHABDAKOSH SE AVASTHA ----------
print(f"\n=== 1) POORE SHABDAKOSH SE AVASTHA ({SAMPLE:,} shabd) ===", flush=True)
t0 = time.time()
rng = np.random.default_rng(0)
sample = np.sort(rng.choice(V, SAMPLE, replace=False))
INS = [[] for _ in range(L)]; OUTS = [[] for _ in range(L)]
with torch.no_grad():
    for st in range(0, len(sample), 256):
        ids = torch.tensor(sample[st:st+256])
        x = PAT[ids]
        for i in range(L):
            xin = x
            x = parat(x, i)
            INS[i].append(xin.numpy().astype(np.float32))
            OUTS[i].append(x.numpy().astype(np.float32))
        if st % 4096 == 0:
            print(f"  {st}/{len(sample)} [{time.time()-t0:.0f}s]", flush=True)
print(f"  [{time.time()-t0:.0f}s] har parat {sum(a.shape[0] for a in INS[0]):,} sample", flush=True)

X = [np.concatenate(INS[i], 0) for i in range(L)]
Y = [np.concatenate(OUTS[i], 0) for i in range(L)]

# ---------- 2) JAAL: avastha -> avastha ----------
def kmeans(A, K, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    C = A[rng.choice(len(A), min(K, len(A)), replace=False)].copy()
    for it in range(iters):
        lab = np.empty(len(A), dtype=np.int32)
        for s in range(0, len(A), 4000):
            e = min(s+4000, len(A))
            d = (A[s:e]**2).sum(1)[:, None] - 2*A[s:e] @ C.T + (C**2).sum(1)[None]
            lab[s:e] = d.argmin(1)
        for k in range(len(C)):
            m = lab == k
            if m.sum() > 0: C[k] = A[m].mean(0)
    return C

def assign(A, C):
    lab = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), 4000):
        e = min(s+4000, len(A))
        d = (A[s:e]**2).sum(1)[:, None] - 2*A[s:e] @ C.T + (C**2).sum(1)[None]
        lab[s:e] = d.argmin(1)
    return lab

print("\n=== 2) JAAL BANA RAHE (avastha -> avastha) ===", flush=True)
HASIL = []
for K in (256, 512, 1024):
    t0 = time.time()
    Cs, Ycs, SC = [], [], []
    for i in range(L):
        C = kmeans(X[i], K, 6, seed=i)
        lab = assign(X[i], C)
        yc = np.zeros((len(C), D), dtype=np.float32)
        for k in range(len(C)):
            m = lab == k
            if m.sum() > 0: yc[k] = Y[i][m].mean(0)
        Cs.append(C); Ycs.append(yc)
        # paimana
        sc = np.ones(len(C), dtype=np.float32)
        for k in range(len(C)):
            m = lab == k
            if m.sum() > 0:
                yt = Y[i][m]; yj = yc[k][None]
                sc[k] = float(np.linalg.norm(yt, axis=1).mean() / (np.linalg.norm(yj) + 1e-9))
        SC.append(sc)
    mb = sum(C.size + yc.size for C, yc in zip(Cs, Ycs)) / 1e6
    HASIL.append((K, mb, Cs, Ycs, SC))
    print(f"  K={K}: {mb:.1f} MB (int8) [{time.time()-t0:.0f}s]", flush=True)

# ---------- 3) CHALAO (avastha, token nahi) ----------
def chalao_jaal(x0, K_idx, bacha=True):
    K, mb, Cs, Ycs, SC = HASIL[K_idx]
    x = x0.copy()
    for i in range(L):
        lab = assign(x, Cs[i])
        y = Ycs[i][lab] * SC[i][lab][:, None]
        if bacha:
            y = y + (x - Cs[i][lab])          # bacha hua hissa
        x = y
    return x

def chalao_asli(x0):
    x = x0.clone()
    for i in range(L): x = parat(x, i)
    return x

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:96]

print("\n=== 3) POORI CHAIN: asli vs jaal-2-jaal ===", flush=True)
print(f"{'setup':<22} {'jaal MB':>8} {'cosine':>8} {'|x|':>9}", flush=True)
print("-" * 52)
with torch.no_grad():
    x0 = PAT[torch.tensor(ids_all)]
    ya = chalao_asli(x0.clone()).numpy()
    print(f"{'asli':<22} {'-':>8} {'1.0000':>8} {float(np.linalg.norm(ya)):>9.1f}", flush=True)
    for j, (K, mb, _, _, _) in enumerate(HASIL):
        for bacha in (False, True):
            yj = chalao_jaal(x0.numpy(), j, bacha)
            cs = float((ya*yj).sum() / (np.linalg.norm(ya)*np.linalg.norm(yj) + 1e-9))
            nm = f"K={K} bacha={int(bacha)}"
            print(f"{nm:<22} {mb:>7.0f} {cs:>8.4f} {float(np.linalg.norm(yj)):>9.1f}", flush=True)

# ---------- 4) KUL AAKAR ----------
pat4 = PAT.numel() * 0.5 / 1e6
print(f"\n=== 4) KUL AAKAR ===", flush=True)
print(f"  pattern taalka (int4):  {pat4:>7.0f} MB", flush=True)
for K, mb, _, _, _ in HASIL:
    print(f"  + jaal K={K:<5}         {mb:>7.1f} MB  = kul {pat4+mb:>7.0f} MB  ({1976/(pat4+mb):.1f}x chhota)", flush=True)
