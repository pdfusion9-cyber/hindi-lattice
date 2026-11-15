#!/usr/bin/env python3
"""parat_jaal.py -- HAR PARAT ki jagah ek JAAL, aur jaal waise jude jaise paratein.

Hamari architecture:
  parat 0 -> JAAL 0 -> parat 1 -> JAAL 1 -> ... -> parat 23 -> JAAL 23

Har jaal us parat ka KAAM kaise rakhta hai:
  - Us parat ke input ko K hisson (cluster) me baanto
  - Har hisse ka AUSAT output yaad rakho
  - Ganana = sabse paas ka hissa dhoondo, uska output do

Yeh "kadiyon" ka jaal hai: cluster -> output
Aakar: K x D x 2 byte (int8) per parat
"""
import json, math, os, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
KCLUST = 256        # har parat me kitne hisse

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
print(f"  {len(W)} tensors | {NP:,} | fp32 {NP*4/1e6:.0f} MB", flush=True)

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
    """ek poori parat ka kaam"""
    p = f"model.layers.{i}."
    x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, x.shape[0])
    return x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)

# ---------- text se parat ka kaam seekho ----------
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:4000]
print(f"seekhne ka text: {len(ids_all)} token", flush=True)

print("\n=== HAR PARAT KA JAAL BANA RAHE ===", flush=True)
t0 = time.time()
JAAL = []          # har parat: (kendra, ausat_output, scale)
with torch.no_grad():
    # poore text ko 256-token khidki me chalao, har parat ka input/output jama karo
    ins = [[] for _ in range(L)]
    outs = [[] for _ in range(L)]
    for st in range(0, len(ids_all) - 256, 256):
        seg = torch.tensor(ids_all[st:st+256])
        x = W["model.embed_tokens.weight"][seg]
        for i in range(L):
            xin = x
            x = parat(x, i)
            ins[i].append(xin[::8].numpy())        # har 8va token -- halka
            outs[i].append(x[::8].numpy())
    print(f"  jama ho gaya [{time.time()-t0:.0f}s]", flush=True)

for i in range(L):
    X = np.concatenate(ins[i], 0)                  # (N, D)
    Y = np.concatenate(outs[i], 0)
    # k-means (saada)
    rng = np.random.default_rng(i)
    idx = rng.choice(len(X), KCLUST, replace=False)
    C = X[idx].copy()
    for it in range(6):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(X) < 3000 else None
        if d is None:
            # bade data par tukdon me
            lab = np.empty(len(X), dtype=np.int32)
            for s in range(0, len(X), 2000):
                e = min(s+2000, len(X))
                dd = ((X[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
                lab[s:e] = dd.argmin(1)
        else:
            lab = d.argmin(1)
        for k in range(KCLUST):
            m = lab == k
            if m.sum() > 0: C[k] = X[m].mean(0)
    # har cluster ka ausat output
    Yc = np.zeros_like(C)
    cnt = np.zeros(KCLUST)
    lab_all = np.empty(len(X), dtype=np.int32)
    for s in range(0, len(X), 2000):
        e = min(s+2000, len(X))
        dd = ((X[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab_all[s:e] = dd.argmin(1)
    for k in range(KCLUST):
        m = lab_all == k
        if m.sum() > 0:
            Yc[k] = Y[m].mean(0); cnt[k] = m.sum()
    JAAL.append((C.astype(np.float32), Yc.astype(np.float32), cnt))
    if i % 6 == 0:
        print(f"  parat {i:>2}: {KCLUST} hisse | bhare {int((cnt>0).sum())} [{time.time()-t0:.0f}s]", flush=True)

# ---------- aakar ----------
jaal_mb = sum(C.nbytes + Yc.nbytes for C, Yc, _ in JAAL) / 1e6
jaal_int8 = sum(C.size + Yc.size for C, Yc, _ in JAAL) / 1e6      # 1 byte per value
print(f"\n=== AAKAR ===", flush=True)
print(f"  Qwen (fp32):     {NP*4/1e6:>9.0f} MB", flush=True)
print(f"  Qwen (int4):     {NP*0.5/1e6:>9.0f} MB", flush=True)
print(f"  JAAL (fp32):     {jaal_mb:>9.2f} MB", flush=True)
print(f"  JAAL (int8):     {jaal_int8:>9.2f} MB", flush=True)
print(f"  JAAL ghatav:     {NP*4/1e6/jaal_int8:>9.0f}x", flush=True)

# ---------- jaal se chalao ----------
def jaal_chalao(x):
    """parat ki jagah jaal -- sabse paas ka hissa dhoondo"""
    for C, Yc, cnt in JAAL:
        d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(-1)     # (T, K)
        k = d.argmin(1)                                         # (T,)
        x = Yc[k] + (x - C[k]) * 0.0                            # hisse ka output
    return x

def asli_chalao(x):
    for i in range(L):
        x = parat(x, i)
    return x

print("\n=== TULNA: asli parat vs jaal ===", flush=True)
seg = torch.tensor(ids_all[:128])
with torch.no_grad():
    x0 = W["model.embed_tokens.weight"][seg]
    y_asli = asli_chalao(x0.clone())
    t0 = time.time(); y_jaal = jaal_chalao(x0.numpy()); t_j = time.time()-t0
    cos = float((y_asli * torch.tensor(y_jaal)).sum() / (y_asli.norm() * torch.tensor(y_jaal).norm() + 1e-9))
    err = float((y_asli - torch.tensor(y_jaal)).norm() / (y_asli.norm() + 1e-9))
print(f"  cosine milaan: {cos:.4f}  (1.0 = bilkul same)", flush=True)
print(f"  galti:         {err*100:.1f}%", flush=True)
print(f"  jaal ka samay: {t_j*1000:.0f} ms | asli: (dekho upar)", flush=True)
print(f"  |asli| {float(y_asli.norm()):.1f} | |jaal| {float(np.linalg.norm(y_jaal)):.1f}", flush=True)
