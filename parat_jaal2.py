#!/usr/bin/env python3
"""parat_jaal2.py -- SUDHRA: saajha kendra + bacha hua hissa.

Teen sudhar:
  1. KENDRA SAAJHA -- sab parat ek hi kendra-set use karein (aakar aadha)
  2. BACHA HUA HISSA -- x = Yc[k] + (x - C[k])   (mera bug tha)
  3. ZYADA HISSE -- 256 vs 1024, tulna

Naap: aakar aur cosine milaan
"""
import json, math, os, struct, sys, time
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
ids_all = tok.encode(text).ids[:6000]
print(f"text: {len(ids_all)} token", flush=True)

# ---------- har parat ka input/output jama ----------
print("\njama kar rahe...", flush=True)
t0 = time.time()
ins = [[] for _ in range(L)]; outs = [[] for _ in range(L)]
with torch.no_grad():
    for st in range(0, len(ids_all) - 256, 256):
        seg = torch.tensor(ids_all[st:st+256])
        x = W["model.embed_tokens.weight"][seg]
        for i in range(L):
            xin = x
            x = parat(x, i)
            ins[i].append(xin[::4].numpy())
            outs[i].append(x[::4].numpy())
print(f"  [{time.time()-t0:.0f}s]", flush=True)

X = [np.concatenate(ins[i], 0) for i in range(L)]
Y = [np.concatenate(outs[i], 0) for i in range(L)]
print(f"  har parat: {X[0].shape[0]:,} sample", flush=True)


def kmeans(A, K, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    C = A[rng.choice(len(A), K, replace=False)].copy()
    for it in range(iters):
        lab = np.empty(len(A), dtype=np.int32)
        for s in range(0, len(A), 2000):
            e = min(s+2000, len(A))
            dd = ((A[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab[s:e] = dd.argmin(1)
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


def chalao_jaal(x, Cs, Ycs, bacha):
    for i in range(L):
        Ci = Cs[i]                            # <-- parat ka apna kendra
        lab = assign(x.numpy(), Ci)
        y = Ycs[i][lab]
        if bacha:
            y = y + (x.numpy() - Ci[lab])     # bacha hua hissa jodo
        x = torch.tensor(y)
    return x


print("\n=== TULNA ===", flush=True)
seg = torch.tensor(ids_all[:128])
with torch.no_grad():
    x0 = W["model.embed_tokens.weight"][seg]
    ya = x0.clone()
    for i in range(L): ya = parat(ya, i)
    ya_np = ya.numpy()

print(f"{'setup':<30} {'jaal MB':>9} {'cosine':>8} {'galti':>7}", flush=True)
print("-" * 60)

for K in (256, 1024):
    for saajha in (False, True):
        for bacha in (False, True):
            t0 = time.time()
            if saajha:
                Call = kmeans(np.concatenate(X, 0)[:20000], K, 6, seed=1)
                Cs = [Call] * L
            else:
                Cs = [kmeans(X[i], K, 6, seed=i) for i in range(L)]
            Ycs = []
            for i in range(L):
                lab = assign(X[i], Cs[i])
                yc = np.zeros((K, D), dtype=np.float32)
                for k in range(K):
                    m = lab == k
                    if m.sum() > 0: yc[k] = Y[i][m].mean(0)
                Ycs.append(yc)
            # aakar (int8)
            if saajha:
                mb = (K*D + L*K*D) / 1e6
            else:
                mb = (2 * L * K * D) / 1e6
            yj = chalao_jaal(x0.clone(), Cs, Ycs, bacha)
            cos = float((torch.tensor(ya_np) * yj).sum() / (torch.tensor(ya_np).norm() * yj.norm() + 1e-9))
            err = float((torch.tensor(ya_np) - yj).norm() / (torch.tensor(ya_np).norm() + 1e-9))
            nm = f"K={K} saajha={int(saajha)} bacha={int(bacha)}"
            print(f"{nm:<30} {mb:>8.1f} {cos:>8.4f} {err*100:>6.1f}% [{time.time()-t0:.0f}s]", flush=True)
