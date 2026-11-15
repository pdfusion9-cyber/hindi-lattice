#!/usr/bin/env python3
"""gyan_jaal.py -- Qwen ke WAZAN se JAAL banao.

Yeh woh kaam hai jo chhoot gaya tha:
  wazan (943 MB)  ->  jaal ke kadiya (kuch MB)

Tarika:
  1. Har token ko model se guzaro -> agla token kya aata hai
  2. Top-k agle token rakho -> yahi KADIYA hain
  3. Sirf kadiya bachao, wazan phenk do

Naapo: aakar kitna? jaal se jawab aata hai?
"""
import json, math, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
TOPK = 8            # har token se kitni kadiya
MAXTOK = 30000      # kitne token par jaal banaayein
BATCH = 128

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
print(f"  {len(W)} tensors | param {sum(t.numel() for t in W.values()):,}", flush=True)
print(f"  aakar: {sum(t.numel() for t in W.values())*4/1e6:.0f} MB (fp32)", flush=True)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)

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

def forward(ids):
    T = ids.shape[0]
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

# ---------- JAAL BANAO ----------
print(f"\n=== JAAL BANA RAHE ({MAXTOK:,} token x top-{TOPK}) ===", flush=True)
t0 = time.time()
src_all = []
dst_all = []
wgt_all = []
done = 0
with torch.no_grad():
    for st in range(0, MAXTOK, BATCH):
        ids = torch.arange(st, min(st + BATCH, MAXTOK))
        if len(ids) == 0: break
        lg = forward(ids)                        # (B, V)
        top = torch.topk(lg, TOPK + 1, dim=-1)   # +1 kyunki khud bhi aa sakta hai
        idx = top.indices
        val = torch.softmax(top.values, -1)
        for r in range(len(ids)):
            s = int(ids[r])
            for c in range(TOPK + 1):
                d = int(idx[r, c])
                if d == s: continue
                src_all.append(s); dst_all.append(d)
                wgt_all.append(float(val[r, c]))
                if len(src_all) % (TOPK * MAXTOK // TOPK) >= MAXTOK: break
        done += len(ids)
        if done % 5000 == 0:
            print(f"  {done:,}/{MAXTOK:,} | kadiya {len(src_all):,} | {time.time()-t0:.0f}s", flush=True)
        if len(src_all) >= MAXTOK * TOPK:
            break

print(f"  kul kadiya: {len(src_all):,} | samay {time.time()-t0:.0f}s", flush=True)

# ---------- SAAFA -- sirf majboot kadiya ----------
src = np.array(src_all, dtype=np.int32)
dst = np.array(dst_all, dtype=np.int32)
wgt = np.array(wgt_all, dtype=np.float32)
print(f"\n=== SAAF KAR RAHE ===", flush=True)
print(f"  pehle: {len(src):,} kadiya", flush=True)
# har src ke top-K
order = np.lexsort((-wgt, src))
src, dst, wgt = src[order], dst[order], wgt[order]
keep = np.zeros(len(src), dtype=bool)
seen = {}
for i in range(len(src)):
    s = int(src[i])
    c = seen.get(s, 0)
    if c < TOPK:
        keep[i] = True; seen[s] = c + 1
src, dst, wgt = src[keep], dst[keep], wgt[keep]
print(f"  baad me: {len(src):,} kadiya | sel {len(np.unique(src)):,}", flush=True)

# ---------- AAKAR ----------
# int8 wazan, int32 target
bytes_edges = len(src) * (4 + 1)          # dst (4) + wazan (1)
print(f"\n=== AAKAR ===", flush=True)
print(f"  wazan (fp32):      {sum(t.numel() for t in W.values())*4/1e6:>8.1f} MB", flush=True)
print(f"  jaal (kadiya):     {bytes_edges/1e6:>8.2f} MB", flush=True)
print(f"  ghatav:            {(sum(t.numel() for t in W.values())*4)/max(bytes_edges,1):>8.0f}x", flush=True)

np.savez_compressed("jaal.npz",
                    src=src, dst=dst,
                    w=(np.clip(wgt * 127, 0, 127)).astype(np.uint8))
import os
print(f"  file: {os.path.getsize('jaal.npz')/1e6:.2f} MB", flush=True)

# ---------- JAAL SE JAWAB ----------
print(f"\n=== JAAL SE JAWAB ===", flush=True)
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
from collections import defaultdict
G = defaultdict(list)
for s, d, w in zip(src, dst, wgt):
    G[int(s)].append((float(w), int(d)))
for s in G: G[s].sort(reverse=True)

def jaal_banao(prompt, naya=25):
    ids = tok.encode(prompt).ids
    out = list(ids)
    for _ in range(naya):
        cur = out[-1]
        if cur not in G: break
        out.append(G[cur][0][1])
    txt = tok.decode(out)
    return txt[len(prompt):] if txt.startswith(prompt) else txt

for s in ("The capital of India is", "def add(a, b):\n    return", "The sun rises in the"):
    try:
        print(f"  {s!r:<32} -> {jaal_banao(s)[:80]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}", flush=True)
