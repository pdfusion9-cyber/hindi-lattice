#!/usr/bin/env python3
"""gyan_jaal2.py -- ASLI PATH se jaal banao (sudhra hua).

Pehli koshish me galti: har shabd AKELA model me daala.
Ab: asli vaakyon me model chalao, wahan se kadiya nikaalo.

Tarika:
  1. Asli path lo (10 lakh token)
  2. Model ko 32-token khidki me chalao
  3. Har jagah jo agla shabd aaya -> wahi kadi
  4. Do tarah ki kadiya:
       a) ek shabd se agla      (bigram)
       b) do shabd se agla      (trigram) -- behtar
  5. Aakar naapo, jawab dekho
"""
import json, math, os, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
TOPK = 4          # har jagah kitni kadiya
WIN = 32          # khidki ka aakar
MAXTOK = 400_000  # kitne token par jaal banaayein

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
NPARAM = sum(t.numel() for t in W.values())
print(f"  {len(W)} tensors | param {NPARAM:,} | {NPARAM*4/1e6:.0f} MB", flush=True)

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

# ---------- asli path ----------
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:MAXTOK]
print(f"\nkosh: {len(ids_all):,} token", flush=True)

# ---------- khidki me chalao ----------
print(f"\n=== ASLI PATH SE KADIYA NIKAAL RAHE ({len(ids_all):,} token) ===", flush=True)
t0 = time.time()
src1, dst1, w1 = [], [], []          # ek shabd se agla
src2, dst2, w2 = [], [], []          # do shabd se agla
step = WIN // 2
done = 0
with torch.no_grad():
    for st in range(0, max(1, len(ids_all) - WIN), step):
        win = torch.tensor(ids_all[st:st+WIN])
        if len(win) < 4: break
        lg = forward(win)                        # (WIN, V)
        pr = torch.softmax(lg, -1)
        top = torch.topk(pr, TOPK + 1, -1)
        for t in range(len(win) - 1):            # aakhri chhod do
            cur = int(win[t]); prev = int(win[t-1]) if t > 0 else -1
            nxt_real = int(win[t+1])
            for c in range(TOPK + 1):
                d = int(top.indices[t, c])
                pv = float(top.values[t, c])
                if d == cur: continue
                # bigram
                src1.append(cur); dst1.append(d); w1.append(pv)
                # trigram
                if prev >= 0:
                    key = (prev << 17) ^ cur
                    src2.append(key); dst2.append(d); w2.append(pv)
        done += step
        if done % 40000 == 0:
            print(f"  {done:,} | bigram {len(src1):,} | trigram {len(src2):,} | {time.time()-t0:.0f}s", flush=True)
        if done >= len(ids_all):
            break
print(f"  bigram: {len(src1):,} | trigram: {len(src2):,} | {time.time()-t0:.0f}s", flush=True)

# ---------- saaf ----------
def saaf(src, dst, wgt, k=TOPK):
    src = np.array(src, dtype=np.int64); dst = np.array(dst, dtype=np.int32)
    wgt = np.array(wgt, dtype=np.float32)
    order = np.lexsort((-wgt, src))
    src, dst, wgt = src[order], dst[order], wgt[order]
    keep = np.zeros(len(src), dtype=bool)
    seen = {}
    for i in range(len(src)):
        s = int(src[i]); c = seen.get(s, 0)
        if c < k:
            keep[i] = True; seen[s] = c + 1
    return src[keep], dst[keep], wgt[keep]

s1, d1, ww1 = saaf(src1, dst1, w1)
s2, d2, ww2 = saaf(src2, dst2, w2)
print(f"\nsaaf: bigram {len(s1):,} ({len(np.unique(s1)):,} sel) | trigram {len(s2):,} ({len(np.unique(s2)):,} sel)", flush=True)

# ---------- aakar ----------
b1 = len(s1) * 5
b2 = len(s2) * 5
print(f"\n=== AAKAR ===", flush=True)
print(f"  wazan:   {NPARAM*4/1e6:>9.1f} MB", flush=True)
print(f"  bigram:  {b1/1e6:>9.2f} MB", flush=True)
print(f"  trigram: {b2/1e6:>9.2f} MB", flush=True)
print(f"  dono:    {(b1+b2)/1e6:>9.2f} MB  ({(NPARAM*4)/(b1+b2):.0f}x chhota)", flush=True)
np.savez_compressed("jaal2.npz", s1=s1, d1=d1, w1=np.clip(ww1*127,0,127).astype(np.uint8),
                    s2=s2, d2=d2, w2=np.clip(ww2*127,0,127).astype(np.uint8))
print(f"  file: {os.path.getsize('jaal2.npz')/1e6:.2f} MB", flush=True)

# ---------- jawab ----------
print(f"\n=== JAAL SE JAWAB ===", flush=True)
from collections import defaultdict
G1 = defaultdict(list)
for a, b, c in zip(s1, d1, ww1): G1[int(a)].append((float(c), int(b)))
for k in G1: G1[k].sort(reverse=True)
G2 = defaultdict(list)
for a, b, c in zip(s2, d2, ww2): G2[int(a)].append((float(c), int(b)))
for k in G2: G2[k].sort(reverse=True)

def banao(prompt, naya=20, tri=True):
    out = tok.encode(prompt).ids
    for _ in range(naya):
        cur = out[-1]; prev = out[-2] if len(out) > 1 else -1
        key = ((prev << 17) ^ cur) if prev >= 0 else -1
        if tri and key in G2:
            out.append(G2[key][0][1])
        elif cur in G1:
            out.append(G1[cur][0][1])
        else:
            break
    txt = tok.decode(out)
    return txt[len(prompt):] if txt.startswith(prompt) else txt

for s in ("The capital of India is", "def add(a, b):\n    return", "The sun rises in the",
          "Machine learning is", "Python is a"):
    try:
        print(f"  {s!r:<34} -> {banao(s)[:80]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}: {str(e)[:50]}", flush=True)
