#!/usr/bin/env python3
"""jaal2jaal.py -- JAAL SE JAAL network + transformer ka poora VOCAB pattern taalka me.

Theek wahi jo kaha gaya:
  1. PATTERN TAALKA = transformer ka poora shabdakosh (151,936 token)
  2. JAAL-2-JAAL    = har parat ka ek jaal, jaal se jaal jude
  3. KUL kaam       = jaal transformer + pattern taalka

Dikhata hai:
  - pattern taalka ka aakar
  - jaal network ka aakar
  - jaal se jaal chalne par kya nikalta hai
"""
import json, math, os, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
V = 151936

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

print("=== 1) PATTERN TAALKA = TRANSFORMER KA POORA SHABDAKOSH ===", flush=True)
PAT = load("model.embed_tokens.weight")          # (151936, 896)
print(f"  pattern taalka: {tuple(PAT.shape)}", flush=True)
print(f"  fp32: {PAT.numel()*4/1e6:.0f} MB", flush=True)
print(f"  int8: {PAT.numel()*1/1e6:.0f} MB", flush=True)
print(f"  int4: {PAT.numel()*0.5/1e6:.0f} MB", flush=True)

# ---------- baaki wazan ----------
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

def forward(ids):
    T = len(ids)
    x = PAT[ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rms(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rms(x, W["model.norm.weight"])
    return x @ PAT.T

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:3000]
print(f"\ntext: {len(ids_all)} token", flush=True)

# ---------- 2) JAAL-2-JAAL: har parat ka jaal ----------
print("\n=== 2) JAAL-2-JAAL BANA RAHE (har parat ka apna jaal) ===", flush=True)
t0 = time.time()
JAAL = []          # har parat: (src token, dst token, bhaar)
with torch.no_grad():
    for st in range(0, len(ids_all) - 64, 64):
        seg = torch.tensor(ids_all[st:st+64]); T = len(seg)
        x = PAT[seg]
        for i in range(L):
            p = f"model.layers.{i}."
            x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, T)
            x = x + mlp(rms(x, W[p+"post_attention_layernorm.weight"]), i)
            if st % 192 == 0:
                # is parat ka jaal: kaun sa token kis se juda
                xn = rms(x, W["model.norm.weight"])
                lg = xn @ PAT.T                            # (T, V)
                top = torch.topk(lg, 4, -1)
                for t in range(T - 1):
                    s_tok = int(seg[t])
                    for c in range(4):
                        JAAL.append((i, s_tok, int(top.indices[t, c]), float(top.values[t, c])))
        if st % 960 == 0:
            print(f"  {st}/{len(ids_all)} | kadiya {len(JAAL):,} [{time.time()-t0:.0f}s]", flush=True)
print(f"  kul kadiya: {len(JAAL):,} [{time.time()-t0:.0f}s]", flush=True)

# ---------- saaf karo ----------
print("\n=== SAAF ===", flush=True)
per_layer = defaultdict(lambda: defaultdict(list))
for i, s, d, w in JAAL:
    per_layer[i][s].append((w, d))
KAD = {}
for i in range(L):
    out = {}
    for s, lst in per_layer[i].items():
        lst.sort(reverse=True)
        seen = []
        for w, d in lst:
            if d not in seen: seen.append(d)
            if len(seen) >= 4: break
        out[s] = seen
    KAD[i] = out
nkad = sum(len(v) for v in KAD.values())
nsel = sum(len(v) for d in KAD.values() for v in d.values())
print(f"  jaal: {nkad:,} sel | {nsel:,} kadiya", flush=True)

# ---------- 3) AAKAR ----------
pat_int8 = PAT.numel() * 1 / 1e6
pat_int4 = PAT.numel() * 0.5 / 1e6
jaal_mb = nsel * (4 + 1) / 1e6      # dst (4 byte) + bhaar (1 byte)
print("\n=== 3) AAKAR ===", flush=True)
print(f"  pattern taalka (vocab) int8:  {pat_int8:>8.0f} MB", flush=True)
print(f"  pattern taalka (vocab) int4:  {pat_int4:>8.0f} MB", flush=True)
print(f"  jaal-2-jaal network:          {jaal_mb:>8.2f} MB", flush=True)
print(f"  kul (int4 + jaal):            {pat_int4 + jaal_mb:>8.0f} MB", flush=True)
print(f"  Qwen fp32 se ghatav:          {1976/(pat_int4+jaal_mb):>8.1f}x", flush=True)

# ---------- 4) JAAL SE JAAL CHALAO ----------
print("\n=== 4) JAAL-2-JAAL SE JAWAB ===", flush=True)
def jaal_banao(prompt, naya=20):
    """jaal 0 -> jaal 1 -> ... -> jaal 23, har baar agla token"""
    out = tok.encode(prompt).ids
    for _ in range(naya):
        cur = out[-1]
        nxt = cur
        for i in range(L):                      # jaal se jaal
            if nxt in KAD[i] and KAD[i][nxt]:
                nxt = KAD[i][nxt][0]
        if nxt == cur: break
        out.append(nxt)
    txt = tok.decode(out)
    return txt[len(prompt):] if txt.startswith(prompt) else txt

for s in ("The capital of India is", "def add(a, b):\n    return", "The sun rises in the"):
    try:
        print(f"  {s!r:<34} -> {jaal_banao(s, 12)[:70]}", flush=True)
    except Exception as e:
        print(f"  {s!r:<34} -> fail {type(e).__name__}", flush=True)
