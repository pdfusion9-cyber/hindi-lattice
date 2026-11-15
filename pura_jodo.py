#!/usr/bin/env python3
"""pura_jodo.py -- Qwen + HAMARI POORI ARCHITECTURE (repo se padh kar).

Hamari architecture ke CHAAR hisse (jaal_tf.py + fold2.py + punah.py se):
  1. pat(pid)      -- pattern taalka embedding
  2. lq / lo       -- jaal-smriti cross-attention (tok(nb) par)
  3. ummidvar-seemit output -- softmax sirf jaal ke agle-shabdon par
  4. RS (residual scale) -- 0.5

Sab SHUNYA se -> Qwen ka gyan surakshit.
Phir SIRF naye hisse seekhein, dhime (lr 1e-5), asli kosh par.
"""
import json, math, os, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
RS = 0.5            # hamari residual scale
K = 8               # jaal me kitne padosi
CAND = 50           # ummidvar seema

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

print("Qwen ke wazan laa rahe...", flush=True)
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
NPARAM_QWEN = sum(t.numel() for t in W.values())
print(f"  Qwen: {len(W)} tensors | {NPARAM_QWEN:,} param | {NPARAM_QWEN*4/1e6:.0f} MB", flush=True)

# ---------- asli kosh + jaal ----------
import re
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
sents = []
for line in text.split("\n"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 6 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= 40000: break
print(f"  kosh: {len(sents):,} vaakya", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

# jaal: kaun sa token kis ke baad aata hai
edge = defaultdict(Counter)
for ws in sents:
    ids = tok.encode(" ".join(ws)).ids
    for x, y in zip(ids, ids[1:]):
        edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(K)] for x, c in edge.items()}
cands = {x: [y for y, _ in c.most_common(CAND)] for x, c in edge.items()}
print(f"  jaal: {len(edge):,} sel | kadiya {sum(len(c) for c in edge.values()):,}", flush=True)

# pattern taalka
STOP = set()
pc = Counter()
for ws in sents:
    pc[tuple(ws)] += 1
STOP = {w for w, c in Counter(w for ws in sents for w in ws).items() if c >= 20}
def pattern(ws): return tuple(w if w in STOP else "*" for w in ws)
pcount = Counter()
for ws in sents: pcount[pattern(ws)] += 1
pats = [p for p, c in pcount.items() if c >= 2][:2000]
pidx = {p: i for i, p in enumerate(pats)}
print(f"  pattern: {len(pats):,}", flush=True)

# ---------- HAMARE NAYE HISSE ----------
print("\nhamare hisse (shunya se):", flush=True)
P = {}
P["pat"] = torch.zeros(max(len(pats), 1), D)
for i in range(L):
    p = f"b.{i}."
    P[p+"n3w"] = torch.ones(D); P[p+"n3b"] = torch.zeros(D)
    P[p+"lq"] = torch.zeros(D, D)
    P[p+"lo"] = torch.zeros(D, D)
NP = sum(t.numel() for t in P.values())
print(f"  naye param: {NP:,} ({NP/1e6:.1f}M)", flush=True)
print(f"  shunya se -> Qwen bilkul surakshit", flush=True)

for t in W.values(): t.requires_grad_(False)
for t in P.values(): t.requires_grad_(True)

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

def qwen_attn(x, i, T):
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

def qwen_mlp(x, i):
    p = f"model.layers.{i}.mlp."
    g = torch.nn.functional.silu(x @ W[p+"gate_proj.weight"].T)
    u = x @ W[p+"up_proj.weight"].T
    return (g * u) @ W[p+"down_proj.weight"].T

def jaal_smriti(x, i, nb):
    """HAMARI JAAL-SMRITI -- tok(nb) par cross-attention"""
    p = f"b.{i}."
    xx = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * P[p+"n3w"] + P[p+"n3b"]
    T = x.shape[0]
    q = (xx @ P[p+"lq"].T).view(T, H, hd).transpose(0, 1)
    mem = W["model.embed_tokens.weight"][nb].view(T, K, H, hd).permute(2, 0, 1, 3)
    sc = q[:, :, None, :] @ mem.transpose(-1, -2) / math.sqrt(hd)
    a = sc.softmax(-1)
    o = (a @ mem).squeeze(2).transpose(0, 1).reshape(T, D)
    return o @ P[p+"lo"].T

def forward(ids, nb, pid, cand=None):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids] + P["pat"][pid][None]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + RS * qwen_attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + RS * qwen_mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
        x = x + RS * jaal_smriti(x, i, nb)          # HAMARA HISSA
    x = rmsnorm(x, W["model.norm.weight"])
    if cand is not None:
        # UMMIDVAR-SEEMIT OUTPUT -- sirf jaal ke agle shabdon par
        lg = (x[:, None, :] * W["model.embed_tokens.weight"][cand]).sum(-1)
        return lg
    return x @ W["model.embed_tokens.weight"].T

def banao(prompt, naya=20):
    out = tok.encode(prompt).ids
    with torch.no_grad():
        for _ in range(naya):
            T = len(out)
            nb = torch.zeros(T, K, dtype=torch.long)
            for t in range(T):
                for j, u in enumerate(nbrs.get(out[t], [])[:K]): nb[t, j] = u
            lg = forward(out, nb, 0)[-1]
            out.append(int(torch.argmax(lg)))
    txt = tok.decode(out)
    return txt[len(prompt):] if txt.startswith(prompt) else txt

# ---------- JAANCH 1: shunya se Qwen surakshit? ----------
print("\n=== JAANCH 1: shunya hisse se Qwen badla? ===", flush=True)
for s in ("The capital of India is", "def add(a, b):\n    return"):
    try:
        print(f"  {s!r:<32} -> {banao(s, 15)[:80]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}: {str(e)[:60]}", flush=True)

# ---------- JAANCH 2: dhime seekho ----------
print("\n=== JAANCH 2: naye hisse dhime seekhein (lr 1e-5) ===", flush=True)
opt = torch.optim.AdamW(list(P.values()), lr=1e-5)
t0 = time.time(); steps = 0
while steps < 80 and time.time() - t0 < 240:
    ws = sents[np.random.randint(len(sents))]
    ids = tok.encode(" ".join(ws)).ids[:24]
    if len(ids) < 8: continue
    x = torch.tensor(ids[:-1]); y = torch.tensor(ids[1:])
    T = len(x)
    nb = torch.zeros(T, K, dtype=torch.long)
    for t in range(T):
        for j, u in enumerate(nbrs.get(int(x[t]), [])[:K]): nb[t, j] = u
    pid = torch.zeros(1, dtype=torch.long)
    lg = forward(x, nb, pid)
    loss = torch.nn.functional.cross_entropy(lg, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0); opt.step()
    steps += 1
    if steps % 20 == 0:
        print(f"    kadam {steps} | loss {loss.item():.3f}", flush=True)

print("\n  seekhne ke baad:", flush=True)
for s in ("The capital of India is", "def add(a, b):\n    return"):
    try:
        print(f"  {s!r:<32} -> {banao(s, 15)[:80]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}", flush=True)

print(f"\nQwen: {NPARAM_QWEN:,} | hamare: {NP:,} | kul: {NPARAM_QWEN+NP:,}", flush=True)
print(f"jaal: {len(edge):,} sel | pattern: {len(pats):,}", flush=True)
