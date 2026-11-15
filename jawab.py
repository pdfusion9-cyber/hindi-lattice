#!/usr/bin/env python3
"""jawab.py -- hamare code me Qwen se asli jawab lo.

Sawaal poochho, jawab likhwa kar dekho.
Greedy + thoda sampling, dono.
"""
import json, math, struct, sys, time
import numpy as np
import torch

D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1_000_000.0; EPS = 1e-6
PATH = "m.safetensors"
torch.manual_seed(0)

with open(PATH, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n

def load(name):
    v = hdr[name]
    with open(PATH, "rb") as f:
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

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T, off=0):
    pos = torch.arange(off, off + T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin; o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)

def attn(x, i, T, off=0):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rope(q.view(T, H, hd), T, off).transpose(0, 1)
    k = rope(k.view(T, KVH, hd), T, off).transpose(0, 1)
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

def forward(ids, off=0):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T, off)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

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
print(f"lade {len(W)} tensors | param {sum(t.numel() for t in W.values()):,}", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
EOT = 151643


def banao(prompt, naya=40, garam=0.0):
    ids = tok.encode(prompt).ids
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            logits = forward(out)[-1]
            if garam <= 0:
                nxt = int(torch.argmax(logits))
            else:
                p = torch.softmax(logits / garam, -1)
                nxt = int(torch.multinomial(p, 1))
            if nxt == EOT: break
            out.append(nxt)
    return tok.decode(out)


print("\n" + "=" * 60, flush=True)
print("  HAMARE CODE ME QWEN -- ASLI JAWAB", flush=True)
print("=" * 60, flush=True)

SAWAAL = [
    "The capital of India is",
    "Q: What is 2 + 2?\nA:",
    "Write a Python function to add two numbers:\n\ndef add(a, b):",
    "The sun rises in the",
    "India is a country in",
]
for s in SAWAAL:
    t0 = time.time()
    try:
        a = banao(s, 32, 0.0)
        print(f"\n[poochha] {s}", flush=True)
        print(f"[jawab ] {a[len(s):][:300] if a.startswith(s) else a[:300]}", flush=True)
        print(f"         ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"\n[poochha] {s}\n[fail  ] {type(e).__name__}: {str(e)[:80]}", flush=True)

print("\n" + "=" * 60, flush=True)
print("  THODA GARAM (0.7) -- alag jawab", flush=True)
print("=" * 60, flush=True)
for s in ("Once upon a time", "The best way to learn programming is"):
    try:
        a = banao(s, 40, 0.7)
        print(f"\n[poochha] {s}", flush=True)
        print(f"[jawab ] {a[len(s):][:300] if a.startswith(s) else a[:300]}", flush=True)
    except Exception as e:
        print(f"fail {type(e).__name__}", flush=True)
