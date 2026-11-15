#!/usr/bin/env python3
"""mool.py -- ORIGINAL Qwen se asli jawab (bina kisi badlav ke).

Do tarah se poochhte hain:
  A) SADA -- kachcha vaakya (jo hum pehle kar rahe the)
  B) CHAT -- Qwen ka asli dhancha (<|im_start|>...)

Isse pata chalega: gadbadi model me hai ya hamare poochhne ke tarike me.
"""
import json, math, struct, sys, time
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
print(f"  {len(W)} tensors | {sum(t.numel() for t in W.values()):,} param", flush=True)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T, off=0):
    pos = torch.arange(off, off + T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)

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

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
EOT = 151643

def banao(ids, naya=60, garam=0.0, dand=1.15):
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            lg = forward(out)[-1].clone()
            if dand > 1.0:
                for t in set(out[-30:]):
                    lg[t] = lg[t] / dand if lg[t] > 0 else lg[t] * dand
            if garam <= 0:
                nxt = int(torch.argmax(lg))
            else:
                nxt = int(torch.multinomial(torch.softmax(lg / garam, -1), 1))
            if nxt == EOT: break
            out.append(nxt)
    return out

SAWAAL = [
    "What is the capital of India?",
    "What is 2 + 2?",
    "Write a Python function to add two numbers.",
    "Which continent is India in?",
    "Where does the sun rise?",
]

print("\n" + "=" * 62, flush=True)
print("  A) SADA POOCHHNA (kachcha vaakya) -- jo hum pehle kar rahe the", flush=True)
print("=" * 62, flush=True)
for s in SAWAAL:
    ids = tok.encode(s).ids
    a = tok.decode(banao(ids, 30))
    print(f"\n[poochha] {s}", flush=True)
    print(f"[jawab ] {a[len(s):][:200] if a.startswith(s) else a[:200]}", flush=True)

print("\n" + "=" * 62, flush=True)
print("  B) CHAT DHANCHA -- Qwen ka asli tarika", flush=True)
print("=" * 62, flush=True)
SYS = "You are a helpful assistant."
for s in SAWAAL:
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
              f"<|im_start|>user\n{s}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tok.encode(prompt).ids
    out = banao(ids, 80, 0.7, 1.15)
    txt = tok.decode(out)
    jawab = txt[len(prompt):] if txt.startswith(prompt) else txt
    print(f"\n[poochha] {s}", flush=True)
    print(f"[jawab ] {jawab[:250]}", flush=True)

print("\n" + "=" * 62, flush=True)
print("  C) HINDI ME CHAT", flush=True)
print("=" * 62, flush=True)
for s in ("भारत की राजधानी क्या है?", "2 + 2 कितना होता है?"):
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
              f"<|im_start|>user\n{s}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tok.encode(prompt).ids
    out = banao(ids, 80, 0.7, 1.15)
    txt = tok.decode(out)
    jawab = txt[len(prompt):] if txt.startswith(prompt) else txt
    print(f"\n[poochha] {s}", flush=True)
    print(f"[jawab ] {jawab[:250]}", flush=True)
