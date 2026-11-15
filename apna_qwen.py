#!/usr/bin/env python3
"""apna_qwen.py -- Qwen ki ARCHITECTURE apne code me banao, gyan sahit.

Yeh transfer nahi hai -- hum uski banawat ko apne haath se likhte hain,
phir uske wazan daal kar chalate hain. Agar sahi chale, matlab
architecture hamare kabze me hai.

Qwen2 ke hisse:
  RMSNorm, RoPE (theta 1e6), GQA (14 Q / 2 KV), SwiGLU
"""
import json, math, struct, sys
import numpy as np
import torch

D = 896; L = 24; H = 14; KVH = 2; FF = 4864; V = 151936
hd = D // H
THETA = 1_000_000.0
EPS = 1e-6
PATH = "m.safetensors"

with open(PATH, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

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

# ---------- hamari apni likhi parat ----------
def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
    """x: (T, H, hd) -- ghoomta hua sthan"""
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)                       # (T, hd/2)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return out

def attn(x, i, T):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rope(q.view(T, H, hd), T).transpose(0, 1)      # (H,T,hd)
    k = rope(k.view(T, KVH, hd), T).transpose(0, 1)    # (KVH,T,hd)
    v = v.view(T, KVH, hd).transpose(0, 1)
    # GQA: KVH -> H (har KV group ko dohrao)
    rep = H // KVH
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask[None], -1e9)
    a = sc.softmax(-1)
    o = (a @ v).transpose(0, 1).reshape(T, D)
    return o @ W[p+"o_proj.weight"].T

def mlp(x, i):
    p = f"model.layers.{i}.mlp."
    g = torch.nn.functional.silu(x @ W[p+"gate_proj.weight"].T)
    u = x @ W[p+"up_proj.weight"].T
    return (g * u) @ W[p+"down_proj.weight"].T

def forward(ids):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

# ---------- wazan ek-ek karke laao ----------
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
for j, nm in enumerate(need):
    W[nm] = load(nm)
    if j % 20 == 0: print(f"  {j}/{len(need)}", flush=True)
print(f"lade: {len(W)} tensors", flush=True)
tot = sum(t.numel() for t in W.values())
print(f"param: {tot:,}", flush=True)

# ---------- tokenizer ----------
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

# ---------- chalao ----------
print("\n=== HAMARE CODE ME QWEN ===")
for pr in ("भारत की राजधानी", "The capital of India is", "def ", "नमस्ते"):
    try:
        ids = tok.encode(pr).ids
        with torch.no_grad():
            logits = forward(ids)[-1]
        top = torch.topk(logits, 6).indices.tolist()
        words = [tok.decode([i]) for i in top]
        print(f"  {pr!r:<28} -> {words}")
    except Exception as e:
        print(f"  {pr!r:<28} -> fail {type(e).__name__}: {str(e)[:60]}")

# ---------- saaf ginti ----------
print(f"\nhamara dhancha: {D} dim, {L} parat, {H} head, {KVH} KV")
print(f"wazan: {tot*4/1e6:.0f} MB (fp32)")
