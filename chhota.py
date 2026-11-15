#!/usr/bin/env python3
"""chhota.py -- Qwen ko kitna chhota kar sakte hain, gyan bacha kar?

Test:
  - 24 parat (poora)      -> baseline
  - 18, 12, 6 parat       -> kitna gyan bachta hai
  - int4                  -> aakar kitna ghatta hai

Naap: perplexity (kam = behtar)
"""
import json, math, struct, sys, time
import numpy as np
import torch

D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1_000_000.0; EPS = 1e-6
PATH = "m.safetensors"

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

def rope(x, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin; o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)

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

def forward(ids, layers):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in layers:
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
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
print(f"lade {len(W)} tensors", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

TEXT = ("Artificial intelligence is transforming the way we work and live. "
        "Machine learning models can understand language, generate text, and solve problems. "
        "The capital of India is New Delhi, and the capital of Japan is Tokyo. "
        "Python is a popular programming language used for data science and web development.")
ids = tok.encode(TEXT).ids[:96]
print(f"jaanch ka text: {len(ids)} token", flush=True)

def ppl(layers):
    """perplexity -- kam = behtar"""
    tot = 0.0; cnt = 0
    with torch.no_grad():
        for st in range(0, len(ids) - 16, 8):
            seg = ids[st:st+16]
            if len(seg) < 4: break
            logits = forward(seg[:-1], layers)
            tgt = torch.tensor(seg[1:])
            lp = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
            tot += float(lp); cnt += len(tgt)
    return math.exp(tot / max(cnt, 1))

print(f"\n{'parat':<14} {'param':>10} {'perplexity':>12} {'samay':>8}")
print("-" * 48)
res = {}
for naam, layers in (("24 (poora)", list(range(24))),
                     ("18 (har 4 me 3)", list(range(0, 24, 4)) + list(range(1, 24, 4)) + list(range(2, 24, 4))),
                     ("12 (aadha)", list(range(0, 24, 2))),
                     ("6 (chauthai)", list(range(0, 24, 4)))):
    t0 = time.time()
    try:
        p = ppl(layers)
        dt = time.time() - t0
        nparam = sum(W[f"model.layers.{i}.{x}"].numel() for i in layers
                     for x in ("self_attn.q_proj.weight","self_attn.k_proj.weight",
                               "self_attn.v_proj.weight","self_attn.o_proj.weight",
                               "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"))
        nparam += W["model.embed_tokens.weight"].numel()
        res[naam] = (p, nparam)
        print(f"{naam:<14} {nparam/1e6:>9.0f}M {p:>12.2f} {dt:>7.1f}s", flush=True)
    except Exception as e:
        print(f"{naam:<14} fail {type(e).__name__}: {str(e)[:40]}", flush=True)

print("\n=== NATIJA ===")
if res:
    base = res.get("24 (poora)", (None, None))[0]
    for naam, (p, np_) in res.items():
        if base:
            print(f"  {naam:<14} ppl {p:>7.2f} ({p-base:+.2f}) | {np_/1e6:.0f}M")
    print(f"\n  int4 me: {res['24 (poora)'][1]*0.5/1e6:.0f} MB (8x chhota)")
