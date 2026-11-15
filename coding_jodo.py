#!/usr/bin/env python3
"""coding_jodo.py -- FIXED Qwen + hamari architecture, CODING par tulna.

Do model, dono me SUDHRA hua RoPE:
  A) QWEN      -- sirf Qwen (baseline)
  B) QWEN+JAAL -- Qwen + hamari jaal-smriti + pattern taalka

Naapo: coding ke jawab, dono me.
"""
import json, math, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
RS = 0.5
K = 8

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

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
SP = {t.content: i for i, t in tok.get_added_tokens_decoder().items()}
IM_START = SP.get("<|im_start|>", 151644)
IM_END = SP.get("<|im_end|>", 151645)
EOT = SP.get("<|endoftext|>", 151643)

# ---------- kosh se jaal ----------
import re
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
sents = []
for line in text.split("\n"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 4 <= len(ws) <= 60: sents.append(ws)
    if len(sents) >= 30000: break
edge = defaultdict(Counter)
for ws in sents:
    ids = tok.encode(" ".join(ws)).ids
    for x, y in zip(ids, ids[1:]): edge[x][y] += 1
nbrs = {x: [y for y, _ in c.most_common(K)] for x, c in edge.items()}
print(f"  kosh {len(sents):,} vaakya | jaal {len(edge):,} sel | kadiya {sum(len(c) for c in edge.values()):,}", flush=True)

# ---------- hamare hisse (shunya) ----------
P = {}
P["pat"] = torch.zeros(1, D)
for i in range(L):
    p = f"b.{i}."
    P[p+"n3w"] = torch.ones(D); P[p+"n3b"] = torch.zeros(D)
    P[p+"lq"] = torch.zeros(D, D)
    P[p+"lo"] = torch.zeros(D, D)
NP = sum(t.numel() for t in P.values())
print(f"  hamare param: {NP:,} ({NP/1e6:.1f}M)", flush=True)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
    """SUDHRA: pehla aadha + doosra aadha (Qwen/Llama tarika)"""
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

def jaal_smriti(x, i, nb):
    p = f"b.{i}."
    xx = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * P[p+"n3w"] + P[p+"n3b"]
    T = x.shape[0]
    q = (xx @ P[p+"lq"].T).view(T, H, hd).transpose(0, 1)
    mem = W["model.embed_tokens.weight"][nb].view(T, K, H, hd).permute(2, 0, 1, 3)
    sc = q[:, :, None, :] @ mem.transpose(-1, -2) / math.sqrt(hd)
    o = (sc.softmax(-1) @ mem).squeeze(2).transpose(0, 1).reshape(T, D)
    return o @ P[p+"lo"].T

def forward(ids, nb, use_jaal):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids] + P["pat"][0].view(1, D)
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
        if use_jaal:
            x = x + RS * jaal_smriti(x, i, nb)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

def chat_ids(user, system="You are a helpful assistant."):
    ids = [IM_START] + tok.encode("system\n" + system).ids + [IM_END] + tok.encode("\n").ids
    ids += [IM_START] + tok.encode("user\n" + user).ids + [IM_END] + tok.encode("\n").ids
    ids += [IM_START] + tok.encode("assistant\n").ids
    return ids

def banao(ids, use_jaal, naya=100, garam=0.7, dand=1.1):
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            T = len(out)
            nb = torch.zeros(T, K, dtype=torch.long)
            for t in range(T):
                for j, u in enumerate(nbrs.get(out[t], [])[:K]): nb[t, j] = u
            lg = forward(out, nb, use_jaal)[-1].clone()
            if dand > 1.0:
                for t in set(out[-40:]):
                    lg[t] = lg[t] / dand if lg[t] > 0 else lg[t] * dand
            nxt = int(torch.multinomial(torch.softmax(lg / garam, -1), 1)) if garam > 0 else int(torch.argmax(lg))
            if nxt in (EOT, IM_END): break
            out.append(nxt)
    return tok.decode(out[len(ids):])

SAWAAL = [
    "Write a Python function to add two numbers.",
    "Write a Python function that reverses a string.",
    "Write a Python function to check if a number is even.",
    "How do I read a file in Python?",
]
print("\n" + "=" * 64, flush=True)
print("  CODING TULNA -- Qwen vs Qwen+Jaal", flush=True)
print("=" * 64, flush=True)
for s in SAWAAL:
    ids = chat_ids(s)
    for use in (False, True):
        t0 = time.time()
        try:
            a = banao(ids, use, 110, 0.7, 1.1)
        except Exception as e:
            a = f"FAIL {type(e).__name__}: {str(e)[:60]}"
        tag = "QWEN+JAAL" if use else "QWEN     "
        print(f"\n[{tag}] {s}", flush=True)
        print(f"  {a[:280]}", flush=True)
        print(f"  ({time.time()-t0:.1f}s)", flush=True)

print(f"\nQwen: {sum(t.numel() for t in W.values()):,} | hamare: {NP:,}", flush=True)
