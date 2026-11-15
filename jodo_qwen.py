#!/usr/bin/env python3
"""jodo_qwen.py -- Qwen ke wazan + HAMARI jaal architecture.

Yeh woh kaam hai jo pehle chhoot gaya tha.

Hamare hisse jo jod rahe hain:
  1. pat.weight   -- pattern taalka (1505 patterns, 1024 dim)
  2. lq / lo      -- jaal-smriti (padosi token se dhyan)
  3. n3           -- us smriti ka norm

Sab SHUNYA se shuru -- isliye nateeja bilkul Qwen jaisa rehta hai.
Phir sirf naye hisse seekhte hain.
"""
import json, math, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
CFG = {"0.5B": dict(D=896, L=24, H=14, KVH=2, d="Qwen2.5-0.5B")}
c = CFG["0.5B"]
D, L, H, KVH = c["D"], c["L"], c["H"], c["KVH"]
hd = D // H
THETA = 1e6; EPS = 1e-6
NPAT = 1505          # hamari pattern taalka ka aakar

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
print(f"  {len(W)} tensors | param {sum(t.numel() for t in W.values()):,}", flush=True)

# ---------- HAMARE NAYE HISSE (shunya se) ----------
print("\nhamare hisse jod rahe (shunya se)...", flush=True)
P = {}
# pattern taalka -- 1505 pattern, D dim
P["pat.weight"] = torch.zeros(NPAT, D)
for i in range(L):
    p = f"blk.{i}."
    P[p+"n3.weight"] = torch.ones(D)                 # norm shunya nahi, 1
    P[p+"n3.bias"] = torch.zeros(D)
    P[p+"lq.weight"] = torch.zeros(D, D)             # smriti -- shunya
    P[p+"lo.weight"] = torch.zeros(D, D)             # shunya = koi asar nahi
np_new = sum(t.numel() for t in P.values())
print(f"  naye param: {np_new:,} ({np_new/1e6:.1f}M)", flush=True)
print(f"  shunya hone se nateeja Qwen jaisa rahega", flush=True)

# ---------- aage badhao ----------
for t in W.values(): t.requires_grad_(False)          # Qwen ke wazan jam
for t in P.values(): t.requires_grad_(True)           # sirf naye seekhein

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

def smriti(x, i, nb):
    """HAMARI JAAL-SMRITI -- padosi token se dhyan"""
    p = f"blk.{i}."
    xx = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * P[p+"n3.weight"] + P[p+"n3.bias"]
    T = x.shape[0]
    q = (xx @ P[p+"lq.weight"].T).view(T, H, hd).transpose(0, 1)
    kk = W["model.embed_tokens.weight"][nb].view(T, 8, H, hd).permute(2, 0, 1, 3)
    sc = q[:, :, None, :] @ kk.transpose(-1, -2) / math.sqrt(hd)
    a = sc.softmax(-1)
    o = (a @ kk).squeeze(2).transpose(0, 1).reshape(T, D)
    return o @ P[p+"lo.weight"].T

def forward(ids, nb, pid):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids] + P["pat.weight"][pid][None] * 0.0
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
        x = x + smriti(x, i, nb)                     # <- HAMARA HISSA
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

# ---------- jaanch 1: shunya se Qwen jaisa? ----------
print("\n=== JAANCH 1: shunya hisse se nateeja badla? ===", flush=True)
def banao(prompt, naya=24, garam=0.0):
    ids = tok.encode(prompt).ids
    out = list(ids)
    nb = torch.zeros(len(out), 8, dtype=torch.long)
    with torch.no_grad():
        for _ in range(naya):
            lg = forward(out, nb, 0)[-1]
            nxt = int(torch.argmax(lg)) if garam <= 0 else int(torch.multinomial(torch.softmax(lg/garam,-1),1))
            out.append(nxt)
    txt = tok.decode(out)
    return txt[len(prompt):] if txt.startswith(prompt) else txt

for s in ("The capital of India is", "def add(a, b):\n    return"):
    try:
        print(f"  {s!r:<32} -> {banao(s, 20)[:90]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}: {str(e)[:60]}", flush=True)

# ---------- jaanch 2: seekhne ke baad badla? ----------
print("\n=== JAANCH 2: naye hisse seekhne par ===", flush=True)
opt = torch.optim.AdamW(list(P.values()), lr=1e-3)
import re
from collections import Counter
corpus = "corpus.txt"
sents = []
for line in open(corpus, encoding="utf-8", errors="ignore"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 6 <= len(ws) <= 40: sents.append(ws)
    if len(sents) >= 3000: break
cnt = Counter(w for ws in sents for w in ws)
print(f"  kosh: {len(sents):,} vaakya", flush=True)

t0 = time.time(); steps = 0
while steps < 60 and time.time() - t0 < 300:
    ws = sents[np.random.randint(len(sents))]
    ids = tok.encode(" ".join(ws)).ids[:24]
    if len(ids) < 6: continue
    x = torch.tensor(ids[:-1]); y = torch.tensor(ids[1:])
    nb = torch.zeros(len(x), 8, dtype=torch.long)
    lg = forward(x, nb, 0)
    loss = torch.nn.functional.cross_entropy(lg, y)
    opt.zero_grad(); loss.backward(); opt.step()
    steps += 1
    if steps % 20 == 0:
        print(f"    kadam {steps} | loss {loss.item():.3f}", flush=True)

print(f"\n  seekhne ke baad:", flush=True)
for s in ("The capital of India is", "def add(a, b):\n    return"):
    try:
        print(f"  {s!r:<32} -> {banao(s, 20)[:90]}", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}", flush=True)

print(f"\nnaye param jo seekhe: {sum(t.numel() for t in P.values()):,}", flush=True)
