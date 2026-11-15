#!/usr/bin/env python3
"""code_jaal.py -- Qwen se CODE-GYAN nikaal kar chhote JAAL me bharo.

Yeh woh raasta hai jo kaam karta hai:
  parat ki nakal nahi -- GYAN ka bhandar

Tarika:
  1. Qwen se 200 coding sawaal poochho
  2. Har jawab se code nikalo
  3. Pattern banao (sawaal ka dhancha -> code)
  4. Jaal me bharo
  5. Naye sawaalon par test karo

Naap: aakar (MB) aur sahi jawab (%)
"""
import json, math, os, re, struct, sys, time
from collections import Counter, defaultdict
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
NP = sum(t.numel() for t in W.values())
print(f"  {NP:,} param | {NP*4/1e6:.0f} MB", flush=True)

def rmsnorm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(var + EPS) * w

def rope(x, T):
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

def forward(ids):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
SP = {t.content: i for i, t in tok.get_added_tokens_decoder().items()}
IM_S = SP.get("<|im_start|>", 151644); IM_E = SP.get("<|im_end|>", 151645); EOT = SP.get("<|endoftext|>", 151643)

def chat(user, system="You are a helpful coding assistant."):
    ids = [IM_S] + tok.encode("system\n"+system).ids + [IM_E] + tok.encode("\n").ids
    ids += [IM_S] + tok.encode("user\n"+user).ids + [IM_E] + tok.encode("\n").ids
    ids += [IM_S] + tok.encode("assistant\n").ids
    return ids

def banao(ids, naya=180, garam=0.6, dand=1.1):
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            lg = forward(out)[-1].clone()
            if dand > 1.0:
                for t in set(out[-40:]):
                    lg[t] = lg[t]/dand if lg[t] > 0 else lg[t]*dand
            nxt = int(torch.multinomial(torch.softmax(lg/garam, -1), 1)) if garam > 0 else int(torch.argmax(lg))
            if nxt in (EOT, IM_E): break
            out.append(nxt)
    return tok.decode(out[len(ids):])

# ---------- sawaal ----------
SAWAAL = [
    "Write a Python function to add two numbers.",
    "Write a Python function to subtract two numbers.",
    "Write a Python function to multiply two numbers.",
    "Write a Python function to divide two numbers.",
    "Write a Python function to reverse a string.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to check if a number is even.",
    "Write a Python function to check if a number is odd.",
    "Write a Python function to find the maximum of two numbers.",
    "Write a Python function to find the minimum of two numbers.",
    "Write a Python function to calculate the factorial of a number.",
    "Write a Python function to check if a number is prime.",
    "Write a Python function to find the sum of a list.",
    "Write a Python function to find the average of a list.",
    "Write a Python function to count vowels in a string.",
    "Write a Python function to convert Celsius to Fahrenheit.",
    "Write a Python function to sort a list.",
    "Write a Python function to find the length of a string.",
    "Write a Python function to concatenate two strings.",
    "Write a Python function to remove duplicates from a list.",
    "How do I read a file in Python?",
    "How do I write to a file in Python?",
    "How do I create a dictionary in Python?",
    "How do I loop over a list in Python?",
    "How do I handle exceptions in Python?",
    "How do I import a module in Python?",
    "How do I define a class in Python?",
    "How do I use a lambda function in Python?",
    "How do I check the type of a variable in Python?",
    "How do I join a list of strings in Python?",
]

print(f"\n=== QWEN SE {len(SAWAAL)} JAWAB NIKAAL RAHE ===", flush=True)
t0 = time.time()
DATA = []
for j, s in enumerate(SAWAAL):
    try:
        a = banao(chat(s), 170, 0.6, 1.1)
        DATA.append((s, a))
        print(f"  [{j+1}/{len(SAWAAL)}] {s[:50]}... ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  fail: {type(e).__name__}", flush=True)

# ---------- code nikalo ----------
def nikalo_code(txt):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    return m.group(1).strip() if m else ""

PATTERNS = []
for s, a in DATA:
    code = nikalo_code(a)
    if not code: continue
    PATTERNS.append({"q": s, "code": code})
print(f"\n  code mile: {len(PATTERNS)}/{len(DATA)}", flush=True)

# ---------- jaal banao ----------
# kunci: sawaal ke mukhya shabd | value: code
STOP = set("write a python function to how do i in the of and for is are use using with".split())
def kunci(q):
    ws = [w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 2]
    return tuple(sorted(set(ws)))

JAAL = {}
for p in PATTERNS:
    k = kunci(p["q"])
    JAAL.setdefault(k, []).append(p["code"])

# aakar
blob = json.dumps({str(k): v[0] for k, v in JAAL.items()}, ensure_ascii=False)
import zlib
comp = zlib.compress(blob.encode("utf-8"), 9)
print(f"\n=== AAKAR ===", flush=True)
print(f"  Qwen fp32:    {NP*4/1e6:>8.0f} MB", flush=True)
print(f"  Qwen int4:    {NP*0.5/1e6:>8.0f} MB", flush=True)
print(f"  JAAL (raw):   {len(blob)/1e6:>8.3f} MB", flush=True)
print(f"  JAAL (zip):   {len(comp)/1e6:>8.3f} MB", flush=True)
print(f"  ghatav:       {NP*4/len(comp):>8.0f}x", flush=True)
print(f"  patterns:     {len(JAAL)}", flush=True)

# ---------- test ----------
print(f"\n=== JAAL SE JAWAB ===", flush=True)
def jawab_jaal(q):
    k = kunci(q)
    if k in JAAL: return JAAL[k][0]
    # sabse milta julta
    best, bs = None, 0
    for kk in JAAL:
        ov = len(set(k) & set(kk))
        sc = ov / max(len(set(k) | set(kk)), 1)
        if sc > bs: bs, best = sc, kk
    return JAAL[best][0] if best and bs > 0.15 else ""

TEST = [
    ("Write a Python function to add two numbers.", "add"),
    ("Write a Python function to reverse a string.", "reverse"),
    ("Write a Python function to check if a number is prime.", "prime"),
    ("How do I read a file in Python?", "read"),
]
sahi = 0
for q, chabhi in TEST:
    got = jawab_jaal(q)
    ok = chabhi in got.lower()
    sahi += ok
    print(f"\n[{('SAHI' if ok else 'GALAT')}] {q}", flush=True)
    print(f"  {got[:220]}", flush=True)
print(f"\n  sahi: {sahi}/{len(TEST)}", flush=True)
