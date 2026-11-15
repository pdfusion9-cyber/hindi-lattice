#!/usr/bin/env python3
"""code_jaal3.py -- TF-IDF + stop-word + kunci saamanya karan.

Sudhar (aakar nahi badhta):
  1. TF-IDF      -- mahatvapurn shabd ka bhaar zyada
  2. STOP WORD   -- 'write', 'python', 'function' hatao (sab me hain)
  3. SAMAANYA    -- add/adding/added -> add | reverse/reversing -> revers
  4. COSINE      -- bhaar wale vector se milaan

Naap: sahi jawab (%)
"""
import json, math, os, re, struct, sys, time, zlib
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

def banao(ids, naya=140, garam=0.5, dand=1.15):
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
GEN = [
    "Write a Python function to add two numbers.",
    "Write a Python function to subtract two numbers.",
    "Write a Python function to multiply two numbers.",
    "Write a Python function to divide two numbers.",
    "Write a Python function to reverse a string.",
    "Write a Python function to check if a number is even.",
    "Write a Python function to check if a number is odd.",
    "Write a Python function to calculate the factorial of a number.",
    "Write a Python function to check if a number is prime.",
    "Write a Python function to find the sum of a list.",
    "Write a Python function to count vowels in a string.",
    "Write a Python function to sort a list.",
    "Write a Python function to remove duplicates from a list.",
    "How do I read a file in Python?",
    "How do I write to a file in Python?",
    "How do I create a dictionary in Python?",
    "How do I handle exceptions in Python?",
    "How do I join a list of strings in Python?",
]

print(f"\n=== {len(GEN)} JAWAB NIKAAL RAHE ===", flush=True)
t0 = time.time()
DATA = []
for j, s in enumerate(GEN):
    try:
        a = banao(chat(s), 140, 0.5, 1.15)
        DATA.append((s, a))
        print(f"  [{j+1}/{len(GEN)}] {s[:45]}... ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  fail {type(e).__name__}", flush=True)

def nikalo_code(txt):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    if m: return m.group(1).strip()
    # bina ``` ke bhi
    lines = [l for l in txt.split("\n") if re.match(r"\s*(def |class |for |if |with |return |import |from )", l)]
    return "\n".join(lines).strip() if len(lines) >= 2 else ""

PAT = [(s, nikalo_code(a)) for s, a in DATA]
PAT = [(s, c) for s, c in PAT if c]
print(f"\n  code mile: {len(PAT)}/{len(DATA)}", flush=True)

# ---------- 1. SAMAANYA KARAN (normalize) ----------
STOP = set("""write a an the how do i in to of and for is are use using with python function
create make get give show me please can you code program script method return""".split())

SUFFIX = [("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("e", "")]
SYN = {"addition": "add", "adding": "add", "added": "add", "sum": "add", "total": "add",
       "subtraction": "subtract", "subtracting": "subtract", "minus": "subtract", "difference": "subtract",
       "multiplication": "multiply", "multiplying": "multiply", "product": "multiply", "times": "multiply",
       "division": "divide", "dividing": "divide", "quotient": "divide",
       "reversing": "reverse", "reversed": "reverse", "backward": "reverse",
       "even": "even", "odd": "odd", "prime": "prime", "factorial": "factorial",
       "vowel": "vowel", "vowels": "vowel", "duplicate": "duplicate", "duplicates": "duplicate",
       "unique": "duplicate", "sorting": "sort", "sorted": "sort",
       "reading": "read", "reading": "read", "writing": "write", "written": "write",
       "exception": "exception", "error": "exception", "errors": "exception", "try": "exception",
       "dictionary": "dictionary", "dict": "dictionary", "map": "dictionary",
       "joining": "join", "concatenate": "join", "concat": "join", "merge": "join",
       "string": "string", "strings": "string", "text": "string", "list": "list", "array": "list",
       "number": "number", "numbers": "number", "integer": "number", "int": "number",
       "file": "file", "files": "file"}

def samaanya(w):
    w = w.lower()
    if w in SYN: return SYN[w]
    for suf, rep in SUFFIX:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:len(w)-len(suf)] + rep
    return w

def shabd(q):
    ws = re.findall(r"[a-z]+", q.lower())
    out = []
    for w in ws:
        if w in STOP: continue
        n = samaanya(w)
        if n not in STOP and len(n) > 1: out.append(n)
    return out

# ---------- 2. TF-IDF ----------
docs = [shabd(s) for s, _ in PAT]
df = Counter()
for d in docs:
    for w in set(d): df[w] += 1
N = len(docs)
idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}

def vector(q):
    ws = shabd(q)
    tf = Counter(ws)
    v = {}
    for w, c in tf.items():
        v[w] = (1 + math.log(c)) * idf.get(w, math.log(N + 1) + 1.0)
    # lambai saamanya
    nrm = math.sqrt(sum(x*x for x in v.values())) or 1.0
    return {w: x/nrm for w, x in v.items()}

JAAL = [(vector(s), c) for s, c in PAT]

def jawab(q):
    vq = vector(q)
    best, bs = None, 0.0
    for vp, code in JAAL:
        # cosine
        s = sum(x * vp.get(w, 0.0) for w, x in vq.items())
        if s > bs: bs, best = s, code
    return (best or ""), bs

# ---------- aakar ----------
blob = json.dumps([c for _, c in PAT], ensure_ascii=False)
comp = zlib.compress(blob.encode("utf-8"), 9)
print(f"\n=== AAKAR ===", flush=True)
print(f"  Qwen fp32:   {NP*4/1e6:>8.0f} MB", flush=True)
print(f"  JAAL (zip):  {len(comp)/1e6:>8.4f} MB", flush=True)
print(f"  ghatav:      {NP*4/len(comp):>8.0f}x", flush=True)
print(f"  patterns:    {len(JAAL)}", flush=True)
print(f"  shabdakosh:  {len(idf)} shabd", flush=True)

# ---------- TEST (naye shabdon me) ----------
TEST = [
    ("Write a function that adds two numbers.", "add", "subtract"),
    ("Create a Python function to reverse a string.", "revers", "sort"),
    ("How can I check whether a number is prime?", "prime", "even"),
    ("Write code to subtract two numbers.", "subtract", "add"),
    ("How do I read from a file?", "read", "write"),
    ("Write a function to find even numbers.", "even", "odd"),
    ("How to sort a list in Python?", "sort", "revers"),
    ("Write a function to count vowels.", "vowel", "sort"),
]
print(f"\n=== TEST ({len(TEST)} naye sawaal) ===", flush=True)
sahi = 0
for q, chahiye, nahi in TEST:
    got, sc = jawab(q)
    ok = (chahiye in got.lower()) and (nahi not in got.lower())
    sahi += ok
    print(f"\n[{'SAHI ' if ok else 'GALAT'}] {q}  (milaan {sc:.2f})", flush=True)
    print(f"  {got[:170].replace(chr(10),' | ')}", flush=True)
print(f"\n  SAHI: {sahi}/{len(TEST)} = {sahi/len(TEST)*100:.0f}%", flush=True)
