#!/usr/bin/env python3
"""poora_dam.py -- model ki POORI kshamata jaga kar test karo.

Gadbadi: <|im_start|> ko saadharan text samajh liya tha.
Sudhar: vishesh token ki asli ID daalo.

Qwen ke vishesh token:
  <|endoftext|>  <|im_start|>  <|im_end|>
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

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")

# ---------- VIS HESH TOKEN KI ASLI ID DHOONDO ----------
print("\n=== VIS HESH TOKEN ===", flush=True)
SPECIAL = {}
for name, tid in tok.get_added_tokens_decoder().items():
    if tid.special or tid.content.startswith("<|"):
        SPECIAL[tid.content] = name
        print(f"  {tid.content!r:<20} id={name}", flush=True)

IM_START = SPECIAL.get("<|im_start|>")
IM_END = SPECIAL.get("<|im_end|>")
EOT = SPECIAL.get("<|endoftext|>")
print(f"\n  im_start={IM_START} im_end={IM_END} eot={EOT}", flush=True)
if IM_START is None:
    # manual guess
    for g in (151644, 151646):
        try:
            print(f"  guess {g}: {tok.decode([g])!r}", flush=True)
        except Exception: pass
    IM_START = 151644; IM_END = 151645; EOT = 151643

def enc_plain(s):
    return tok.encode(s).ids

def chat_ids(user, system="You are a helpful assistant."):
    """ASLI chat dhancha -- vishesh token ki asli ID ke saath"""
    ids = []
    if system:
        ids += [IM_START] + enc_plain("system\n" + system) + [IM_END] + enc_plain("\n")
    ids += [IM_START] + enc_plain("user\n" + user) + [IM_END] + enc_plain("\n")
    ids += [IM_START] + enc_plain("assistant\n")
    return ids

# jaanch: dhancha sahi bana?
test_ids = chat_ids("hi")
print(f"\n  chat dhancha (pehle 12): {test_ids[:12]}", flush=True)
print(f"  decode: {tok.decode(test_ids[:12])!r}", flush=True)

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

def forward(ids):
    T = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rmsnorm(x, W[p+"input_layernorm.weight"]), i, T)
        x = x + mlp(rmsnorm(x, W[p+"post_attention_layernorm.weight"]), i)
    x = rmsnorm(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

def banao(ids, naya=100, garam=0.7, dand=1.1):
    out = list(ids)
    with torch.no_grad():
        for _ in range(naya):
            lg = forward(out)[-1].clone()
            if dand > 1.0:
                for t in set(out[-40:]):
                    lg[t] = lg[t] / dand if lg[t] > 0 else lg[t] * dand
            if garam <= 0:
                nxt = int(torch.argmax(lg))
            else:
                nxt = int(torch.multinomial(torch.softmax(lg / garam, -1), 1))
            if nxt in (EOT, IM_END): break
            out.append(nxt)
    return out

print("\n" + "=" * 62, flush=True)
print("  ASLI CHAT DHANCHA -- poori kshamata", flush=True)
print("=" * 62, flush=True)
for s in ("What is the capital of India?",
          "What is 2 + 2?",
          "Write a Python function to add two numbers.",
          "Which continent is India in?",
          "Where does the sun rise?"):
    t0 = time.time()
    ids = chat_ids(s)
    out = banao(ids, 100, 0.7, 1.1)
    txt = tok.decode(out[len(ids):])
    print(f"\n[poochha] {s}", flush=True)
    print(f"[jawab ] {txt[:300]}", flush=True)
    print(f"         ({time.time()-t0:.1f}s)", flush=True)

print("\n" + "=" * 62, flush=True)
print("  HINDI ME", flush=True)
print("=" * 62, flush=True)
for s in ("भारत की राजधानी क्या है?", "2 + 2 कितना होता है?", "पाइथन में जोड़ने का फंक्शन लिखो।"):
    t0 = time.time()
    ids = chat_ids(s)
    out = banao(ids, 100, 0.7, 1.1)
    txt = tok.decode(out[len(ids):])
    print(f"\n[poochha] {s}", flush=True)
    print(f"[jawab ] {txt[:300]}", flush=True)
    print(f"         ({time.time()-t0:.1f}s)", flush=True)
