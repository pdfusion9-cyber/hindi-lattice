#!/usr/bin/env python3
"""mlp_jaal.py -- Qwen ke MLP se SEEDHA gyan nikaalo (jawab banaye bina).

Kyun tez: koi token-by-token generation nahi. Ek hi pass me poora gyan.

Tathya: har MLP neuron ek KADI hai
    kunci  = W_gate ki pankti (896)   -- kaun sa pattern isko jagata hai
    maan   = W_down ka stambh (896)   -- wo kya likhta hai
    bhaar  = W_up ki pankti           -- kitna likhta hai

Naap:
  1. kitne neuron asli path par jagte hain (sparsity)
  2. kaun se neuron sabse zyada kaam ke
  3. chhote jaal me kitna gyan bachta hai
"""
import json, math, os, struct, sys, time
import numpy as np
import torch

torch.manual_seed(0)
D = 896; L = 24; H = 14; KVH = 2; hd = D // H
THETA = 1e6; EPS = 1e-6
FF = 4864

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

# ---------- 1. KITNE NEURON JAGTE HAIN ----------
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:3000]
print(f"\ntext: {len(ids_all)} token", flush=True)

print("\n=== 1) NEURON KI JAAG (asli path par) ===", flush=True)
t0 = time.time()
act_count = np.zeros((L, FF), dtype=np.int32)
total = 0
with torch.no_grad():
    for st in range(0, len(ids_all) - 64, 64):
        seg = torch.tensor(ids_all[st:st+64])
        x = W["model.embed_tokens.weight"][seg]
        for i in range(L):
            p = f"model.layers.{i}."
            # parat ka pehla hissa
            xx = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * W[p+"input_layernorm.weight"]
            q = xx @ W[p+"self_attn.q_proj.weight"].T + W[p+"self_attn.q_proj.bias"]
            k = xx @ W[p+"self_attn.k_proj.weight"].T + W[p+"self_attn.k_proj.bias"]
            v = xx @ W[p+"self_attn.v_proj.weight"].T + W[p+"self_attn.v_proj.bias"]
            T = len(seg)
            pos = torch.arange(T, dtype=torch.float32)
            inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
            ang = torch.outer(pos, inv)
            cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
            half = hd // 2
            def rp(z):
                z1 = z[..., :half]; z2 = z[..., half:]
                return torch.cat([z1*cos - z2*sin, z1*sin + z2*cos], dim=-1)
            q = rp(q.view(T, H, hd)).transpose(0, 1)
            k = rp(k.view(T, KVH, hd)).transpose(0, 1)
            v = v.view(T, KVH, hd).transpose(0, 1)
            rep = H // KVH
            k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
            sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            sc = sc.masked_fill(mask[None], -1e9)
            o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, D)
            x = x + o @ W[p+"self_attn.o_proj.weight"].T
            # MLP ki jaag
            xn = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * W[p+"post_attention_layernorm.weight"]
            g = torch.nn.functional.silu(xn @ W[p+"mlp.gate_proj.weight"].T)
            u = xn @ W[p+"mlp.up_proj.weight"].T
            hid = (g * u).abs()
            act_count[i] += (hid > hid.mean(-1, keepdim=True) * 0.1).sum(0).numpy()
            x = x + (g * u) @ W[p+"mlp.down_proj.weight"].T
        total += len(seg)
        if total % 960 == 0:
            print(f"  {total}/{len(ids_all)} [{time.time()-t0:.0f}s]", flush=True)
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ---------- 2. SPARSITY ----------
print("\n=== 2) SPARSITY (kitne neuron zaroori) ===", flush=True)
frac = act_count / max(total, 1)
for th in (0.5, 0.2, 0.1, 0.05, 0.02):
    keep = (frac > th).sum()
    print(f"  {th*100:>4.0f}% se zyada jagte: {keep:>7,}/{L*FF:,} ({keep/(L*FF)*100:>5.1f}%)", flush=True)

# top-k neuron per layer
print(f"\n{'top-k/layer':<14} {'neurons':>10} {'int8 MB':>9}", flush=True)
print("-" * 36)
for k in (256, 512, 1024, 2048):
    tot_n = k * L
    mb = tot_n * (D + D) / 1e6      # kunci + maan, int8
    print(f"{k:<14} {tot_n:>10,} {mb:>8.1f}", flush=True)

# ---------- 3. SABSE ZAROORI NEURON ----------
print("\n=== 3) SABSE ZAROORI NEURON (per parat) ===", flush=True)
SEL = []
for i in range(L):
    idx = np.argsort(-act_count[i])[:512]
    SEL.append(idx)
    if i % 6 == 0:
        top = act_count[i][idx[0]]
        print(f"  parat {i:>2}: sabse zyada {top:,} baar | 512 chune", flush=True)

# ---------- 4. KADI BANAO ----------
print("\n=== 4) KADI BANAO (kunci -> maan) ===", flush=True)
t0 = time.time()
KAD = []
for i in range(L):
    p = f"model.layers.{i}.mlp."
    idx = torch.tensor(SEL[i])
    kg = W[p+"gate_proj.weight"][idx].numpy()       # (512, 896) kunci
    ku = W[p+"up_proj.weight"][idx].numpy()         # (512, 896) bhaar
    vv = W[p+"down_proj.weight"][:, idx].numpy().T  # (512, 896) maan
    KAD.append((kg, ku, vv))
print(f"  {L} parat x 512 kadi | [{time.time()-t0:.0f}s]", flush=True)

mb_int8 = sum(k.size + v.size for k, _, v in KAD) / 1e6
mb_f32 = sum(k.nbytes + v.nbytes for k, _, v in KAD) / 1e6
print(f"\n=== AAKAR ===", flush=True)
print(f"  Qwen fp32:      {NP*4/1e6:>8.0f} MB", flush=True)
print(f"  Qwen int4:      {NP*0.5/1e6:>8.0f} MB", flush=True)
print(f"  MLP jaal int8:  {mb_int8:>8.1f} MB", flush=True)
print(f"  MLP jaal fp32:  {mb_f32:>8.1f} MB", flush=True)
print(f"  ghatav:         {NP*4/1e6/mb_int8:>8.0f}x", flush=True)

# ---------- 5. JAAL SE CHALAO ----------
def mlp_jaal(xn, i):
    kg, ku, vv = KAD[i]
    K = torch.tensor(kg); U = torch.tensor(ku); V = torch.tensor(vv)
    g = torch.nn.functional.silu(xn @ K.T)
    u = xn @ U.T
    return (g * u) @ V

print("\n=== 5) MLP JAAL vs ASLI MLP ===", flush=True)
seg = torch.tensor(ids_all[:96])
with torch.no_grad():
    x = W["model.embed_tokens.weight"][seg]
    # asli pehli parat ka MLP
    p = "model.layers.0."
    xx = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * W[p+"input_layernorm.weight"]
    q = xx @ W[p+"self_attn.q_proj.weight"].T + W[p+"self_attn.q_proj.bias"]
    k = xx @ W[p+"self_attn.k_proj.weight"].T + W[p+"self_attn.k_proj.bias"]
    v = xx @ W[p+"self_attn.v_proj.weight"].T + W[p+"self_attn.v_proj.bias"]
    T = len(seg)
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    half = hd // 2
    def rp(z):
        z1 = z[..., :half]; z2 = z[..., half:]
        return torch.cat([z1*cos - z2*sin, z1*sin + z2*cos], dim=-1)
    q = rp(q.view(T, H, hd)).transpose(0, 1)
    k = rp(k.view(T, KVH, hd)).transpose(0, 1)
    v = v.view(T, KVH, hd).transpose(0, 1)
    rep = H // KVH
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask[None], -1e9)
    o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, D)
    x = x + o @ W[p+"self_attn.o_proj.weight"].T
    xn = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * W[p+"post_attention_layernorm.weight"]
    g = torch.nn.functional.silu(xn @ W[p+"mlp.gate_proj.weight"].T)
    u = xn @ W[p+"mlp.up_proj.weight"].T
    y_asli = (g * u) @ W[p+"mlp.down_proj.weight"].T
    y_jaal = mlp_jaal(xn, 0)
cos = float((y_asli * y_jaal).sum() / (y_asli.norm() * y_jaal.norm() + 1e-9))
err = float((y_asli - y_jaal).norm() / (y_asli.norm() + 1e-9))
print(f"  cosine: {cos:.4f}  (512/4864 neuron = 10.5%)", flush=True)
print(f"  galti:  {err*100:.1f}%", flush=True)
print(f"  |asli| {float(y_asli.norm()):.1f} | |jaal| {float(y_jaal.norm()):.1f}", flush=True)
