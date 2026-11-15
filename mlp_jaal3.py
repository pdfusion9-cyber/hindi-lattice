#!/usr/bin/env python3
"""mlp_jaal3.py -- SAMUH-WAR PAIMANA (har parat ke liye ek nahi).

Sudhar: neuron ko G samuhon me baanto, har samuh ka apna paimana.

    y = sum_g ( jaal_g * scale_g )

Paimana ki keemat: L x G numbers (naganya)
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

print("wazan...", flush=True)
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
print(f"  {NP:,} | {NP*4/1e6:.0f} MB", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tok.json")
text = open("corpus.txt", encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text).ids[:4000]

def rms(x, w):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

def rp(z, T):
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[:, None, :]; sin = ang.sin()[:, None, :]
    half = hd // 2
    z1 = z[..., :half]; z2 = z[..., half:]
    return torch.cat([z1*cos - z2*sin, z1*sin + z2*cos], dim=-1)

def attn(x, i, T):
    p = f"model.layers.{i}.self_attn."
    q = x @ W[p+"q_proj.weight"].T + W[p+"q_proj.bias"]
    k = x @ W[p+"k_proj.weight"].T + W[p+"k_proj.bias"]
    v = x @ W[p+"v_proj.weight"].T + W[p+"v_proj.bias"]
    q = rp(q.view(T, H, hd), T).transpose(0, 1)
    k = rp(k.view(T, KVH, hd), T).transpose(0, 1)
    v = v.view(T, KVH, hd).transpose(0, 1)
    rep = H // KVH
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    sc = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask[None], -1e9)
    o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, D)
    return o @ W[p+"o_proj.weight"].T

# ---------- neuron ki jaag ----------
print("\nneuron ki jaag...", flush=True)
t0 = time.time()
act = np.zeros((L, FF), dtype=np.int32)
with torch.no_grad():
    for st in range(0, len(ids_all) - 64, 64):
        seg = torch.tensor(ids_all[st:st+64]); T = len(seg)
        x = W["model.embed_tokens.weight"][seg]
        for i in range(L):
            p = f"model.layers.{i}."
            x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, T)
            xn = rms(x, W[p+"post_attention_layernorm.weight"])
            g = torch.nn.functional.silu(xn @ W[p+"mlp.gate_proj.weight"].T)
            u = xn @ W[p+"mlp.up_proj.weight"].T
            h = (g * u).abs()
            act[i] += (h > h.mean(-1, keepdim=True)).sum(0).numpy()
            x = x + (g * u) @ W[p+"mlp.down_proj.weight"].T
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ---------- chuno ----------
KAD = {}
for K in (512, 1024, 2048):
    kad = []
    for i in range(L):
        p = f"model.layers.{i}.mlp."
        idx = torch.tensor(np.argsort(-act[i])[:K])
        kg = W[p+"gate_proj.weight"][idx].numpy()
        ku = W[p+"up_proj.weight"][idx].numpy()
        vv = W[p+"down_proj.weight"][:, idx].numpy().T
        kad.append((kg, ku, vv))
    KAD[K] = kad
    mb = sum(k.size + v.size for k, _, v in kad) / 1e6
    print(f"  K={K}: {mb:.1f} MB", flush=True)

# ---------- paimana seekho ----------
SEG_TR = torch.tensor(ids_all[:128]); TT = len(SEG_TR)
print("\n=== PAIMANA SEEKH RAHE (samuh-war) ===", flush=True)

def seekho(G):
    """har parat x har samuh ka paimana"""
    SC = {K: [[1.0]*G for _ in range(L)] for K in KAD}
    with torch.no_grad():
        x = W["model.embed_tokens.weight"][SEG_TR]
        for i in range(L):
            p = f"model.layers.{i}."
            x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, TT)
            xn = rms(x, W[p+"post_attention_layernorm.weight"])
            g = torch.nn.functional.silu(xn @ W[p+"mlp.gate_proj.weight"].T)
            u = xn @ W[p+"mlp.up_proj.weight"].T
            y_true = (g * u) @ W[p+"mlp.down_proj.weight"].T
            for K in KAD:
                kg, ku, vv = KAD[K][i]
                gj = torch.nn.functional.silu(xn @ torch.tensor(kg).T)
                uj = xn @ torch.tensor(ku).T
                Vt = torch.tensor(vv)                     # (K, D)
                per = (gj * uj).unsqueeze(-1) * Vt.unsqueeze(0)   # (T, K, D)
                step = max(1, K // G)
                for gi in range(G):
                    a, b = gi*step, min((gi+1)*step, K)
                    yg = per[:, a:b, :].sum(1)            # (T, D)
                    SC[K][i][gi] = float(y_true.norm() / (yg.norm() + 1e-9)) if b > a else 1.0
            x = x + y_true
    return SC

SCALES = {G: seekho(G) for G in (1, 8, 32, 128)}
for G in SCALES:
    v = np.mean([np.mean(SCALES[G][512][i]) for i in range(L)])
    print(f"  G={G:>3}: ausat paimana {v:.2f}", flush=True)

# ---------- chain ----------
def chain(K=None, G=None):
    x = W["model.embed_tokens.weight"][SEG_TR]
    for i in range(L):
        p = f"model.layers.{i}."
        x = x + attn(rms(x, W[p+"input_layernorm.weight"]), i, TT)
        xn = rms(x, W[p+"post_attention_layernorm.weight"])
        if K is None:
            g = torch.nn.functional.silu(xn @ W[p+"mlp.gate_proj.weight"].T)
            u = xn @ W[p+"mlp.up_proj.weight"].T
            y = (g * u) @ W[p+"mlp.down_proj.weight"].T
        else:
            kg, ku, vv = KAD[K][i]
            gj = torch.nn.functional.silu(xn @ torch.tensor(kg).T)
            uj = xn @ torch.tensor(ku).T
            Vt = torch.tensor(vv)
            step = max(1, K // G)
            y = torch.zeros(TT, D)
            for gi in range(G):
                a, b = gi*step, min((gi+1)*step, K)
                if b <= a: continue
                yg = ((gj[:, a:b] * uj[:, a:b]).unsqueeze(-1) * Vt[a:b].unsqueeze(0)).sum(1)
                y = y + yg * SCALES[G][K][i][gi]
        x = x + y
    x = rms(x, W["model.norm.weight"])
    return x @ W["model.embed_tokens.weight"].T

print("\n=== POORI CHAIN ===", flush=True)
print(f"{'setup':<24} {'MB':>6} {'cosine':>8} {'|h|':>8}", flush=True)
print("-" * 50)
with torch.no_grad():
    yt = chain()
    print(f"{'asli':<24} {'-':>6} {'1.0000':>8} {float(yt.norm()):>8.1f}", flush=True)
    for K in KAD:
        mb = sum(k.size + v.size for k, _, v in KAD[K]) / 1e6
        for G in (1, 8, 32, 128):
            if G > K: continue
            y = chain(K, G)
            cs = float((yt * y).sum() / (yt.norm() * y.norm() + 1e-9))
            print(f"{'K='+str(K)+' G='+str(G):<24} {mb:>5.0f} {cs:>8.4f} {float(y.norm()):>8.1f}", flush=True)
