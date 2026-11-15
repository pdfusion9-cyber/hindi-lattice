#!/usr/bin/env python3
"""shabd_jaal.py -- shabd-taalka ko JAAL banao (kendra + kadi).

Vichar:
  har shabd ek point hai 896-dim jagah me
  kuch KENDRA chuno
  har shabd apne sabse paas ke kendra se juda
  shabd = kendra ki id + thoda bacha hua hissa

Aakar: K x 896 (kendra) + 151936 x 2 byte (id) + 151936 x D (bacha)
Agar bacha hata dein: sirf kendra + id!

Naap: cosine (shabd kitna sahi wapas aata hai)
"""
import json, math, os, struct, sys, time
import numpy as np

D = 896
V = 151936
PATH = "m.safetensors"

with open(PATH, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n

v = hdr["model.embed_tokens.weight"]
print(f"embed: {v['shape']}", flush=True)

with open(PATH, "rb") as f:
    f.seek(base + v["data_offsets"][0])
    raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
ne = len(raw) // 2
E = np.empty(ne, dtype=np.float32)
CH = 4_000_000
for st in range(0, ne, CH):
    en = min(st + CH, ne)
    u = np.frombuffer(raw[st*2:en*2], dtype=np.uint16)
    E[st:en] = (u.astype(np.uint32) << 16).view(np.float32)
E = E.reshape(V, D)
print(f"  lada: {E.shape} | fp32 {E.nbytes/1e6:.0f} MB", flush=True)

def kmeans(A, K, iters=10, seed=0, chunk=20000):
    rng = np.random.default_rng(seed)
    C = A[rng.choice(len(A), K, replace=False)].copy()
    for it in range(iters):
        lab = np.empty(len(A), dtype=np.int32)
        for s in range(0, len(A), chunk):
            e = min(s+chunk, len(A))
            d = (A[s:e]**2).sum(1)[:, None] - 2 * A[s:e] @ C.T + (C**2).sum(1)[None]
            lab[s:e] = d.argmin(1)
        for k in range(K):
            m = lab == k
            if m.sum() > 0: C[k] = A[m].mean(0)
    return C, lab

def assign(A, C, chunk=20000):
    lab = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), chunk):
        e = min(s+chunk, len(A))
        d = (A[s:e]**2).sum(1)[:, None] - 2 * A[s:e] @ C.T + (C**2).sum(1)[None]
        lab[s:e] = d.argmin(1)
    return lab

print(f"\n{'K':>7} {'kendra MB':>10} {'id MB':>7} {'kad MB':>8} {'cosine':>8} {'bahut achha':>12}", flush=True)
print("-" * 62)
HASIL = []
for K in (1024, 4096, 16384):
    t0 = time.time()
    C, lab = kmeans(E, K, 8, seed=1)
    # 1) sirf kendra (koi bacha nahi)
    R = C[lab]
    cs = float((E * R).sum(1).mean() / (np.linalg.norm(E, axis=1).mean() * np.linalg.norm(R, axis=1).mean()))
    # theek cosine
    num = (E * R).sum(1)
    den = np.linalg.norm(E, axis=1) * np.linalg.norm(R, axis=1) + 1e-9
    cs = float((num / den).mean())
    acc = float((num / den > 0.9).mean() * 100)
    mb_c = K * D / 1e6                    # int8 kendra
    mb_id = V * 2 / 1e6                   # uint16 id
    HASIL.append((K, mb_c + mb_id, cs, acc))
    print(f"{K:>7} {mb_c:>9.1f} {mb_id:>6.1f} {'-':>8} {cs:>8.4f} {acc:>11.1f}% [{time.time()-t0:.0f}s]", flush=True)

# ---------- bacha hua hissa bhi (2-bit) ----------
print(f"\n=== BACHA HUA HISSA JODKAR (2-bit) ===", flush=True)
print(f"{'K':>7} {'kad+2bit MB':>13} {'cosine':>8} {'bahut achha':>12}", flush=True)
print("-" * 48)
for K, _, _, _ in HASIL:
    C, lab = kmeans(E, K, 8, seed=1)
    R = E - C[lab]
    # har shabd ka apna paimana, 2-bit
    sc = np.abs(R).max(1) / 1.0
    sc[sc == 0] = 1
    q = np.clip(np.round(R / sc[:, None] * 1.0), -1, 1)   # 3 level = 2 bit
    Rr = q * sc[:, None]
    R2 = C[lab] + Rr
    num = (E * R2).sum(1); den = np.linalg.norm(E, axis=1) * np.linalg.norm(R2, axis=1) + 1e-9
    cs = float((num/den).mean()); acc = float((num/den > 0.9).mean()*100)
    mb = K * D / 1e6 + V * 2 / 1e6 + V * D * 0.25 / 1e6 + V * 4 / 1e6
    print(f"{K:>7} {mb:>12.1f} {cs:>8.4f} {acc:>11.1f}%", flush=True)

# ---------- kadi (neighbour) ----------
print(f"\n=== KADI (har shabd ke padosi) ===", flush=True)
K = 4096
C, lab = kmeans(E, K, 8, seed=1)
# kendra ke padosi
C2 = (C**2).sum(1)[:, None] - 2 * C @ C.T + (C**2).sum(1)[None]
np.fill_diagonal(C2, 1e9)
nb = np.argsort(C2, 1)[:, :8]
mb_kadi = K * 8 * 2 / 1e6
print(f"  {K} kendra x 8 padosi = {mb_kadi:.2f} MB (uint16)", flush=True)
print(f"  + id {V*2/1e6:.1f} MB = kul {mb_kadi + V*2/1e6:.1f} MB", flush=True)
print(f"  int4 shabd-taalka se: {68.0/(mb_kadi + V*2/1e6):.1f}x chhota", flush=True)
