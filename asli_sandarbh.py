#!/usr/bin/env python3
"""asli_sandarbh.py -- SANDARBH-ID JAAL asli paath par (Hindi).

Tulna:
  1. BIGRAM TABLE  -- sirf pichhla shabd
  2. CONTEXT-ID    -- pichhle K shabd + khud = unique ID
  3. ADAPTIVE      -- kam sandarbh se shuru, bharosa na ho to gehra

Naap: sahi (%), jaal cell, reuse, kadam
"""
import hashlib, random, re, sys, time
from collections import Counter, defaultdict

random.seed(0)
import os
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"

print("=== KOSH PADH RAHE ===", flush=True)
sents = []
with open(CORPUS, encoding="utf-8", errors="ignore") as f:
    for line in f:
        ws = re.findall(r"[^\s।,?!;:\"']+", line)
        if 5 <= len(ws) <= 60:
            sents.append(ws)
print(f"  vaakya: {len(sents):,}", flush=True)

random.shuffle(sents)
cut = int(len(sents) * 0.8)
train, test = sents[:cut], sents[cut:]
print(f"  prashikshan {len(train):,} | pariksha {len(test):,}", flush=True)

vocab = Counter(w for ws in train for w in ws)
print(f"  shabd: {len(vocab):,} | token: {sum(vocab.values()):,}", flush=True)

# ---------- 1) BIGRAM TABLE ----------
print("\n=== 1) BIGRAM TABLE (sirf pichhla shabd) ===", flush=True)
t0 = time.time()
big = defaultdict(Counter)
for ws in train:
    for a, b in zip(ws, ws[1:]):
        big[a][b] += 1
print(f"  cell: {len(big):,} [{time.time()-t0:.0f}s]", flush=True)

def naapo_bigram():
    sahi = kul = 0
    for ws in test:
        for a, b in zip(ws, ws[1:]):
            if a in big:
                pred = big[a].most_common(1)[0][0]
                sahi += (pred == b)
                kul += 1
    return sahi, kul

s, k = naapo_bigram()
print(f"  sahi: {s:,}/{k:,} = {s/max(k,1)*100:.1f}%", flush=True)

# ---------- 2) CONTEXT-ID JAAL ----------
print("\n=== 2) CONTEXT-ID JAAL (K shabd + khud = ID) ===", flush=True)

def cid(ws, i, K):
    """pichhle K shabd + khud ka unique ID"""
    lo = max(0, i - K)
    key = "\x1f".join(ws[lo:i+1])
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()

HASIL = {}
for K in (1, 2, 3):
    t0 = time.time()
    lat = defaultdict(Counter)
    for ws in train:
        for i in range(len(ws) - 1):
            lat[cid(ws, i, K)][ws[i+1]] += 1
    HASIL[K] = lat
    print(f"  K={K}: {len(lat):,} cell [{time.time()-t0:.0f}s]", flush=True)

print(f"\n{'K':>3} {'cell':>12} {'sahi':>8} {'reuse':>8} {'MB':>7}", flush=True)
print("-" * 44)
for K, lat in HASIL.items():
    sahi = kul = reuse = 0
    for ws in test:
        for i in range(len(ws) - 1):
            key = cid(ws, i, K)
            if key in lat:
                reuse += 1
                pred = lat[key].most_common(1)[0][0]
                sahi += (pred == ws[i+1])
                kul += 1
    mb = len(lat) * (8 + 2 + 1) / 1e6      # ID + shabd-id + ginti
    print(f"{K:>3} {len(lat):>12,} {sahi/max(kul,1)*100:>7.1f}% {reuse/len(test):>7.1f}% {mb:>7.2f}", flush=True)

# ---------- 3) ADAPTIVE ----------
print("\n=== 3) ADAPTIVE (kam se shuru, bharosa na ho to gehra) ===", flush=True)
def jawab(ws, i, maxK=3, conf=0.6):
    steps = 0
    for K in range(1, maxK + 1):
        steps += 1
        key = cid(ws, i, K)
        lat = HASIL.get(K)
        if lat and key in lat:
            c = lat[key]
            tot = sum(c.values())
            best, cnt = c.most_common(1)[0]
            if cnt / tot >= conf:
                return best, steps, K
    return None, steps, 0

sahi = kul = 0; kadam = 0; gehrai = Counter()
t0 = time.time()
for ws in test:
    for i in range(len(ws) - 1):
        pred, st, K = jawab(ws, i)
        kadam += st
        gehrai[K] += 1
        if pred is not None:
            sahi += (pred == ws[i+1]); kul += 1
print(f"  sahi: {sahi:,}/{kul:,} = {sahi/max(kul,1)*100:.1f}%", flush=True)
print(f"  ausat kadam: {kadam/max(kul,1):.2f}", flush=True)
print(f"  gehrai: {dict(sorted(gehrai.items()))}", flush=True)
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ---------- 4) KUL AAKAR ----------
print("\n=== 4) AAKAR ===", flush=True)
for K, lat in HASIL.items():
    mb = len(lat) * 11 / 1e6
    print(f"  K={K}: {len(lat):>10,} cell | {mb:>7.2f} MB (11 byte/cell)", flush=True)
print(f"  bigram: {len(big):>10,} cell | {len(big)*11/1e6:>7.2f} MB", flush=True)
