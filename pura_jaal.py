#!/usr/bin/env python3
"""pura_jaal.py -- POORE jaal se adaptive jawab (ek lookup nahi).

Pichhli galti: maine ek-ek gehrai par ALAG lookup kiya.
Ab: SAARI gehrai ka sabut JODO, adaptive tarike se.

    score(shabd) = sum_K  w_K * P(shabd | sandarbh_K)

  w_K = bharosemandi ka bhaar (gehra sandarbh = zyada bhaar)
  adaptive: margin kam ho to aur gehrai dekho, zyada ho to ruk jao

Tulna: bigram vs ek-lookup vs POORA JAAL
"""
import hashlib, random, re, sys, time
from collections import Counter, defaultdict

random.seed(0)
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"

print("=== KOSH ===", flush=True)
sents = []
with open(CORPUS, encoding="utf-8", errors="ignore") as f:
    for line in f:
        ws = re.findall(r"[^\s।,?!;:\"']+", line)
        if 5 <= len(ws) <= 60: sents.append(ws)
print(f"  vaakya: {len(sents):,}", flush=True)

random.shuffle(sents)
n = len(sents)
tr, dv, te = sents[:int(n*.7)], sents[int(n*.7):int(n*.85)], sents[int(n*.85):]
print(f"  prashikshan {len(tr):,} | vikas {len(dv):,} | pariksha {len(te):,}", flush=True)

# ---------- jaal: har gehrai ka apna ----------
MAXK = 3
def cid(ws, i, K):
    lo = max(0, i - K)
    return "\x1f".join(ws[lo:i+1])

print("\n=== JAAL BANA RAHE (K=1,2,3) ===", flush=True)
LAT = {}
for K in range(1, MAXK + 1):
    lat = defaultdict(Counter)
    for ws in tr:
        for i in range(len(ws) - 1):
            lat[cid(ws, i, K)][ws[i+1]] += 1
    LAT[K] = lat
    print(f"  K={K}: {len(lat):,} cell", flush=True)

big = defaultdict(Counter)
for ws in tr:
    for a, b in zip(ws, ws[1:]): big[a][b] += 1
print(f"  bigram: {len(big):,} cell", flush=True)

# ---------- 1) BASELINE: bigram ----------
def naapo_bigram():
    s = k = 0
    for ws in te:
        for i in range(len(ws) - 1):
            if ws[i] in big:
                s += (big[ws[i]].most_common(1)[0][0] == ws[i+1]); k += 1
    return s, k
s, k = naapo_bigram()
print(f"\n=== 1) BIGRAM ===\n  sahi {s:,}/{k:,} = {s/max(k,1)*100:.1f}%", flush=True)

# ---------- 2) EK LOOKUP (gehra se shuru) ----------
def naapo_ek():
    s = k = 0
    for ws in te:
        for i in range(len(ws) - 1):
            for K in range(MAXK, 0, -1):
                key = cid(ws, i, K)
                if key in LAT[K]:
                    s += (LAT[K][key].most_common(1)[0][0] == ws[i+1]); k += 1
                    break
    return s, k
s, k = naapo_ek()
print(f"=== 2) EK LOOKUP ===\n  sahi {s:,}/{k:,} = {s/max(k,1)*100:.1f}%", flush=True)

# ---------- 3) POORA JAAL: saara sabut jodo ----------
def bhaar_seekho():
    """vikas set par: har gehrai ka bhaar seekho"""
    best_w = {K: 1.0 for K in LAT}
    # grid: 0.2 se 3.0 tak
    grid = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    for K in LAT:
        bs, bw = -1, 1.0
        for w in grid:
            s = k = 0
            for ws in dv:
                for i in range(len(ws) - 1):
                    key = cid(ws, i, K)
                    if key not in LAT[K]: continue
                    c = LAT[K][key]; tot = sum(c.values())
                    sc = {t: w * (n_/tot) for t, n_ in c.items()}
                    if sc and max(sc, key=sc.get) == ws[i+1]: s += 1
                    k += 1
            if k and s/k > bs: bs, bw = s/k, w
        best_w[K] = bw
    return best_w

W = bhaar_seekho()
print(f"\n=== 3) POORA JAAL ===\n  seekhe hue bhaar: {W}", flush=True)

def score_jaal(ws, i, gehrai=None, margin_ruko=0.25):
    """saari gehrai ka sabut jodo; adaptive rukna"""
    Kmax = gehrai or MAXK
    sc = defaultdict(float)
    steps = 0
    for K in range(1, Kmax + 1):
        steps += 1
        key = cid(ws, i, K)
        if key not in LAT[K]: continue
        c = LAT[K][key]; tot = sum(c.values())
        for t, n_ in c.items():
            sc[t] += W[K] * (n_ / tot)
        # adaptive: margin kaafi hai to ruk jao
        if len(sc) >= 2:
            top = sorted(sc.values(), reverse=True)
            if top[0] > 0 and (top[0] - top[1]) / top[0] >= margin_ruko:
                break
    return sc, steps

def naapo_pura(margin=0.25):
    s = k = 0; kadam = 0; ruke = Counter()
    for ws in te:
        for i in range(len(ws) - 1):
            sc, st = score_jaal(ws, i, margin_ruko=margin)
            kadam += st; ruke[st] += 1
            if sc:
                s += (max(sc, key=sc.get) == ws[i+1]); k += 1
    return s, k, kadam, ruke

for m in (0.0, 0.15, 0.25, 0.5):
    s, k, kad, ruke = naapo_pura(m)
    print(f"  margin={m:<5} sahi {s:,}/{k:,} = {s/max(k,1)*100:>5.1f}%  ausat kadam {kad/max(k,1):.2f}  {dict(sorted(ruke.items()))}", flush=True)

# ---------- 4) AAKAR ----------
print("\n=== AAKAR ===", flush=True)
for K in LAT:
    print(f"  K={K}: {len(LAT[K]):>10,} cell | {len(LAT[K])*11/1e6:>7.2f} MB", flush=True)
print(f"  bigram: {len(big):>10,} cell | {len(big)*11/1e6:>7.2f} MB", flush=True)
