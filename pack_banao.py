#!/usr/bin/env python3
"""pack_banao.py -- सब कुछ एक फ़ाइल में: शब्दकोश + पाथ (varint) + रिंगें (MinHash, तेज़)।

  python3 pack_banao.py --text data/corpus.txt --out store/gyaan.pack
"""
import argparse, re, struct, time, os, hashlib
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="hi_namuna.txt")
ap.add_argument("--out", default="gyaan.pack")
ap.add_argument("--max-paths", type=int, default=10_000_000)
ap.add_argument("--max-rings", type=int, default=200000)
ap.add_argument("--sig", type=int, default=16)
a = ap.parse_args()

t0 = time.time()
sents = []
for line in open(a.text, encoding="utf-8"):
    ws = tuple(re.findall(r"[^\s।,?!]+", line))
    if 4 <= len(ws) <= 25:
        sents.append(ws)
sents = list(dict.fromkeys(sents))[:a.max_paths]
print(f"यूनीक पाथ {len(sents):,} | {time.time()-t0:.1f}s", flush=True)

vocab = {}
for ws in sents:
    for w in ws:
        if w not in vocab:
            vocab[w] = len(vocab)
words = sorted(vocab, key=vocab.get)
print(f"शब्दकोश {len(words):,}", flush=True)

def vint(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break

# --- पाथ: सजाकर, साझा उपसर्ग छोड़कर, varint ---
t0 = time.time()
keyed = sorted(tuple(vocab[w] for w in ws) for ws in sents)
body = bytearray()
prev = ()
for p in keyed:
    k = 0
    while k < min(len(prev), len(p)) and prev[k] == p[k]:
        k += 1
    vint(k, body)
    vint(len(p) - k, body)
    for x in p[k:]:
        vint(x, body)
    prev = p
print(f"पाथ-भाग {len(body)/1e6:.2f} MB | {time.time()-t0:.1f}s", flush=True)

# --- रिंगें: MinHash + LSH (तेज़, बिना पूरी सूची छाने) ---
t0 = time.time()
SIG, BANDS = a.sig, 4
ROWS = max(1, SIG // BANDS)

def h32(w, seed):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{w}".encode(), digest_size=4).digest(), "little")

sigs = []
for ws in sents:
    u = list(set(ws))
    sigs.append([min(h32(w, s) for w in u) for s in range(SIG)])
print(f"हस्ताक्षर {time.time()-t0:.1f}s", flush=True)

buckets = defaultdict(list)
for i, sig in enumerate(sigs):
    for b in range(BANDS):
        buckets[(b, tuple(sig[b*ROWS:(b+1)*ROWS]))].append(i)

cand = defaultdict(list)
for key, ids in buckets.items():
    if 1 < len(ids) <= 40:
        for i in ids:
            cand[i].extend(j for j in ids if j != i)
print(f"उम्मीदवार {time.time()-t0:.1f}s | औसत {sum(len(v) for v in cand.values())/max(len(cand),1):.1f}", flush=True)

t0 = time.time()
used, rings = set(), []
for i in range(len(sents)):
    if i in used:
        continue
    grp = [i]
    for j in cand.get(i, ()):
        if j not in used and j not in grp and len(grp) < 7:
            grp.append(j)
    k = 0
    while len(grp) < 7 and k < len(sents):
        if k not in used and k not in grp:
            grp.append(k)
        k += 1
    if len(grp) == 7:
        rings.append(grp)
        used.update(grp)
    if len(rings) >= a.max_rings:
        break
print(f"रिंगें {len(rings):,} | {time.time()-t0:.1f}s", flush=True)

ring_bytes = bytearray()
for g in rings:
    for j in g:
        vint(j, ring_bytes)

voc = bytearray()
for w in words:
    b = w.encode("utf-8")
    vint(len(b), voc)
    voc += b

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "wb") as f:
    f.write(b"GPACK1")
    f.write(struct.pack("<IIII", len(words), len(keyed), len(rings), 0))
    for part in (voc, body, ring_bytes):
        f.write(struct.pack("<I", len(part)))
        f.write(bytes(part))
print(f"\nपैक: {a.out} | {os.path.getsize(a.out)/1e6:.2f} MB", flush=True)
print(f"  पाथ {len(keyed):,} | शब्द {len(words):,} | रिंगें {len(rings):,}", flush=True)
print(f"  {(len(voc)+len(body)+len(ring_bytes))/len(keyed):.1f} बाइट/पाथ", flush=True)
