#!/usr/bin/env python3
"""pack_banao.py -- GPACK5: शब्द-जाल एक बार + वाक्य कड़ी-क्रम में + mmap-योग्य सूचकांक।

सुधार (2026-11-11):
  - कड़ियाँ काटी नहीं जातीं (पहले 24 पर काटने से वाक्य बिगड़ते थे)
  - हर वाक्य में जितनी कड़ियाँ लिखी गईं, उतनी गिनकर सहेजी जाती हैं
  - शुरू-शब्द uint32 सरणी में (O(1) पहुँच)
  - हर वाक्य का bit offset (mmap से सीधे पढ़ने के लिए)
"""
import argparse, math, os, re, struct, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="data/corpus.txt")
ap.add_argument("--out", default="store/gyaan.pack")
ap.add_argument("--max-paths", type=int, default=3_000_000)
ap.add_argument("--max-links", type=int, default=1000000)
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--min-char", type=int, default=3)
a = ap.parse_args()

t0 = time.time()
sents = []
for line in open(a.text, encoding="utf-8"):
    ws = tuple(re.findall(r"[^\s।,?!]+", line))
    if 4 <= len(ws) <= 25:
        sents.append(ws)
sents = list(dict.fromkeys(sents))[:a.max_paths]
print(f"वाक्य {len(sents):,} | {time.time()-t0:.0f}s", flush=True)

vocab = {}
for ws in sents:
    for w in ws:
        if w not in vocab:
            vocab[w] = len(vocab)
V = len(vocab)
words = sorted(vocab, key=vocab.get)
print(f"शब्द {V:,}", flush=True)

edges = defaultdict(Counter)
for ws in sents:
    for x, y in zip(ws, ws[1:]):
        edges[vocab[x]][vocab[y]] += 1
order = {k: [b for b, _ in c.most_common(a.max_links)] for k, c in edges.items()}
width = {k: max(1, math.ceil(math.log2(max(len(l), 2)))) for k, l in order.items()}
print(f"कड़ियाँ {sum(len(c) for c in edges.values()):,}", flush=True)


def vint(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break


class Bits:
    def __init__(s):
        s.buf = bytearray(); s.acc = 0; s.n = 0; s.bits = 0

    def put(s, v, b):
        s.acc |= (v << s.n); s.n += b; s.bits += b
        while s.n >= 8:
            s.buf.append(s.acc & 0xFF); s.acc >>= 8; s.n -= 8

    def end(s):
        if s.n:
            s.buf.append(s.acc & 0xFF)
        return bytes(s.buf)


bt = Bits(); starts = []; blocks = []; lens = []; missed = 0
for i, ws in enumerate(sents):
    blocks.append(bt.bits)
    ids = [vocab[w] for w in ws]
    starts.append(ids[0])
    lens.append(0)
    wrote = 0
    for x, y in zip(ids, ids[1:]):
        lst = order.get(x, [])
        try:
            k = lst.index(y)
        except ValueError:
            missed += 1
            continue
        bt.put(k, width.get(x, 1))
        wrote += 1
    lens[-1] = wrote
body = bt.end()
starts = b"".join(struct.pack("<I", x) for x in starts)
blk = b"".join(struct.pack("<I", b) for b in blocks)
lens = bytes(min(x, 255) for x in lens)

lat = bytearray(); lat_off = []
for k in range(V):
    lat_off.append(len(lat))
    lst = order.get(k, [])
    vint(len(lst), lat)
    for x in lst:
        vint(x, lat)
lat_off.append(len(lat))
latidx = b"".join(struct.pack("<I", o) for o in lat_off)

cnt = Counter(c for w in words for c in w)
chars = sorted([c for c, n in cnt.items() if n >= a.min_char]) or sorted(cnt)
cmap = {c: i for i, c in enumerate(chars)}
cbits = max(1, math.ceil(math.log2(len(chars) + 1)))
STOP = len(chars)
cb = Bits()
for w in words:
    for c in w:
        cb.put(cmap.get(c, 0), cbits)
    cb.put(STOP, cbits)
voc = cb.end()

parts = [b"".join(c.encode() for c in chars), voc, lat, latidx, starts, body, blk, lens]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "wb") as f:
    f.write(b"GPACK5")
    f.write(struct.pack("<IIIIIIII", V, len(sents), len(chars), cbits, missed, len(blocks), a.block, len(lens)))
    for p in parts:
        f.write(struct.pack("<I", len(p)))
        f.write(p)

tot = sum(len(p) for p in parts)
print(f"\nपैक {a.out} | {os.path.getsize(a.out)/1e6:.2f} MB", flush=True)
print(f"  वाक्य {len(sents):,} | शब्द {V:,} | कड़ियाँ {sum(len(c) for c in edges.values()):,} | छूटे {missed}", flush=True)
print(f"  {tot/len(sents):.2f} बाइट/वाक्य | 250 MB में {250e6/(tot/len(sents)):,.0f} वाक्य", flush=True)
