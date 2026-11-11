#!/usr/bin/env python3
"""pack_banao.py -- GPACK4: शब्द-जाल एक बार + वाक्य सिर्फ़ कड़ी-क्रम के रूप में।

तीन सिद्धांत:
  1. शब्द एक ही बार लिखे जाते हैं (अक्षर-कोड से, 7 बिट प्रति अक्षर)
  2. जाल (हर शब्द से कड़ियाँ) भी एक ही बार
  3. वाक्य = शुरू-शब्द + हर कदम की कड़ी-संख्या (शब्द दोबारा नहीं)
"""
import argparse, math, os, re, struct, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="data/corpus.txt")
ap.add_argument("--out", default="store/gyaan.pack")
ap.add_argument("--max-paths", type=int, default=10_000_000)
ap.add_argument("--max-links", type=int, default=24)
a = ap.parse_args()

t0 = time.time()
sents = []
for line in open(a.text, encoding="utf-8"):
    ws = tuple(re.findall(r"[^\s।,?!]+", line))
    if 4 <= len(ws) <= 25:
        sents.append(ws)
sents = list(dict.fromkeys(sents))[:a.max_paths]
print(f"वाक्य {len(sents):,} | {time.time()-t0:.1f}s", flush=True)

vocab = {}
for ws in sents:
    for w in ws:
        if w not in vocab:
            vocab[w] = len(vocab)
V = len(vocab)
words = sorted(vocab, key=vocab.get)
print(f"शब्द {V:,}", flush=True)

# --- जाल ---
t0 = time.time()
edges = defaultdict(Counter)
for ws in sents:
    for x, y in zip(ws, ws[1:]):
        edges[vocab[x]][vocab[y]] += 1
order = {k: [b for b, _ in c.most_common(a.max_links)] for k, c in edges.items()}
width = {k: max(1, math.ceil(math.log2(max(len(l), 2)))) for k, l in order.items()}
print(f"जाल: {sum(len(c) for c in edges.values()):,} कड़ियाँ | {time.time()-t0:.1f}s", flush=True)


def vint(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break


class Bits:
    def __init__(s):
        s.buf = bytearray(); s.acc = 0; s.n = 0

    def put(s, v, b):
        s.acc |= (v << s.n); s.n += b
        while s.n >= 8:
            s.buf.append(s.acc & 0xFF); s.acc >>= 8; s.n -= 8

    def end(s):
        if s.n:
            s.buf.append(s.acc & 0xFF)
        return bytes(s.buf)


# --- वाक्य: शुरू-आईडी + कड़ी-क्रम ---
t0 = time.time()
bt = Bits(); starts = bytearray(); missed = 0
for ws in sents:
    ids = [vocab[w] for w in ws]
    vint(ids[0], starts)
    for x, y in zip(ids, ids[1:]):
        lst = order.get(x, [])
        try:
            i = lst.index(y)
        except ValueError:
            missed += 1
            continue
        bt.put(i, width.get(x, 1))
body = bt.end()
print(f"वाक्य-भाग {len(body)/1e6:.2f} MB | छूटे {missed:,} | {time.time()-t0:.1f}s", flush=True)

# --- जाल-भाग ---
lat = bytearray()
for k in range(V):
    lst = order.get(k, [])
    vint(len(lst), lat)
    for x in lst:
        vint(x, lat)

# --- शब्दकोश: 7 बिट प्रति अक्षर ---
chars = sorted({c for w in words for c in w})
cmap = {c: i for i, c in enumerate(chars)}
cb = Bits()
for w in words:
    for c in w:
        cb.put(cmap[c], 7)
    cb.put(127, 7)          # शब्द-विराम
voc = cb.end()
print(f"शब्दकोश {len(voc)/1e6:.2f} MB | अलग अक्षर {len(chars)} | जाल {len(lat)/1e6:.2f} MB", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "wb") as f:
    f.write(b"GPACK4")
    f.write(struct.pack("<IIIII", V, len(sents), len(chars), missed, 0))
    for part in (bytes(chars_utf8) if False else b"".join(c.encode() for c in chars), voc, lat, starts, body):
        f.write(struct.pack("<I", len(part)))
        f.write(part)
tot = len(voc) + len(lat) + len(starts) + len(body)
print(f"\nपैक: {a.out} | {os.path.getsize(a.out)/1e6:.2f} MB", flush=True)
print(f"  वाक्य {len(sents):,} | शब्द {V:,} | {tot/len(sents):.2f} बाइट/वाक्य", flush=True)
print(f"  250 MB में: {250e6/(tot/len(sents)):,.0f} वाक्य", flush=True)
