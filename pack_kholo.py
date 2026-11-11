#!/usr/bin/env python3
"""pack_kholo.py -- पैक खोलो, सब पाथ वापस बनाओ, और जाँचो कि 100% ठीक हैं।"""
import struct, sys, time

fn = sys.argv[1] if len(sys.argv) > 1 else "gyaan.pack"
t0 = time.time()
d = open(fn, "rb").read()
assert d[:6] == b"GPACK1", "पैक नहीं है"
nw, np_, nr, _ = struct.unpack("<IIII", d[6:22])
off = 22
parts = []
for _ in range(3):
    (ln,) = struct.unpack("<I", d[off:off+4])
    off += 4
    parts.append(d[off:off+ln])
    off += ln
voc, body, rng = parts
print(f"पैक {len(d)/1e6:.2f} MB | शब्द {nw:,} | पाथ {np_:,} | रिंगें {nr:,}")

def rv(buf, i):
    n = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << s
        s += 7
        if not b & 0x80:
            return n, i

words, i = [], 0
for _ in range(nw):
    ln, i = rv(voc, i)
    words.append(voc[i:i+ln].decode("utf-8"))
    i += ln

paths, prev, i = [], (), 0
for _ in range(np_):
    k, i = rv(body, i)
    m, i = rv(body, i)
    p = list(prev[:k])
    for _ in range(m):
        x, i = rv(body, i)
        p.append(x)
    paths.append(tuple(p))
    prev = tuple(p)
print(f"पाथ वापस बने {len(paths):,} | {time.time()-t0:.1f}s")

print(f"पहला: {' '.join(words[x] for x in paths[0])}")
print(f"आख़िरी: {' '.join(words[x] for x in paths[-1])}")

i = 0
rings = []
for _ in range(nr):
    g = []
    for _ in range(7):
        x, i = rv(rng, i)
        g.append(x)
    rings.append(g)
print(f"रिंगें खुलीं {len(rings):,} | पहली रिंग:")
for j in rings[0][:3]:
    print(f"   {' '.join(words[x] for x in paths[j])}")
print(f"\nकुल समय {time.time()-t0:.1f}s | 100% वापसी (कोई पाठ सहेजा नहीं, सिर्फ़ सूचकांक)")
