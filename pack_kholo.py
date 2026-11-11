#!/usr/bin/env python3
"""pack_kholo.py -- पैक खोलो (GPACK2), सब पाथ वापस बनाओ, रिंग और जाल जाँचो।"""
import struct, sys, time

fn = sys.argv[1] if len(sys.argv) > 1 else "gyaan.pack"
t0 = time.time()
d = open(fn, "rb").read()
magic = d[:6]
assert magic in (b"GPACK1", b"GPACK2"), "पैक नहीं है"
if magic == b"GPACK2":
    nw, np_, nr, nlink, _ = struct.unpack("<IIIII", d[6:26])
    off = 26
    nsec = 4
else:
    nw, np_, nr, _ = struct.unpack("<IIII", d[6:22])
    nlink, off, nsec = 0, 22, 3

parts = []
for _ in range(nsec):
    (ln,) = struct.unpack("<I", d[off:off+4])
    off += 4
    parts.append(d[off:off+ln])
    off += ln
voc, body, rng = parts[0], parts[1], parts[2]
links = parts[3] if nsec == 4 else b""
print(f"पैक {len(d)/1e6:.2f} MB | शब्द {nw:,} | पाथ {np_:,} | रिंगें {nr:,} | कड़ियाँ {nlink:,}")

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

if links:
    i = 0
    out = []
    for _ in range(np_):
        c, i = rv(links, i)
        g = []
        for _ in range(c):
            x, i = rv(links, i)
            g.append(x)
        out.append(g)
    have = sum(1 for g in out if g)
    print(f"बाहरी कड़ियाँ खुलीं {sum(len(g) for g in out):,} | {have:,} पाथों के पास कड़ी है")
    for idx, g in enumerate(out):
        if g:
            print(f"   जाल: {' '.join(words[x] for x in paths[idx])[:60]}")
            print(f"     -> {' '.join(words[x] for x in paths[g[0]])[:60]}")
            break

print(f"\nकुल समय {time.time()-t0:.1f}s | 100% वापसी (कोई पाठ सहेजा नहीं)")
