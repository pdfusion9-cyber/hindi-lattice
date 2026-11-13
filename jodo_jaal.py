#!/usr/bin/env python3
"""jodo_jaal.py -- कई जाल + पैटर्न-तालिकाएँ एक में जोड़ो (डुप्लीकेट हटाकर)।

  N पैक पढ़ो -> पाथ शब्दों में बदलो -> यूनीक करो -> एक पैक लिखो
  साथ में सबकी पैटर्न-तालिका एक में मिलाओ
"""
import argparse, glob, json, mmap, os, struct
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--packs", nargs="+", required=True)
ap.add_argument("--out", default="store/merged.pack")
ap.add_argument("--pat-out", default="store/merged_pats.json")
ap.add_argument("--minpat", type=int, default=5)
a = ap.parse_args()


def read_pack(path):
    f = open(path, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic = mm[:6]

    def rv(buf, i):
        n = 0; sh = 0
        while True:
            b = buf[i]; i += 1; n |= (b & 0x7F) << sh
            if not (b & 0x80): return n, i
            sh += 7

    if magic == b"GPACK1":
        nw, np_, nr, _x = struct.unpack("<IIII", mm[6:22])
        off = 22; secs = []
        for _ in range(3):
            (k,) = struct.unpack("<I", mm[off:off+4]); off += 4
            secs.append(mm[off:off+k]); off += k
        voc, body, rng = secs
        words, i = [], 0
        for _ in range(nw):
            ln, i = rv(voc, i)
            words.append(voc[i:i+ln].decode("utf-8")); i += ln
        out, prev, i = [], (), 0
        for _ in range(np_):
            k, i = rv(body, i); m, i = rv(body, i)
            p = list(prev[:k])
            for _ in range(m):
                x, i = rv(body, i); p.append(x)
            out.append(tuple(p)); prev = tuple(p)
        return [tuple(words[x] if x < len(words) else f"<{x}>" for x in p)
                for p in out if 2 <= len(p) <= 80]

    if magic != b"GPACK5":
        return None
    (V, ns, nch, cbits, miss, nblk, block, nlen) = struct.unpack("<IIIIIIII", mm[6:38])
    off = 38; parts = {}
    for nm in ("chars", "voc", "lat", "latidx", "starts", "body", "blk", "lens"):
        (k,) = struct.unpack("<I", mm[off:off+4]); off += 4
        parts[nm] = (off, k); off += k
    co, cn = parts["chars"]; chars = mm[co:co+cn].decode()
    vo, vn = parts["voc"]; data = mm[vo:vo+vn]
    words, cur, acc, n = [], [], 0, 0
    cb, stop = cbits, len(chars)
    for byte in data:
        for kk in range(8):
            acc |= ((byte >> kk) & 1) << n; n += 1
            if n == cb:
                if acc == stop: words.append("".join(cur)); cur = []
                else: cur.append(chars[acc])
                acc = 0; n = 0
    if cur: words.append("".join(cur))

    def vint(d, i):
        v = 0; sh = 0
        while True:
            b = d[i]; i += 1; v |= (b & 0x7F) << sh
            if not (b & 0x80): return v, i
            sh += 7

    lo, _ = parts["lat"]; io, _ = parts["latidx"]
    def links(w):
        (x,) = struct.unpack("<I", mm[io+w*4:io+w*4+4])
        (y,) = struct.unpack("<I", mm[io+(w+1)*4:io+(w+1)*4+4])
        d = mm[lo+x:lo+y]; c, i = vint(d, 0); o = []
        for _ in range(c):
            v, i = vint(d, i); o.append(v)
        return o

    def bits(bp, nb):
        bo, _ = parts["body"]; v = 0
        for i in range(nb):
            p = bp + i; v |= ((mm[bo + (p >> 3)] >> (p & 7)) & 1) << i
        return v

    bo, _ = parts["blk"]; so, _ = parts["starts"]; l2, _ = parts["lens"]
    paths = []
    for idx in range(ns):
        (bp,) = struct.unpack("<I", mm[bo+idx*4:bo+idx*4+4])
        (w,) = struct.unpack("<I", mm[so+idx*4:so+idx*4+4])
        L = mm[l2+idx]; out = [w]
        for _ in range(L):
            lk = links(w)
            b = max(1, (max(len(lk), 2) - 1).bit_length())
            k = bits(bp, b); bp += b
            if k >= len(lk): break
            w = lk[k]; out.append(w)
        if 2 <= len(out) <= 80:
            paths.append(tuple(words[i] if i < len(words) else f"<{i}>" for i in out))
    return paths


all_paths = set()
per = {}
for p in a.packs:
    if not os.path.exists(p):
        print(f"  छोड़ा (नहीं मिला): {p}", flush=True); continue
    ps = read_pack(p)
    if ps is None:
        print(f"  छोड़ा (फ़ॉर्मैट): {p}", flush=True); continue
    before = len(all_paths)
    all_paths.update(ps)
    per[p] = {"paths": len(ps), "new": len(all_paths) - before}
    print(f"  {os.path.basename(p)}: {len(ps):,} पाथ | नए {len(all_paths)-before:,}", flush=True)

print(f"\nकुल यूनीक पाथ: {len(all_paths):,}", flush=True)

# ---------- पैटर्न-तालिका ----------
cnt = Counter(w for p in all_paths for w in p)
STOP = {w for w, c in cnt.items() if c >= 300}
def pattern(q): return tuple(w if w in STOP else "*" for w in q)
pat_slots = defaultdict(lambda: defaultdict(Counter)); pat_count = Counter()
for p in all_paths:
    pt = pattern(p); pat_count[pt] += 1
    for i, w in enumerate(p):
        if pt[i] == "*": pat_slots[pt][i][w] += 1
pats = {tuple(k): v for k, v in pat_count.items() if v >= a.minpat}
print(f"पैटर्न: {len(pats):,}", flush=True)
os.makedirs(os.path.dirname(a.pat_out) or ".", exist_ok=True)
json.dump({"patterns": [list(k) for k in pats],
           "slots": {str(k): {str(i): [w for w, _ in c.most_common(20)] for i, c in v.items()}
                     for k, v in list(pat_slots.items())[:20000]}},
          open(a.pat_out, "w"), ensure_ascii=False)
print(f"पैटर्न-तालिका: {os.path.getsize(a.pat_out)/1e6:.2f} MB", flush=True)

# ---------- एक पैक लिखो (GPACK5) ----------
vocab = {}
for p in all_paths:
    for w in p:
        if w not in vocab: vocab[w] = len(vocab)
V = len(vocab)
words = sorted(vocab, key=vocab.get)
ids = [[vocab[w] for w in p] for p in all_paths]
links = defaultdict(Counter)
for p in ids:
    for x, y in zip(p, p[1:]): links[x][y] += 1

def vint(v, buf):
    while True:
        b = v & 0x7F; v >>= 7
        if v: buf.append(b | 0x80)
        else: buf.append(b); break

chars = sorted({c for w in words for c in w})
chars = chars[:250]
cmap = {c: i for i, c in enumerate(chars)}
STOPC = len(chars)
cbits = max(1, (len(chars) + 1).bit_length() - 1) if len(chars) > 1 else 1
cbits = max(1, (len(chars)).bit_length())
if (1 << cbits) <= len(chars): cbits += 1

cb_buf = bytearray(); acc = 0; nb = 0
def put(v, bits):
    global acc, nb
    acc |= (v << nb); nb += bits
    while nb >= 8:
        cb_buf.append(acc & 0xFF); acc >>= 8; nb -= 8
for w in words:
    for c in w: put(cmap.get(c, 0), cbits)
    put(STOPC, cbits)
if nb: cb_buf.append(acc & 0xFF)

lat = bytearray(); lat_off = []
for k in range(V):
    lat_off.append(len(lat))
    lst = [y for y, _ in links.get(k, Counter()).most_common(1000)]
    vint(len(lst), lat)
    for x in lst: vint(x, lat)
lat_off.append(len(lat))
latidx = b"".join(struct.pack("<I", o) for o in lat_off)

bt = bytearray(); bacc = 0; bn = 0; bbits = 0
starts, blocks, lens = [], [], []
for p in ids:
    blocks.append(bbits)
    starts.append(p[0]); wrote = 0
    for x, y in zip(p, p[1:]):
        lst = [z for z, _ in links.get(x, Counter()).most_common(1000)]
        try: k = lst.index(y)
        except ValueError: continue
        wdt = max(1, (max(len(lst), 2) - 1).bit_length())
        bacc |= (k << bn); bn += wdt; bbits += wdt
        while bn >= 8:
            bt.append(bacc & 0xFF); bacc >>= 8; bn -= 8
        wrote += 1
    lens.append(min(wrote, 255))
if bn: bt.append(bacc & 0xFF)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
parts = [b"".join(c.encode() for c in chars), bytes(cb_buf), bytes(lat), latidx,
         b"".join(struct.pack("<I", x) for x in starts), bytes(bt),
         b"".join(struct.pack("<I", x) for x in blocks), bytes(lens)]
with open(a.out, "wb") as f:
    f.write(b"GPACK5")
    f.write(struct.pack("<IIIIIIII", V, len(ids), len(chars), cbits, 0, len(blocks), 256, len(lens)))
    for p in parts:
        f.write(struct.pack("<I", len(p))); f.write(p)

sz = os.path.getsize(a.out)
json.dump({"packs": per, "unique_paths": len(all_paths), "V": V,
           "patterns": len(pats), "pack_bytes": sz}, open(a.out + ".json", "w"), indent=2)
print(f"\n=== मर्ज ===", flush=True)
print(f"  यूनीक पाथ: {len(all_paths):,}", flush=True)
print(f"  शब्द: {V:,}", flush=True)
print(f"  कड़ियाँ: {sum(len(c) for c in links.values()):,}", flush=True)
print(f"  पैक: {sz/1e6:.2f} MB -> {a.out}", flush=True)
