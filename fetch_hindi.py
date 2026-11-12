#!/usr/bin/env python3
"""fetch_hindi.py -- fineweb-2 हिंदी का एक टुकड़ा (शार्ड) उतारकर पाठ निकालो।

  python3 fetch_hindi.py --shard 3 --rows 150000 --out data/web.txt

हर शार्ड = अलग parquet फ़ाइल = बिल्कुल नया डेटा (कोई दोहराव नहीं)।
"""
import argparse, os, re, urllib.request

BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/hin_Deva/train"
UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--rows", type=int, default=150000)
ap.add_argument("--out", default="data/web.txt")
a = ap.parse_args()

name = f"{a.shard:03d}_00000.parquet"
url = f"{BASE}/{name}"
local = f"/tmp/{name}"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

if not os.path.exists(local):
    print(f"उतार रहे: {name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(local, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
print(f"आकार: {os.path.getsize(local)/1e6:.0f} MB", flush=True)

import pyarrow.parquet as pq
pf = pq.ParquetFile(local)
print(f"पंक्तियाँ: {pf.metadata.num_rows:,}", flush=True)

n = 0
with open(a.out, "w", encoding="utf-8") as o:
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if not t:
                continue
            for s in re.split(r"[।\n]+", t):
                ws = s.split()
                if 5 <= len(ws) <= 30 and not re.search(r"[A-Za-z]{5,}", s):
                    o.write(" ".join(ws) + "।\n")
                    n += 1
        if n >= a.rows:
            break
print(f"वाक्य {n:,} -> {a.out}", flush=True)
os.remove(local)
print("अस्थायी फ़ाइल हटाई", flush=True)
