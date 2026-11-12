#!/usr/bin/env python3
"""loop_fetch.py -- एक डोमेन का डेटा: parquet + सारा उपलब्ध repo डेटा।

डेटा कभी कम न पड़े:
  - parquet (hin_Deva 000-007)
  - gyaan_base / gyaan_sab / gyaan_ghana  (तथ्य)
  - हर शार्ड-ब्रांच का baat.txt            (डोमेन सवाल-जवाब)
  - document डोमेन को अतिरिक्त: सभी baat + gyaan
"""
import argparse, os, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--domain", default="general")
a = ap.parse_args()
os.makedirs("data", exist_ok=True)

BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/hin_Deva/train"
RAW = "https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice"

def get(url, out, minsize=1000):
    if os.path.exists(out) and os.path.getsize(out) >= minsize:
        return True
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, out)
            if os.path.getsize(out) >= minsize:
                return True
        except Exception:
            pass
    return False

idx = a.shard if a.shard <= 7 else (a.shard - 8)
got = 0
if get(f"{BASE}/{idx:03d}_00000.parquet", f"data/p{idx}.parquet", 100000):
    got += 1
    print("parquet", idx, os.path.getsize(f"data/p{idx}.parquet") // 1024, "KB", flush=True)

# सारा उपलब्ध तथ्य-डेटा
for name, url in (
    ("data/gyaan_base.txt", f"{RAW}/main/data/gyaan_base.txt"),
    ("data/gyaan_sab.txt", f"{RAW}/facts/data/gyaan_sab.txt"),
    ("data/gyaan_ghana.txt", f"{RAW}/facts/data/gyaan_ghana.txt"),
):
    if get(url, name, 1000):
        print("got", name, os.path.getsize(name) // 1024, "KB", flush=True)

# हर शार्ड-ब्रांच का baat (डोमेन सवाल-जवाब)
DOMAINS = ["science","math","reasoning","news","qa","general","coding","document","history","geography"]
for d in DOMAINS:
    out = f"data/baat_{d}.txt"
    if get(f"{RAW}/shard-{d}/store/baat.txt", out, 500):
        print("got", out, os.path.getsize(out) // 1024, "KB", flush=True)

print("parquet:", got, flush=True)
if got == 0:
    print("parquet नहीं मिला -- txt से ही काम चलेगा", flush=True)
