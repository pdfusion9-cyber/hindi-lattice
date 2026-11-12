#!/usr/bin/env python3
"""loop_fetch.py -- एक डोमेन के लिए अलग डेटा भंडार।

hin_Deva parquet में 8 फ़ाइलें (000-007)। 10 डोमेन के लिए:
  shard 0-7 -> फ़ाइल 000-007
  shard 8-9 -> फ़ाइल 000/001 फिर से (डोमेन-QA और तथ्य अलग रहते हैं)
"""
import argparse, os, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--domain", default="general")
ap.add_argument("--files", type=int, default=1)
a = ap.parse_args()
os.makedirs("data", exist_ok=True)

BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/hin_Deva/train"
RAW = "https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice"

idx = a.shard if a.shard <= 7 else (a.shard - 8)
got = 0
url = f"{BASE}/{idx:03d}_00000.parquet"
out = f"data/p{idx}.parquet"
if os.path.exists(out) and os.path.getsize(out) > 100000:
    got = 1
    print("have", idx, os.path.getsize(out) // 1024, "KB", flush=True)
else:
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, out)
            if os.path.getsize(out) < 100000:
                raise IOError("too small")
            got = 1
            print("got", idx, os.path.getsize(out) // 1024, "KB", flush=True)
            break
        except Exception as e:
            print("miss", idx, attempt, type(e).__name__, flush=True)

for name, url2 in (
    ("data/baat.txt", f"{RAW}/shard-{a.domain}/store/baat_{a.domain}.txt"),
    ("data/gyaan.txt", f"{RAW}/facts/data/gyaan_sab.txt"),
    ("data/gyaan2.txt", f"{RAW}/facts/data/gyaan_ghana.txt"),
):
    try:
        urllib.request.urlretrieve(url2, name)
        print("got", name, os.path.getsize(name) // 1024, "KB", flush=True)
    except Exception:
        open(name, "w").write("")

print("स्रोत:", got, flush=True)
if got == 0:
    print("कोई डेटा नहीं मिला", flush=True)
    sys.exit(1)
