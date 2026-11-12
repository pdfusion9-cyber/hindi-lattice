#!/usr/bin/env python3
"""loop_fetch.py -- एक डोमेन के लिए अलग डेटा भंडार (कभी कम न पड़े)।"""
import argparse, os, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--domain", default="general")
ap.add_argument("--files", type=int, default=6)
ap.add_argument("--out", default="data/corpus.txt")
a = ap.parse_args()
os.makedirs("data", exist_ok=True)

BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/hin_Hani/train"
got = 0
for k in range(a.files):
    idx = a.shard * a.files + k
    url = f"{BASE}/{idx:03d}_00000.parquet"
    out = f"data/p{idx}.parquet"
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        got += 1
        continue
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out)
            got += 1
            print("got", idx, os.path.getsize(out) // 1024, "KB", flush=True)
            break
        except Exception as e:
            print("miss", idx, type(e).__name__, flush=True)

for name, url in (
    ("data/baat.txt", f"https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice/shard-{a.domain}/store/baat_{a.domain}.txt"),
    ("data/gyaan.txt", "https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice/facts/data/gyaan_sab.txt"),
    ("data/gyaan2.txt", "https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice/facts/data/gyaan_ghana.txt"),
):
    try:
        urllib.request.urlretrieve(url, name)
        print("got", name, os.path.getsize(name) // 1024, "KB", flush=True)
    except Exception:
        open(name, "w").write("")
print("parquet:", got, flush=True)
