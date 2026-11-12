#!/usr/bin/env python3
"""loop_fetch.py -- एक डोमेन के लिए अलग डेटा भंडार।

हिंदी (hin_Deva) parquet में सिर्फ़ 8 फ़ाइलें हैं (000-007), इसलिए:
  shard 0-7  -> parquet फ़ाइल 000-007 (हर एक ~100-200 MB)
  shard 8-9  -> repo की vakya.txt, अलग हिस्से
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

got = 0
if a.shard <= 7:
    idx = a.shard
    url = f"{BASE}/{idx:03d}_00000.parquet"
    out = f"data/p{idx}.parquet"
    if os.path.exists(out) and os.path.getsize(out) > 100000:
        got = 1
        print("have", idx, os.path.getsize(out) // 1024, "KB", flush=True)
    else:
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(url, out)
                if os.path.getsize(out) < 100000:
                    raise IOError("too small")
                got = 1
                print("got", idx, os.path.getsize(out) // 1024, "KB", flush=True)
                break
            except Exception as e:
                print("miss", idx, attempt, type(e).__name__, flush=True)
else:
    # 8, 9 -- repo की vakya.txt के अलग हिस्से
    for br in ("shard-math", "shard-science"):
        try:
            urllib.request.urlretrieve(f"{RAW}/{br}/store/vakya.txt", f"data/v_{br}.txt")
            got += 1
            print("got vakya", br, os.path.getsize(f"data/v_{br}.txt") // 1024, "KB", flush=True)
        except Exception as e:
            print("miss vakya", br, type(e).__name__, flush=True)
    # अलग हिस्सा: shard 8 पहला आधा, shard 9 दूसरा आधा
    src = "data/v_shard-math.txt"
    if os.path.exists(src):
        lines = open(src, encoding="utf-8", errors="ignore").readlines()
        half = len(lines) // 2
        part = lines[:half] if a.shard == 8 else lines[half:]
        with open("data/vakya_part.txt", "w", encoding="utf-8") as f:
            f.writelines(part)
        print("vakya part", a.shard, len(part), "पंक्तियाँ", flush=True)

for name, url in (
    ("data/baat.txt", f"{RAW}/shard-{a.domain}/store/baat_{a.domain}.txt"),
    ("data/gyaan.txt", f"{RAW}/facts/data/gyaan_sab.txt"),
    ("data/gyaan2.txt", f"{RAW}/facts/data/gyaan_ghana.txt"),
):
    try:
        urllib.request.urlretrieve(url, name)
        print("got", name, os.path.getsize(name) // 1024, "KB", flush=True)
    except Exception:
        open(name, "w").write("")

print("स्रोत:", got, flush=True)
if got == 0:
    print("कोई डेटा नहीं मिला", flush=True)
    sys.exit(1)
