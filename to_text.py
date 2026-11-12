#!/usr/bin/env python3
"""to_text.py -- parquet + txt से सादा पाठ। खाली हो तो साफ़ त्रुटि।"""
import argparse, glob, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data")
ap.add_argument("--out", default="data/corpus.txt")
ap.add_argument("--max-chars", type=int, default=200_000_000)
a = ap.parse_args()
tot = 0
with open(a.out, "w", encoding="utf-8") as f:
    # पहले parquet
    try:
        import pyarrow.parquet as pq
        for fn in sorted(glob.glob(os.path.join(a.dir, "*.parquet"))):
            try:
                t = pq.read_table(fn, columns=["text"])
            except Exception as e:
                print("skip", fn, type(e).__name__, file=sys.stderr); continue
            for v in t.column("text").to_pylist():
                if not v: continue
                for ln in v.split("\n"):
                    ln = ln.strip()
                    if len(ln) < 20: continue
                    f.write(ln + "\n"); tot += len(ln)
                if tot >= a.max_chars: break
            print(f"{fn}: {tot/1e6:.0f}M", flush=True)
            if tot >= a.max_chars: break
    except ImportError:
        print("pyarrow नहीं -- सिर्फ़ txt", file=sys.stderr)
    # फिर txt
    if tot < a.max_chars:
        for fn in sorted(glob.glob(os.path.join(a.dir, "*.txt"))):
            if os.path.basename(fn) in ("corpus.txt",): continue
            try:
                for ln in open(fn, encoding="utf-8", errors="ignore"):
                    ln = ln.strip()
                    if len(ln) < 20 or "|" in ln: continue
                    f.write(ln + "\n"); tot += len(ln)
                    if tot >= a.max_chars: break
            except Exception as e:
                print("skip", fn, type(e).__name__, file=sys.stderr)
            print(f"{fn}: {tot/1e6:.0f}M", flush=True)
            if tot >= a.max_chars: break
print(f"कुल {tot/1e6:.1f}M अक्षर -> {a.out}", flush=True)
if tot < 1_000_000:
    print("कोष बहुत छोटा -- रुक रहे हैं", flush=True)
    sys.exit(1)
