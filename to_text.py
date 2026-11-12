#!/usr/bin/env python3
"""to_text.py -- parquet + txt से सादा पाठ, टुकड़ों में (रैम सीमित रहे)।"""
import argparse, glob, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data")
ap.add_argument("--out", default="data/corpus.txt")
ap.add_argument("--max-chars", type=int, default=150_000_000)
ap.add_argument("--batch", type=int, default=2000)
a = ap.parse_args()
tot = 0

with open(a.out, "w", encoding="utf-8") as f:
    try:
        import pyarrow.parquet as pq
        for fn in sorted(glob.glob(os.path.join(a.dir, "*.parquet"))):
            try:
                pf = pq.ParquetFile(fn)
            except Exception as e:
                print("skip", fn, type(e).__name__, flush=True)
                continue
            print(f"{os.path.basename(fn)}: {pf.metadata.num_row_groups} row-groups", flush=True)
            stop = False
            try:
                for b in pf.iter_batches(batch_size=a.batch, columns=["text"]):
                    for v in b.column("text").to_pylist():
                        if not v:
                            continue
                        for ln in v.split("\n"):
                            ln = ln.strip()
                            if len(ln) < 20:
                                continue
                            f.write(ln + "\n")
                            tot += len(ln)
                    if tot >= a.max_chars:
                        stop = True
                        break
            except Exception as e:
                print("read stop", type(e).__name__, flush=True)
            print(f"  {os.path.basename(fn)} -> {tot/1e6:.0f}M", flush=True)
            if stop or tot >= a.max_chars:
                break
    except ImportError:
        print("pyarrow नहीं", flush=True)

    if tot < a.max_chars:
        for fn in sorted(glob.glob(os.path.join(a.dir, "*.txt"))):
            if os.path.basename(fn) == "corpus.txt":
                continue
            try:
                with open(fn, encoding="utf-8", errors="ignore") as g:
                    for ln in g:
                        if "|" in ln:
                            continue
                        ln = ln.strip()
                        if len(ln) < 20:
                            continue
                        f.write(ln + "\n")
                        tot += len(ln)
                        if tot >= a.max_chars:
                            break
            except Exception:
                pass
            print(f"  {os.path.basename(fn)} -> {tot/1e6:.0f}M", flush=True)
            if tot >= a.max_chars:
                break

print(f"कुल {tot/1e6:.1f}M अक्षर -> {a.out}", flush=True)
if tot < 1_000_000:
    print("कोष बहुत छोटा", flush=True)
    sys.exit(1)
