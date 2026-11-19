#!/usr/bin/env python3
"""Fetch domain training text from the repo's shard branch, with Hindi parquet fallback."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request

UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"
DOMAINS = {"science", "math", "reasoning", "news", "qa", "general",
           "coding", "document", "history", "geography"}
RAW = "https://raw.githubusercontent.com/pdfusion9-cyber/hindi-lattice"

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
ap.add_argument("--out", default="store")
ap.add_argument("--per-dataset", type=int, default=8000)
ap.add_argument("--shard", type=int, default=0)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
out = os.path.join(a.out, f"baat_{a.domain}.txt")
sources = [
    f"{RAW}/shard-{a.domain}/store/baat.txt",
    f"{RAW}/shard-{a.domain}/store/baat_{a.domain}.txt",
]

def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)
    return os.path.getsize(path)

# Prefer the already curated domain Q&A from its shard branch.
for i, url in enumerate(sources):
    tmp = out + ".tmp"
    try:
        size = download(url, tmp)
        if size >= 100:
            os.replace(tmp, out)
            print(f"domain source: {url} | {size:,} bytes", flush=True)
            break
    except Exception as exc:
        print(f"domain source unavailable ({type(exc).__name__}): {url}", flush=True)
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
else:
    # Fallback to a fresh Hindi FineWeb-2 shard. This keeps the training job
    # useful even when an older shard branch has not published Q&A text.
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("fallback के लिए pyarrow जरूरी है") from exc
    idx = a.shard % 8
    name = f"{idx:03d}_00000.parquet"
    url = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/hin_Deva/train/{name}"
    local = f"/tmp/hindi-domain-{idx}.parquet"
    print(f"domain branch खाली; Hindi corpus fallback: {name}", flush=True)
    if not os.path.isfile(local) or os.path.getsize(local) < 100000:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as response, open(local, "wb") as f:
            shutil.copyfileobj(response, f)
    pf = pq.ParquetFile(local)
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                for sentence in re.split(r"[।\n]+", text):
                    words = sentence.split()
                    if 5 <= len(words) <= 80:
                        f.write(" ".join(words) + "\n")
                        written += 1
                        if written >= a.per_dataset:
                            break
                if written >= a.per_dataset:
                    break
            if written >= a.per_dataset:
                break
    try:
        os.remove(local)
    except OSError:
        pass
    print(f"Hindi fallback rows: {written:,}", flush=True)

if not os.path.isfile(out) or os.path.getsize(out) < 100:
    raise SystemExit(f"domain corpus empty: {out}")
print(f"saved: {out} | {os.path.getsize(out):,} bytes", flush=True)
