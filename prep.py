#!/usr/bin/env python3
"""prep.py -- विकिपीडिया डंप से साफ़ हिंदी पाठ बनाओ (रनर के अंदर)।"""
import argparse, bz2, os, re, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--articles", type=int, default=60000)
ap.add_argument("--out", default="data/corpus.txt")
a = ap.parse_args()

URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"
if not os.path.exists("dump.bz2"):
    print("डंप डाउनलोड...", flush=True)
    urllib.request.urlretrieve(URL, "dump.bz2")

def saaf(t):
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://[^\]]*\]", "", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"==+[^=]*==+", "।", t)
    t = re.sub(r"^[*#:;].*$", "", t, flags=re.M)
    return re.sub(r"[ \t]+", " ", t)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
art = 0
with bz2.open("dump.bz2", "rt", encoding="utf-8", errors="ignore") as f, \
     open(a.out, "w", encoding="utf-8") as o:
    buf = []
    for line in f:
        if "<text" in line or buf:
            buf.append(line)
            if "</text>" in line:
                t = "".join(buf); buf = []
                t = t.split(">", 1)[-1].rsplit("</text>", 1)[0]
                if len(t) < 200:
                    continue
                art += 1
                o.write(saaf(t)); o.write("\n")
                if art >= a.articles:
                    break
print(f"लेख {art:,} -> {a.out}", flush=True)
