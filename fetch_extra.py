#!/usr/bin/env python3
"""fetch_extra.py -- हिंदी के बड़े QA डेटासेट HuggingFace से उतारो (सवाल | जवाब)।"""
import argparse, json, os, time, urllib.parse, urllib.request

UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"
DS = [
    "kaifahmad/indian-history-hindi-QA-3.4k",
    "AvAkanksh/dharampal_hindi_qa_bharatgen",
    "deepanshumiglani0408/hindi-qa-dataset",
    "KshitizKhatri/hindi_qa_kaggle",
]


def rows(ds, offset, length=100):
    url = ("https://datasets-server.huggingface.co/rows?" +
           urllib.parse.urlencode({"dataset": ds, "config": "default", "split": "train",
                                   "offset": offset, "length": length}))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="store/baat_qa.txt")
ap.add_argument("--per-dataset", type=int, default=4000)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

QF = ("Question", "question", "प्रश्न", "question_hi", "input", "instruction")
AF = ("Answer", "answer", "उत्तर", "answer_hi", "output", "response")


def pick(row, names):
    for n in names:
        if n in row and isinstance(row[n], str) and row[n].strip():
            return row[n].strip()
    return None


total = 0
with open(a.out, "w", encoding="utf-8") as f:
    f.write("# HuggingFace हिंदी QA\n")
    for ds in DS:
        got = 0
        for off in range(0, a.per_dataset, 100):
            try:
                d = rows(ds, off)
            except Exception as e:
                print(f"{ds} @{off}: {type(e).__name__}", flush=True)
                break
            rs = d.get("rows", [])
            if not rs:
                break
            for x in rs:
                r = x.get("row", {})
                q = pick(r, QF)
                ans = pick(r, AF)
                if not q or not ans:
                    continue
                q = " ".join(q.split())[:300]
                ans = " ".join(ans.split())[:500]
                f.write(f"{q} | {ans}\n")
                got += 1
            time.sleep(0.2)
        print(f"{ds}: {got}", flush=True)
        total += got
print(f"कुल जोड़े: {total:,} -> {a.out}", flush=True)
