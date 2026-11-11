#!/usr/bin/env python3
"""build_store.py -- उसी पाठ से जाल-भंडार: पाथ, तथ्य, बातचीत।

यही विशेष वास्तुकला है: मॉडल वज़न में भाषा सीखता है,
यह हिस्सा जाल में ज्ञान भरता है -- दोनों एक ही डेटा से।
"""
import argparse, os, re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/corpus.txt")
ap.add_argument("--out", default="store")
ap.add_argument("--max-sent", type=int, default=200000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

CH = {"में","से","पर","को","ने","का","की","के","और","या","तक","लिए","द्वारा","भी","ही","यह","वह","इस","उस"}
KR = {"है","हैं","था","थी","थे","हो","हुआ","हुई","गया","गई","कर","किया","जाता","जाती","रहा","रही","स्थित","कहलाता"}
NEG = {"नहीं","न"}

NIYAM = [
    ("की राजधानी", re.compile(r"^(\S+)\s+की\s+राजधानी\s+(\S+)\s+है")),
    ("में स्थित है", re.compile(r"^(\S+)\s+(\S+)\s+में\s+स्थित\s+है")),
    ("की भाषा", re.compile(r"^(\S+)\s+की\s+(?:राजभाषा|भाषा)\s+(\S+)\s+है")),
    ("का जिला", re.compile(r"^(\S+)\s+(\S+)\s+जिले?\s+में\s+है")),
]

txt = open(a.data, encoding="utf-8").read()
sents = []
for s in re.split(r"[।\n]+", txt):
    ws = s.split()
    if 5 <= len(ws) <= 30 and not re.search(r"[A-Za-z]{5,}", s):
        sents.append(" ".join(ws) + "।")
    if len(sents) >= a.max_sent:
        break
uni = Counter(w for s in sents for w in s.split())
print(f"वाक्य {len(sents):,}", flush=True)

def ok(w):
    return (w and w not in CH and w not in KR and w not in NEG
            and not w.isdigit() and len(w) >= 3 and "." not in w and uni[w] >= 5)

tathya = Counter()
for s in sents:
    for naam, rx in NIYAM:
        m = rx.match(s.replace("।", ""))
        if not m:
            continue
        x, y = m.group(1), m.group(2)
        if ok(x) and ok(y) and x != y:
            tathya[(x, naam, y)] += 1

with open(f"{a.out}/gyaan.txt", "w", encoding="utf-8") as f:
    f.write("# पाठ से निकाले तथ्य\n")
    for (x, r, y), c in tathya.most_common(20000):
        f.write(f"{x} | {r} | {y}\n")

with open(f"{a.out}/vakya.txt", "w", encoding="utf-8") as f:
    for s in sents[:100000]:
        f.write(s + "\n")

n = 0
with open(f"{a.out}/baat.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(sents):
        if s.startswith("क्या ") and i + 1 < len(sents):
            f.write(f"{s.replace('।','')} | {sents[i+1]}\n")
            n += 1
            if n >= 5000:
                break

print(f"तथ्य {len(tathya):,} | वाक्य 100000 | बातचीत {n:,}", flush=True)
