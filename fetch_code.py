#!/usr/bin/env python3
"""fetch_code.py -- कोड डेटा सीधे (datasets-server बंद हो तब भी चले)।

  python3 fetch_code.py --out store --rows 25000
"""
import argparse, json, os, urllib.request

UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"
SRC = [
    ("https://huggingface.co/datasets/flytech/python-codes-25k/resolve/main/python-codes-25k.jsonl", "jsonl"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="store")
ap.add_argument("--rows", type=int, default=25000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

total = 0
with open(f"{a.out}/baat_coding.txt", "w", encoding="utf-8") as o:
    o.write("# कोडिंग डोमेन\n")
    for url, kind in SRC:
        local = "/tmp/code_ds." + kind
        if not os.path.exists(local):
            print(f"उतार रहे: {url.split('/')[-1]}", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=600) as r, open(local, "wb") as f:
                    while True:
                        c = r.read(1 << 20)
                        if not c:
                            break
                        f.write(c)
            except Exception as e:
                print(f"  डाउनलोड विफल: {type(e).__name__}", flush=True)
                continue
        print(f"आकार: {os.path.getsize(local)/1e6:.0f} MB", flush=True)
        got = 0
        with open(local, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if got >= a.rows:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ins = (r.get("instruction") or "").strip()
                out = (r.get("output") or "").strip()
                if not ins or not out:
                    continue
                o.write(" ".join(ins.split())[:300] + " | " + " ".join(out.split())[:500] + "\n")
                got += 1
        print(f"  जोड़े: {got:,}", flush=True)
        total += got
print(f"कोडिंग: कुल {total:,} -> {a.out}/baat_coding.txt", flush=True)
