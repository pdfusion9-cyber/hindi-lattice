#!/usr/bin/env python3
"""fetch_domain.py -- डोमेन-वार डेटा, दोबारा-कोशिश और डेटा-कर्सर के साथ।

  python3 fetch_domain.py --domain math --out store --shard 2
"""
import argparse, json, os, time, urllib.parse, urllib.request

UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"

DOMAINS = {
    "science": [("model2me/scienceqa-multilingual-hindi", None, None)],
    "math": [("BhabhaAI/orca-math-word-problems-200k-hindi-filtered", "question", "answer"),
             ("GenVRadmin/Aryabhatta-Orca-Maths-Hindi", "question", "response")],
    "reasoning": [("Athekunal/english-hindi-reasoning-dataset", "question", "answer")],
    "news": [("AIhnIndicRag/Hindi_cc_news", None, None)],
    "qa": [("kaifahmad/indian-history-hindi-QA-3.4k", "Question", "Answer"),
           ("AvAkanksh/dharampal_hindi_qa_bharatgen", "question", "answer")],
    "translate": [("ai4bharat/samanantar", None, None)],
    "literature": [("ai4bharat/IndicCorpV2", None, None)],
    "general": [("kaifahmad/indian-history-hindi-QA-3.4k", "Question", "Answer")],
    "coding": [("iamtarun/python_code_instructions_18k_alpaca", "instruction", "output"),
               ("flytech/python-codes-25k", "instruction", "output")],
}
QF = ("Question", "question", "प्रश्न", "instruction", "input", "text", "en")
AF = ("Answer", "answer", "उत्तर", "output", "response", "code", "hi")


def rows(ds, offset, length=100):
    url = ("https://datasets-server.huggingface.co/rows?" +
           urllib.parse.urlencode({"dataset": ds, "config": "default", "split": "train",
                                   "offset": offset, "length": length}))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def rows_retry(ds, offset, tries=4):
    """सेवा बंद हो तो रुक-रुककर दोबारा कोशिश।"""
    for t in range(tries):
        try:
            return rows(ds, offset)
        except Exception as e:
            code = getattr(e, "code", None)
            wait = 5 * (t + 1)
            print(f"  {ds} @{offset} कोशिश {t+1}/{tries}: {type(e).__name__} {code or ''} -- {wait}s रुकें", flush=True)
            if t < tries - 1:
                time.sleep(wait)
    return None


def pick(row, names):
    for n in names:
        if n in row and isinstance(row[n], str) and row[n].strip():
            return row[n].strip()
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True, choices=list(DOMAINS))
ap.add_argument("--out", default="store")
ap.add_argument("--per-dataset", type=int, default=8000)
ap.add_argument("--shard", type=int, default=0, help="कर्सर: कौन-सा टुकड़ा (हर रन नया डेटा)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

total = 0
with open(f"{a.out}/baat_{a.domain}.txt", "w", encoding="utf-8") as f:
    f.write(f"# डोमेन: {a.domain} | शार्ड: {a.shard}\n")
    for ds, qf, af in DOMAINS[a.domain]:
        got = 0
        # कर्सर: हर शार्ड अलग हिस्से से शुरू
        start = a.shard * a.per_dataset
        for off in range(start, start + a.per_dataset, 100):
            d = rows_retry(ds, off)
            if d is None:
                print(f"  {ds}: सेवा नहीं मिली, आगे बढ़ रहे", flush=True)
                break
            rs = d.get("rows", [])
            if not rs:
                break
            for x in rs:
                r = x.get("row", {})
                q = pick(r, (qf,) if qf else QF)
                ans = pick(r, (af,) if af else AF)
                if not q or not ans:
                    vals = [v for v in r.values() if isinstance(v, str) and len(v) > 30]
                    if vals:
                        f.write(f"{vals[0][:200]} | {vals[0][:400]}\n")
                        got += 1
                    continue
                f.write(f"{' '.join(q.split())[:300]} | {' '.join(ans.split())[:500]}\n")
                got += 1
            time.sleep(0.2)
        print(f"{ds}: {got}", flush=True)
        total += got
print(f"डोमेन {a.domain} शार्ड {a.shard}: कुल {total:,} -> {a.out}/baat_{a.domain}.txt", flush=True)
