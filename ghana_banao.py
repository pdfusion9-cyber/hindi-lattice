#!/usr/bin/env python3
"""ghana_banao.py -- घने तथ्य-समूह: हर डोमेन की 100-300 इकाइयाँ, 15-25 गुण।

  python3 ghana_banao.py --domain desh      --out gyaan_desh.txt
  python3 ghana_banao.py --domain sab       --out gyaan_ghana_all.txt
"""
import argparse, json, os, time, urllib.parse, urllib.request

UA = "HindiLatticeBot/1.0 (https://github.com/pdfusion9-cyber/hindi-lattice)"
EP = "https://query.wikidata.org/sparql"

# हर डोमेन: (इकाई-वर्ग, [(गुण, हिंदी नाम)], सीमा)
D = {
 "desh": ("wd:Q6256", [("P36","राजधानी"),("P1082","जनसंख्या"),("P2046","क्षेत्रफल"),
          ("P30","महाद्वीप"),("P38","मुद्रा"),("P37","राजभाषा"),("P571","स्थापना"),
          ("P297","देश कोड"),("P474","दूरभाष कोड"),("P78","वाहन दिशा")], 300),
 "rajya": ("wd:Q12443800", [("P36","राजधानी"),("P1082","जनसंख्या"),("P2046","क्षेत्रफल"),
           ("P37","राजभाषा"),("P571","स्थापना"),("P17","देश")], 200),
 "nadi": ("wd:Q4022", [("P403","मुहाना"),("P17","देश"),("P2043","लंबाई"),
          ("P885","स्रोत"),("P974","सहायक नदी")], 400),
 "parvat": ("wd:Q8502", [("P4552","पर्वत श्रृंखला"),("P2044","ऊंचाई"),("P17","देश"),
            ("P625","निर्देशांक")], 400),
 "tatva": ("wd:Q11344", [("P246","प्रतीक"),("P1086","परमाणु संख्या"),("P2067","परमाणु भार"),
           ("P61","खोजकर्ता"),("P575","खोज वर्ष")], 200),
 "film": ("wd:Q11424", [("P57","निर्देशक"),("P577","वर्ष"),("P495","देश"),
          ("P364","भाषा"),("P161","कलाकार")], 500),
 "pustak": ("wd:Q571", [("P50","लेखक"),("P577","वर्ष"),("P407","भाषा"),
            ("P123","प्रकाशक"),("P1104","पृष्ठ")], 400),
 "shahar": ("wd:Q515", [("P17","देश"),("P1082","जनसंख्या"),("P131","जिला"),
            ("P571","स्थापना"),("P625","निर्देशांक")], 500),
 "vishwavidyalaya": ("wd:Q3918", [("P17","देश"),("P571","स्थापना"),("P131","स्थान"),
                     ("P2196","छात्र संख्या")], 300),
 "jheel": ("wd:Q23397", [("P17","देश"),("P2046","क्षेत्रफल"),("P4519","गहराई"),
           ("P2043","लंबाई")], 200),
}

def chala(q):
    url = EP + "?" + urllib.parse.urlencode({"query": q, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=100) as r:
        return json.load(r)

def saaf(v):
    if not v: return None
    v = str(v).strip()
    if not v or (v.startswith("Q") and v[1:].isdigit()) or len(v) < 2 or "http" in v: return None
    return v[:60]

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

doms = list(D) if a.domain == "sab" else [a.domain]
f = open(a.out, "w", encoding="utf-8")
total = 0
for i, dname in enumerate(doms):
    cls, props, lim = D[dname]
    lim = a.limit or lim
    sel = " ".join(f"OPTIONAL{{?x wdt:{p} ?v{j}}}" for j, (p, _) in enumerate(props))
    q = f"""SELECT ?xLabel {' '.join(f'?v{j}' for j in range(len(props)))} WHERE {{
      ?x wdt:P31 {cls} . {sel}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "hi,en". }} }} LIMIT {lim}"""
    got = 0
    for att in range(2):
        try:
            d = chala(q)
            for b in d["results"]["bindings"]:
                x = saaf(b.get("xLabel", {}).get("value"))
                if not x: continue
                for j, (p, rel) in enumerate(props):
                    v = saaf(b.get(f"v{j}", {}).get("value"))
                    if v and v != x:
                        f.write(f"{x} | {rel} | {v}\n"); got += 1
            break
        except Exception as e:
            print(f"  {dname} प्रयास{att+1}: {type(e).__name__}", flush=True)
            if att == 0: time.sleep(66)
    f.flush()
    print(f"{dname}: {got:,} तथ्य", flush=True)
    total += got
    if i < len(doms) - 1: time.sleep(66)
f.close()
print(f"कुल {total:,} -> {a.out} ({os.path.getsize(a.out)/1024:.0f} KB)", flush=True)
