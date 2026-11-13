#!/usr/bin/env python3
"""wd_jama.py -- Wikidata se thah aur kism jama karo (dheeraj se)."""
import json, os, sys, time, urllib.parse, urllib.request

UA = "hindi-lattice/1.0 (research; github.com/pdfusion9-cyber/hindi-lattice)"
LANG = "hi"
GAP = 8

REL = [("P36","राजधानी"),("P30","महाद्वीप"),("P1082","जनसंख्या"),("P2046","क्षेत्रफल"),
       ("P37","राजभाषा"),("P571","स्थापना"),("P17","देश"),("P131","स्थित"),
       ("P2043","लंबाई"),("P2044","ऊंचाई"),("P1082","जनसंख्या")]
TYPE_Q = ["Q6256","Q3624078","Q515","Q1549591","Q4022","Q8502","Q46831",
          "Q34770","Q5","Q7275","Q5107","Q23442","Q23397"]

def q(sp, tries=5):
    url = "https://query.wikidata.org/sparql" + urllib.parse.quote(sp)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            if isinstance(d, dict):
                return d.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"    (try {i+1}: {type(e).__name__})", flush=True)
            time.sleep(GAP * (i + 1))
    return []

facts = set(); kism = {}
t0 = time.time()
LIM = 60000

print("=== kism ===", flush=True)
vals = " ".join(f"wd:{k}" for k in TYPE_Q)
sp = f'''SELECT ?sL ?tL WHERE {{
  ?s wdt:P31 ?t . VALUES ?t {{ {vals} }}
  ?s rdfs:label ?sL . FILTER(LANG(?sL) = "{LANG}")
  ?t rdfs:label ?tL . FILTER(LANG(?tL) = "{LANG}")
}} LIMIT {LIM}'''
rows = q(sp)
for x in rows:
    try:
        s = x["sL"]["value"].strip(); t = x["tL"]["value"].strip()
        if s and t: kism[s] = t
    except Exception: pass
print(f"  kism {len(kism):,}", flush=True)
time.sleep(GAP)

print("=== thah ===", flush=True)
for prop, naam in REL:
    if time.time() - t0 > 14000: break
    sp = f'''SELECT ?sL ?oL WHERE {{
  ?s wdt:{prop} ?o .
  ?s rdfs:label ?sL . FILTER(LANG(?sL) = "{LANG}")
  ?o rdfs:label ?oL . FILTER(LANG(?oL) = "{LANG}")
}} LIMIT {LIM}'''
    rows = q(sp)
    n0 = len(facts)
    for x in rows:
        try:
            s = x["sL"]["value"].strip(); o = x["oL"]["value"].strip()
            if s and o and s != o: facts.add((s, naam, o))
        except Exception: pass
    print(f"  {naam:<10} +{len(facts)-n0:>7,} | kul {len(facts):,}", flush=True)
    with open("gyaan_wd.txt","w",encoding="utf-8") as f:
        f.write("# Wikidata\n")
        for s,r,o in sorted(facts): f.write(f"{s} | {r} | {o}\n")
    time.sleep(GAP)

with open("gyaan_wd.txt","w",encoding="utf-8") as f:
    f.write("# Wikidata\n")
    for s,r,o in sorted(facts): f.write(f"{s} | {r} | {o}\n")
with open("gyaan_wd.txt.kism.txt","w",encoding="utf-8") as f:
    for s,t in sorted(kism.items()): f.write(f"{s} | {t}\n")
print(f"kul thah {len(facts):,} | kism {len(kism):,}")
