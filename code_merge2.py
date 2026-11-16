#!/usr/bin/env python3
"""code_merge2.py -- SUDHRA milaan: virodhi-shabd bhaar + jodi pehchan.

Do sudhar:
  1. VIRODHI BHAAR -- alag karne wale shabd (gcd, maximum, factorial) ka bhaar 3x
  2. JODI          -- akela shabd nahi, jodi bhi (find+gcd)  [muft, aakar nahi badhta]
"""
import glob, json, math, os, re, sys, zlib
from collections import Counter, defaultdict

RES = []
for p in sorted(glob.glob("in/shard*.json")):
    try:
        RES += json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  {p} fail", flush=True)

seen = set(); UNIQ = []
for r in RES:
    k = r["q"].strip().lower()
    if k in seen: continue
    seen.add(k); UNIQ.append(r)
print(f"kul {len(RES)} | unique {len(UNIQ)}", flush=True)

STOP = set("""write a an the how do i in to of and for is are use using with python function
create make get give show me please can you code program script method return that whether""".split())

# ---------- 1) VIRODHI SHABD (alag karne wale) ----------
VIRODHI = set("""add subtract multiply divide reverse palindrome prime even odd factorial fibonacci
vowel consonant duplicate sort read write exception dictionary key value join merge
upper lower capitalize space trim char length count max min square root power abs round
avg median mode variance std convert encode decode binary decimal roman ascii base64
date time day week month year age random password email phone validate search find
index insert remove delete append copy exists class object inheritance lambda map
filter reduce comprehension enumerate zip any all type check compare swap leap
celsius fahrenheit interest area circle rectangle perimeter anagram compress longest
word sentence punctuation snake camel case balanced parentheses stack queue bubble
insertion selection quick linear missing pair triplet zero intersection union prefix
suffix equal group kth none invert hash bmi tax timer countdown hour second valid
days solve permutation combination subset path grid tower hanoi recursion http
request json regex regular expression environment variable command argument line
measure execution sleep format current generate numeric start end repeat pad swap
gcd lcm divisor divisor armstrong digit sum""".split())

SUFFIX = [("ies","y"),("sses","ss"),("ing",""),("ed",""),("es",""),("s",""),("e","")]
SYN = {"addition":"add","adding":"add","added":"add","sum":"add","total":"add","plus":"add",
 "subtraction":"subtract","minus":"subtract","difference":"subtract",
 "multiplication":"multiply","product":"multiply","times":"multiply",
 "division":"divide","quotient":"divide","remainder":"remainder","modulo":"remainder",
 "reversing":"reverse","reversed":"reverse","backward":"reverse",
 "vowels":"vowel","consonants":"consonant","duplicates":"duplicate","unique":"duplicate",
 "sorting":"sort","sorted":"sort","reading":"read","writing":"write","written":"write",
 "errors":"exception","error":"exception","try":"exception","custom":"custom",
 "dict":"dictionary","keys":"key","values":"value","joining":"join","concatenate":"join",
 "concat":"join","strings":"string","text":"string","array":"list","numbers":"number",
 "integer":"number","int":"number","digits":"digit","files":"file","folder":"directory",
 "uppercase":"upper","lowercase":"lower","capital":"capitalize","spaces":"space",
 "trim":"trim","strip":"trim","characters":"char","letter":"char","letters":"char",
 "size":"length","counting":"count","frequency":"count","occurrence":"count",
 "maximum":"max","largest":"max","minimum":"min","smallest":"min","absolute":"abs",
 "average":"avg","mean":"avg","squares":"square","deviation":"std","conversion":"convert",
 "position":"index","search":"find","exists":"exists","sorting":"sort"}

def samaanya(w):
    w = w.lower()
    if w in SYN: return SYN[w]
    for suf, rep in SUFFIX:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:len(w)-len(suf)] + rep
    return w

def shabd(q):
    out = []
    for w in re.findall(r"[a-z]+", q.lower()):
        if w in STOP: continue
        n = samaanya(w)
        if n not in STOP and len(n) > 1: out.append(n)
    return out

# ---------- TF-IDF + virodhi bhaar + jodi ----------
docs = [shabd(r["q"]) for r in UNIQ]
df = Counter()
for d in docs:
    for w in set(d): df[w] += 1
N = len(docs)
idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}

def vector(q):
    ws = shabd(q)
    tf = Counter(ws)
    v = {}
    for w, c in tf.items():
        bhaar = (1 + math.log(c)) * idf.get(w, math.log(N + 1) + 1.0)
        if w in VIRODHI: bhaar *= 3.0            # 1) virodhi shabd
        v[w] = bhaar
    # 2) JODI -- paas-paas aane wale shabd
    for a, b in zip(ws, ws[1:]):
        if a in STOP or b in STOP: continue
        key = f"{a}+{b}"
        v[key] = v.get(key, 0.0) + (idf.get(a, 1.0) + idf.get(b, 1.0)) * 0.8
    nrm = math.sqrt(sum(x*x for x in v.values())) or 1.0
    return {w: x/nrm for w, x in v.items()}

JAAL = [(vector(r["q"]), r["code"], r["q"]) for r in UNIQ]

def jawab(q):
    vq = vector(q)
    best, bs, bq = None, 0.0, ""
    for vp, code, orig in JAAL:
        s = sum(x * vp.get(w, 0.0) for w, x in vq.items())
        if s > bs: bs, best, bq = s, code, orig
    return (best or ""), bs, bq

blob = json.dumps([r["code"] for r in UNIQ], ensure_ascii=False)
comp = zlib.compress(blob.encode("utf-8"), 9)
print(f"\n=== AAKAR ===", flush=True)
print(f"  JAAL zip: {len(comp)/1e6:.4f} MB ({len(comp)/1e3:.1f} KB)", flush=True)
print(f"  ghatav:   {1976*1e6/len(comp):.0f}x", flush=True)
print(f"  patterns: {len(JAAL)} | shabdakosh: {len(idf)}", flush=True)

TEST = [
 ("Write a function that adds two numbers.", ["add"], ["subtract","multiply"]),
 ("Create a Python function to reverse a string.", ["revers"], ["sort"]),
 ("How can I check whether a number is prime?", ["prime"], ["even","odd"]),
 ("Write code to subtract two numbers.", ["subtract"], ["add","multiply"]),
 ("How do I read from a file?", ["read"], ["write"]),
 ("How to sort a list in Python?", ["sort"], ["revers"]),
 ("Write a function to count vowels.", ["vowel"], ["consonant"]),
 ("Find the factorial of a number.", ["factorial"], ["fibonacci"]),
 ("Check if a string is a palindrome.", ["palindrome"], ["anagram"]),
 ("How do I create a dictionary?", ["dict"], ["list"]),
 ("Write a function to find maximum in a list.", ["max"], ["min"]),
 ("How do I handle exceptions?", ["except","try","error","Exception"], []),
 ("Write a function to convert Celsius to Fahrenheit.", ["celsius","fahrenheit","C","F"], []),
 ("Write a function to implement bubble sort.", ["bubble"], ["quick","merge"]),
 ("How do I use a lambda function?", ["lambda"], []),
 ("Write a function to calculate area of circle.", ["circle"], ["rectangle"]),
 ("Write a function to check leap year.", ["leap"], []),
 ("How do I parse JSON in Python?", ["json"], []),
 ("Write a function to find GCD of two numbers.", ["gcd","divisor"], ["lcm","max"]),
 ("How do I generate a random number?", ["random"], []),
]
print(f"\n=== TEST ({len(TEST)}) ===", flush=True)
sahi = 0; GALAT = []
for q, chahiye, nahi in TEST:
    got, sc, orig = jawab(q)
    low = got.lower()
    ok = any(c.lower() in low for c in chahiye) and not any(n.lower() in low for n in nahi)
    sahi += ok
    if not ok: GALAT.append((q, orig, got[:90]))
    print(f"[{'SAHI ' if ok else 'GALAT'}] {q[:50]:<52} {sc:.2f}", flush=True)
print(f"\n  SAHI: {sahi}/{len(TEST)} = {sahi/len(TEST)*100:.0f}%   (pichhla 70%)", flush=True)
if GALAT:
    print("\n  GALAT:", flush=True)
    for q, o, g in GALAT:
        print(f"    {q[:42]:<44} -> mila: {o[:38]}", flush=True)
