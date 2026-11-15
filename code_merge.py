#!/usr/bin/env python3
"""code_merge.py -- sab shard jodo, TF-IDF jaal banao, test karo."""
import glob, json, math, os, re, sys, zlib
from collections import Counter

# ---------- shard jodo ----------
RES = []
for p in sorted(glob.glob("in/shard*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
        RES += d
        print(f"  {os.path.basename(p)}: {len(d)}", flush=True)
    except Exception as e:
        print(f"  {p} fail {type(e).__name__}", flush=True)

# duplicate hatao (same sawaal)
seen = set(); UNIQ = []
for r in RES:
    k = r["q"].strip().lower()
    if k in seen: continue
    seen.add(k); UNIQ.append(r)
print(f"\nkul: {len(RES)} | unique: {len(UNIQ)}", flush=True)

# ---------- samaanya karan ----------
STOP = set("""write a an the how do i in to of and for is are use using with python function
create make get give show me please can you code program script method return that whether""".split())

SUFFIX = [("ies","y"),("sses","ss"),("ing",""),("ed",""),("es",""),("s",""),("e","")]
SYN = {"addition":"add","adding":"add","added":"add","sum":"add","total":"add","plus":"add",
 "subtraction":"subtract","subtracting":"subtract","minus":"subtract","difference":"subtract",
 "multiplication":"multiply","multiplying":"multiply","product":"multiply","times":"multiply",
 "division":"divide","dividing":"divide","quotient":"divide","remainder":"remainder","modulo":"remainder",
 "reversing":"reverse","reversed":"reverse","backward":"reverse","palindrome":"palindrome",
 "even":"even","odd":"odd","prime":"prime","factorial":"factorial","fibonacci":"fibonacci",
 "vowel":"vowel","vowels":"vowel","consonant":"consonant","duplicate":"duplicate","duplicates":"duplicate",
 "unique":"duplicate","sorting":"sort","sorted":"sort","reading":"read","writing":"write","written":"write",
 "exception":"exception","error":"exception","errors":"exception","try":"exception","custom":"custom",
 "dictionary":"dictionary","dict":"dictionary","map":"dictionary","key":"key","keys":"key","value":"value",
 "joining":"join","concatenate":"join","concat":"join","merge":"merge","string":"string","strings":"string",
 "text":"string","list":"list","array":"list","tuple":"tuple","set":"set",
 "number":"number","numbers":"number","integer":"number","int":"number","float":"float","digit":"digit",
 "digits":"digit","file":"file","files":"file","directory":"directory","folder":"directory","path":"path",
 "uppercase":"upper","lowercase":"lower","capitalize":"capitalize","capital":"capitalize",
 "whitespace":"space","space":"space","spaces":"space","trim":"trim","strip":"trim",
 "character":"char","characters":"char","char":"char","letter":"char","letters":"char",
 "length":"length","size":"length","count":"count","counting":"count","frequency":"count","occurrence":"count",
 "maximum":"max","max":"max","minimum":"min","min":"min","largest":"max","smallest":"min",
 "square":"square","root":"root","power":"power","absolute":"abs","abs":"abs","round":"round",
 "average":"avg","avg":"avg","mean":"avg","median":"median","mode":"mode","variance":"variance",
 "standard":"std","deviation":"std","squares":"square","cube":"cube",
 "convert":"convert","conversion":"convert","encode":"encode","decode":"decode","binary":"binary",
 "decimal":"decimal","roman":"roman","ascii":"ascii","bytes":"bytes","base64":"base64",
 "date":"date","time":"time","day":"day","week":"week","month":"month","year":"year","age":"age",
 "random":"random","password":"password","email":"email","phone":"phone","validate":"validate",
 "search":"search","find":"find","index":"index","position":"index","insert":"insert","remove":"remove",
 "delete":"delete","append":"append","copy":"copy","exists":"exists","create":"create",
 "class":"class","object":"object","inheritance":"inheritance","method":"method","lambda":"lambda",
 "map":"map","filter":"filter","reduce":"reduce","comprehension":"comprehension","enumerate":"enumerate",
 "zip":"zip","any":"any","all":"all","type":"type","check":"check","compare":"compare",
 "swap":"swap","leap":"leap","celsius":"celsius","fahrenheit":"fahrenheit","interest":"interest",
 "area":"area","circle":"circle","rectangle":"rectangle","perimeter":"perimeter",
 "anagram":"anagram","compress":"compress","longest":"longest","word":"word","words":"word",
 "sentence":"sentence","punctuation":"punctuation","snake":"snake","camel":"camel","case":"case",
 "balanced":"balanced","parentheses":"parentheses","stack":"stack","queue":"queue",
 "binary":"binary","bubble":"bubble","insertion":"insertion","selection":"selection","quick":"quick",
 "linear":"linear","missing":"missing","pair":"pair","triplet":"triplet","zeros":"zero","zero":"zero",
 "intersection":"intersection","union":"union","prefix":"prefix","suffix":"suffix","equal":"equal",
 "group":"group","kth":"kth","none":"none","invert":"invert","hash":"hash","bmi":"bmi","tax":"tax",
 "timer":"timer","countdown":"countdown","hours":"hour","hour":"hour","seconds":"second","second":"second",
 "valid":"valid","add":"add","days":"day","solve":"solve","permutations":"permutation",
 "permutation":"permutation","combinations":"combination","combination":"combination","subsets":"subset",
 "paths":"path","grid":"grid","tower":"tower","hanoi":"hanoi","recursion":"recursion","recursive":"recursion",
 "http":"http","request":"request","json":"json","regex":"regex","regular":"regex","expression":"regex",
 "environment":"environment","variable":"variable","variables":"variable","command":"command",
 "arguments":"argument","line":"line","measure":"measure","execution":"execution","sleep":"sleep",
 "format":"format","current":"current","generate":"generate","number":"number","numeric":"numeric",
 "starts":"start","ends":"end","repeat":"repeat","pad":"pad","zero":"zero","swap":"swap"}

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

# ---------- TF-IDF ----------
docs = [shabd(r["q"]) for r in UNIQ]
df = Counter()
for d in docs:
    for w in set(d): df[w] += 1
N = len(docs)
idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}

def vector(q):
    tf = Counter(shabd(q))
    v = {w: (1 + math.log(c)) * idf.get(w, math.log(N + 1) + 1.0) for w, c in tf.items()}
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

# ---------- aakar ----------
blob = json.dumps([r["code"] for r in UNIQ], ensure_ascii=False)
comp = zlib.compress(blob.encode("utf-8"), 9)
print(f"\n=== AAKAR ===", flush=True)
print(f"  Qwen fp32:   1976.0 MB", flush=True)
print(f"  JAAL raw:    {len(blob)/1e6:.3f} MB", flush=True)
print(f"  JAAL zip:    {len(comp)/1e6:.4f} MB", flush=True)
print(f"  ghatav:      {1976*1e6/len(comp):.0f}x", flush=True)
print(f"  patterns:    {len(JAAL)}", flush=True)
print(f"  shabdakosh:  {len(idf)}", flush=True)

# ---------- TEST ----------
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
 ("How do I handle exceptions?", ["except","try","error"], []),
 ("Write a function to convert Celsius to Fahrenheit.", ["celsius","fahrenheit"], []),
 ("Write a function to implement bubble sort.", ["bubble"], ["quick","merge"]),
 ("How do I use a lambda function?", ["lambda"], []),
 ("Write a function to calculate area of circle.", ["circle"], ["rectangle"]),
 ("Write a function to check leap year.", ["leap"], []),
 ("How do I parse JSON in Python?", ["json"], []),
 ("Write a function to find GCD of two numbers.", ["gcd","divisor"], ["lcm"]),
 ("How do I generate a random number?", ["random"], []),
]
print(f"\n=== TEST ({len(TEST)} sawaal) ===", flush=True)
sahi = 0; GALAT = []
for q, chahiye, nahi in TEST:
    got, sc, orig = jawab(q)
    low = got.lower()
    ok = any(c in low for c in chahiye) and not any(n in low for n in nahi)
    sahi += ok
    if not ok: GALAT.append((q, orig, got[:100]))
    print(f"[{'SAHI ' if ok else 'GALAT'}] {q[:52]:<54} milaan {sc:.2f}", flush=True)
print(f"\n  SAHI: {sahi}/{len(TEST)} = {sahi/len(TEST)*100:.0f}%", flush=True)
if GALAT:
    print("\n  GALAT:" if False else "\n  GALAT:", flush=True)
    for q, o, g in GALAT:
        print(f"    {q[:45]} -> mila: {o[:40]}", flush=True)
