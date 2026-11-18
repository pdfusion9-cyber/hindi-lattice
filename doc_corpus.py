"""doc_corpus.py -- OPEN SOURCE DOKUMENT DATA se corpus banao.
   Teen srot: instructions (likhna/edit karna), kahani (prose), factual docs.
   Koi fallback nahi -- agar data na mile to saaf fail karo.
"""
import json, os, sys, gzip
import pyarrow.parquet as pq

OUT="data/corpus.txt"
os.makedirs("data",exist_ok=True)
parts=[]; stats={}

# 1) DOLLY: instructions -- likhna, edit karna, saransh, vargikaran
p="dolly.jsonl"
if os.path.exists(p):
    n=0; buf=[]
    for line in open(p,encoding="utf-8",errors="ignore"):
        try: d=json.loads(line)
        except: continue
        txt=" ".join(str(d.get(k,"")) for k in ("instruction","context","response"))
        if txt.strip(): buf.append(txt); n+=1
    parts.append("\n".join(buf)); stats["dolly"]=(n,sum(len(x) for x in buf))
    print(f"dolly: {n:,} document, {stats['dolly'][1]:,} char",flush=True)

# 2) TINYSTORIES: saaf prose (likhna seekhne ke liye)
p="tinystories.txt"
if os.path.exists(p):
    with open(p,encoding="utf-8",errors="ignore") as f:
        txt=f.read(300_000_000)          # ~300 MB kaafi hai
    parts.append(txt); stats["tinystories"]=(txt.count("<|endoftext|>"),len(txt))
    print(f"tinystories: {stats['tinystories'][0]:,} kahani, {len(txt):,} char",flush=True)

# 3) WIKITEXT: tathya-parak document
p="wt.parquet"
if os.path.exists(p):
    t=pq.read_table(p); col=t.column_names[0]
    lines=[str(x) for x in t.column(col).to_pylist()]
    txt="\n".join(lines)
    parts.append(txt); stats["wikitext"]=(len(lines),len(txt))
    print(f"wikitext: {len(lines):,} line, {len(txt):,} char",flush=True)

if not parts:
    print("KOI DOKUMENT DATA NAHI MILA -- fail",flush=True)
    sys.exit(1)

txt="\n".join(parts)
open(OUT,"w",encoding="utf-8").write(txt)
print(f"\ncorpus: {len(txt):,} char | {txt.count(chr(10))+1:,} line",flush=True)
print(f"srot: {list(stats.keys())}",flush=True)
