"""emb_banao.py -- Qwen embedding + VOCAB, ek hi npz me.
   Jo file pehle thi usme vocab hi nahi tha, isliye random shuruat hui thi.
"""
import json, struct, os, subprocess, re
from collections import Counter
import numpy as np

# 1) shabdakosh: corpus ke top 30,000 shabd
txt=""
for p in ("data/corpus.txt","w1.txt"):
    if os.path.exists(p):
        with open(p,encoding="utf-8",errors="ignore") as f:
            txt=f.read(200_000_000)
        break
print(f"corpus {len(txt):,} char",flush=True)
words=re.findall(r"[\u0900-\u097F]+|[A-Za-z]+",txt)
cnt=Counter(words)
VOC=[w for w,_ in cnt.most_common(30000)]
print(f"vokeb {len(VOC):,} (top shabd)",flush=True)

# 2) Qwen 0.5B embedding (sirf uska range)
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"
if not os.path.exists("hdr.bin"):
    subprocess.run(["curl","-sL","-r","0-200000","-o","hdr.bin",URL],check=True)
raw=open("hdr.bin","rb").read(); n=struct.unpack("<Q",raw[:8])[0]
hdr=json.loads(raw[8:8+n]); v=hdr["model.embed_tokens.weight"]
o0,o1=v["data_offsets"]; QV,QD=v["shape"]
print(f"Qwen embedding {QV:,}x{QD:,}",flush=True)
if not os.path.exists("qemb.bin"):
    subprocess.run(["curl","-sL","-r",f"{8+n+o0}-{8+n+o1-1}","-o","qemb.bin",URL],check=True)
mm=np.memmap("qemb.bin",dtype=np.uint16,mode="r",shape=(QV,QD))

# 3) tokenizer
from tokenizers import Tokenizer
tk=Tokenizer.from_file("tokenizer.json")

D=1024
E=np.zeros((len(VOC),D),np.float32); hit=0
for i,w in enumerate(VOC):
    e=tk.encode(" "+w)
    if not e.ids: continue
    t=e.ids[0]
    if t>=QV: continue
    row=np.array(mm[t],dtype=np.uint16).view(np.float16).astype(np.float32)
    E[i,:min(D,QD)]=row[:D]
    hit+=1
print(f"  bhare {hit:,}/{len(VOC):,} = {hit/len(VOC)*100:.1f}%",flush=True)

# 4) int8 + per-row scale, vocab ke saath
mx=np.abs(E).max(1,keepdims=True); mx[mx==0]=1
s=(mx/127.0).astype(np.float32)
q=np.clip(np.round(E/s),-127,127).astype(np.int8)
np.savez_compressed("emb_q.npz", q=q, s=s, vocab=np.array(json.dumps(VOC,ensure_ascii=False)))
print(f"  emb_q.npz {os.path.getsize('emb_q.npz')/1e6:.1f} MB | q {q.shape} | vocab {len(VOC):,}",flush=True)
