"""ctx.py -- LAMBA SANDARBH (1..4) + adaptive bridge. Tez version."""
import re, math, json, struct
from collections import Counter, defaultdict
import numpy as np
np.random.seed(0)

txt=""
for p in ("wt.txt","sh.txt","c1.txt"):
    try: txt+=open(p,encoding="utf-8",errors="ignore").read()+"\n"
    except: pass
words=re.findall(r"[A-Za-z]+",txt.lower())
cnt=Counter(words)
VOC=[w for w,_ in cnt.most_common(3000)]; stoi={w:i for i,w in enumerate(VOC)}; V=len(stoi)
stream=[stoi[w] for w in words if w in stoi]
cut=int(len(stream)*0.9); tr,te=stream[:cut],stream[cut:]
print(f"vokeb {V:,} | train {len(tr):,} | test {len(te):,}",flush=True)

P="model.safetensors"
f=open(P,"rb"); n=struct.unpack("<Q",f.read(8))[0]
hdr=json.loads(f.read(n)); hdr.pop("__metadata__",None); BASE=8+n
v=hdr["model.embed_tokens.weight"]; o0,_=v["data_offsets"]; QV,QD=v["shape"]
def rows(i0,i1):
    f.seek(BASE+o0+i0*QD*2); raw=f.read((i1-i0)*QD*2)
    u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)
    return (u<<16).view(np.float32).reshape(i1-i0,QD)
from tokenizers import Tokenizer
tok=Tokenizer.from_file("tokenizer.json")
E=np.zeros((V,QD),np.float32)
for w,i in stoi.items():
    e=tok.encode(" "+w)
    if e.ids: E[i]=rows(e.ids[0],e.ids[0]+1)[0]
E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
SEM=(E@E.T).astype(np.float32)

# sparse continuation tables
CONT={}; CTOT={}
for k in (1,2,3,4):
    d=defaultdict(Counter)
    for i in range(k,len(tr)+1):
        g=tuple(tr[i-k:i]); d[g[:-1]][g[-1]]+=1
    CONT[k]={c:dict(v) for c,v in d.items()}
    CTOT[k]={c:sum(v.values()) for c,v in d.items()}
    print(f"  level {k}: {len(d):,} context",flush=True)

uni=Counter(tr); tot=len(tr)
uni_arr=np.maximum(np.array([uni.get(i,0) for i in range(V)],np.float64)/tot,1e-9)

def dist(ctx,K,alpha=3.0,beta=0.5,L=0.4):
    p=uni_arr.copy()
    for k in range(1,K+1):
        key=tuple(ctx[-k:])
        conts=CONT[k].get(key)
        if not conts: continue
        t=CTOT[k][key]
        p=p*(1-L)
        for j,c in conts.items(): p[j]+=L*c/t
    # adaptive weight: K-gram count
    key=tuple(ctx[-K:])
    nk=CONT[K].get(key[:-1],{}).get(key[-1],0)
    w=nk/(nk+1.0)
    s=SEM[ctx[-1]]*alpha; s=s-s.max(); ps=np.exp(s); ps/=ps.sum()
    pb=(1-beta)*p + beta*ps
    return w*p + (1-w)*pb

N_SAMP=min(12000,len(te)-6)
pos=np.random.choice(range(6,len(te)),N_SAMP,replace=False)
print(f"\n{'ctx':>4} {'ppl':>10} {'top-1':>7} {'top-5':>7} {'median rank':>12}",flush=True)
print("-"*46,flush=True)
for K in (1,2,3,4):
    lps=[]; t1=t5=0; ranks=[]
    for i in pos:
        ctx=te[i-K:i]; tgt=te[i]
        p=dist(ctx,K)
        lps.append(math.log(max(p[tgt],1e-12)))
        order=np.argsort(-p)
        r=int(np.where(order==tgt)[0][0])+1
        ranks.append(r)
        if r==1: t1+=1
        if r<=5: t5+=1
    print(f"{K:>4} {math.exp(-np.mean(lps)):>10,.2f} {t1/N_SAMP*100:>6.1f}% {t5/N_SAMP*100:>6.1f}% {int(np.median(ranks)):>12,}",flush=True)
f.close()
