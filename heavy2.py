"""heavy2.py -- DO-STARIYA CONNECTION MODEL, BHARI DATA PAR.
   Sudhaar: probability-space interpolation (log add nahi).
   Scaling: 25% / 50% / 100% data.
"""
import re, math, time
from collections import Counter
import sys

parts=[]
for p in ("c1.txt","c2.txt","c3.txt"):
    try: parts.append(open(p,encoding="utf-8",errors="ignore").read())
    except: pass
txt="\n".join(parts)
words=re.findall(r"[A-Za-z]+",txt.lower())
cnt=Counter(words)
VOC=[w for w,_ in cnt.most_common(3000)]; stoi={w:i for i,w in enumerate(VOC)}; V=len(stoi)
stream=[stoi[w] for w in words if w in stoi]
cut=int(len(stream)*0.9); full_tr,te=stream[:cut],stream[cut:]
print(f"corpus {len(txt):,} char | shabd {len(words):,} | vokeb {V:,}",flush=True)
print(f"train {len(full_tr):,} | test {len(te):,}",flush=True)

def build(tr):
    uni=Counter(tr); c1=Counter(); c2=Counter()
    for i in range(len(tr)):
        if i>=1: c1[(tr[i-1],tr[i])]+=1
        if i>=2: c2[(tr[i-2],tr[i-1],tr[i])]+=1
    T1=Counter(); T2=Counter()
    for k,v in c1.items(): T1[k[:1]]+=v
    for k,v in c2.items(): T2[k[:2]]+=v
    return uni,c1,c2,T1,T2,len(tr)

def ppl_two(tr,L1=0.4,L2=0.2):
    uni,c1,c2,T1,T2,tot=build(tr)
    lp=0.0;n=0
    for i in range(2,len(te)):
        tgt=te[i]; ctx=te[i-2:i]
        p=uni.get(tgt,0)/tot
        t=T1.get((ctx[1],),0)
        if t: p=(1-L1)*p + L1*(c1.get((ctx[1],tgt),0)/t)
        t=T2.get((ctx[0],ctx[1]),0)
        if t: p=(1-L2)*p + L2*(c2.get((ctx[0],ctx[1],tgt),0)/t)
        lp+=math.log(max(p,1e-12)); n+=1
    return math.exp(-lp/n), len(c1), len(c2), tot

def ppl_one(tr,L1=0.5):
    uni,c1,c2,T1,T2,tot=build(tr)
    lp=0.0;n=0
    for i in range(1,len(te)):
        tgt=te[i]; a=te[i-1]
        p=uni.get(tgt,0)/tot
        t=T1.get((a,),0)
        if t: p=(1-L1)*p + L1*(c1.get((a,tgt),0)/t)
        lp+=math.log(max(p,1e-12)); n+=1
    return math.exp(-lp/n)

print(f"\n{'data':>7} {'token':>10} {'level-1':>10} {'level-2':>10} {'ppl L1':>11} {'ppl L2':>11}",flush=True)
print("-"*64,flush=True)
for frac in (0.25,0.5,1.0):
    tr=full_tr[:int(len(full_tr)*frac)]
    p2,n1,n2,tot=ppl_two(tr)
    p1=ppl_one(tr)
    print(f"{frac*100:>6.0f}% {tot:>10,} {n1:>10,} {n2:>10,} {p1:>11,.2f} {p2:>11,.2f}",flush=True)

print(f"\n  dono star ka faayda dekhо: ppl L2 - ppl L1",flush=True)
