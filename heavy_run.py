"""heavy_run.py -- BHARI DATA par 3-star connection model, sharing ke saath.

Har unique connection EK baar banta hai (id = (a,b) / (a,b,c) / (a,b,c,d)),
baar baar wahi copy chalti hai.
"""
import re, math, random, time
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0); torch.set_num_threads(4)

# ---------- corpus ----------
parts=[]
for p in ("corpus_a.txt","corpus_b.txt"):
    try: parts.append(open(p,encoding="utf-8",errors="ignore").read())
    except: pass
txt="\n".join(parts)
words=re.findall(r"[A-Za-z\u0900-\u097F]+",txt)
cnt=Counter(words)
VOC=[w for w,_ in cnt.most_common(8000)]; stoi={w:i for i,w in enumerate(VOC)}; V=len(stoi)
stream=[stoi[w] for w in words if w in stoi]
cut=int(len(stream)*0.9); full_train,test=stream[:cut],stream[cut:]
print(f"corpus {len(txt):,} char | shabd {len(words):,} | vokeb {V:,}",flush=True)
print(f"train {len(full_train):,} | test {len(test):,}",flush=True)

class Conn(nn.Module):
    def __init__(s,W1,n1,n2,n3,i2,i3,lv):
        super().__init__()
        s.E=nn.Embedding(V,64); nn.init.normal_(s.E.weight,std=0.01)
        s.W1=nn.Parameter(W1.clone()); s.lv=lv
        if lv>=2: s.W2=nn.Embedding(n2,1); s.W2.weight.data.copy_(i2)
        if lv>=3: s.W3=nn.Embedding(n3,1); s.W3.weight.data.copy_(i3)
        s.register_buffer("mask1",W1>-12)
    def forward(s,x,byab,byabc):
        B,T=x.shape; c=x[:,-1]
        lg=s.E(c)@s.E.weight.T + s.W1[c]
        if s.lv>=2 and T>=3:
            b=x[:,-2]
            for i in range(B):
                l=byab.get((int(b[i]),int(c[i])))
                if not l: continue
                cs=torch.tensor([v for v,_ in l]); ix=torch.tensor([j for _,j in l])
                lg[i,cs]=lg[i,cs]+s.W2(ix).squeeze(-1)
        if s.lv>=3 and T>=4:
            a=x[:,-3]; b=x[:,-2]
            for i in range(B):
                l=byabc.get((int(a[i]),int(b[i]),int(c[i])))
                if not l: continue
                cs=torch.tensor([v for v,_ in l]); ix=torch.tensor([j for _,j in l])
                lg[i,cs]=lg[i,cs]+s.W3(ix).squeeze(-1)
        return lg

def naapo(m,order,byab,byabc):
    m.eval(); lp=0.0;n=0
    with torch.no_grad():
        for i in range(order,len(test)):
            x=torch.tensor([test[i-order:i]])
            lp+=F.log_softmax(m(x,byab,byabc),-1)[0,test[i]].item(); n+=1
    return math.exp(-lp/n)

print(f"\n{'data':>7} {'token':>9} {'uniq c1':>9} {'uniq c2':>9} {'uniq c3':>9} {'uses':>9} {'ppl L3':>8}",flush=True)
print("-"*66,flush=True)
for frac in (0.25,0.5,1.0):
    tr=full_train[:int(len(full_train)*frac)]
    c1=Counter(); c2=Counter(); c3=Counter()
    ca=Counter(); cab=Counter(); cabc=Counter()
    for i in range(len(tr)):
        if i>=1: c1[(tr[i-1],tr[i])]+=1; ca[tr[i-1]]+=1
        if i>=2: c2[(tr[i-2],tr[i-1],tr[i])]+=1; cab[(tr[i-2],tr[i-1])]+=1
        if i>=3: c3[(tr[i-3],tr[i-2],tr[i-1],tr[i])]+=1; cabc[(tr[i-3],tr[i-2],tr[i-1])]+=1
    tri=list(c2.keys()); quad=list(c3.keys())
    byab=defaultdict(list); byabc=defaultdict(list)
    for i,(a,b,c) in enumerate(tri): byab[(a,b)].append((c,i))
    for i,(a,b,c,d) in enumerate(quad): byabc[(a,b,c)].append((d,i))
    W1=torch.full((V,V),-12.0)
    for (a,b),n in c1.items(): W1[a,b]=math.log(n/ca[a])
    i2=torch.zeros(len(tri),1); i3=torch.zeros(len(quad),1)
    for i,(a,b,c) in enumerate(tri):
        pt=c2[(a,b,c)]/cab[(a,b)]; pb=c1.get((b,c),0)/max(ca.get(b,1),1)
        i2[i,0]=math.log(pt)-math.log(max(pb,1e-12))
    for i,(a,b,c,d) in enumerate(quad):
        p4=c3[(a,b,c,d)]/cabc[(a,b,c)]; p2=c2.get((b,c,d),0)/max(cab.get((b,c),1),1)
        i3[i,0]=math.log(p4)-math.log(max(p2,1e-12))
    m=Conn(W1,len(c1),len(tri),len(quad),i2,i3,3)
    W=[tr[i-4:i+1] for i in range(4,len(tr))]
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    t0=time.time()
    for st in range(1,401):
        batch=random.sample(W,min(64,len(W)))
        x=torch.tensor([r[:-1] for r in batch]); y=torch.tensor([r[-1] for r in batch])
        loss=F.cross_entropy(m(x,byab,byabc),y)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            if m.W1.grad is not None: m.W1.grad.mul_(m.mask1)
        opt.step()
    p=naapo(m,4,byab,byabc)
    uses=sum(c1.values())
    print(f"{frac*100:>6.0f}% {len(tr):>9,} {len(c1):>9,} {len(c2):>9,} {len(c3):>9,} {uses:>9,} {p:>8.2f}",flush=True)
    print(f"        ({time.time()-t0:.0f}s) | sharing: {uses/max(len(c1),1):.1f}x reuse",flush=True)
