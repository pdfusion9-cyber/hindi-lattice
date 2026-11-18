"""gtrans4.py -- KOI SEEMA NAHI. Har shabd har shabd se jud sakta hai.

  W : (V,V) learnable -- poora connection matrix, koi top-K cap nahi
  Attention : saare V par (koi paas/door nahi)
"""
import re, math, time
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

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

# poora connection matrix, warm start = log-count
c1=Counter(); ca=Counter()
for i in range(len(tr)):
    if i>=1: c1[(tr[i-1],tr[i])]+=1; ca[tr[i-1]]+=1
W0=np.full((V,V),-12.0,np.float32)
for (a,b),c in c1.items(): W0[a,b]=math.log(c/ca[a])
print(f"  connection matrix: {V*V:,} (= {V}x{V}), dekhe gaye {len(c1):,}",flush=True)
print(f"  memory: {V*V*4/1e6:.0f} MB fp32",flush=True)

d=32
class FullTF(nn.Module):
    def __init__(s,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.W=nn.Parameter(torch.tensor(W0))        # <-- POORA connection matrix
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.out=nn.Linear(d,V,bias=False)
    def forward(s,x):
        h=s.E(x)
        for l in range(len(s.q)):
            Q=s.q[l](h)
            K=s.k[l](s.E.weight); Vv=s.v[l](s.E.weight)
            sc=Q@K.T/math.sqrt(d) + s.W[x]        # (B,V) -- sabse connection
            w=torch.softmax(sc,-1)
            h=s.n[l](h+s.o[l](w@Vv))
        return s.out(h)

m=FullTF(); npar=sum(p.numel() for p in m.parameters())
print(f"  parameter: {npar:,} (isme {V*V:,} connection)",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
t0=time.time(); LIMIT=1200.0
for st in range(1,2001):
    i=np.random.randint(0,len(tr)-1,128)
    x=torch.tensor([tr[j] for j in i]); y=torch.tensor([tr[j+1] for j in i])
    loss=F.cross_entropy(m(x),y)
    opt.zero_grad(); loss.backward(); opt.step()
    if st%200==0: print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
    if time.time()-t0>LIMIT: print(f"  samay hadd -- kadam {st}",flush=True); break

m.eval(); N=4000
pos=np.random.choice(range(1,len(te)),N,replace=False)
lps=[];t1=t5=0;ranks=[]
with torch.no_grad():
    for k in range(0,N,250):
        idx=pos[k:k+250]
        x=torch.tensor([te[j-1] for j in idx]); y=torch.tensor([te[j] for j in idx])
        lg=m(x); lp=F.log_softmax(lg,-1)
        lps.append(lp.gather(1,y[:,None]).squeeze(1).numpy())
        order=torch.argsort(-lg,dim=1)
        for r in range(len(idx)):
            rr=int((order[r]==y[r]).nonzero()[0][0])+1
            ranks.append(rr); t1+=(rr==1); t5+=(rr<=5)
lps=np.concatenate(lps)
print(f"\n=== POORA CONNECTION MATRIX (koi seema nahi) ===",flush=True)
print(f"  ppl         : {math.exp(-lps.mean()):,.2f}",flush=True)
print(f"  top-1       : {t1/N*100:.1f}%",flush=True)
print(f"  top-5       : {t5/N*100:.1f}%",flush=True)
print(f"  median rank : {int(np.median(ranks)):,}",flush=True)
print(f"\n  top-32 cap wala: ppl 781.40 | top-1 6.5% | rank 86",flush=True)
print(f"  bigram        : ppl 485.40 | context-4 rank 140",flush=True)
