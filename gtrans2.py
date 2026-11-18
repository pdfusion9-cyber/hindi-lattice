"""gtrans2.py -- CONNECTION par transformer, TEZ version.
   Attention sirf graph ke top-32 padosiyon par (batch-only, poore V par nahi).
"""
import re, math, json, struct, time
from collections import Counter, defaultdict
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

# graph -> top-K padosi prati node
MAXD=32
edges=Counter()
for i in range(len(tr)-1): edges[(tr[i],tr[i+1])]+=1
nb=defaultdict(list)
for (a,b),c in edges.items(): nb[a].append((b,c))
ADJ=np.zeros((V,MAXD),np.int64); WGT=np.full((V,MAXD),-12.0,np.float32); MSK=np.zeros((V,MAXD),bool)
for a,lst in nb.items():
    lst.sort(key=lambda x:-x[1]); lst=lst[:MAXD]
    for k,(b,c) in enumerate(lst):
        ADJ[a,k]=b; WGT[a,k]=math.log(c); MSK[a,k]=True
print(f"  connection: {len(edges):,} | top-{MAXD} padosi/node",flush=True)

ADJt=torch.tensor(ADJ); WGTt=torch.tensor(WGT); MSKt=torch.tensor(MSK)
d=32
class GTF(nn.Module):
    def __init__(s,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.out=nn.Linear(d,V,bias=False)
    def forward(s,x):
        B=x.shape[0]
        h=s.E(x)                       # (B,d)
        A=ADJt[x]; W=WGTt[x]; M=MSKt[x]   # (B,MAXD)
        for l in range(len(s.q)):
            Q=s.q[l](h)                        # (B,d)
            K=s.k[l](s.E.weight)[A]            # (B,MAXD,d)
            Vv=s.v[l](s.E.weight)[A]           # (B,MAXD,d)
            sc=(K*Q[:,None,:]).sum(-1)/math.sqrt(d) + W
            sc=sc.masked_fill(~M,-1e9)
            w=torch.softmax(sc,-1)
            agg=(w[:,:,None]*Vv).sum(1)        # (B,d)
            h=s.n[l](h+s.o[l](agg))
        return s.out(h)

m=GTF(); npar=sum(p.numel() for p in m.parameters())
print(f"  parameter: {npar:,} (isme {len(edges):,} connection)",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
t0=time.time(); LIMIT=600.0
for st in range(1,401):
    i=np.random.randint(0,len(tr)-1,256)
    x=torch.tensor([tr[j] for j in i]); y=torch.tensor([tr[j+1] for j in i])
    loss=F.cross_entropy(m(x),y)
    opt.zero_grad(); loss.backward(); opt.step()
    if st%50==0: print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
    if time.time()-t0>LIMIT: print(f"  samay hadd -- kadam {st}",flush=True); break

m.eval(); N=6000
pos=np.random.choice(range(1,len(te)),N,replace=False)
lps=[];t1=t5=0;ranks=[]
with torch.no_grad():
    for k in range(0,N,500):
        idx=pos[k:k+500]
        x=torch.tensor([te[j-1] for j in idx]); y=torch.tensor([te[j] for j in idx])
        lg=m(x); lp=F.log_softmax(lg,-1)
        lps.append(lp.gather(1,y[:,None]).squeeze(1).numpy())
        order=torch.argsort(-lg,dim=1)
        for r in range(len(idx)):
            rr=int((order[r]==y[r]).nonzero()[0][0])+1
            ranks.append(rr); t1+= (rr==1); t5+= (rr<=5)
lps=np.concatenate(lps)
print(f"\n=== CONNECTION TRANSFORMER ===",flush=True)
print(f"  ppl         : {math.exp(-lps.mean()):,.2f}",flush=True)
print(f"  top-1       : {t1/N*100:.1f}%",flush=True)
print(f"  top-5       : {t5/N*100:.1f}%",flush=True)
print(f"  median rank : {int(np.median(ranks)):,}",flush=True)
print(f"\n  tulna: bigram ppl 485.40 | context-4 rank 140",flush=True)
