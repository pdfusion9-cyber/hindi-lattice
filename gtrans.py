"""gtrans.py -- ASLI TRANSFORMER JO CONNECTION PAR CHALTA HAI.

  Connection = parameter (sparse edge weights)
  Node state = chhota vector
  Attention SIRF graph ke padosiyon par (sparse)
  Phir output distribution

  Naap: ppl, top-1, top-5, median rank
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

# ---------- graph: connection = parameter ----------
edges=Counter()
for i in range(len(tr)-1): edges[(tr[i],tr[i+1])]+=1
elist=list(edges.keys()); eidx={e:i for i,e in enumerate(elist)}
E_SRC=torch.tensor([e[0] for e in elist]); E_DST=torch.tensor([e[1] for e in elist])
NE=len(elist)
print(f"  connection: {NE:,} (parameter)",flush=True)

# adjacency: node -> edge indices (outgoing)
out_edges=defaultdict(list)
for i,(a,b) in enumerate(elist): out_edges[a].append(i)

d=64
class GraphTF(nn.Module):
    def __init__(s,layers=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.W=nn.Embedding(NE,1); nn.init.zeros_(s.W.weight)   # CONNECTION WEIGHTS
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        s.out=nn.Linear(d,V,bias=False)
    def forward(s,x):          # x: (B,) last token
        h=s.E(x)               # (B,d)
        for l in range(len(s.q)):
            Q=s.q[l](h); K=s.k[l](s.E.weight); Val=s.v[l](s.E.weight)
            # sirf graph ke jodiyon par attention score
            sc=(K[E_SRC]*Q[:,None,:]).sum(-1)/math.sqrt(d) + s.W.weight.squeeze(-1)[None,:]
            # aggregate: har dst par max/softmax
            agg=torch.zeros(V,d)
            mx=torch.full((V,),-1e9)
            mx.index_reduce_(0,E_DST,sc[0],'amax')
            e=torch.exp(sc[0]-mx[E_DST])
            den=torch.zeros(V); den.index_add_(0,E_DST,e)
            w=e/den[E_DST].clamp_min(1e-9)
            contrib=w[:,None]*Val[E_SRC]
            agg.index_add_(0,E_DST,contrib)
            upd=s.o[l](agg[x])                  # (B,d)
            h=s.n[l](h+upd)
        return s.out(h)

m=GraphTF()
npar=sum(p.numel() for p in m.parameters())
print(f"  parameter: {npar:,} (isme {NE:,} connection)",flush=True)

opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
B=256
t0=time.time()
LIMIT=1500.0
for st in range(1,601):
    i=np.random.randint(0,len(tr)-1,B)
    x=torch.tensor([tr[j] for j in i]); y=torch.tensor([tr[j+1] for j in i])
    loss=F.cross_entropy(m(x),y)
    opt.zero_grad(); loss.backward(); opt.step()
    if st%100==0:
        print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
    if time.time()-t0 > LIMIT:
        print(f"  samay ki hadd ({LIMIT:.0f}s) -- kadam {st} par roka",flush=True)
        break

# ---------- naap ----------
m.eval()
N=min(8000,len(te)-1)
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
            ranks.append(rr)
            if rr==1: t1+=1
            if rr<=5: t5+=1
lps=np.concatenate(lps)
print(f"\n=== GRAPH TRANSFORMER ===",flush=True)
print(f"  ppl         : {math.exp(-lps.mean()):,.2f}",flush=True)
print(f"  top-1       : {t1/N*100:.1f}%",flush=True)
print(f"  top-5       : {t5/N*100:.1f}%",flush=True)
print(f"  median rank : {int(np.median(ranks)):,}",flush=True)
print(f"\n  tulna: bigram corpus  ppl 485.40 | median rank 42-2,656",flush=True)
print(f"  tulna: adaptive bridge ppl 379.86",flush=True)
