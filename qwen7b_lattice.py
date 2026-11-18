"""qwen7b_lattice.py -- QWEN 7B ke embedding se bani jaal sthiti par model.
   A/B: random sthiti vs 7B transfer.
"""
import math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
from tokenizers import Tokenizer
import pyarrow.parquet as pq

tok=Tokenizer.from_file("tokenizer.json")
V=tok.get_vocab_size()
print(f"Qwen vokeb {V:,}",flush=True)

import base64
if not os.path.exists("pos7.npy") and os.path.exists("pos7.b64"):
    open("pos7.npy","wb").write(base64.b64decode(open("pos7.b64").read()))
pos7=np.load("pos7.npy").astype(np.float32)
print(f"7B sthiti {pos7.shape} = {pos7.nbytes/1e6:.1f} MB",flush=True)
K=pos7.shape[1]

# corpus
txt=""
for f in ("w1.parquet","w2.parquet"):
    if not os.path.exists(f): continue
    t=pq.read_table(f); col=t.column_names[0]
    txt+="\n".join(str(x) for x in t.column(col).to_pylist()[:400000])+"\n"
print(f"corpus {len(txt):,} char",flush=True)
ids=tok.encode(txt).ids
del txt
stream=np.array(ids,dtype=np.int64); del ids
print(f"token {len(stream):,}",flush=True)
cut=int(len(stream)*0.9); tr=stream[:cut]; te=stream[cut:]
print(f"train {len(tr):,} | test {len(te):,}",flush=True)
seen=len(np.unique(tr)); print(f"dikhe alag token {seen:,}/{V:,} = {seen/V*100:.1f}%",flush=True)

d=16
class LT(nn.Module):
    def __init__(s,pos_init,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.pos=nn.Parameter(torch.tensor(pos_init))
        s.lsig=nn.Parameter(torch.zeros(V))
        s.amp=nn.Parameter(torch.zeros(V))
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.out=nn.Linear(d,V,bias=False)
    def conn(s,a):
        d2=((s.pos[a][:,None,:]-s.pos[None,:,:])**2).sum(-1)
        sg=torch.exp(s.lsig[a]).clamp(0.05,3.0)[:,None]
        am=torch.exp(s.amp[a])[:,None]*torch.exp(s.amp)[None,:]
        return am*torch.exp(-d2/(2*sg*sg))
    def forward(s,x):
        h=s.E(x); C=s.conn(x)
        for l in range(len(s.q)):
            Q=s.q[l](h)
            Kk=s.k[l](s.E.weight); Vv=s.v[l](s.E.weight)
            sc=Q@Kk.T/math.sqrt(d)+C
            w=torch.softmax(sc,-1)
            h=s.n[l](h+s.o[l](w@Vv))
        return s.out(h)

def run(nm,pos_init,steps=600):
    m=LT(pos_init); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    t0=time.time(); LIM=1500.0
    for st in range(1,steps+1):
        i=np.random.randint(0,len(tr)-1,64)
        x=torch.tensor(tr[i]); y=torch.tensor(tr[i+1])
        loss=F.cross_entropy(m(x),y)
        opt.zero_grad(); loss.backward(); opt.step()
        if st%200==0: print(f"  [{nm}] kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
        if time.time()-t0>LIM: print(f"  [{nm}] samay hadd",flush=True); break
    m.eval(); N=800
    idx=np.random.choice(range(1,len(te)),N,replace=False)
    lps=[];t1=t5=0;ranks=[]
    with torch.no_grad():
        for k in range(0,N,100):
            j=idx[k:k+100]
            x=torch.tensor(te[j-1]); y=torch.tensor(te[j])
            lg=m(x); lp=F.log_softmax(lg,-1)
            lps.append(lp.gather(1,y[:,None]).squeeze(1).numpy())
            order=torch.argsort(-lg,dim=1)
            for r in range(len(j)):
                rr=int((order[r]==y[r]).nonzero()[0][0])+1
                ranks.append(rr); t1+=(rr==1); t5+=(rr<=5)
    lps=np.concatenate(lps)
    print(f"\n=== {nm} ===",flush=True)
    print(f"  ppl {math.exp(-lps.mean()):,.2f} | top-1 {t1/N*100:.2f}% | top-5 {t5/N*100:.2f}% | median rank {int(np.median(ranks)):,}",flush=True)
    return math.exp(-lps.mean()), int(np.median(ranks))

# A: 7B transfer se shuru
p7,r7=run("7B EMBEDDING SE TRANSFER",pos7)
# B: random sthiti (tulna)
rnd=(np.random.rand(V,K)*2-1).astype(np.float32)
p0,r0=run("RANDOM STHITI (tulna)",rnd)
print(f"\n=== ANTAR ===",flush=True)
print(f"  7B transfer : ppl {p7:,.2f} | rank {r7:,}",flush=True)
print(f"  random      : ppl {p0:,.2f} | rank {r0:,}",flush=True)
print(f"  sudhaar     : ppl {p0/p7:.2f}x | rank {r0/max(r7,1):.2f}x",flush=True)
