"""qwen_big.py -- 5-10 CRORE token par jaal-connection model, poori Qwen dictionary."""
import re, math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
import pyarrow.parquet as pq
from tokenizers import Tokenizer

tok=Tokenizer.from_file("tokenizer.json")
V=tok.get_vocab_size()
print(f"Qwen vokeb: {V:,}",flush=True)

CAP=60_000_000
ids=[]
tot_chars=0
for f in ("w1.parquet","w2.parquet"):
    if not os.path.exists(f): continue
    t=pq.read_table(f); col=t.column_names[0]
    lines=[str(x) for x in t.column(col).to_pylist()]
    txt="\n".join(lines); tot_chars+=len(txt)
    print(f"  {f}: {len(txt):,} char",flush=True)
    for i in range(0,len(txt),2_000_000):
        ids.extend(tok.encode(txt[i:i+2_000_000]).ids)
        if len(ids)>=CAP: break
    if len(ids)>=CAP: break
    del txt, lines
    print(f"    -> {len(ids):,} token",flush=True)

stream=np.array(ids[:CAP],dtype=np.int64)
del ids
print(f"corpus {tot_chars:,} char | token {len(stream):,}",flush=True)
cut=int(len(stream)*0.9); tr=stream[:cut]; te=stream[cut:]
print(f"train {len(tr):,} | test {len(te):,}",flush=True)
seen=len(np.unique(tr))
print(f"train me dikhe alag token: {seen:,}/{V:,} = {seen/V*100:.1f}%",flush=True)

d=16
class LatticeTF(nn.Module):
    def __init__(s,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.pos=nn.Parameter(torch.rand(V,3)*2-1)
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
            K=s.k[l](s.E.weight); Vv=s.v[l](s.E.weight)
            sc=Q@K.T/math.sqrt(d)+C
            w=torch.softmax(sc,-1)
            h=s.n[l](h+s.o[l](w@Vv))
        return s.out(h)

m=LatticeTF()
lp=V*3+V+V
print(f"  jaal parameter: {lp:,} ({lp*4/1024/1024:.2f} MB fp32)",flush=True)
print(f"  kul parameter : {sum(p.numel() for p in m.parameters()):,}",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
t0=time.time(); LIMIT=2400.0; B=64
for st in range(1,1501):
    i=np.random.randint(0,len(tr)-1,B)
    x=torch.tensor(tr[i]); y=torch.tensor(tr[i+1])
    loss=F.cross_entropy(m(x),y)
    opt.zero_grad(); loss.backward(); opt.step()
    if st%100==0: print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
    if time.time()-t0>LIMIT: print(f"  samay hadd -- kadam {st}",flush=True); break

m.eval(); N=1500
pos=np.random.choice(range(1,len(te)),N,replace=False)
lps=[];t1=t5=0;ranks=[]
with torch.no_grad():
    for k in range(0,N,100):
        idx=pos[k:k+100]
        x=torch.tensor(te[idx-1]); y=torch.tensor(te[idx])
        lg=m(x); lp_=F.log_softmax(lg,-1)
        lps.append(lp_.gather(1,y[:,None]).squeeze(1).numpy())
        order=torch.argsort(-lg,dim=1)
        for r in range(len(idx)):
            rr=int((order[r]==y[r]).nonzero()[0][0])+1
            ranks.append(rr); t1+=(rr==1); t5+=(rr<=5)
lps=np.concatenate(lps)
print(f"\n=== JAAL: 6 CRORE TOKEN, POORI QWEN DICTIONARY ===",flush=True)
print(f"  ppl         : {math.exp(-lps.mean()):,.2f}",flush=True)
print(f"  top-1       : {t1/N*100:.2f}%",flush=True)
print(f"  top-5       : {t5/N*100:.2f}%",flush=True)
print(f"  median rank : {int(np.median(ranks)):,}",flush=True)
print(f"\n  pichhla (28 lakh token): ppl 13,786 | top-1 1.27% | rank 1,189",flush=True)
print(f"  random: ppl {V:,} | rank {V//2:,}",flush=True)
