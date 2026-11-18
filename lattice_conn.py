"""lattice_conn.py -- JAAL SE KANEXAN: sthiti se bane, koi matrix store nahi.

  Har shabd ki jagah 3D jaal me (seekhi hui)
  Connection = dono ki field ka overlap (ganana se, storage 0)
  Attention bias = wahi connection
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

d=32
class LatticeTF(nn.Module):
    def __init__(s,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        # JAAL: har shabd ki jagah + field ka failav (seekhne yogya)
        s.pos=nn.Parameter(torch.rand(V,3)*2-1)
        s.lsig=nn.Parameter(torch.zeros(V))          # log-sigma
        s.amp=nn.Parameter(torch.zeros(V))           # log-amplitude
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.out=nn.Linear(d,V,bias=False)
    def conn(s,a):                                   # (B,V) -- ganna se banti connection
        d2=((s.pos[a][:,None,:]-s.pos[None,:,:])**2).sum(-1)
        sg=torch.exp(s.lsig[a]).clamp(0.05,3.0)[:,None]
        am=torch.exp(s.amp[a])[:,None]*torch.exp(s.amp)[None,:]
        return am*torch.exp(-d2/(2*sg*sg))
    def forward(s,x):
        h=s.E(x)
        C=s.conn(x)                                  # (B,V)
        for l in range(len(s.q)):
            Q=s.q[l](h)
            K=s.k[l](s.E.weight); Vv=s.v[l](s.E.weight)
            sc=Q@K.T/math.sqrt(d) + C
            w=torch.softmax(sc,-1)
            h=s.n[l](h+s.o[l](w@Vv))
        return s.out(h)

m=LatticeTF()
npar=sum(p.numel() for p in m.parameters())
lattice_params=V*3+V+V
print(f"  jaal parameter: {lattice_params:,} ({lattice_params*4/1024:.0f} KB fp32)",flush=True)
print(f"  kul parameter : {npar:,}",flush=True)
print(f"  connection matrix STORE nahi hota -- ganna se banti hai",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
t0=time.time(); LIMIT=1000.0
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
print(f"\n=== JAAL SE KANEXAN (sthiti se, store nahi) ===",flush=True)
print(f"  ppl         : {math.exp(-lps.mean()):,.2f}",flush=True)
print(f"  top-1       : {t1/N*100:.1f}%",flush=True)
print(f"  top-5       : {t5/N*100:.1f}%",flush=True)
print(f"  median rank : {int(np.median(ranks)):,}",flush=True)
print(f"\n  ghana matrix (90 lakh, 36 MB): ppl 771.49 | top-1 6.5% | rank 90",flush=True)
print(f"  bigram corpus                : ppl 485.40 | rank 140",flush=True)
# jaal ki asli sthiti dekho
with torch.no_grad():
    P=m.pos.cpu().numpy(); S=torch.exp(m.lsig).cpu().numpy()
print(f"\n  jaal me shabd ka failav: {P.min():.2f}..{P.max():.2f}",flush=True)
print(f"  sigma: {S.min():.3f}..{S.max():.3f} (ausat {S.mean():.3f})",flush=True)
