"""qwen7b_pca.py -- 7B transfer: PCA (random projection ki jagah) + 2000 kadam.
   A/B: 7B PCA sthiti vs random sthiti, dono 2000 kadam.
"""
import json, struct, math, time, os, subprocess
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
from tokenizers import Tokenizer
import pyarrow.parquet as pq

SHARD="model-00001-of-00004.safetensors"
URL=f"https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/{SHARD}"

if not os.path.exists("hdr.bin"):
    subprocess.run(["curl","-sL","-r","0-200000","-o","hdr.bin",URL],check=True)
raw=open("hdr.bin","rb").read(); n=struct.unpack("<Q",raw[:8])[0]
hdr=json.loads(raw[8:8+n]); v=hdr["model.embed_tokens.weight"]
o0,o1=v["data_offsets"]; V7,D7=v["shape"]
print(f"7B embedding {V7:,}x{D7:,} = {(o1-o0)/1e9:.2f} GB",flush=True)
if not os.path.exists("emb7b.bin"):
    subprocess.run(["curl","-sL","-r",f"{8+n+o0}-{8+n+o1-1}","-o","emb7b.bin",URL],check=True)
mm=np.memmap("emb7b.bin",dtype=np.uint16,mode="r",shape=(V7,D7))

def readrows(i0,i1):
    x=np.array(mm[i0:i1],dtype=np.uint16).view(np.float16).astype(np.float32)
    x/= (np.linalg.norm(x,axis=1,keepdims=True)+1e-9)
    return x

K=64
# --- PCA basis: 25,000 row ke sample se ---
print("PCA basis bana rahe...",flush=True)
S=25000
samp=readrows(0,S)
mu=samp.mean(0,keepdims=True)
C=((samp-mu).T@(samp-mu))/S
w,Vb=np.linalg.eigh(C)
idx=np.argsort(w)[::-1][:K]
print(f"  top-{K} variance share: {w[idx].sum()/w.sum()*100:.1f}%",flush=True)
B=Vb[:,idx].astype(np.float32)
del samp,C,Vb,w

pos=np.empty((V7,K),np.float32); CH=2000
for s in range(0,V7,CH):
    e=min(s+CH,V7); pos[s:e]=(readrows(s,e)-mu)@B
pos/= (np.std(pos,axis=0,keepdims=True)+1e-9)
print(f"  jaal sthiti {pos.shape} = {pos.nbytes/1e6:.1f} MB | transfer {1.09e9/pos.nbytes:.0f}x",flush=True)

# --- corpus ---
tok=Tokenizer.from_file("tokenizer.json"); V=tok.get_vocab_size()
txt=""
if os.path.exists("w1.parquet"):
    t=pq.read_table("w1.parquet"); col=t.column_names[0]
    txt="\n".join(str(x) for x in t.column(col).to_pylist()[:300000])
ids=tok.encode(txt).ids; del txt
stream=np.array(ids,dtype=np.int64); del ids
cut=int(len(stream)*0.9); tr=stream[:cut]; te=stream[cut:]
print(f"vokeb {V:,} | token {len(stream):,} | dikhe {len(np.unique(tr)):,} = {len(np.unique(tr))/V*100:.1f}%",flush=True)
pos=pos[:V]

d=16
class LT(nn.Module):
    def __init__(s,pos_init,L=2):
        super().__init__()
        s.E=nn.Embedding(V,d); nn.init.normal_(s.E.weight,std=0.02)
        s.pos=nn.Parameter(torch.tensor(pos_init))
        s.lsig=nn.Parameter(torch.zeros(V)); s.amp=nn.Parameter(torch.zeros(V))
        s.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(L)])
        s.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.out=nn.Linear(d,V,bias=False)
    def conn(s,a):
        P=s.pos
        d2=((P[a]**2).sum(-1)[:,None]+(P**2).sum(-1)[None,:]-2*P[a]@P.T).clamp_min(0)
        sg=torch.exp(s.lsig[a]).clamp(0.05,3.0)[:,None]
        am=torch.exp(s.amp[a])[:,None]*torch.exp(s.amp)[None,:]
        return am*torch.exp(-d2/(2*sg*sg))
    def forward(s,x):
        h=s.E(x); C=s.conn(x)
        for l in range(len(s.q)):
            Q=s.q[l](h); Kk=s.k[l](s.E.weight); Vv=s.v[l](s.E.weight)
            sc=Q@Kk.T/math.sqrt(d)+C
            w=torch.softmax(sc,-1)
            h=s.n[l](h+s.o[l](w@Vv))
        return s.out(h)

def run(nm,pos_init,steps=2000,B_=32):
    m=LT(pos_init); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    t0=time.time(); LIM=2400.0
    print(f"\n[{nm}] param {sum(p.numel() for p in m.parameters()):,}",flush=True)
    for st in range(1,steps+1):
        i=np.random.randint(0,len(tr)-1,B_)
        x=torch.tensor(tr[i]); y=torch.tensor(tr[i+1])
        loss=F.cross_entropy(m(x),y)
        opt.zero_grad(); loss.backward(); opt.step()
        if st%400==0: print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s",flush=True)
        if time.time()-t0>LIM: print(f"  samay hadd -- kadam {st}",flush=True); break
    m.eval(); N=500
    ix=np.random.choice(range(1,len(te)),N,replace=False)
    lps=[];t1=t5=0;ranks=[]
    with torch.no_grad():
        for k in range(0,N,50):
            j=ix[k:k+50]
            x=torch.tensor(te[j-1]); y=torch.tensor(te[j])
            lg=m(x); lp=F.log_softmax(lg,-1)
            lps.append(lp.gather(1,y[:,None]).squeeze(1).numpy())
            order=torch.argsort(-lg,dim=1)
            for r in range(len(j)):
                rr=int((order[r]==y[r]).nonzero()[0][0])+1
                ranks.append(rr); t1+=(rr==1); t5+=(rr<=5)
    lps=np.concatenate(lps)
    print(f"  => ppl {math.exp(-lps.mean()):,.2f} | top-1 {t1/N*100:.2f}% | top-5 {t5/N*100:.2f}% | median rank {int(np.median(ranks)):,}",flush=True)
    return math.exp(-lps.mean()), int(np.median(ranks))

p7,r7=run("7B PCA TRANSFER K=64, 2000 kadam",pos)
rnd=(np.random.rand(V,K)*2-1).astype(np.float32)
p0,r0=run("RANDOM K=64, 2000 kadam",rnd)
print(f"\n=== ANTAR (2000 kadam) ===",flush=True)
print(f"  7B PCA transfer : ppl {p7:,.2f} | rank {r7:,}",flush=True)
print(f"  random          : ppl {p0:,.2f} | rank {r0:,}",flush=True)
print(f"  transfer ka labh: ppl {p0/p7:.2f}x | rank {r0/max(r7,1):.2f}x",flush=True)
print(f"\n  purana (random projection, 350 kadam): transfer rank 143 | random rank 63",flush=True)
