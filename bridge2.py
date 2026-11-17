"""bridge2.py -- SAHI BRIDGE: probability space me interpolate, log add NAHI.

  p_count = L*P(b|a) + (1-L)*P_uni
  p_sem   = softmax(alpha * cosine(emb_a, emb_b))
  p_final = (1-beta)*p_count + beta*p_sem

  Aur khaas naap: UNSEEN jodiyon par bridge kitna sudhaar deta hai?
"""
import re, math, json, struct
from collections import Counter
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

# Qwen embedding
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
print(f"  Qwen embedding ready {E.shape}",flush=True)

# counts
uni=Counter(tr); c1=Counter(); T1=Counter()
for i in range(len(tr)):
    uni[tr[i]]+=1
    if i>=1: c1[(tr[i-1],tr[i])]+=1
for k,v in c1.items(): T1[k[:1]]+=v
tot=len(tr)
uni_arr=np.array([uni.get(i,0) for i in range(V)],np.float64)/tot
uni_arr=np.maximum(uni_arr,1e-9)

def naapo(beta,alpha,L=0.4):
    """probability space interpolation"""
    seen_lp=0.0; seen_n=0; uns_lp=0.0; uns_n=0
    for i in range(1,len(te)):
        a,b=te[i-1],te[i]
        p_uni=uni_arr[b]
        t=T1.get((a,),0)
        p_big=(c1.get((a,b),0)/t) if t else 0.0
        p=(1-L)*p_uni + L*p_big                       # <-- PROBABILITY space
        if beta>0:
            s=SEM[a]*alpha
            s=s-s.max(); ps=np.exp(s); ps/=ps.sum()
            p=(1-beta)*p + beta*ps[b]                 # <-- phir interpolate
        p=max(p,1e-12)
        if (a,b) in c1: seen_lp+=math.log(p); seen_n+=1
        else: uns_lp+=math.log(p); uns_n+=1
    return (math.exp(-seen_lp/max(seen_n,1)), math.exp(-uns_lp/max(uns_n,1)), seen_n, uns_n)

print(f"\n{'setup':<28} {'ppl(dekhe)':>12} {'ppl(naye)':>11}",flush=True)
print("-"*54,flush=True)
s0,u0,sn,un=naapo(0.0,0)
print(f"{'corpus only':<28} {s0:>12,.2f} {u0:>11,.2f}",flush=True)
print(f"  (dekhe jodi {sn:,} | nayi jodi {un:,})",flush=True)
best=None
for al in (1.0,3.0):
    for be in (0.1,0.2,0.3,0.5):
        s,u,_,_=naapo(be,al)
        print(f"{'bridge a=%.0f b=%.1f'%(al,be):<28} {s:>12,.2f} {u:>11,.2f}",flush=True)
        score=s*0.5+u*0.5
        if best is None or score<best[0]: best=(score,al,be,s,u)
print(f"\n=== SABSE ACCHA ===",flush=True)
print(f"  alpha={best[1]} beta={best[2]} -> dekhe {best[3]:,.2f} | naye {best[4]:,.2f}",flush=True)
print(f"  corpus only       -> dekhe {s0:,.2f} | naye {u0:,.2f}",flush=True)
print(f"  NAYI jodiyon par sudhaar: {u0-best[4]:+,.2f} ({u0/best[4]:.2f}x)",flush=True)
print(f"  DEKHI jodiyon par:        {s0-best[3]:+,.2f} ({s0/best[3]:.2f}x)",flush=True)
f.close()
