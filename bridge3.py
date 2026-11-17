"""bridge3.py -- ADAPTIVE BRIDGE: corpus jahan hai wahan corpus, jahan nahi wahan Qwen.
   Aur nayi jodiyon par THOS UDAHARAN.
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

uni=Counter(tr); c1=Counter(); T1=Counter()
for i in range(len(tr)):
    uni[tr[i]]+=1
    if i>=1: c1[(tr[i-1],tr[i])]+=1
for k,v in c1.items(): T1[k[:1]]+=v
tot=len(tr)
uni_arr=np.maximum(np.array([uni.get(i,0) for i in range(V)],np.float64)/tot,1e-9)

def sem_prob(a,alpha=3.0):
    s=SEM[a]*alpha; s=s-s.max(); p=np.exp(s); return p/p.sum()

def naapo(mode,k=1.0,L=0.4,alpha=3.0,beta=0.5):
    lp=0.0;n=0
    for i in range(1,len(te)):
        a,b=te[i-1],te[i]
        pu=uni_arr[b]
        c=c1.get((a,b),0); t=T1.get((a,),0)
        pbig=(c/t) if t else 0.0
        pc=(1-L)*pu + L*pbig
        if mode=="corpus": p=pc
        elif mode=="bridge": p=(1-beta)*pc + beta*sem_prob(a,alpha)[b]
        else:  # adaptive
            w=c/(c+k)                      # jitni baar dekha, utna corpus par bharosa
            p=w*pc + (1-w)*((1-beta)*pc + beta*sem_prob(a,alpha)[b])
        lp+=math.log(max(p,1e-12)); n+=1
    return math.exp(-lp/n)

print(f"\n{'setup':<26} {'ppl':>10}",flush=True)
print("-"*38,flush=True)
for nm,mode in (("corpus only","corpus"),("bridge sab jagah","bridge"),("ADAPTIVE","adaptive")):
    print(f"{nm:<26} {naapo(mode):>10,.2f}",flush=True)

# ---- THOS UDAHARAN: nayi jodiyan ----
print(f"\n=== UDAHARAN (jahan jodi corpus me NAHI thi) ===",flush=True)
inv={i:w for w,i in stoi.items()}
shown=0
for i in range(1,len(te)):
    a,b=te[i-1],te[i]
    if (a,b) in c1: continue
    if shown>=6: break
    shown+=1
    pc=(1-0.4)*uni_arr + 0.4*np.array([ (c1.get((a,j),0)/T1.get((a,),1)) for j in range(V)])
    pb=(1-0.5)*pc + 0.5*sem_prob(a)
    tc=[inv[j] for j in np.argsort(-pc)[:4]]
    tb=[inv[j] for j in np.argsort(-pb)[:4]]
    rank_c=int(np.where(np.argsort(-pc)==b)[0][0])+1
    rank_b=int(np.where(np.argsort(-pb)==b)[0][0])+1
    print(f"  '{inv[a]}' -> sahi: '{inv[b]}'",flush=True)
    print(f"      corpus : {tc}  (sahi shabd rank {rank_c})",flush=True)
    print(f"      bridge : {tb}  (sahi shabd rank {rank_b})",flush=True)
f.close()
