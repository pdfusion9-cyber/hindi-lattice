"""bridge.py -- HIBRID BRIDGE: pretrained model ka gyaan connection weights me.

  W1_count    = corpus se (syntactic connection)
  W1_semantic = Qwen embedding se (semantic connection)
  Bridge      = dono ko jodna

  Aur jaanch: jo seekh raha hai, usse jawab sahi aate hain?
"""
import re, math, json, struct
from collections import Counter
import numpy as np
np.random.seed(0)

# ---------- corpus ----------
txt=""
for p in ("wt.txt","sh.txt","c1.txt"):
    try: txt+=open(p,encoding="utf-8",errors="ignore").read()+"\n"
    except: pass
words=re.findall(r"[A-Za-z]+",txt.lower())
cnt=Counter(words)
VOC=[w for w,_ in cnt.most_common(3000)]; stoi={w:i for i,w in enumerate(VOC)}; V=len(stoi)
stream=[stoi[w] for w in words if w in stoi]
cut=int(len(stream)*0.9); tr,te=stream[:cut],stream[cut:]
print(f"corpus {len(txt):,} char | vokeb {V:,} | train {len(tr):,} | test {len(te):,}",flush=True)

# ---------- Qwen embedding (bridge) ----------
print("Qwen embedding load...",flush=True)
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
# har shabd ka pehla Qwen token id
ids=np.zeros(V,np.int64); ok=np.zeros(V,bool)
for w,i in stoi.items():
    e=tok.encode(" "+w)
    if e.ids: ids[i]=e.ids[0]; ok[i]=True
print(f"  {ok.sum():,}/{V:,} shabd ka Qwen token mila",flush=True)
# embedding nikalo (sirf zaroori ids)
uniq=np.unique(ids[ok])
E=np.zeros((V,QD),np.float32)
got=rows(0,0) if False else None
for t in uniq:
    E[ids==t]=rows(int(t),int(t)+1)[0]
E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
print(f"  embedding ready {E.shape}",flush=True)

# ---------- connection weights ----------
uni=Counter(tr); c1=Counter(); ca=Counter()
for i in range(len(tr)):
    uni[tr[i]]+=1
    if i>=1: c1[(tr[i-1],tr[i])]+=1; ca[tr[i-1]]+=1

W1=np.full((V,V),-12.0,np.float32)
for (a,b),k in c1.items(): W1[a,b]=math.log(k/ca[a])
SEM=(E@E.T).astype(np.float32)      # semantic connection
print(f"  W1_count + SEM ready | SEM range {SEM.min():.2f}..{SEM.max():.2f}",flush=True)

def naapo(alpha,nm):
    """logit = W1_count + alpha*SEM + unigram"""
    uni_arr=np.array([uni.get(i,0) for i in range(V)],np.float32)+1e-6
    ulp=np.log(uni_arr/uni_arr.sum())
    lp=0.0;n=0
    for i in range(1,len(te)):
        a,b=te[i-1],te[i]
        lg=W1[a]+alpha*SEM[a]+ulp
        mx=lg.max(); z=lg-mx
        lp+=(z[b]-math.log(np.exp(z).sum())); n+=1
    p=math.exp(-lp/n)
    print(f"  {nm:<34} ppl {p:>10,.2f}",flush=True)
    return p

print(f"\n=== BRIDGE KA ASAR ===",flush=True)
base=naapo(0.0,"sirf corpus (bridge band)")
res={}
for al in (0.5,1.0,2.0,4.0):
    res[al]=naapo(al,f"bridge alpha={al}")
best=min(res.items(),key=lambda x:x[1])
print(f"\n  sabse achha alpha={best[0]} -> ppl {best[1]:,.2f}",flush=True)
print(f"  sudhaar: {base-best[1]:+,.2f} ({base/best[1]:.2f}x)",flush=True)

# ---------- JAANCH: jawab sahi aate hain? ----------
print(f"\n=== JAANCH: agla shabd kya batata hai? ===",flush=True)
uni_arr=np.array([uni.get(i,0) for i in range(V)],np.float32)+1e-6
ulp=np.log(uni_arr/uni_arr.sum())
inv={i:w for w,i in stoi.items()}
tests=["the","of","in","he","she","king","water","time","love","war"]
for w in tests:
    if w not in stoi: continue
    a=stoi[w]
    for nm,al in (("corpus",0.0),("bridge",best[0])):
        lg=W1[a]+al*SEM[a]+ulp
        top=np.argsort(-lg)[:5]
        print(f"  '{w}' [{nm:>6}] -> {[inv[t] for t in top]}",flush=True)
    print("",flush=True)
f.close()
