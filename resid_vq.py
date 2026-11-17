"""resid_vq.py -- RESIDUAL quantisation (NCM) vs INDEPENDENT, asli Qwen vocab par."""
import json, struct, time
import numpy as np
np.random.seed(0)
P="model.safetensors"
f=open(P,"rb"); n=struct.unpack("<Q",f.read(8))[0]
hdr=json.loads(f.read(n)); hdr.pop("__metadata__",None); BASE=8+n
v=hdr["model.embed_tokens.weight"]; o0,_=v["data_offsets"]
V,D=v["shape"]; DP=D+((-D)%3)
print(f"=== VOCAB {V:,} x {D} (pad {DP}) ===",flush=True)
f.seek(BASE+o0); raw=f.read(V*D*2)
u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)
E=(u<<16).view(np.float32).reshape(V,D)
if DP>D: E=np.concatenate([E,np.zeros((V,DP-D),np.float32)],1)
A=E.reshape(-1,3).astype(np.float32)
print(f"  triples {len(A):,}",flush=True)

def kmeans(X,K,iters=10):
    C=X[np.random.permutation(len(X))[:K]].copy()
    for _ in range(iters):
        d=(X*X).sum(1)[:,None]-2*X@C.T+(C*C).sum(1)[None]
        lab=d.argmin(1)
        for j in range(K):
            m=lab==j
            if m.any(): C[j]=X[m].mean(0)
    return C

def quant(X,C):
    Cn=(C*C).sum(1)[None]; out=np.empty_like(X); B=300000
    for s0 in range(0,len(X),B):
        s1=min(s0+B,len(X)); Q=X[s0:s1]
        d=(Q*Q).sum(1)[:,None]-2*Q@C.T+Cn
        out[s0:s1]=C[d.argmin(1)]
    return out

KL=4
# ---------- codebook: har level apna (residual se fit) ----------
print("codebook fit...",flush=True)
CB=[]
R=A.copy()
for k in range(KL):
    S=R[np.random.permutation(len(R))[:15000]]
    CB.append(kmeans(S,64))
    R=R-quant(R,CB[-1])
    print(f"  level {k+1} codebook bana | resid std {R.std():.5f}",flush=True)
del R

def chalane(mode,K):
    num=[0.0]*(K+1); d1=[0.0]*(K+1); d2=[0.0]*(K+1)
    R=A.copy(); rec=np.zeros_like(A)
    for k in range(K):
        q=quant(R if mode=="resid" else A, CB[k])
        if mode=="resid": R=R-q
        rec+=q
        num[k+1]=float((A*rec).sum()); d1[k+1]=float((A*A).sum()); d2[k+1]=float((rec*rec).sum())
    return [num[k]/(np.sqrt(d1[k])*np.sqrt(d2[k])+1e-9) if d1[k]>0 else 0 for k in range(K+1)]

print("\n=== COSINE (2-bit/level) ===",flush=True)
print(f"{'levels':>7} {'bits/w':>7} {'residual':>10} {'independent':>12}",flush=True)
print("-"*40,flush=True)
Rz=chalane("resid",KL); Id=chalane("indep",KL)
for k in range(1,KL+1):
    print(f"{k:>7} {k*2:>7} {Rz[k]:>10.4f} {Id[k]:>12.4f}",flush=True)

print("\n=== TULNA (pehle naapa hua) ===",flush=True)
print(f"  int4  (4 bit/w) : 0.9871",flush=True)
print(f"  int8  (8 bit/w) : 1.0000",flush=True)
print(f"  residual 2 level (4 bit/w) : {Rz[2]:.4f}",flush=True)
print(f"  residual 4 level (8 bit/w) : {Rz[4]:.4f}",flush=True)

print("\n=== NATIJA ===",flush=True)
print(f"  4 bit/w par: residual {Rz[2]:.4f} vs int4 0.9871 -> {Rz[2]-0.9871:+.4f}",flush=True)
print(f"  8 bit/w par: residual {Rz[4]:.4f} vs int8 1.0000 -> {Rz[4]-1.0:+.4f}",flush=True)
f.close()
