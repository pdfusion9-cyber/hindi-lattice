import argparse, json, os, pickle, zipfile
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt",required=True); ap.add_argument("--out",default="int4.npz")
a=ap.parse_args()
zf=zipfile.ZipFile(a.ckpt)
pkl=[n for n in zf.namelist() if n.endswith("data.pkl")][0]; pre=pkl[:-len("data.pkl")]
class R:
    __slots__=("key","numel","dtype")
    def __init__(s,key,numel,dtype): s.key,s.numel,s.dtype=key,numel,dtype
class T:
    __slots__=("st","off","size")
    def __init__(s,st,off,size): s.st,s.off,s.size=st,off,tuple(size)
class U(pickle.Unpickler):
    def persistent_load(s,pid):
        if pid[0]=="storage": return R(pid[2],pid[4],str(pid[1]))
        raise pickle.UnpicklingError(str(pid))
    def find_class(s,m,n):
        if n=="_rebuild_tensor_v2": return lambda st,off,size,stride,*r: T(st,off,size)
        if m=="collections" and n=="OrderedDict":
            import collections; return collections.OrderedDict
        return super().find_class(m,n)
with zf.open(pkl) as f: obj=U(f).load()
sd=obj["m"] if "m" in obj else obj
meta={k:v for k,v in obj.items() if not isinstance(v,dict)}
q={}; sc={}; shp={}; tot=0
for name,t in sd.items():
    if not isinstance(t,T): continue
    raw=zf.read(f"{pre}data/{t.st.key}")
    if "Half" in t.st.dtype: arr=np.frombuffer(raw,dtype=np.float16).astype(np.float32)
    elif "Double" in t.st.dtype: arr=np.frombuffer(raw,dtype=np.float64).astype(np.float32)
    else: arr=np.frombuffer(raw,dtype=np.float32)
    n=int(np.prod(t.size)) if t.size else 1
    arr=arr[t.off:t.off+n].reshape(t.size) if t.size else arr[t.off:t.off+1]
    mx=float(np.abs(arr).max()) if arr.size else 0.0
    s=mx/7.0 if mx>0 else 1.0
    qi=np.clip(np.round(arr/s),-7,7).astype(np.int8)+8
    flat=qi.flatten()
    if flat.size%2: flat=np.concatenate([flat,np.zeros(1,dtype=np.int8)])
    q[name]=((flat[0::2].astype(np.uint8)<<4)|(flat[1::2].astype(np.uint8)&0x0F)).astype(np.uint8)
    sc[name]=np.float32(s); shp[name]=list(t.size); tot+=arr.size
    del arr
np.savez_compressed(a.out,**{f"q__{k}":v for k,v in q.items()},**{f"s__{k}":v for k,v in sc.items()},
    __meta__=np.array(json.dumps({"shapes":shp,"meta":{k:str(v) for k,v in meta.items()}})))
print(f"int4: {os.path.getsize(a.out)/1e6:.1f} MB | मान {tot/1e6:.1f}M | टेंसर {len(q)}")
