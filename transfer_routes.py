"""transfer_routes.py -- 7B gyaan ko TEZ kaise utaarein: chaar raaste naapo.

  1. seedhi copy       : file copy ki raftaar
  2. sparse            : kitne % wazan chhodne par bhi gyaan bacha
  3. low-rank          : har matrix ko rank-r se kitna bacha
  4. ek linear map     : A.W.B se ek hi pass me utaarna
"""
import json, struct, os, subprocess, time
import numpy as np

URL="https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors"
if not os.path.exists("hdr7.bin"):
    subprocess.run(["curl","-sL","-r","0-400000","-o","hdr7.bin",URL],check=True)
raw=open("hdr7.bin","rb").read(); n=struct.unpack("<Q",raw[:8])[0]
hdr=json.loads(raw[8:8+n]); hdr.pop("__metadata__",None)
BASE=8+n
print(f"shard-1 me {len(hdr)} tensor",flush=True)

# bade matrices chuno (attention/MLP)
cands=[(k,v) for k,v in hdr.items() if len(v["shape"])==2 and v["shape"][0]*v["shape"][1]>1_000_000]
cands.sort(key=lambda kv:-kv[1]["shape"][0]*kv[1]["shape"][1])
print(f"bade 2D tensor: {len(cands)}",flush=True)

def read_tensor(v):
    o0,o1=v["data_offsets"]; sh=v["shape"]
    if not os.path.exists("sh7.bin"):
        subprocess.run(["curl","-sL","-r",f"{BASE+o0}-{BASE+o1-1}","-o","sh7.bin",URL],check=True)
    u=np.fromfile("sh7.bin",dtype=np.uint16)
    x=(u.astype(np.uint32)<<16).view(np.float32).reshape(sh)
    os.remove("sh7.bin")
    return x

print(f"\n{'tensor':<46} {'shape':>16} {'energy@top1%':>13} {'rank@99%':>9} {'rank@95%':>9}",flush=True)
print("-"*100,flush=True)
for k,v in cands[:8]:
    try:
        W=read_tensor(v)
        flat=np.abs(W).ravel()
        thr=np.percentile(flat,99)
        e_top=(flat[flat>=thr]**2).sum()/ (flat**2).sum()*100
        s=np.linalg.svd(W.astype(np.float32),compute_uv=False)
        e=np.cumsum(s**2)/np.sum(s**2)
        r99=int(np.searchsorted(e,0.99))+1
        r95=int(np.searchsorted(e,0.95))+1
        print(f"{k[-45:]:<46} {str(v['shape']):>16} {e_top:>12.1f}% {r99:>9} {r95:>9}",flush=True)
        del W,flat,s,e
    except Exception as ex:
        print(f"{k[-45:]:<46} {str(v['shape']):>16}  {type(ex).__name__} {str(ex)[:30]}",flush=True)

# copy raftaar
print("\n=== COPY RAFTAAR ===",flush=True)
for mb in (200,):
    a=np.random.rand(mb*1024*1024//4).astype(np.float32)
    a.tofile("t.bin")
    t0=time.time(); b=np.fromfile("t.bin",dtype=np.float32); dt=time.time()-t0
    print(f"  {mb} MB read: {dt:.2f}s -> {mb/dt:.0f} MB/s",flush=True)
    os.remove("t.bin"); del a,b
print("\n  7B fp16 (15.2 GB) isi raftaar se: %.0f second"%(15200/(mb/dt)),flush=True)
print("  7B int4 (3.8 GB)  isi raftaar se: %.0f second"%(3800/(mb/dt)),flush=True)
