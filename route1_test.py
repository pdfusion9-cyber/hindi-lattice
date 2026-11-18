"""route1_test.py -- TAREEKA 1 alag se: Qwen ke tensor SEEDHE padho, badlo mat.
   Naap: kitni RAM lagti hai, kitna samay, aur kya poora model chal sakta hai.
"""
import json, struct, os, subprocess, time, resource
import numpy as np

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

URL="https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors"
if not os.path.exists("shard1.bin"):
    print("shard-1 download (3.9 GB)...",flush=True)
    subprocess.run(["curl","-sL","-o","shard1.bin",URL],check=True)
sz=os.path.getsize("shard1.bin")
print(f"shard-1: {sz/1e9:.2f} GB on disk",flush=True)

# header padho
raw=open("shard1.bin","rb").read(400000)
n=struct.unpack("<Q",raw[:8])[0]
hdr=json.loads(raw[8:8+n]); hdr.pop("__metadata__",None); BASE=8+n
print(f"tensor: {len(hdr)}",flush=True)

mm=np.memmap("shard1.bin",dtype=np.uint8,mode="r")
print(f"mmap ready | RSS shuru me: {rss_mb():.0f} MB",flush=True)

def get(name):
    v=hdr[name]; o0,o1=v["data_offsets"]; sh=v["shape"]
    buf=mm[BASE+o0:BASE+o1]
    x=np.frombuffer(buf,dtype=np.uint16).view(np.float16)
    return x.reshape(sh) if sh else x

# 1) EK tensor padho -- RSS kitna badha
print("\n=== 1) EK TENSOR ===",flush=True)
t0=time.time(); E=get("model.embed_tokens.weight").astype(np.float32); dt=time.time()-t0
print(f"  embed_tokens {E.shape} = {E.nbytes/1e6:.0f} MB float32 | {dt:.2f}s | RSS {rss_mb():.0f} MB",flush=True)
del E
print(f"  (delete ke baad RSS abhi bhi {rss_mb():.0f} MB -- allocator chhodta nahi)",flush=True)

# 2) STREAMING: ek-ek tensor, chhodo, aage badho
print("\n=== 2) STREAMING (ek ke baad ek, chhodte hue) ===",flush=True)
names=[k for k,v in hdr.items() if len(v["shape"])==2]
tot=0; t0=time.time(); peak=0
for i,k in enumerate(names):
    x=get(k); tot+=x.size
    peak=max(peak,rss_mb())
    del x
    if i%20==0: print(f"  {i+1}/{len(names)} {k[-32:]:<34} RSS {rss_mb():.0f} MB",flush=True)
dt=time.time()-t0
print(f"\n  kul {len(names)} tensor, {tot/1e6:.1f}M number",flush=True)
print(f"  samay {dt:.1f}s | peak RSS {peak:.0f} MB",flush=True)
print(f"  (file {sz/1e9:.2f} GB thi, RAM sirf {peak:.0f} MB)",flush=True)

# 3) poora shard RAM me -- mumkin hai ya nahi
print("\n=== 3) POORA RAM ME (koshish) ===",flush=True)
try:
    t0=time.time(); big=np.frombuffer(mm[0:sz],dtype=np.uint8).copy()
    print(f"  {big.nbytes/1e9:.2f} GB RAM me aa gaya | {time.time()-t0:.1f}s | RSS {rss_mb():.0f} MB",flush=True)
    del big
except MemoryError:
    print(f"  MemoryError -- {sz/1e9:.2f} GB RAM me nahi aaya | RSS {rss_mb():.0f} MB",flush=True)

print(f"\n=== NATIJA ===",flush=True)
print(f"  file on disk     : {sz/1e9:.2f} GB (shard-1 only)",flush=True)
print(f"  streaming peak   : {peak:.0f} MB",flush=True)
print(f"  anupat          : {sz/1e6/max(peak,1):.0f}x kam",flush=True)
