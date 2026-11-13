#!/usr/bin/env python3
"""chhota_karo.py -- चेकपॉइंट को int8 और int4 में बदलो (टुकड़ों में)।

  fp32 वज़न  -> int8  (1 बाइट/मान, प्रति-टेंसर scale)
             -> int4  (0.5 बाइट/मान, दो निबल एक बाइट में)
हर टुकड़ा 90 MB से छोटा -- git में जा सके।
"""
import argparse, glob, json, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/sampurna.pt")
ap.add_argument("--out", default="store")
ap.add_argument("--part-mb", type=int, default=90)
a = ap.parse_args()

if not os.path.exists(a.ckpt):
    print("चेकपॉइंट नहीं:", a.ckpt, flush=True); sys.exit(1)

print("लोड हो रहा:", a.ckpt, os.path.getsize(a.ckpt)//1024//1024, "MB", flush=True)
d = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
sd = d["m"]
meta = {"step": d.get("step"), "V": d.get("V"), "dim": d.get("dim"), "tensors": list(sd)}
print(f"टेंसर {len(sd)} | मान {sum(v.numel() for v in sd.values())/1e6:.1f}M", flush=True)

os.makedirs(a.out, exist_ok=True)

# ---------- int8 ----------
q8, s8 = {}, {}
for k, v in sd.items():
    t = v.detach().float()
    mx = t.abs().max().item()
    sc = mx / 127.0 if mx > 0 else 1.0
    q8[k] = torch.clamp(torch.round(t / sc), -127, 127).to(torch.int8)
    s8[k] = sc
torch.save({"q": q8, "s": s8, "meta": meta}, os.path.join(a.out, "m_int8.pt"))
print("int8:", os.path.getsize(os.path.join(a.out, "m_int8.pt"))//1024//1024, "MB", flush=True)

# ---------- int4 (दो निबल एक बाइट) ----------
q4, s4 = {}, {}
for k, v in sd.items():
    t = v.detach().float()
    mx = t.abs().max().item()
    sc = mx / 7.0 if mx > 0 else 1.0
    q = torch.clamp(torch.round(t / sc), -7, 7).to(torch.int8) + 8      # 0..15
    flat = q.flatten()
    if flat.numel() % 2: flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
    packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
    q4[k] = packed.to(torch.uint8)
    s4[k] = sc
torch.save({"q": q4, "s": s4, "meta": meta}, os.path.join(a.out, "m_int4.pt"))
print("int4:", os.path.getsize(os.path.join(a.out, "m_int4.pt"))//1024//1024, "MB", flush=True)

# ---------- टुकड़ों में बाँटो (90 MB से छोटा) ----------
def split(path, tag):
    data = open(path, "rb").read()
    step = a.part_mb * 1024 * 1024
    n = (len(data) + step - 1) // step
    for i in range(n):
        with open(os.path.join(a.out, f"{tag}.part{i:02d}"), "wb") as f:
            f.write(data[i*step:(i+1)*step])
    os.remove(path)
    print(f"{tag}: {n} टुकड़े", flush=True)
    return n

n8 = split(os.path.join(a.out, "m_int8.pt"), "int8")
n4 = split(os.path.join(a.out, "m_int4.pt"), "int4")
json.dump({"int8_parts": n8, "int4_parts": n4, "meta": meta},
          open(os.path.join(a.out, "ckpt_meta.json"), "w"))
print("तैयार:", a.out, flush=True)
