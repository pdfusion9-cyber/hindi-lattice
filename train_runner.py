#!/usr/bin/env python3
"""train_runner.py -- रनर पर प्रशिक्षण: कम रैम, सुरक्षित चेकपॉइंट, साफ़ कारण।"""
import argparse, gc, os, time, traceback
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", default="ckpt/model.pt")
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--steps", type=int, default=300000)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--budget", type=int, default=8000)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--max-mb", type=int, default=20)
a = ap.parse_args()
torch.set_num_threads(a.threads)
print(f"थ्रेड {a.threads} | कोर {os.cpu_count()}", flush=True)

# --- पाठ: सीमा में पढ़ो ---
limit = a.max_mb * 1_000_000
parts, size = [], 0
with open(a.data, encoding="utf-8", errors="ignore") as f:
    while size < limit:
        c = f.read(1_000_000)
        if not c:
            break
        parts.append(c)
        size += len(c.encode("utf-8"))
txt = "".join(parts)
del parts; gc.collect()
print(f"पाठ {len(txt):,} अक्षर ({size/1e6:.0f} MB)", flush=True)

chars = sorted(set(txt))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
dtype = np.uint16 if V < 65000 else np.int32
arr = np.fromiter((stoi.get(c, 0) for c in txt), dtype=dtype, count=len(txt))
del txt, stoi; gc.collect()
print(f"शब्दावली {V} | सरणी {arr.nbytes/1e6:.0f} MB ({np.dtype(dtype).name})", flush=True)


class M(nn.Module):
    def __init__(s, V, d):
        super().__init__()
        s.e = nn.Embedding(V, d)
        s.c = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=2),
        )
        s.n = nn.LayerNorm(d)
        s.o = nn.Linear(d, V)

    def forward(s, x):
        h = s.e(x).transpose(1, 2)
        h = s.n((h + s.c(h)).transpose(1, 2))
        return s.o(h)


m = M(V, a.dim)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
step = 0
if os.path.exists(a.ckpt):
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("chars") == chars and ck.get("m"):
            m.load_state_dict(ck["m"])
            opt.load_state_dict(ck["o"])
            step = int(ck.get("step", 0))
            print(f"चेकपॉइंट से: कदम {step:,}", flush=True)
        else:
            print("शब्दावली बदली -- नए सिरे से", flush=True)
    except Exception as e:
        print(f"चेकपॉइंट नहीं लदा ({type(e).__name__}) -- नए सिरे से", flush=True)

print(f"पैरामीटर {sum(p.numel() for p in m.parameters()):,}", flush=True)
t0 = time.time()
try:
    while step < a.steps and time.time() - t0 < a.budget:
        i = np.random.randint(0, len(arr) - a.seq - 1, a.batch)
        x = torch.from_numpy(np.stack([arr[j:j + a.seq] for j in i]).astype(np.int64))
        y = torch.from_numpy(np.stack([arr[j + 1:j + a.seq + 1] for j in i]).astype(np.int64))
        loss = nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"कदम {step:,} | loss {loss.item():.3f} | {step/(time.time()-t0):.1f} कदम/सेकंड", flush=True)
except Exception:
    traceback.print_exc()
    print("प्रशिक्षण में गड़बड़ी -- जो बना है वह सहेजा जा रहा है", flush=True)

os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
torch.save({"m": m.state_dict(), "o": opt.state_dict(), "step": step, "chars": chars}, a.ckpt)
print(f"सहेजा: कदम {step:,}", flush=True)
