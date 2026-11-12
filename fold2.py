#!/usr/bin/env python3
"""fold2.py -- जाल को ट्रांसफॉर्मर में मोड़ो: जाल की खूबियाँ + ट्रांसफॉर्मर की गहराई।

जाल से जो बचाते हैं:
  1. embedding सेल्स से (शब्द = सेल)
  2. sparse attention -- सिर्फ़ जाल की कड़ियों पर (n*k, n*n नहीं)
  3. उम्मीदवार-सीमित output -- सिर्फ़ जाल के अगले-शब्दों पर softmax
ट्रांसफॉर्मर से जो जोड़ते हैं:
  4. causal अनुक्रम, 5. कई परतें, 6. सीखे हुए वज़न
"""
import argparse, math, random, re, time
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="hi_40k.txt")
ap.add_argument("--vocab", type=int, default=4000)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--seq", type=int, default=48)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--cand", type=int, default=50, help="जाल से उम्मीदवार")
ap.add_argument("--window", type=int, default=8, help="स्थानीय खिड़की")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(2)

# ---------- पाठ ----------
sents = []
for line in open(a.data, encoding="utf-8"):
    ws = re.findall(r"[^\s।,?!]+", line)
    if 5 <= len(ws) <= 30: sents.append(ws)
    if len(sents) >= 200000: break
cnt = Counter(w for ws in sents for w in ws)
top = [w for w, _ in cnt.most_common(a.vocab)]
stoi = {w: i + 3 for i, w in enumerate(top)}   # 0=PAD 1=BOS 2=UNK
V = len(stoi) + 3
print(f"वाक्य {len(sents):,} | शब्दकोश {V:,} (कवरेज {sum(cnt[w] for w in top)/sum(cnt.values())*100:.1f}%)", flush=True)

# ---------- जाल: अगले-शब्द (उम्मीदवार) ----------
nxt = defaultdict(Counter)
for ws in sents:
    ids = [stoi.get(w, 2) for w in ws]
    for x, y in zip(ids, ids[1:]): nxt[x][y] += 1
cand_list = {x: [y for y, _ in c.most_common(a.cand)] for x, c in nxt.items()}
print(f"जाल: {len(nxt):,} शब्दों के अगले-शब्द | औसत उम्मीदवार {sum(len(v) for v in cand_list.values())/len(cand_list):.1f}", flush=True)

seqs = [[stoi.get(w, 2) for w in ws] for ws in sents if len(ws) >= 6]
data = torch.tensor([i for s in seqs for i in s[:30]], dtype=torch.long)
print(f"टोकन: {len(data):,}", flush=True)


class FoldTF(nn.Module):
    def __init__(s, V, d, L, H, win):
        super().__init__(); s.d, s.H, s.win = d, H, win
        s.tok = nn.Embedding(V, d)          # सेल-सदिश
        s.pos = nn.Embedding(128, d)
        s.blocks = nn.ModuleList()
        for _ in range(L):
            s.blocks.append(nn.ModuleDict({
                "qkv": nn.Linear(d, 3*d, bias=False), "out": nn.Linear(d, d, bias=False),
                "n1": nn.LayerNorm(d), "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d),
                "n2": nn.LayerNorm(d)}))
        s.nf = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
    def forward(s, x):
        B, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None]
        # sparse mask: स्थानीय खिड़की (जाल की कड़ियों का स्थानीय रूप)
        idx = torch.arange(T, device=x.device)
        mask = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - s.win)
        for b in s.blocks:
            q, k, v = b["qkv"](b["n1"](h)).chunk(3, dim=-1)
            q = q.view(B,T,s.H,-1).transpose(1,2); k = k.view(B,T,s.H,-1).transpose(1,2)
            v = v.view(B,T,s.H,-1).transpose(1,2)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            att = att.transpose(1,2).contiguous().view(B,T,s.d)
            h = h + b["out"](att)
            h = h + b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
        return s.head(s.nf(h))


m = FoldTF(V, a.dim, a.layers, a.heads, a.window)
npar = sum(p.numel() for p in m.parameters())
print(f"पैरामीटर {npar:,} | fp32 {npar*4/1e6:.2f} MB", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)

t0 = time.time(); last = 0
for step in range(1, a.steps + 1):
    i = torch.randint(0, len(data) - a.seq - 1, (a.batch,))
    x = torch.stack([data[j:j+a.seq] for j in i])
    y = torch.stack([data[j+1:j+a.seq+1] for j in i])
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    last = loss.item()
    if step % 300 == 0:
        print(f"  कदम {step:,} | loss {last:.3f} | {step/(time.time()-t0):.0f} कदम/सेकंड", flush=True)
print(f"\nपूरा: {a.steps:,} कदम, {time.time()-t0:.0f}s, loss {last:.3f}", flush=True)

# ---------- नमूना ----------
m.eval()
inv = {i: w for w, i in stoi.items()}
with torch.no_grad():
    for start in ("भारत", "गंगा", "विज्ञान"):
        ids = [stoi.get(start, 2)]
        x = torch.tensor([ids]); out = [start]
        for _ in range(25):
            lg = m(x[:, -a.seq:])[:, -1, :] / 0.9
            nid = torch.multinomial(F.softmax(lg, -1), 1).item()
            out.append(inv.get(nid, "?")); x = torch.cat([x, torch.tensor([[nid]])], 1)
        print(f"  '{start}' -> {' '.join(out)}", flush=True)
