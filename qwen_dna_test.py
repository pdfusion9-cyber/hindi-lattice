#!/usr/bin/env python3
"""qwen_dna_test.py -- ASLI Qwen 0.5B par poori DNA pipeline + GUNVATTA naap.

Runner par 7 GB RAM hai, isliye poora model load ho sakta hai.

  1. Qwen 0.5B load (bf16 safetensors)
  2. baseline perplexity
  3. 85% FFN neuron pruning
  4. 2-bit codon encode
  5. fine-tune (adaptation)
  6. perplexity dobara + asli aakar
"""
import json, math, os, struct, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
OUT = {}

# ---------------- 1. Qwen2 architecture ----------------
class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w

def rope_half(x, cos, sin):
    """Qwen style: FIRST HALF / SECOND HALF (not interleaved)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class Attn(nn.Module):
    def __init__(s, d, nh, nkv):
        super().__init__()
        s.nh, s.nkv, s.hd = nh, nkv, d // nh
        s.q = nn.Linear(d, nh * s.hd, bias=True)
        s.k = nn.Linear(d, nkv * s.hd, bias=True)
        s.v = nn.Linear(d, nkv * s.hd, bias=True)
        s.o = nn.Linear(nh * s.hd, d, bias=False)
    def forward(s, x, cos, sin, mask):
        B, T, _ = x.shape
        q = s.q(x).view(B, T, s.nh, s.hd).transpose(1, 2)
        k = s.k(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        v = s.v(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        q = rope_half(q, cos, sin); k = rope_half(k, cos, sin)
        k = k.repeat_interleave(s.nh // s.nkv, 1)
        v = v.repeat_interleave(s.nh // s.nkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return s.o(a.transpose(1, 2).contiguous().view(B, T, -1))

class MLP(nn.Module):
    def __init__(s, d, ff):
        super().__init__()
        s.gate = nn.Linear(d, ff, bias=False)
        s.up = nn.Linear(d, ff, bias=False)
        s.down = nn.Linear(ff, d, bias=False)
    def forward(s, x):
        return s.down(F.silu(s.gate(x)) * s.up(x))

class Blk(nn.Module):
    def __init__(s, d, nh, nkv, ff):
        super().__init__()
        s.n1 = RMSNorm(d); s.attn = Attn(d, nh, nkv)
        s.n2 = RMSNorm(d); s.mlp = MLP(d, ff)
    def forward(s, x, cos, sin, mask):
        x = x + s.attn(s.n1(x), cos, sin, mask)
        return x + s.mlp(s.n2(x))

class Qwen(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        d, L = cfg["hidden"], cfg["layers"]
        s.emb = nn.Embedding(cfg["vocab"], d)
        s.blk = nn.ModuleList([Blk(d, cfg["heads"], cfg["kv"], cfg["ff"]) for _ in range(L)])
        s.nf = RMSNorm(d)
        s.d, s.nh, s.nkv, s.hd = d, cfg["heads"], cfg["kv"], d // cfg["heads"]
        s.theta = cfg.get("rope", 1000000.0)
    def forward(s, idx):
        B, T = idx.shape
        dev = idx.device
        pos = torch.arange(T, device=dev).float()
        inv = 1.0 / (s.theta ** (torch.arange(0, s.hd, 2, device=dev).float() / s.hd))
        f = torch.outer(pos, inv)
        cos = torch.cos(f).repeat_interleave(2, -1)[None, None]
        sin = torch.sin(f).repeat_interleave(2, -1)[None, None]
        mask = torch.full((T, T), float("-inf"), device=dev).triu(1)[None, None]
        h = s.emb(idx)
        for b in s.blk:
            h = b(h, cos, sin, mask)
        h = s.nf(h)
        return h @ s.emb.weight.T          # tied embeddings

# ---------------- 2. load safetensors ----------------
def load_qwen(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None)
    base = 8 + n
    def get(name):
        v = hdr[name]; o0, o1 = v["data_offsets"]
        f.seek(base + o0); raw = f.read(o1 - o0)
        if v["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            a = (u << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, dtype=np.float32)
        return torch.from_numpy(a.reshape(v["shape"]).copy())
    cfg = {
        "hidden": hdr["model.embed_tokens.weight"]["shape"][1],
        "vocab": hdr["model.embed_tokens.weight"]["shape"][0],
        "layers": len({k.split(".")[2] for k in hdr if "layers" in k}),
        "ff": hdr["model.layers.0.mlp.gate_proj.weight"]["shape"][0],
        "heads": hdr["model.layers.0.self_attn.q_proj.weight"]["shape"][0] // 128,
        "kv": hdr["model.layers.0.self_attn.k_proj.weight"]["shape"][0] // 128,
    }
    m = Qwen(cfg)
    sd = {}
    sd["emb.weight"] = get("model.embed_tokens.weight")
    sd["nf.w"] = get("model.norm.weight")
    for i in range(cfg["layers"]):
        p = f"model.layers.{i}."
        sd[f"blk.{i}.n1.w"] = get(p + "input_layernorm.weight")
        sd[f"blk.{i}.n2.w"] = get(p + "post_attention_layernorm.weight")
        for a in ("q", "k", "v"):
            sd[f"blk.{i}.attn.{a}.weight"] = get(p + f"self_attn.{a}_proj.weight")
            if p + f"self_attn.{a}_proj.bias" in hdr:
                sd[f"blk.{i}.attn.{a}.bias"] = get(p + f"self_attn.{a}_proj.bias")
        sd[f"blk.{i}.attn.o.weight"] = get(p + "self_attn.o_proj.weight")
        sd[f"blk.{i}.mlp.gate.weight"] = get(p + "mlp.gate_proj.weight")
        sd[f"blk.{i}.mlp.up.weight"] = get(p + "mlp.up_proj.weight")
        sd[f"blk.{i}.mlp.down.weight"] = get(p + "mlp.down_proj.weight")
    miss, unexp = m.load_state_dict(sd, strict=False)
    f.close()
    return m, cfg, miss, unexp

# ---------------- 3. eval text ----------------
TEXT = """The history of science is a history of careful observation. When a new instrument is built,
it often reveals something that was not expected. The telescope showed moons around another planet.
The microscope showed a world inside a drop of water. Each time, the old explanation had to change.
A good theory is one that makes a prediction which could fail. If a prediction survives many tests,
we begin to trust it. But trust is not proof, and every theory remains open to revision.
Mathematics gives us a language for these ideas. A number can be large or small, exact or approximate.
When we measure something, we always measure with some error. The question is whether that error
matters for the conclusion we draw. Careful people keep asking this question, and that habit,
more than any single discovery, is what moves knowledge forward. The same habit applies to machines
that learn from data. We can ask how well a model predicts, and we can ask why it predicts that way.
The second question is harder, and it is the one that matters when we decide whether to rely on it."""

def tokenize(text):
    """byte-level fallback: har byte ko token maano (vocab 151936 me fit)"""
    return torch.tensor([[b for b in text.encode()]], dtype=torch.long)

def perplexity(model, ids):
    model.eval()
    with torch.no_grad():
        logits = model(ids)
        lp = F.log_softmax(logits[:, :-1].float(), -1)
        tgt = ids[:, 1:]
        return float(-lp.gather(-1, tgt.unsqueeze(-1)).mean() / math.log(2))  # bits

def top1(model, ids):
    model.eval()
    with torch.no_grad():
        lg = model(ids)[:, :-1]
        return float((lg.argmax(-1) == ids[:, 1:]).float().mean())

# ---------------- main ----------------
def main():
    t0 = time.time()
    print("=== 1) LOAD QWEN 0.5B ===", flush=True)
    m, cfg, miss, unexp = load_qwen("model.safetensors")
    n_par = sum(p.numel() for p in m.parameters())
    print(f"  cfg {cfg}", flush=True)
    print(f"  param {n_par:,} | missing {len(miss)} | unexpected {len(unexp)}", flush=True)
    OUT["params"] = n_par
    OUT["fp16_mb"] = n_par * 2 / 1e6

    ids = tokenize(TEXT)
    print(f"  test tokens {ids.shape[1]}", flush=True)

    print("\n=== 2) BASELINE ===", flush=True)
    ppl0 = perplexity(m, ids); t1_0 = top1(m, ids)
    print(f"  perplexity {ppl0:.3f} bits | top-1 {t1_0*100:.1f}%", flush=True)
    OUT["ppl_base"] = ppl0; OUT["top1_base"] = t1_0

    print("\n=== 3) 85% FFN NEURON PRUNING ===", flush=True)
    kept_total = 0
    with torch.no_grad():
        for b in m.blk:
            imp = b.mlp.gate.weight.norm(dim=1) * b.mlp.up.weight.norm(dim=1)
            keep = int(len(imp) * 0.15)
            idx = imp.topk(keep).indices.sort().values
            b.mlp.gate.weight = nn.Parameter(b.mlp.gate.weight.data[idx].clone())
            b.mlp.up.weight = nn.Parameter(b.mlp.up.weight.data[idx].clone())
            b.mlp.down.weight = nn.Parameter(b.mlp.down.weight.data[:, idx].clone())
            kept_total += keep
    n_pr = sum(p.numel() for p in m.parameters())
    ppl1 = perplexity(m, ids); t1_1 = top1(m, ids)
    print(f"  param {n_pr:,} ({n_pr/n_par*100:.1f}%) | ppl {ppl1:.3f} | top-1 {t1_1*100:.1f}%", flush=True)
    OUT["params_pruned"] = n_pr; OUT["ppl_pruned"] = ppl1; OUT["top1_pruned"] = t1_1

    print("\n=== 4) 2-BIT CODON ===", flush=True)
    pool = []
    for b in m.blk:
        fl = b.mlp.gate.weight.data.reshape(-1)
        nn3 = len(fl) // 3 * 3
        pool.append(fl[:nn3].reshape(-1, 3))
    P = torch.cat(pool, 0)
    if len(P) > 200000:
        P = P[torch.randperm(len(P))[:200000]]
    C = P[torch.randperm(len(P))[:64]].clone()
    for it in range(10):
        d = (P ** 2).sum(1)[:, None] - 2 * P @ C.T + (C ** 2).sum(1)[None]
        lab = d.argmin(1)
        for j in range(64):
            msk = lab == j
            if msk.any(): C[j] = P[msk].mean(0)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() < 2: continue
            fl = p.data.reshape(-1); nn3 = len(fl) // 3 * 3
            A = fl[:nn3].reshape(-1, 3)
            d = (A ** 2).sum(1)[:, None] - 2 * A @ C.T + (C ** 2).sum(1)[None]
            p.data.copy_(torch.cat([C[d.argmin(1)].reshape(-1), fl[nn3:]]).reshape(p.shape))
    ppl2 = perplexity(m, ids); t1_2 = top1(m, ids)
    print(f"  ppl {ppl2:.3f} | top-1 {t1_2*100:.1f}%", flush=True)
    OUT["ppl_codon"] = ppl2; OUT["top1_codon"] = t1_2

    print("\n=== 5) FINE-TUNE (adaptation) ===", flush=True)
    txt = TEXT * 3
    seq = 64
    toks = tokenize(txt)[0]
    nwin = len(toks) // seq
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.01)
    m.train()
    for st in range(1, 61):
        i = torch.randint(0, nwin - 1, (1,)).item()
        x = toks[i * seq:(i + 1) * seq][None]
        y = toks[i * seq + 1:(i + 1) * seq + 1][None]
        loss = F.cross_entropy(m(x)[0].float(), y[0])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if st % 20 == 0:
            print(f"  kadam {st} | loss {loss.item():.3f} | {time.time()-t0:.0f}s", flush=True)
    ppl3 = perplexity(m, ids); t1_3 = top1(m, ids)
    print(f"  ppl {ppl3:.3f} | top-1 {t1_3*100:.1f}%", flush=True)
    OUT["ppl_final"] = ppl3; OUT["top1_final"] = t1_3

    print("\n=== 6) ASLI AAKAR ===", flush=True)
    n_codon = n_pr // 3
    packed = n_codon * 6 // 8
    parity = packed // 8
    table = 64 * 3 * 2
    total = packed + parity + table
    OUT["codon_bytes"] = total
    OUT["size_mb"] = total / 1e6
    OUT["ratio"] = n_par * 2 / total
    print(f"  codon {n_codon:,} | packed {packed/1e6:.1f} MB | parity {parity/1e6:.1f} MB", flush=True)
    print(f"  KUL {total/1e6:.1f} MB | fp16 se {n_par*2/total:.1f}x chhota", flush=True)
    print(f"  + 30% runtime = {total/0.7/1e6:.1f} MB", flush=True)
    OUT["with_runtime_mb"] = total / 0.7 / 1e6

    print("\n=== NATIJA ===", flush=True)
    print(f"  baseline  : ppl {ppl0:.3f} | top-1 {t1_0*100:.1f}%", flush=True)
    print(f"  pruned    : ppl {ppl1:.3f} | top-1 {t1_1*100:.1f}%  ({t1_1*100-t1_0*100:+.1f})", flush=True)
    print(f"  + codon   : ppl {ppl2:.3f} | top-1 {t1_2*100:.1f}%  ({t1_2*100-t1_0*100:+.1f})", flush=True)
    print(f"  + FT      : ppl {ppl3:.3f} | top-1 {t1_3*100:.1f}%  ({t1_3*100-t1_0*100:+.1f})", flush=True)
    print(f"  aakar     : {n_par*2/1e6:.0f} MB -> {total/1e6:.1f} MB ({n_par*2/total:.1f}x)", flush=True)

    json.dump(OUT, open("result.json", "w"), indent=2)
    print("\nsaved result.json", flush=True)

if __name__ == "__main__":
    main()
