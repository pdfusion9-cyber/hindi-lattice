#!/usr/bin/env python3
"""Shared connection-model trainer: real context + on-demand geometric links.

The geometric connection table is never materialized. For a source token, its
link scores to the vocabulary are calculated from learned 3D positions.
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus(paths):
    pieces = []
    for path in paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                pieces.append(f.read())
            print(f"corpus: {path} | {os.path.getsize(path):,} bytes", flush=True)
        else:
            print(f"corpus file missing (skip): {path}", flush=True)
    if not pieces:
        raise SystemExit("कोई corpus file नहीं मिली")
    return "\n".join(pieces)


def make_stream(args, text):
    if args.mode == "tokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("tokenizers package जरूरी है: pip install tokenizers") from exc
        if not os.path.isfile(args.tokenizer):
            raise SystemExit(f"tokenizer नहीं मिला: {args.tokenizer}")
        tok_bytes = open(args.tokenizer, "rb").read()
        tok = Tokenizer.from_file(args.tokenizer)
        vocab_size = tok.get_vocab_size()
        ids = tok.encode(text).ids
        stream = np.asarray(ids, dtype=np.int32)
        vocab_hash = sha256_bytes(tok_bytes)
        print(f"Qwen/BPE vocabulary: {vocab_size:,}", flush=True)
    else:
        words = re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower())
        if len(words) < 100:
            raise SystemExit("regex corpus में पर्याप्त हिंदी/अंग्रेज़ी शब्द नहीं मिले")
        counts = Counter(words)
        vocab = [w for w, _ in counts.most_common(args.vocab_limit)]
        stoi = {w: i for i, w in enumerate(vocab)}
        stream = np.fromiter((stoi[w] for w in words if w in stoi), dtype=np.int32)
        vocab_size = len(vocab)
        vocab_hash = sha256_bytes("\n".join(vocab).encode("utf-8"))
        print(f"word vocabulary: {vocab_size:,} | raw words: {len(words):,}", flush=True)

    if vocab_size < 8:
        raise SystemExit(f"vocabulary बहुत छोटा: {vocab_size}")
    if len(stream) <= args.seq * 4:
        raise SystemExit(f"टोकन बहुत कम: {len(stream):,}; seq={args.seq}")
    if stream.min() < 0 or stream.max() >= vocab_size:
        raise SystemExit("token ID vocabulary सीमा से बाहर है")
    stream_hash = sha256_bytes(stream.tobytes())
    print(f"tokens: {len(stream):,} | unique seen: {len(np.unique(stream)):,}/{vocab_size:,}", flush=True)
    return stream, vocab_size, vocab_hash, stream_hash


class CausalBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, 4 * dim)
        self.ff2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        hd = d // self.heads
        q = q.view(b, t, self.heads, hd).transpose(1, 2)
        k = k.view(b, t, self.heads, hd).transpose(1, 2)
        v = v.view(b, t, self.heads, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(a)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class ConnectionLM(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, seq):
        super().__init__()
        if dim % heads:
            raise ValueError("dim को heads से पूरा विभाजित होना चाहिए")
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq = seq
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq, dim)
        self.xyz = nn.Parameter(torch.empty(vocab_size, 3))
        nn.init.uniform_(self.xyz, -1.0, 1.0)
        self.log_sigma = nn.Parameter(torch.zeros(vocab_size))
        self.log_amp = nn.Parameter(torch.zeros(vocab_size))
        self.geo_scale = nn.Parameter(torch.tensor(0.1))
        self.blocks = nn.ModuleList([CausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def connection_scores(self, source_ids):
        p_all = self.xyz
        p_src = p_all[source_ids]
        d2 = (
            (p_src * p_src).sum(-1, keepdim=True)
            + (p_all * p_all).sum(-1)[None, :]
            - 2.0 * (p_src @ p_all.T)
        ).clamp_min(0.0)
        sigma = self.log_sigma[source_ids].clamp(-3.0, 1.1).exp()[:, None]
        amp_src = self.log_amp[source_ids].clamp(-3.0, 1.1).exp()[:, None]
        amp_all = self.log_amp.clamp(-3.0, 1.1).exp()[None, :]
        score = amp_src * amp_all * torch.exp(-d2 / (2.0 * sigma.square()))
        return score - score.mean(dim=-1, keepdim=True)

    def forward(self, x, use_geometry=True):
        if x.ndim != 2 or x.shape[1] > self.seq:
            raise ValueError(f"input shape {tuple(x.shape)}; max context={self.seq}")
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h[:, -1, :])
        logits = F.linear(h, self.tok.weight) / math.sqrt(self.dim)
        if use_geometry:
            logits = logits + torch.tanh(self.geo_scale) * self.connection_scores(x[:, -1])
        return logits


def make_batch(stream, positions, seq, device):
    offsets = np.arange(seq, dtype=np.int64)[None, :]
    x = stream[positions[:, None] - seq + offsets]
    y = stream[positions]
    return torch.as_tensor(x, dtype=torch.long, device=device), torch.as_tensor(y, dtype=torch.long, device=device)


def evaluate(model, stream, positions, seq, batch_size, device, use_geometry):
    model.eval()
    total_loss = 0.0
    total = top1 = top5 = 0
    ranks = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            idx = positions[start:start + batch_size]
            x, y = make_batch(stream, idx, seq, device)
            logits = model(x, use_geometry=use_geometry)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(-1)
            top1 += (pred == y).sum().item()
            k = min(5, logits.shape[-1])
            top5 += (logits.topk(k, dim=-1).indices == y[:, None]).any(-1).sum().item()
            true_score = logits.gather(1, y[:, None])
            ranks.extend(((logits > true_score).sum(-1) + 1).cpu().tolist())
            total += len(y)
    mean_loss = total_loss / max(total, 1)
    return {
        "samples": total,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "top1_percent": 100.0 * top1 / max(total, 1),
        "top5_percent": 100.0 * top5 / max(total, 1),
        "median_rank": int(np.median(ranks)) if ranks else None,
    }


def save_checkpoint(path, model, optimizer, step, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "meta": meta,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
    }, tmp)
    os.replace(tmp, path)


def main(default_mode="regex", default_vocab_limit=3000, default_ckpt="checkpoints/connection.pt", argv=None):
    ap = argparse.ArgumentParser(description="Context-aware low-parameter geometric connection LM")
    ap.add_argument("--mode", choices=("regex", "tokenizer"), default=default_mode)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--corpus", nargs="+", default=None)
    ap.add_argument("--vocab-limit", type=int, default=default_vocab_limit)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--metrics", default="results/connection_metrics.json")
    args = ap.parse_args(argv)

    if args.seq < 2 or args.dim < 8 or args.layers < 1 or args.heads < 1 or args.batch < 1:
        raise SystemExit("seq, dim, layers, heads और batch की values जाँचें")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    corpus_paths = args.corpus or ["data/web.txt", "data/gyaan_base.txt"]
    text = read_corpus(corpus_paths)
    stream, vocab_size, vocab_hash, stream_hash = make_stream(args, text)
    del text

    cut = int(len(stream) * 0.9)
    if cut <= args.seq or len(stream) - (cut + args.seq) < 10:
        raise SystemExit("train/test split के लिए data कम है")
    test_all = np.arange(cut + args.seq, len(stream), dtype=np.int64)
    rng = np.random.default_rng(args.seed + 1)
    eval_n = min(args.eval_samples, len(test_all))
    test_positions = np.sort(rng.choice(test_all, size=eval_n, replace=False))

    meta = {
        "version": 2,
        "mode": args.mode,
        "vocab_size": int(vocab_size),
        "vocab_hash": vocab_hash,
        "stream_hash": stream_hash,
        "seq": args.seq,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
    }
    model = ConnectionLM(vocab_size, args.dim, args.layers, args.heads, args.seq).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_params = sum(p.numel() for p in model.parameters())
    lattice_params = vocab_size * 5
    potential_pairs = vocab_size * vocab_size
    print(f"\n=== CONNECTION MODEL V2 ===", flush=True)
    print(f"vocabulary: {vocab_size:,} | directed potential pairs (incl self): {potential_pairs:,}", flush=True)
    print(f"lattice parameters: {lattice_params:,} | total trainable parameters: {total_params:,}", flush=True)
    print(f"pair matrix stored: NO | geometry dimension: 3 | context: {args.seq} tokens", flush=True)
    print(f"train tokens: {cut:,} | held-out tokens: {len(stream)-cut:,} | eval samples: {eval_n:,}", flush=True)

    step0 = 0
    if os.path.isfile(args.ckpt):
        try:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if ck.get("meta") == meta:
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                step0 = int(ck.get("step", 0))
                print(f"compatible checkpoint restored: step {step0:,}", flush=True)
            else:
                print("old checkpoint incompatible (data/model metadata differs); safe fresh start", flush=True)
        except Exception as exc:
            print(f"checkpoint restore failed ({type(exc).__name__}); safe fresh start", flush=True)

    t0 = time.time()
    step = step0
    model.train()
    while step < args.steps and time.time() - t0 < args.budget:
        positions = rng.integers(args.seq, cut, size=args.batch, dtype=np.int64)
        x, y = make_batch(stream, positions, args.seq, device)
        logits = model(x, use_geometry=True)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step:,} | loss {loss.item():.4f} | elapsed {time.time()-t0:.0f}s", flush=True)
        if step % args.save_every == 0:
            save_checkpoint(args.ckpt, model, optimizer, step, meta)
    save_checkpoint(args.ckpt, model, optimizer, step, meta)

    print(f"\ntraining stopped at step {step:,} | elapsed {time.time()-t0:.0f}s", flush=True)
    geo = evaluate(model, stream, test_positions, args.seq, args.eval_batch, device, True)
    no_geo = evaluate(model, stream, test_positions, args.seq, args.eval_batch, device, False)
    result = {
        "meta": meta,
        "potential_directed_pairs_including_self": potential_pairs,
        "lattice_parameters": lattice_params,
        "total_trainable_parameters": total_params,
        "training_step": step,
        "training_seconds": round(time.time() - t0, 2),
        "geometry_on": geo,
        "geometry_off_ablation": no_geo,
        "uniform_ppl_reference": vocab_size,
        "checkpoint": args.ckpt,
    }
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n=== HELD-OUT METRICS ===", flush=True)
    print("geometry ON : " + json.dumps(geo, ensure_ascii=False), flush=True)
    print("geometry OFF: " + json.dumps(no_geo, ensure_ascii=False), flush=True)
    print(f"uniform reference PPL: {vocab_size:,}", flush=True)
    print(f"checkpoint saved: {args.ckpt}", flush=True)
    print(f"metrics saved: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
