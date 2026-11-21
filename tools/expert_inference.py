#!/usr/bin/env python3
"""Read-only inference for one independent CHAKRA expert checkpoint.

This mirrors punah.py's inference-relevant forward path, including MQA,
residual scaling, graph-neighbour attention, and the sparse local/graph mask.
It never trains, saves, averages, or quantizes a checkpoint.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_RE = re.compile(r"[^\s।,?!]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _map_get(mapping: dict, key: int, default=None):
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def pattern_id_for(tokens: list[str], vocab: list[str], vocab_counts: list[int],
                   patterns: list, pattern_to_id: dict) -> tuple[int, bool]:
    """Return exact training-style pattern ID; unknown/incomplete -> safe ID 0."""
    counts = {w: int(vocab_counts[i]) for i, w in enumerate(vocab)
              if i < len(vocab_counts)}
    stop = {w for w, count in counts.items() if count >= 300}
    pattern = tuple(w if w in stop else "*" for w in tokens)
    idx = pattern_to_id.get(pattern)
    if idx is None:
        return 0, False
    return int(idx), True


class ExpertModel(nn.Module):
    """Inference-equivalent mirror of class M in punah.py."""
    def __init__(self, V: int, P: int, d: int, layers: int, heads: int,
                 K: int, res_scale: float):
        super().__init__()
        self.d, self.H, self.K = d, heads, K
        self.res_scale = float(res_scale)
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(256, d)
        self.pat = nn.Embedding(max(P, 1), d)
        self.blk = nn.ModuleList()
        for _ in range(layers):
            self.blk.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "q": nn.Linear(d, d, bias=False),
                "k": nn.Linear(d, d // heads, bias=False),
                "v": nn.Linear(d, d // heads, bias=False),
                "out": nn.Linear(d, d, bias=False), "n2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, 4*d), "ff2": nn.Linear(4*d, d),
                "n3": nn.LayerNorm(d), "lq": nn.Linear(d, d, bias=False),
                "lo": nn.Linear(d, d, bias=False),
            }))
        self.nf = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, nb: torch.Tensor, pid: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        h = h + self.pat(pid)[:, None, :]
        for b in self.blk:
            xx = b["n1"](h)
            q = b["q"](xx).view(B,T,self.H,-1).transpose(1,2)
            k = b["k"](xx).view(B,T,1,-1).transpose(1,2)
            v = b["v"](xx).view(B,T,1,-1).transpose(1,2)
            att = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)
            h = h + self.res_scale*b["out"](
                att.transpose(1,2).contiguous().view(B,T,self.d))
            h = h + self.res_scale*b["ff2"](F.gelu(b["ff1"](b["n2"](h))))
            mem = self.tok(nb)
            q2 = b["lq"](b["n3"](h)).view(B,T,self.H,-1)
            k2 = mem.view(B,T,self.K,self.H,-1).permute(0,1,3,2,4)
            sc = (q2.unsqueeze(3) @ k2.transpose(-1,-2))/math.sqrt(self.d//self.H)
            h = h + self.res_scale*b["lo"](
                (sc.softmax(-1) @ k2).squeeze(3).reshape(B,T,self.d))
        return self.nf(h)


class ExpertRunner:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        # Only trusted, internally-produced expert packages should be loaded.
        self.ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if self.ck.get("format") != "chakra-independent-expert-v1":
            raise ValueError("Unsupported checkpoint package format")
        self.domain = str(self.ck["domain"])
        self.vocab = list(self.ck["vocab"])
        self.vocab_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.patterns = [tuple(p) for p in self.ck["pats"]]
        self.pattern_to_id = {p: i for i, p in enumerate(self.patterns)}
        self.vocab_counts = list(self.ck.get("vocab_counts") or [])
        self.graph = self.ck.get("graph") or {}
        self.config = self.ck.get("config") or {}
        self.seq_len = 16  # punah.py default and training shape
        self.sparse_win = int(self.config.get("sparse_win", 6))
        self.nbr_count = int(self.config["nbr"])
        self.candidate_limit = int(self.config.get("cand", 50)) + int(
            self.config.get("pat_cand", 20))
        self.slot_candidates = {}
        for row in self.graph.get("slot_cand", []):
            key = (tuple(row.get("pattern", [])), int(row.get("position", -1)))
            self.slot_candidates[key] = [int(v) for v in row.get("ids", [])
                                         if 0 <= int(v) < len(self.vocab)]
        self.model = ExpertModel(
            int(self.ck["V"]), len(self.patterns), int(self.ck["dim"]),
            int(self.ck["layers"]), int(self.ck["heads"]), self.nbr_count,
            float(self.config.get("res_scale", 0.5)))
        self.model.load_state_dict(self.ck["m"], strict=True)
        self.model.to(self.device).eval()

    def _build_inputs(self, token_ids: list[int], pid: int):
        # punah.py right-pads contexts and predicts from output position T-1.
        ctx = token_ids[-self.seq_len:]
        x = torch.zeros((1, self.seq_len), dtype=torch.long)
        nb = torch.zeros((1, self.seq_len, self.nbr_count), dtype=torch.long)
        if ctx:
            x[0, :len(ctx)] = torch.tensor(ctx, dtype=torch.long)
        nbrs = self.graph.get("nbrs", {})
        for t, token_id in enumerate(ctx):
            values = [int(v) for v in _map_get(nbrs, token_id, [])]
            values = (values[:self.nbr_count] + [0] * self.nbr_count)[:self.nbr_count]
            nb[0, t] = torch.tensor(values, dtype=torch.long)

        mask = torch.zeros((1, 1, self.seq_len, self.seq_len), dtype=torch.bool)
        positions = torch.arange(self.seq_len)
        local = ((positions[None, :] >= positions[:, None] - self.sparse_win)
                 & (positions[None, :] <= positions[:, None]))
        mask[0, 0] = local
        # Exactly reproduce punah.py's additional graph links among prior
        # positions in the unpadded context. Padded positions get local links only.
        for t, token_id in enumerate(ctx):
            allowed = set(int(v) for v in _map_get(nbrs, token_id, []))
            for q in range(t):
                if ctx[q] in allowed:
                    mask[0, 0, t, q] = True
        pid_tensor = torch.tensor([pid], dtype=torch.long)
        return (x.to(self.device), nb.to(self.device), pid_tensor.to(self.device),
                mask.to(self.device))

    def _candidate_ids(self, last_token_id: int, pattern_id: int, position: int,
                       exact_pattern: bool) -> list[int]:
        cands = self.graph.get("cands", {})
        nbrs = self.graph.get("nbrs", {})
        values = _map_get(cands, last_token_id, []) or _map_get(nbrs, last_token_id, [])
        out = list(dict.fromkeys(int(v) for v in values
                                 if 0 <= int(v) < len(self.vocab)))
        # Slot candidates are keyed by a complete training-sentence pattern
        # and its absolute position. A prompt prefix cannot safely identify
        # those values; do not inject pattern-0 slots on a failed match.
        if exact_pattern and 0 <= pattern_id < len(self.patterns):
            out.extend(self.slot_candidates.get((self.patterns[pattern_id], position), []))
        out = list(dict.fromkeys(out))
        if not out:
            # Vocab is frequency ordered by the training script.
            out = list(range(min(self.candidate_limit, len(self.vocab))))
        return out[:self.candidate_limit]

    @torch.inference_mode()
    def next_token(self, tokens: list[str]) -> tuple[str, dict[str, Any]]:
        ids = [self.vocab_to_id[w] for w in tokens if w in self.vocab_to_id]
        if len(ids) < 2:
            raise ValueError("At least two in-vocabulary prompt tokens are required")
        pid, exact_pattern = pattern_id_for(
            tokens, self.vocab, self.vocab_counts, self.patterns, self.pattern_to_id)
        x, nb, pid_tensor, mask = self._build_inputs(ids, pid)
        hidden = self.model(x, nb, pid_tensor, mask)[:, -1, :]
        candidates = self._candidate_ids(ids[-1], pid, len(tokens), exact_pattern)
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)
        scores = hidden @ self.model.tok(candidate_tensor).T
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite candidate scores; inference aborted")
        best = int(scores.argmax(-1).item())
        token_id = candidates[best]
        return self.vocab[token_id], {
            "domain": self.domain, "candidate_count": len(candidates),
            "pattern_exact_match": exact_pattern, "pattern_id": pid,
            "score": float(scores[0, best].item()),
            "context_token_count": min(len(ids), self.seq_len),
        }

    def generate(self, prompt: str, max_new_tokens: int = 12) -> dict[str, Any]:
        words = tokenize(prompt)
        known = [w for w in words if w in self.vocab_to_id]
        unknown = [w for w in words if w not in self.vocab_to_id]
        if len(known) < 2:
            raise ValueError("Prompt has fewer than two known vocabulary tokens")
        generated = []
        diagnostics = []
        context = list(known)
        for _ in range(max(0, min(int(max_new_tokens), 64))):
            token, info = self.next_token(context)
            generated.append(token)
            diagnostics.append(info)
            context.append(token)
        return {
            "domain": self.domain, "prompt": prompt,
            "known_prompt_tokens": len(known), "unknown_prompt_tokens": unknown,
            "generated_tokens": generated,
            "text": " ".join(generated),
            "diagnostics": diagnostics,
            "notice": ("Greedy graph-candidate continuation; not a validated answer-quality "
                       "benchmark. Pattern ID falls back to 0 when no exact pattern matches."),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    runner = ExpertRunner(args.checkpoint, args.device)
    import json
    print(json.dumps(runner.generate(args.prompt, args.max_new_tokens),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
