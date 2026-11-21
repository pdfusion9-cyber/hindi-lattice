#!/usr/bin/env python3
"""Small structural probe inspired by NeuroCore-Edge.

NOT the full published Gated DeltaNet implementation and NOT a language-model
accuracy test. It checks a compact recurrent delta memory + local-attention
hybrid for shapes, finite values, gradients, and estimated cache/state bytes.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import torch
from torch import nn
import torch.nn.functional as F


class DeltaMemory(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.dk = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Linear(dim, heads)
        self.write = nn.Linear(dim, heads)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        q = self.q(x).view(bsz, steps, self.heads, self.dk)
        k = F.normalize(self.k(x).view(bsz, steps, self.heads, self.dk), dim=-1)
        v = self.v(x).view(bsz, steps, self.heads, self.dk)
        decay = torch.sigmoid(self.decay(x)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        write = torch.sigmoid(self.write(x)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        state = x.new_zeros((bsz, self.heads, self.dk, self.dk))
        ys = []
        for t in range(steps):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            retrieved = torch.einsum("bhk,bhkv->bhv", kt, state)
            residual = vt - retrieved
            update = torch.einsum("bhk,bhv->bhkv", kt, residual)
            state = decay[:, :, t] * state + write[:, :, t] * update
            ys.append(torch.einsum("bhk,bhkv->bhv", qt, state))
        y = torch.stack(ys, dim=1).reshape(bsz, steps, self.dim)
        return self.out(y)


class LocalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.dim, self.heads, self.window = dim, heads, window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, steps, self.heads, -1).transpose(1, 2)
        k = k.view(bsz, steps, self.heads, -1).transpose(1, 2)
        v = v.view(bsz, steps, self.heads, -1).transpose(1, 2)
        pos = torch.arange(steps, device=x.device)
        allowed = (pos[:, None] >= pos[None, :]) & ((pos[:, None] - pos[None, :]) < self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, steps, dim))


class NeuroCoreEdgeMini(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, window: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.delta = DeltaMemory(dim, heads)
        self.local = LocalAttention(dim, heads, window)
        self.mix = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate = torch.sigmoid(self.mix(h))
        x = x + gate * self.delta(h) + (1.0 - gate) * self.local(h)
        return x + self.ff(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out/neurocore_probe.json")
    args = parser.parse_args()
    torch.manual_seed(7)
    bsz, steps, dim, heads, window = 2, 64, 128, 4, 16
    model = NeuroCoreEdgeMini(dim, heads, window)
    x = torch.randn(bsz, steps, dim, requires_grad=True)
    y = model(x)
    loss = y.float().square().mean()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                   for p in model.parameters())
    forward_ok = (y.shape == x.shape and torch.isfinite(y).all().item()
                  and math.isfinite(loss.item()))

    target_t = 2048
    dh = dim // heads
    full_kv = 2 * bsz * heads * target_t * dh * 4
    local_kv = 2 * bsz * heads * window * dh * 4
    recurrent_state = bsz * heads * dh * dh * 4
    hybrid_working = local_kv + recurrent_state
    report = {
        "probe": "NeuroCore-Edge mini: gated delta memory + sliding-window attention",
        "not_full_reference_implementation": True,
        "test": {"batch": bsz, "sequence": steps, "dim": dim,
                 "heads": heads, "window": window},
        "forward_finite_and_shape_ok": bool(forward_ok),
        "backward_gradients_finite": bool(grads_ok),
        "loss": float(loss.detach()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "fp32_memory_estimate_at_2048_tokens_per_batch": {
            "full_kv_cache_bytes": full_kv,
            "local_window_kv_plus_recurrent_state_bytes": hybrid_working,
            "estimated_ratio_full_to_hybrid": round(full_kv / hybrid_working, 2),
            "note": "state/cache only; excludes activations, weights, allocator overhead and graph memory"
        },
        "accuracy_tested": False
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not (forward_ok and grads_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
