"""NeuroCore-Edge mobile-first sparse MoE scaffold.

This is architecture code only. It contains randomly initialized weights and is
NOT a pretrained language model. Training is deliberately not launched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class NeuroCoreEdgeConfig:
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_experts: int = 20
    experts_per_token: int = 2
    expert_intermediate_size: int = 1024
    attention_heads: int = 16
    kv_heads: int = 4
    sliding_window: int = 2048
    recurrent_state_size: int = 128
    router_aux_loss_coefficient: float = 0.01

    def __post_init__(self):
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts]")
        if self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")


class BGGateTop2(nn.Module):
    """Sparse top-k router with an auxiliary load-balancing statistic."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        logits = self.proj(x.float())
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Switch-style load balance loss; no labels or target injection.
        importance = probs.mean(dim=tuple(range(probs.ndim - 1)))
        assignment = F.one_hot(indices, self.num_experts).float().mean(
            dim=tuple(range(indices.ndim - 1))
        ).sum(dim=0)
        aux_loss = self.num_experts * torch.sum(importance * assignment)
        return indices, weights.to(x.dtype), aux_loss


class SparseExpert(nn.Module):
    """Compact gated expert; quantization/ternary packing is a later export step."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.gate(x)) ** 2 * self.up(x))


class HybridWorkingMemory(nn.Module):
    """Bounded recurrent working-memory state plus sliding-window attention stub.

    This reference implementation is intentionally conservative and portable.
    Production Gated-DeltaNet/Mamba kernels and SnapKV are runtime-specific TODOs.
    """

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        self.state_in = nn.Linear(hidden_size, state_size, bias=False)
        self.state_gate = nn.Linear(hidden_size, state_size, bias=True)
        self.state_out = nn.Linear(state_size, hidden_size, bias=False)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        # x: [batch, time, hidden]; state: [batch, state_size]
        batch = x.shape[0]
        if state is None:
            state = x.new_zeros(batch, self.state_in.out_features)
        outputs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            gate = torch.sigmoid(self.state_gate(xt))
            proposal = torch.tanh(self.state_in(xt))
            state = gate * state + (1.0 - gate) * proposal
            outputs.append(xt + self.state_out(state))
        return torch.stack(outputs, dim=1), state


class NeuroCoreEdgeLayer(nn.Module):
    """Hybrid working-memory layer with top-2 sparse expert feed-forward."""

    def __init__(self, cfg: NeuroCoreEdgeConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.memory = HybridWorkingMemory(cfg.hidden_size, cfg.recurrent_state_size)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.router = BGGateTop2(
            cfg.hidden_size, cfg.num_experts, cfg.experts_per_token
        )
        self.experts = nn.ModuleList([
            SparseExpert(cfg.hidden_size, cfg.expert_intermediate_size)
            for _ in range(cfg.num_experts)
        ])

    def forward(self, x, state=None):
        memory_out, next_state = self.memory(self.norm1(x), state)
        x = x + memory_out
        h = self.norm2(x)
        indices, weights, aux_loss = self.router(h)
        flat_h = h.reshape(-1, h.shape[-1])
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        routed = torch.zeros_like(flat_h)
        # Reference dispatch. Optimized grouped dispatch is required for mobile.
        for expert_id, expert in enumerate(self.experts):
            token_pos, slot_pos = torch.where(flat_indices == expert_id)
            if token_pos.numel() == 0:
                continue
            values = expert(flat_h.index_select(0, token_pos))
            scale = flat_weights[token_pos, slot_pos].unsqueeze(-1)
            routed.index_add_(0, token_pos, values * scale)
        x = x + routed.view_as(x)
        return x, next_state, aux_loss


class NeuroCoreEdgeForCausalLM(nn.Module):
    """Untrained reference model; forward returns logits and router aux loss."""

    def __init__(self, cfg: NeuroCoreEdgeConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            NeuroCoreEdgeLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, states=None):
        x = self.embed(input_ids)
        if states is None:
            states = [None] * len(self.layers)
        next_states, aux_losses = [], []
        for layer, state in zip(self.layers, states):
            x, next_state, aux = layer(x, state)
            next_states.append(next_state)
            aux_losses.append(aux)
        logits = self.lm_head(self.final_norm(x))
        aux_loss = torch.stack(aux_losses).mean()
        return {"logits": logits, "router_aux_loss": aux_loss, "states": next_states}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Small smoke-test shape. Do not instantiate the 3B config on a phone.
    cfg = NeuroCoreEdgeConfig(
        vocab_size=512, hidden_size=128, num_hidden_layers=2,
        num_experts=20, experts_per_token=2, expert_intermediate_size=64,
        attention_heads=4, kv_heads=2, recurrent_state_size=16
    )
    model = NeuroCoreEdgeForCausalLM(cfg)
    sample = torch.randint(0, cfg.vocab_size, (2, 12))
    result = model(sample)
    assert result["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(result["router_aux_loss"])
    print({"status": "SMOKE_PASS", "parameters": count_parameters(model),
           "logits_shape": list(result["logits"].shape)})
