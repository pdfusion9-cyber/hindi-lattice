"""NeuroCore-Edge v0.3.2 canonical architecture reference.

This file implements the sandwich execution topology, grouped-query rotary
attention, bounded recurrent working memory, learned meta-token prefix and
sparse experts. It is still an untrained reference implementation: it does not
include packed W1.58A8 kernels, a trained tokenizer, or a trained checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class NeuroCoreEdgeConfig:
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_hidden_layers: int = 24  # effective operation count, not unique blocks
    num_experts: int = 20
    experts_per_token: int = 2
    expert_intermediate_size: int = 2899
    attention_heads: int = 16
    kv_heads: int = 4
    sliding_window: int = 2048
    recurrent_state_size: int = 128
    router_aux_loss_coefficient: float = 0.01
    prelude_layers: int = 2
    core_unique_layers: int = 4
    core_repeats: int = 5
    coda_layers: int = 2
    meta_token_count: int = 128
    rope_theta: float = 10000.0
    global_attention_layers: tuple[int, ...] = (7, 15, 23)

    def __post_init__(self):
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts]")
        if self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")
        if self.prelude_layers < 0 or self.core_unique_layers < 1 or self.coda_layers < 0:
            raise ValueError("sandwich block counts are invalid")
        if self.core_repeats < 1:
            raise ValueError("core_repeats must be positive")
        expected_unique = self.prelude_layers + self.core_unique_layers + self.coda_layers
        expected_effective = self.prelude_layers + self.core_unique_layers * self.core_repeats + self.coda_layers
        if expected_effective != self.num_hidden_layers:
            raise ValueError(
                f"num_hidden_layers must equal effective sandwich depth "
                f"({expected_effective}), got {self.num_hidden_layers}"
            )
        if self.meta_token_count < 0:
            raise ValueError("meta_token_count cannot be negative")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if any(i < 0 or i >= expected_effective for i in self.global_attention_layers):
            raise ValueError("global attention layer index outside effective depth")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class GQARotaryAttention(nn.Module):
    """Causal grouped-query attention with RoPE and local/global modes."""

    def __init__(
        self,
        hidden_size: int,
        attention_heads: int,
        kv_heads: int,
        sliding_window: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if attention_heads % kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        self.hidden_size = hidden_size
        self.num_heads = attention_heads
        self.num_kv_heads = kv_heads
        self.head_dim = hidden_size // attention_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.sliding_window = sliding_window
        self.rope_theta = float(rope_theta)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        kv_width = kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _apply_rope(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # value: [batch, heads, sequence, head_dim]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, device=value.device, dtype=torch.float32
                )
                / self.head_dim
            )
        )
        frequencies = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(value.dtype)[None, None, :, :]
        sin = embedding.sin().to(value.dtype)[None, None, :, :]
        return value * cos + _rotate_half(value) * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        global_attention: bool = False,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch, sequence, width = hidden_states.shape
        if width != self.hidden_size:
            raise ValueError("hidden dimension does not match attention configuration")
        if sequence == 0:
            return hidden_states

        q = self.q_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(
            position_offset, position_offset + sequence, device=hidden_states.device
        )
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        repeat_factor = self.num_heads // self.num_kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        query_positions = positions[:, None]
        key_positions = positions[None, :]
        allowed = key_positions <= query_positions
        if not global_attention and self.sliding_window > 0:
            allowed = allowed & (
                key_positions > query_positions - self.sliding_window
            )
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.o_proj(attended)


class BGGateTop2(nn.Module):
    """Sparse top-k router with a load-balancing auxiliary statistic."""

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
        importance = probs.mean(dim=tuple(range(probs.ndim - 1)))
        assignment = F.one_hot(indices, self.num_experts).float().mean(
            dim=tuple(range(indices.ndim - 1))
        )
        aux_loss = self.num_experts * torch.sum(importance * assignment)
        return indices, weights.to(x.dtype), aux_loss


class SparseExpert(nn.Module):
    """Gated ReLU-squared expert; mobile quantization is a later export step."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.gate(x)) ** 2 * self.up(x))


class HybridWorkingMemory(nn.Module):
    """Bounded recurrent state reference block; not a fused Gated-DeltaNet kernel."""

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        self.state_in = nn.Linear(hidden_size, state_size, bias=False)
        self.state_gate = nn.Linear(hidden_size, state_size, bias=True)
        self.state_out = nn.Linear(state_size, hidden_size, bias=False)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        batch = x.shape[0]
        if state is None:
            state = x.new_zeros(batch, self.state_in.out_features)
        outputs = []
        for token_index in range(x.shape[1]):
            token = x[:, token_index]
            gate = torch.sigmoid(self.state_gate(token))
            proposal = torch.tanh(self.state_in(token))
            state = gate * state + (1.0 - gate) * proposal
            outputs.append(self.state_out(state))
        if not outputs:
            return x.new_zeros(x.shape), state
        return torch.stack(outputs, dim=1), state


class NeuroCoreEdgeLayer(nn.Module):
    """One unique parameterized block used at one or more effective depths."""

    def __init__(self, cfg: NeuroCoreEdgeConfig):
        super().__init__()
        self.norm_memory = nn.LayerNorm(cfg.hidden_size)
        self.memory = HybridWorkingMemory(cfg.hidden_size, cfg.recurrent_state_size)
        self.norm_attention = nn.LayerNorm(cfg.hidden_size)
        self.attention = GQARotaryAttention(
            cfg.hidden_size,
            cfg.attention_heads,
            cfg.kv_heads,
            cfg.sliding_window,
            cfg.rope_theta,
        )
        self.norm_expert = nn.LayerNorm(cfg.hidden_size)
        self.router = BGGateTop2(
            cfg.hidden_size, cfg.num_experts, cfg.experts_per_token
        )
        self.experts = nn.ModuleList([
            SparseExpert(cfg.hidden_size, cfg.expert_intermediate_size)
            for _ in range(cfg.num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        *,
        global_attention: bool = False,
        force_expert_id: Optional[int] = None,
    ):
        memory_delta, next_state = self.memory(self.norm_memory(x), state)
        x = x + memory_delta
        x = x + self.attention(
            self.norm_attention(x), global_attention=global_attention
        )
        normalized = self.norm_expert(x)

        if force_expert_id is not None:
            if not isinstance(force_expert_id, int) or isinstance(force_expert_id, bool):
                raise TypeError("force_expert_id must be an integer")
            if force_expert_id not in range(len(self.experts)):
                raise ValueError("force_expert_id outside configured expert range")
            return x + self.experts[force_expert_id](normalized), next_state, x.new_zeros(())

        indices, weights, aux_loss = self.router(normalized)
        flat_hidden = normalized.reshape(-1, normalized.shape[-1])
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        routed = torch.zeros_like(flat_hidden)
        for expert_id, expert in enumerate(self.experts):
            token_positions, slot_positions = torch.where(flat_indices == expert_id)
            if token_positions.numel() == 0:
                continue
            values = expert(flat_hidden.index_select(0, token_positions))
            scale = flat_weights[token_positions, slot_positions].unsqueeze(-1)
            routed.index_add_(0, token_positions, values * scale)
        return x + routed.view_as(x), next_state, aux_loss


class NeuroCoreEdgeForCausalLM(nn.Module):
    """Untrained sandwich MoE reference model with tied token embeddings."""

    def __init__(self, cfg: NeuroCoreEdgeConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.meta_token_bank = nn.Parameter(
            torch.empty(cfg.meta_token_count, cfg.hidden_size)
        )
        nn.init.normal_(self.meta_token_bank, mean=0.0, std=0.02)

        self.prelude_layers = nn.ModuleList([
            NeuroCoreEdgeLayer(cfg) for _ in range(cfg.prelude_layers)
        ])
        self.core_layers = nn.ModuleList([
            NeuroCoreEdgeLayer(cfg) for _ in range(cfg.core_unique_layers)
        ])
        self.coda_layers = nn.ModuleList([
            NeuroCoreEdgeLayer(cfg) for _ in range(cfg.coda_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def _execution_plan(self) -> list[tuple[str, int]]:
        plan: list[tuple[str, int]] = []
        plan.extend(("prelude", index) for index in range(len(self.prelude_layers)))
        for _ in range(self.config.core_repeats):
            plan.extend(("core", index) for index in range(len(self.core_layers)))
        plan.extend(("coda", index) for index in range(len(self.coda_layers)))
        return plan

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[Sequence[Optional[torch.Tensor]]] = None,
        force_expert_id: Optional[int] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        x = self.embed(input_ids)
        batch = x.shape[0]

        if self.config.meta_token_count:
            meta = self.meta_token_bank.unsqueeze(0).expand(batch, -1, -1)
            x = torch.cat((meta, x), dim=1)

        plan = self._execution_plan()
        unique_count = (
            len(self.prelude_layers) + len(self.core_layers) + len(self.coda_layers)
        )
        if states is None:
            state_list: list[Optional[torch.Tensor]] = [None] * unique_count
        else:
            if len(states) != unique_count:
                raise ValueError(f"states must contain exactly {unique_count} unique-block states")
            state_list = list(states)

        next_states = list(state_list)
        aux_losses = []
        trace = []
        prelude_count = len(self.prelude_layers)
        core_count = len(self.core_layers)

        for effective_index, (group, block_index) in enumerate(plan):
            if group == "prelude":
                block = self.prelude_layers[block_index]
                state_index = block_index
            elif group == "core":
                block = self.core_layers[block_index]
                state_index = prelude_count + block_index
            else:
                block = self.coda_layers[block_index]
                state_index = prelude_count + core_count + block_index

            is_global = effective_index in self.config.global_attention_layers
            x, next_state, aux_loss = block(
                x,
                next_states[state_index],
                global_attention=is_global,
                force_expert_id=force_expert_id,
            )
            next_states[state_index] = next_state
            aux_losses.append(aux_loss)
            trace.append({
                "effective_index": effective_index,
                "group": group,
                "block_index": block_index,
                "global_attention": is_global,
            })

        logits = self.lm_head(self.final_norm(x))
        if self.config.meta_token_count:
            logits = logits[:, self.config.meta_token_count :, :]
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        return {
            "logits": logits,
            "router_aux_loss": aux_loss,
            "states": next_states,
            "execution_trace": trace,
        }


def count_parameters(model: nn.Module):
    # Tied embedding/lm_head parameters are counted once by parameters().
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    cfg = NeuroCoreEdgeConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=5,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=64,
        attention_heads=4,
        kv_heads=2,
        sliding_window=8,
        recurrent_state_size=16,
        prelude_layers=1,
        core_unique_layers=1,
        core_repeats=3,
        coda_layers=1,
        meta_token_count=8,
        global_attention_layers=(3,),
    )
    model = NeuroCoreEdgeForCausalLM(cfg)
    sample = torch.randint(0, cfg.vocab_size, (2, 12))
    result = model(sample)
    assert result["logits"].shape == (2, 12, cfg.vocab_size)
    assert len(result["execution_trace"]) == 5
    assert torch.isfinite(result["router_aux_loss"])
    print({
        "status": "SMOKE_PASS",
        "parameters": count_parameters(model),
        "logits_shape": list(result["logits"].shape),
        "effective_depth": len(result["execution_trace"]),
        "unique_blocks": 3,
    })
