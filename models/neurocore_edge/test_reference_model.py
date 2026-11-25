import torch
from modeling_neurocore_edge import (
    NeuroCoreEdgeConfig,
    NeuroCoreEdgeForCausalLM,
    count_parameters,
)


def tiny_config():
    return NeuroCoreEdgeConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=5,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=16,
        attention_heads=4,
        kv_heads=2,
        sliding_window=8,
        recurrent_state_size=8,
        prelude_layers=1,
        core_unique_layers=1,
        core_repeats=3,
        coda_layers=1,
        meta_token_count=4,
        global_attention_layers=(3,),
    )


def test_tiny_reference_forward_backward():
    torch.manual_seed(7)
    model = NeuroCoreEdgeForCausalLM(tiny_config())
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    result = model(ids)
    assert result["logits"].shape == (2, 8, model.config.vocab_size)
    assert len(result["execution_trace"]) == 5
    assert len(model.prelude_layers) + len(model.core_layers) + len(model.coda_layers) == 3
    loss = result["logits"].float().square().mean() + result["router_aux_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.embed.weight.grad is not None
    assert model.core_layers[0].router.proj.weight.grad is not None
    assert count_parameters(model) > 0


def test_core_repetitions_reuse_the_same_parameter_objects():
    model = NeuroCoreEdgeForCausalLM(tiny_config())
    plan = model._execution_plan()
    core_calls = [item for item in plan if item[0] == "core"]
    assert len(core_calls) == 3
    assert len({id(model.core_layers[index]) for _, index in core_calls}) == 1
    assert len(model.core_layers) == 1


def test_forced_expert_updates_only_selected_expert():
    torch.manual_seed(13)
    model = NeuroCoreEdgeForCausalLM(tiny_config())
    before = {
        i: [p.detach().clone() for p in expert.parameters()]
        for i, expert in enumerate(model.core_layers[0].experts)
    }
    ids = torch.randint(0, model.config.vocab_size, (1, 5))
    result = model(ids, force_expert_id=7)
    loss = result["logits"].float().square().mean()
    loss.backward()
    optimizer = torch.optim.SGD(
        model.core_layers[0].experts[7].parameters(), lr=1e-3
    )
    optimizer.step()
    for expert_id, expert in enumerate(model.core_layers[0].experts):
        for old, current in zip(before[expert_id], expert.parameters()):
            if expert_id == 7:
                assert not torch.equal(old, current)
            else:
                assert torch.equal(old, current)


def test_invalid_force_expert_id_is_rejected():
    model = NeuroCoreEdgeForCausalLM(tiny_config())
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    try:
        model(ids, force_expert_id=20)
    except ValueError as exc:
        assert "outside configured expert range" in str(exc)
    else:
        raise AssertionError("out-of-range expert id must be rejected")


def test_top2_router_weights_are_normalized():
    from modeling_neurocore_edge import BGGateTop2
    router = BGGateTop2(32, 20, 2)
    ids, weights, aux = router(torch.randn(2, 5, 32))
    assert ids.shape == (2, 5, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.isfinite(aux)


def test_gqa_attention_shapes_and_sliding_global_modes():
    from modeling_neurocore_edge import GQARotaryAttention
    attention = GQARotaryAttention(32, 4, 2, sliding_window=3)
    x = torch.randn(2, 6, 32)
    local = attention(x, global_attention=False)
    global_result = attention(x, global_attention=True)
    assert local.shape == x.shape
    assert global_result.shape == x.shape
    assert torch.isfinite(local).all() and torch.isfinite(global_result).all()
    assert not torch.allclose(local, global_result)


def test_config_rejects_mismatched_effective_depth():
    try:
        NeuroCoreEdgeConfig(
            vocab_size=128, hidden_size=32, num_hidden_layers=24,
            attention_heads=4, kv_heads=2,
            prelude_layers=1, core_unique_layers=1, core_repeats=3,
            coda_layers=1,
        )
    except ValueError as exc:
        assert "effective sandwich depth" in str(exc)
    else:
        raise AssertionError("invalid sandwich depth must be rejected")
