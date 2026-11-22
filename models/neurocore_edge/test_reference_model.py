import torch
from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM, count_parameters

def test_tiny_reference_forward_backward():
    torch.manual_seed(7)
    cfg = NeuroCoreEdgeConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
        num_experts=20, experts_per_token=2, expert_intermediate_size=16,
        attention_heads=4, kv_heads=2, sliding_window=8, recurrent_state_size=8,
        router_aux_loss_coefficient=0.01)
    model = NeuroCoreEdgeForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    result = model(ids)
    assert result["logits"].shape == (2, 8, cfg.vocab_size)
    loss = result["logits"].float().square().mean() + result["router_aux_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert count_parameters(model) > 0

def test_top2_router_weights_are_normalized():
    from modeling_neurocore_edge import BGGateTop2
    router = BGGateTop2(32, 20, 2)
    ids, weights, aux = router(torch.randn(2, 5, 32))
    assert ids.shape == (2, 5, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.isfinite(aux)
