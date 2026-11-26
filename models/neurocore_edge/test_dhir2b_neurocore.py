import types,unittest
import torch
from torch import nn
from dhir2b_neurocore import Dhir2BNeuroCore,validate_dhir_config

class Tok:
    def __len__(self): return 32
    def get_vocab(self): return {str(i):i for i in range(32)}
class Cfg:
    vocab_size=32; hidden_size=16; intermediate_size=32; num_hidden_layers=28
    num_attention_heads=4; num_key_value_heads=2; max_position_embeddings=16384
    rope_theta=1_000_000.; tie_word_embeddings=True; hidden_act="silu"
class Layer(nn.Module):
    def __init__(self): super().__init__(); self.proj=nn.Linear(16,16,bias=False)
    def forward(self,x): return (x+self.proj(x),)
class Body(nn.Module):
    def __init__(self): super().__init__(); self.layers=nn.ModuleList([Layer() for _ in range(28)])
class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.config=Cfg(); self.model=Body()
        self.embed=nn.Embedding(32,16); self.lm_head=nn.Linear(16,32,bias=False)
        self.lm_head.weight=self.embed.weight
    def forward(self,input_ids):
        x=self.embed(input_ids)
        for layer in self.model.layers: x=layer(x)[0]
        return types.SimpleNamespace(logits=self.lm_head(x))
    def generate(self,input_ids,max_new_tokens=1,**kwargs):
        result=self(input_ids).logits[:,-1,:].argmax(-1,keepdim=True)
        return torch.cat((input_ids,result),dim=1)
class Tests(unittest.TestCase):
    def make(self):
        facts={"hidden_size":16,"vocab_size":32,"num_hidden_layers":28}
        return Dhir2BNeuroCore(Backbone(),"test/dhir","a"*40,rank=4,source_facts=facts)
    def test_config_and_tokenizer_vocab(self):
        class RealCfg:
            vocab_size=32; hidden_size=2048; intermediate_size=4096
            num_hidden_layers=28; num_attention_heads=16; num_key_value_heads=8
            max_position_embeddings=16384; rope_theta=1_000_000.
            tie_word_embeddings=True; hidden_act="silu"
        self.assertEqual(validate_dhir_config(RealCfg(),Tok())["vocab_size"],32)
    def test_vocab_mismatch(self):
        class RealCfg:
            vocab_size=32; hidden_size=2048; intermediate_size=4096
            num_hidden_layers=28; num_attention_heads=16; num_key_value_heads=8
            max_position_embeddings=16384; rope_theta=1_000_000.
            tie_word_embeddings=True; hidden_act="silu"
        class Bad(Tok):
            def __len__(self): return 31
        with self.assertRaisesRegex(ValueError,"vocabulary mismatch"): validate_dhir_config(RealCfg(),Bad())
    def test_28_layers_and_zero_init_preserves_logits(self):
        torch.manual_seed(1); model=self.make(); ids=torch.randint(0,32,(1,4))
        a=model.backbone(input_ids=ids).logits
        b=model(input_ids=ids,expert_id=3).logits
        self.assertEqual(len(model.adapter_layers),28); self.assertTrue(torch.allclose(a,b))
    def test_only_fixed_expert_is_trainable(self):
        model=self.make(); params=model.prepare_expert_training(5)
        self.assertEqual(len(params),28*2)
        self.assertTrue(all(p.requires_grad for p in params))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for l in model.adapter_layers for i,e in enumerate(l.experts) if i!=5 for p in e.parameters()))
        ids=torch.randint(0,32,(1,4)); loss=model(input_ids=ids,expert_id=5).logits.float().square().mean()
        loss.backward(); self.assertTrue(any(p.grad is not None for p in params))
    def test_bfloat16_hidden_states_keep_fp32_adapter_weights(self):
        model=self.make()
        hidden=torch.randn(2,3,16,dtype=torch.bfloat16)
        result=model.adapter_layers[0](hidden,expert_id=0)
        self.assertEqual(result.dtype,torch.bfloat16)
        self.assertEqual(result.shape,hidden.shape)
        self.assertTrue(torch.isfinite(result).all())
    def test_expert_state_roundtrip_and_mismatch_guard(self):
        model=self.make()
        state={f"layer_{i}.{k}":v.detach().clone()+0.01
          for i,layer in enumerate(model.adapter_layers)
          for k,v in layer.experts[2].state_dict().items()}
        model.load_expert_state_dict(2,state)
        loaded=model.adapter_layers[2].experts[2].state_dict()
        self.assertTrue(torch.equal(loaded["down.weight"],state["layer_2.down.weight"]))
        with self.assertRaisesRegex(ValueError,"key set mismatch"):
            model.load_expert_state_dict(2,{})
    def test_tokenizer_id_gap_is_rejected(self):
        class Gapped(Tok):
            def get_vocab(self): return {str(i):i+1 for i in range(32)}
        class RealCfg:
            vocab_size=32; hidden_size=2048; intermediate_size=4096
            num_hidden_layers=28; num_attention_heads=16; num_key_value_heads=8
            max_position_embeddings=16384; rope_theta=1_000_000.
            tie_word_embeddings=True; hidden_act="silu"
        with self.assertRaisesRegex(ValueError,"contiguous"):
            validate_dhir_config(RealCfg(),Gapped())
    def test_generate_proxy_forces_expert_and_restores_state(self):
        model=self.make()
        ids=torch.randint(0,32,(1,4))
        generated=model.generate(input_ids=ids,max_new_tokens=1,expert_id=11)
        self.assertEqual(generated.shape,(1,5))
        self.assertIsNone(model._forced_expert_id)
    def test_invalid_generate_expert_is_rejected(self):
        model=self.make()
        ids=torch.randint(0,32,(1,4))
        with self.assertRaisesRegex(ValueError,"invalid expert_id"):
            model.generate(input_ids=ids,max_new_tokens=1,expert_id=20)
    def test_bad_expert_id(self):
        with self.assertRaises(ValueError): self.make().prepare_expert_training(20)
if __name__=="__main__": unittest.main()
