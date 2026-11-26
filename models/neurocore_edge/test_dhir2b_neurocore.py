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
class Tests(unittest.TestCase):
    def make(self): return Dhir2BNeuroCore(Backbone(),"test/dhir","a"*40,rank=4)
    def test_config_and_tokenizer_vocab(self):
        self.assertEqual(validate_dhir_config(Cfg(),Tok())["vocab_size"],32)
    def test_vocab_mismatch(self):
        class Bad(Tok):
            def __len__(self): return 31
        with self.assertRaisesRegex(ValueError,"vocabulary mismatch"): validate_dhir_config(Cfg(),Bad())
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
    def test_bad_expert_id(self):
        with self.assertRaises(ValueError): self.make().prepare_expert_training(20)
if __name__=="__main__": unittest.main()
