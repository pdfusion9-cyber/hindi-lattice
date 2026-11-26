# Copyright 2026 ZenteiQ AiTech Innovations Pvt. Ltd. All rights reserved.
#
# Portions adapted from Hugging Face Transformers. Licensed under Apache License 2.0.
# Full license: https://www.apache.org/licenses/LICENSE-2.0
"""BrahmAI Dhīr model configuration; vendored from pinned upstream source."""
from transformers import PretrainedConfig

class DhirConfig(PretrainedConfig):
    model_type = "dhir"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(self, vocab_size=131072, hidden_size=2048, intermediate_size=8192,
                 num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                 head_dim=128, hidden_act="silu", max_position_embeddings=16384,
                 initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True,
                 tie_word_embeddings=True, rope_theta=1000000.0, attention_bias=False,
                 attention_dropout=0.0, pad_token_id=0, bos_token_id=2,
                 eos_token_id=(1, 12), **kwargs):
        self.vocab_size=vocab_size; self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size; self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads; self.num_key_value_heads=num_key_value_heads
        self.head_dim=head_dim; self.hidden_act=hidden_act
        self.max_position_embeddings=max_position_embeddings
        self.initializer_range=initializer_range; self.rms_norm_eps=rms_norm_eps
        self.use_cache=use_cache; self.rope_theta=rope_theta
        self.attention_bias=attention_bias; self.attention_dropout=attention_dropout
        super().__init__(pad_token_id=pad_token_id,bos_token_id=bos_token_id,
            eos_token_id=list(eos_token_id) if isinstance(eos_token_id,tuple) else eos_token_id,
            tie_word_embeddings=tie_word_embeddings,**kwargs)

__all__=["DhirConfig"]
