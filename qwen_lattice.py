#!/usr/bin/env python3
"""Full Qwen-tokenizer vocabulary connection model (about 23B possible directed pairs)."""
from conn_model_v2 import main

if __name__ == "__main__":
    main(
        default_mode="tokenizer",
        default_vocab_limit=0,
        default_ckpt="checkpoints/qwen_lattice.pt",
    )
