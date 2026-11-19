#!/usr/bin/env python3
"""Small-vocabulary Hindi/English lattice connection experiment, context-aware."""
from conn_model_v2 import main

if __name__ == "__main__":
    main(
        default_mode="regex",
        default_vocab_limit=3000,
        default_ckpt="checkpoints/lattice_conn.pt",
    )
