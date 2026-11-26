"""Vendored Dhīr-2B model implementation.

Source: zenteiq-ai/Dhir-2B
Pinned revision: 7cbe444cd1e45e25197cfa0f683b2cb575858169
Source SHA-256: configuration_dhir.py 9ddceca88e4d7780ccab5d194d93bb1bf557238e4f71b5b7abe4d2b540ba5cee
Source SHA-256: modeling_dhir.py 9836ca6d3996428bbcd7dc768e885268707dbb7d3dc04ca21df494db893bfc45

The upstream source files are kept unmodified, including their Apache-2.0 notices.
They are imported locally; no remote Python code execution is required.
"""
from .configuration_dhir import DhirConfig
from .modeling_dhir import DhirForCausalLM, DhirModel, DhirPreTrainedModel

__all__ = ["DhirConfig", "DhirForCausalLM", "DhirModel", "DhirPreTrainedModel"]
