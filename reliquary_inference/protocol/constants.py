"""Authoritative protocol-level constants for Reliquary Ledger.

These values are mathematical parameters of the proof protocol, not runtime
knobs. No environment overrides are permitted here; any attempt to mutate
these at import time is an error.

Scope: proof sketch, challenge selection, attention kernel, version tag.
Runtime/operational constants (window cadence, scoring, dataset defaults)
live in ``reliquary_inference.constants``.
"""

from __future__ import annotations

LEDGER_PROOF_VERSION: str = "v1"

PRIME_Q: int = 2_147_483_647

CHALLENGE_K: int = 32

LAYER_INDEX: int = -1

PROOF_TOPK: int = 16
PROOF_NUM_BUCKETS: int = 8
PROOF_COEFF_RANGE: int = 127

PROOF_SKETCH_TOLERANCE_BASE: int = 6000
PROOF_SKETCH_TOLERANCE_GROWTH: float = 5.0

ATTN_IMPLEMENTATION: str = "flash_attention_2"

_PROTOCOL_CONSTANTS: tuple[str, ...] = (
    "LEDGER_PROOF_VERSION",
    "PRIME_Q",
    "CHALLENGE_K",
    "LAYER_INDEX",
    "PROOF_TOPK",
    "PROOF_NUM_BUCKETS",
    "PROOF_COEFF_RANGE",
    "PROOF_SKETCH_TOLERANCE_BASE",
    "PROOF_SKETCH_TOLERANCE_GROWTH",
    "ATTN_IMPLEMENTATION",
)
