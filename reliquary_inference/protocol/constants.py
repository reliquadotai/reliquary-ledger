"""Authoritative protocol-level constants for Reliquary Ledger.

These values are mathematical parameters of the proof protocol, not runtime
knobs. No environment overrides are permitted here; any attempt to mutate
these at import time is an error.

Scope: proof sketch, challenge selection, attention kernel, version tag.
Runtime/operational constants (window cadence, scoring, dataset defaults)
live in ``reliquary_inference.constants``.
"""

from __future__ import annotations

LEDGER_PROOF_VERSION: str = "v5"

PRIME_Q: int = 2_147_483_647

CHALLENGE_K: int = 32

LAYER_INDEX: int = -1

PROOF_TOPK: int = 16
PROOF_NUM_BUCKETS: int = 8
PROOF_COEFF_RANGE: int = 127

PROOF_SKETCH_TOLERANCE_BASE: int = 6000
PROOF_SKETCH_TOLERANCE_GROWTH: float = 5.0

ATTN_IMPLEMENTATION: str = "flash_attention_2"

COPYCAT_AMBIGUITY_WINDOW_SECONDS: float = 2.0
COPYCAT_WINDOW_THRESHOLD: float = 0.05
COPYCAT_INTERVAL_THRESHOLD: float = 0.03
COPYCAT_INTERVAL_LENGTH: int = 12
COPYCAT_GATE_DURATION_WINDOWS: int = 12

MESH_STAKE_CAP_FRACTION: float = 0.10
MESH_MIN_QUORUM_STAKE_FRACTION: float = 0.50
MESH_OUTLIER_THRESHOLD: float = 0.25
MESH_OUTLIER_RATE_GATE: float = 0.05

LOGPROB_DRIFT_THRESHOLD: float = 0.15
LOGPROB_DRIFT_QUORUM: float = 0.51
DISTRIBUTION_RATIO_BAND_HIGH: float = 1.15
DISTRIBUTION_RATIO_BAND_LOW: float = 0.85
DISTRIBUTION_MIN_POSITIONS: int = 8

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
    "COPYCAT_AMBIGUITY_WINDOW_SECONDS",
    "COPYCAT_WINDOW_THRESHOLD",
    "COPYCAT_INTERVAL_THRESHOLD",
    "COPYCAT_INTERVAL_LENGTH",
    "COPYCAT_GATE_DURATION_WINDOWS",
    "MESH_STAKE_CAP_FRACTION",
    "MESH_MIN_QUORUM_STAKE_FRACTION",
    "MESH_OUTLIER_THRESHOLD",
    "MESH_OUTLIER_RATE_GATE",
    "LOGPROB_DRIFT_THRESHOLD",
    "LOGPROB_DRIFT_QUORUM",
    "DISTRIBUTION_RATIO_BAND_HIGH",
    "DISTRIBUTION_RATIO_BAND_LOW",
    "DISTRIBUTION_MIN_POSITIONS",
)
