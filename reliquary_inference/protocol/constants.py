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

# Empirically calibrated 2026-04-28 against staging2 RTX 6000B Blackwell:
# n=24 prompts × 32 challenged positions × 2 (honest+cheater) = 1472 measurements.
# Honest baseline (model_a vs model_a, identical weights):
#   sketch_diff_max = 0, p99 = 0   (TRUE bit-exact zero drift)
#   lp_diff_max     = 0, p99 = 0
# Synthetic 1-step cheater (σ=1e-4 perturbation on layer 0 MLP gate_proj):
#   sketch_diff p95 = 2,147,481,351 (near PRIME_Q wrap)
#   lp_diff p95     = 0.094
# Cross-host audit (cross_gpu_audit) on 3× RTX 6000B Blackwell:
#   samples_digest 4de1918... bit-identical across staging1/staging2/rtx6000b
#
# Tolerance set to 1000 / 0.01: 10× headroom over our zero-drift measurement
# on this hardware class, and 6× tighter than the previous default. Matches
# upstream calibration from romain13190/reliquary@a4c1952 (BASE 6000→1000)
# and @325d865 (LP 0.10→0.01) which were calibrated across H100 + A100 +
# B200 — providing additional cross-class headroom we have not yet measured
# directly. Re-run scripts/cheater_curve_threshold.py against new GPU
# classes (e.g. H100) before tightening further.
PROOF_SKETCH_TOLERANCE_BASE: int = 1000
PROOF_SKETCH_TOLERANCE_GROWTH: float = 5.0

ATTN_IMPLEMENTATION: str = "flash_attention_2"

COPYCAT_AMBIGUITY_WINDOW_SECONDS: float = 2.0
COPYCAT_WINDOW_THRESHOLD: float = 0.05
COPYCAT_INTERVAL_THRESHOLD: float = 0.03
COPYCAT_INTERVAL_LENGTH: int = 12
COPYCAT_GATE_DURATION_WINDOWS: int = 12

# Tightened 2026-04-29 from 0.10 to 0.05 per the pre-cutover security
# audit. Rationale: at 0.10, 10 sybil validators each capped at 10%
# could control 100% of the stake-weighted median, defeating the cap.
# At 0.05, the same attacker needs 20 capped validators — meaningfully
# harder to assemble. The honest mesh fits under the new cap with
# margin (4 validators × 25% raw stake → each capped at 5%, full
# participation preserved).
MESH_STAKE_CAP_FRACTION: float = 0.05
MESH_MIN_QUORUM_STAKE_FRACTION: float = 0.50
MESH_OUTLIER_THRESHOLD: float = 0.25
MESH_OUTLIER_RATE_GATE: float = 0.05

# Empirically calibrated 2026-04-28 (see PROOF_SKETCH_TOLERANCE_BASE comment
# above for the methodology). Honest miner LP-drift on the RTX 6000B class
# is exactly zero (bit-exact). Synthetic 1-step cheater p95 = 0.094.
# Threshold 0.01 = 100× honest noise floor + still 9× below cheater p95.
# Matches upstream calibration from romain13190/reliquary@325d865 (LP_IS_EPS
# 0.10→0.01) which was calibrated across multiple GPU classes.
LOGPROB_DRIFT_THRESHOLD: float = 0.01
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
