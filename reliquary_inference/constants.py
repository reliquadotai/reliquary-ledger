"""Runtime and operational constants for reliquary-inference.

Proof-layer constants (sketch, challenge, attention, version) are defined
canonically in ``reliquary_inference.protocol.constants`` and re-exported here
for backward compatibility with callers that import from the package root.
"""

from __future__ import annotations

from .protocol.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    COPYCAT_AMBIGUITY_WINDOW_SECONDS,
    COPYCAT_GATE_DURATION_WINDOWS,
    COPYCAT_INTERVAL_LENGTH,
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_WINDOW_THRESHOLD,
    LAYER_INDEX,
    LEDGER_PROOF_VERSION,
    PRIME_Q,
    PROOF_COEFF_RANGE,
    PROOF_NUM_BUCKETS,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
    PROOF_TOPK,
)

PROOF_VERSION = LEDGER_PROOF_VERSION

RNG_LABEL = {"sketch": b"sketch", "open": b"open", "task": b"task"}

BLOCK_TIME_SECONDS = 12
WINDOW_LENGTH = 30
WEIGHT_SUBMISSION_INTERVAL = 360

SUPERLINEAR_EXPONENT = 4.0
UNIQUE_ROLLOUTS_CAP = 5000
UNIQUE_ROLLOUTS_CAP_ENABLED = True

MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35
ROLLOUT_SAMPLE_RATE = 0.10
ROLLOUT_SAMPLE_MIN = 16
VERIFICATION_BATCH_SIZE = 16
BATCH_FAILURE_THRESHOLD = 0.30

DEFAULT_DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_FALLBACK_PROMPTS = (
    "Summarize the theme of disciplined problem solving in one compact paragraph.",
    "Explain why deterministic verification matters in decentralized AI systems.",
    "Write a concise argument for using content-addressed artifacts in subnets.",
    "Describe the tradeoff between throughput and verifiability in inference networks.",
)

__all__ = [
    "ATTN_IMPLEMENTATION",
    "BATCH_FAILURE_THRESHOLD",
    "BLOCK_TIME_SECONDS",
    "CHALLENGE_K",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_SPLIT",
    "DEFAULT_FALLBACK_PROMPTS",
    "LAYER_INDEX",
    "LEDGER_PROOF_VERSION",
    "MINER_SAMPLE_MAX",
    "MINER_SAMPLE_MIN",
    "MINER_SAMPLE_RATE",
    "PRIME_Q",
    "PROOF_COEFF_RANGE",
    "PROOF_NUM_BUCKETS",
    "PROOF_SKETCH_TOLERANCE_BASE",
    "PROOF_SKETCH_TOLERANCE_GROWTH",
    "PROOF_TOPK",
    "PROOF_VERSION",
    "RNG_LABEL",
    "ROLLOUT_SAMPLE_MIN",
    "ROLLOUT_SAMPLE_RATE",
    "SUPERLINEAR_EXPONENT",
    "UNIQUE_ROLLOUTS_CAP",
    "UNIQUE_ROLLOUTS_CAP_ENABLED",
    "VERIFICATION_BATCH_SIZE",
    "WEIGHT_SUBMISSION_INTERVAL",
    "WINDOW_LENGTH",
    "WEIGHT_SUBMISSION_INTERVAL",
]
