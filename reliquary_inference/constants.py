"""Consensus and runtime constants for reliquary-inference."""

PROOF_VERSION = "v5"

PRIME_Q = 2_147_483_647
CHALLENGE_K = 32
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "task": b"task"}
LAYER_INDEX = -1
PROOF_TOPK = 16
PROOF_NUM_BUCKETS = 8
PROOF_COEFF_RANGE = 127
PROOF_SKETCH_TOLERANCE_BASE = 6000
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0
ATTN_IMPLEMENTATION = "flash_attention_2"

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
