"""Weight computation for the inference runtime."""

import logging

from ..constants import (
    SUPERLINEAR_EXPONENT,
    UNIQUE_ROLLOUTS_CAP,
    UNIQUE_ROLLOUTS_CAP_ENABLED,
)

logger = logging.getLogger(__name__)


def compute_weights(
    miner_scores: dict[str, dict[str, int]],
    superlinear_exponent: float = SUPERLINEAR_EXPONENT,
    unique_cap: int = UNIQUE_ROLLOUTS_CAP,
    cap_enabled: bool = UNIQUE_ROLLOUTS_CAP_ENABLED,
) -> dict[str, float]:
    """Compute normalized weights from miner scores.

    Scoring formula (V1 — no environment, no burn):
        raw_score = min(unique_rollouts, cap) ^ superlinear_exponent
        weight_i = raw_score_i / sum(raw_score_j for all j)

    Args:
        miner_scores: {hotkey: {"unique": int, "valid": int}}
        superlinear_exponent: Sybil resistance exponent (default 4.0)
        unique_cap: Max unique rollouts that count (default 5000)
        cap_enabled: Whether to enforce the cap

    Returns:
        {hotkey: normalized_weight} summing to 1.0
    """
    raw_scores: dict[str, float] = {}

    for hotkey, scores in miner_scores.items():
        unique = scores.get("unique", 0)
        valid = scores.get("valid", 0)

        if unique <= 0 or valid <= 0:
            raw_scores[hotkey] = 0.0
            continue

        capped = min(unique, unique_cap) if cap_enabled else unique
        raw_scores[hotkey] = capped ** superlinear_exponent

    total = sum(raw_scores.values())
    if total == 0:
        return {hk: 0.0 for hk in miner_scores}

    return {hk: score / total for hk, score in raw_scores.items()}
