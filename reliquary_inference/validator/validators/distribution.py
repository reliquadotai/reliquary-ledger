"""Stage 9: distribution validator (soft).

Re-runs the sampler pipeline on cached logits and compares the replayed
probability for each claimed token against the miner's claimed probability.
The stage is SOFT: band crossings accumulate in the pipeline's
``soft_flags`` list but never short-circuit execution.

Spec: private/reliquary-plan/notes/spec-distribution-validator.md.
"""

from __future__ import annotations

import math

from ...protocol.constants import (
    DISTRIBUTION_MIN_POSITIONS,
    DISTRIBUTION_RATIO_BAND_HIGH,
    DISTRIBUTION_RATIO_BAND_LOW,
)
from ..sampler_replay import SamplingParams, median_importance_ratio, replay_probability
from .base import RejectReason, StageContext, StageResult, accept, soft_flag


class DistributionStage:
    name: str = "distribution"

    def check(self, context: StageContext) -> StageResult:
        payload = context.payload
        claimed_logprobs = payload.get("logprobs")
        if claimed_logprobs is None:
            return accept(self.name, {"status": "logprob_unavailable"})

        commitment = payload.get("logits_commitment")
        if not isinstance(commitment, dict):
            return accept(self.name, {"status": "logit_cache_unavailable"})

        cached_logits = context.extras.get("cached_logits")
        if cached_logits is None:
            return accept(self.name, {"status": "logit_cache_not_loaded"})

        sampling_params_raw = payload.get("sampling_params")
        if not isinstance(sampling_params_raw, dict):
            return accept(self.name, {"status": "sampling_params_unavailable"})

        try:
            params = SamplingParams(
                temperature=float(sampling_params_raw["temperature"]),
                top_p=float(sampling_params_raw["top_p"]),
                repetition_penalty=float(sampling_params_raw.get("repetition_penalty", 1.0)),
            )
        except (KeyError, ValueError):
            return accept(self.name, {"status": "invalid_sampling_params"})

        tokens = payload.get("tokens", [])
        prompt_length = int(payload.get("prompt_length", len(tokens) - len(claimed_logprobs)))
        generated = tokens[prompt_length:]

        n = min(len(cached_logits), len(claimed_logprobs), len(generated))
        if n < DISTRIBUTION_MIN_POSITIONS:
            return accept(
                self.name,
                {"status": "insufficient_positions", "positions_available": n,
                 "min_positions": DISTRIBUTION_MIN_POSITIONS},
            )

        replay_probs: list[float] = []
        miner_probs: list[float] = []
        for i in range(n):
            prior = tokens[: prompt_length + i]
            replay_probs.append(
                replay_probability(cached_logits[i], params, int(generated[i]), prior)
            )
            miner_probs.append(math.exp(float(claimed_logprobs[i])))

        ratio = median_importance_ratio(replay_probs, miner_probs)
        metadata = {
            "median_ratio": ratio,
            "band_low": DISTRIBUTION_RATIO_BAND_LOW,
            "band_high": DISTRIBUTION_RATIO_BAND_HIGH,
            "positions_checked": n,
        }
        if DISTRIBUTION_RATIO_BAND_LOW <= ratio <= DISTRIBUTION_RATIO_BAND_HIGH:
            return accept(self.name, metadata)
        return soft_flag(
            self.name,
            RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND,
            metadata,
        )
