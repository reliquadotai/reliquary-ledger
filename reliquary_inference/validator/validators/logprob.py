"""Stage 8: logprob replay.

Walks each generated-token position, re-runs the sampler pipeline on the
miner's committed logit cache, and compares the replayed log-probability
against the miner's claimed one. Hard-fails when the fraction of positions
within tolerance drops below the quorum threshold.

Cached logits are pulled from the miner's ``logits_commitment`` field. If
the miner has not committed logits yet (Tier 1 scaffolding regime) the
stage returns PASS with an availability flag so the pipeline stays green.

Spec: private/reliquary-plan/notes/spec-distribution-validator.md.
"""

from __future__ import annotations

import math
from typing import Any

from ...protocol.constants import LOGPROB_DRIFT_QUORUM, LOGPROB_DRIFT_THRESHOLD
from ..sampler_replay import SamplingParams, replay_logprob
from .base import RejectReason, StageContext, StageResult, accept, reject


class LogprobStage:
    name: str = "logprob"

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
        except (KeyError, ValueError) as exc:
            return reject(
                self.name,
                RejectReason.LOGPROB_MISSING,
                {"reason": "invalid_sampling_params", "detail": str(exc)[:160]},
            )

        tokens = payload.get("tokens", [])
        prompt_length = int(payload.get("prompt_length", len(tokens) - len(claimed_logprobs)))
        generated = tokens[prompt_length:]
        if len(generated) != len(claimed_logprobs):
            return reject(
                self.name,
                RejectReason.LOGPROB_DRIFT_EXCEEDED,
                {"reason": "logprob_length_mismatch",
                 "generated": len(generated),
                 "claimed_logprobs": len(claimed_logprobs)},
            )

        n = min(len(cached_logits), len(claimed_logprobs))
        if n == 0:
            return accept(self.name, {"status": "no_positions"})

        deltas: list[float] = []
        for i in range(n):
            prior = tokens[: prompt_length + i]
            replay_lp = replay_logprob(cached_logits[i], params, int(generated[i]), prior)
            deltas.append(abs(replay_lp - float(claimed_logprobs[i])))

        passing = sum(1 for d in deltas if d < LOGPROB_DRIFT_THRESHOLD)
        fraction = passing / len(deltas)
        metadata: dict[str, Any] = {
            "positions_checked": len(deltas),
            "positions_passed": passing,
            "fraction_passing": fraction,
            "max_abs_delta": max(deltas),
            "median_abs_delta": sorted(deltas)[len(deltas) // 2],
            "drift_threshold": LOGPROB_DRIFT_THRESHOLD,
            "quorum": LOGPROB_DRIFT_QUORUM,
        }

        if fraction < LOGPROB_DRIFT_QUORUM:
            return reject(self.name, RejectReason.LOGPROB_DRIFT_EXCEEDED, metadata)
        return accept(self.name, metadata)
