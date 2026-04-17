"""Stage 9: distribution validator (soft, deferred to Tier 2 Epic 4).

Validator replays the sampler pipeline (repetition penalty + temperature +
top-p) on the committed logits and asserts the median importance ratio
``p_replay(chosen) / p_miner(chosen)`` stays within [0.85, 1.15]. Tampering
with sampled tokens (same logits, different choice) pushes the ratio out of
band.

This is the ONLY stage that is SOFT: failures accumulate across a miner's
rolling window; exceeding ``soft_fail_threshold`` (default 51%) gates the
miner from future scoring.

Full implementation requires committed logit caches (artifact extension) and
parity-checked sampler replay. Until then this stage returns PASS with a
``sampler_replay_not_yet_implemented`` metadata flag.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept


class DistributionStage:
    name: str = "distribution"

    def check(self, context: StageContext) -> StageResult:
        if context.payload.get("logprobs") is None:
            return accept(self.name, {"status": "logprob_unavailable"})

        return accept(
            self.name,
            {"status": "sampler_replay_not_yet_implemented"},
        )
