"""Stage 8: logprob replay (deferred to Tier 2 Epic 4).

Validator re-runs the forward pass on the completion and compares miner-
claimed per-token logprobs against recomputed ones using median importance
sampling. A drift of more than ~15% from unity indicates tampering with
either the logits cache or the token selection.

This stage requires:
  - Miners to commit to logit caches (artifact schema extension pending).
  - Validator-side forward pass producing per-token logprobs (already in
    ``shared/modeling.compute_completion_logprobs``).

Until miners commit to logit caches the stage returns PASS with a
``logprob_unavailable`` metadata flag so the pipeline does not short-circuit.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept


class LogprobStage:
    name: str = "logprob"

    def check(self, context: StageContext) -> StageResult:
        claimed_logprobs = context.payload.get("logprobs")
        if claimed_logprobs is None:
            return accept(self.name, {"status": "logprob_unavailable"})

        return accept(
            self.name,
            {"status": "logprob_replay_not_yet_implemented", "claim_length": len(claimed_logprobs)},
        )
