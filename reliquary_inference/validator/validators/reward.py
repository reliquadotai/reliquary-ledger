"""Stage 7: reward-contract check (placeholder).

Current completion artifacts do not yet carry an explicit reward claim — the
environment stage (6) computes reward inline. Stage 7 will become meaningful
once miners start claiming reward values in their payload so we can cross-
check against the environment's re-computation.

Until then the stage returns PASS with a diagnostic indicating no reward
was claimed.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept, reject


class RewardStage:
    name: str = "reward"

    def check(self, context: StageContext) -> StageResult:
        claimed = context.payload.get("claimed_reward")
        if claimed is None:
            return accept(self.name, {"status": "no_reward_claim"})

        environment_reward = context.extras.get("environment_reward")
        if environment_reward is None:
            return accept(
                self.name,
                {"status": "env_reward_unavailable", "claimed": claimed},
            )

        try:
            claimed_f = float(claimed)
            env_f = float(environment_reward)
        except (TypeError, ValueError):
            return reject(
                self.name,
                RejectReason.REWARD_CONTRACT_VIOLATION,
                {"reason": "unparseable_values", "claimed": claimed, "env": environment_reward},
            )

        tolerance = float(context.extras.get("reward_tolerance", 1e-6))
        if abs(claimed_f - env_f) > tolerance:
            return reject(
                self.name,
                RejectReason.REWARD_CONTRACT_VIOLATION,
                {"claimed": claimed_f, "env": env_f, "tolerance": tolerance},
            )

        return accept(
            self.name,
            {"claimed": claimed_f, "env": env_f, "delta": abs(claimed_f - env_f)},
        )
