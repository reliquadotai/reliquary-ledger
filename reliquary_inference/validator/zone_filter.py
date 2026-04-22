"""DAPO/GRPO-style zone filter for rollout groups.

Adapted from romain13190/reliquary (MIT). A "rollout group" is the set of
completions a single miner produced on a single task (= ``(miner_id, task_id)``
key). The group-relative GRPO advantage is ``(r - mean) / std`` — so a group
whose rewards cluster too tight (small σ) carries no gradient signal and
should be dropped before reaching the trainer.

For binary-reward benchmarks like Hendrycks MATH the threshold ``σ ≥ 0.43``
maps to Bernoulli(k ∈ [2, 6] correct out of 8), which is the canonical
DAPO / GRPO-OOA zone. A bootstrap threshold of ``σ ≥ 0.33`` (k ∈ [1, 7])
loosens the filter when the fleet is new and the batch would otherwise
not fill.

We deliberately keep this a pure helper — group-level decisions are
computed from the verdict list after per-completion validators have
already run, so the per-completion pipeline contract stays unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping


# Steady-state minimum σ. For binary rewards, σ of Bernoulli(k/8) where
# k ∈ [2, 6] is ≈ 0.433. Below that, rollouts cluster too tight for
# meaningful GRPO gradient.
SIGMA_MIN = 0.43

# Bootstrap threshold — matches the old k ∈ [1, 7] gate.
BOOTSTRAP_SIGMA_MIN = 0.33

# Default rollout-group size — also the canonical M in DAPO/GRPO papers.
DEFAULT_M_ROLLOUTS = 8


def rewards_std(rewards: Iterable[float]) -> float:
    """Population standard deviation of a rollout group's rewards.

    Population formula (divide by n) — we want the std of *this* sample,
    not an estimator of the underlying distribution. Degenerate groups
    (n < 2) return 0.0 so the zone filter drops them.
    """
    values = [float(r) for r in rewards]
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((r - mean) ** 2 for r in values) / n
    return variance ** 0.5


def is_in_zone(sigma: float, *, bootstrap: bool = False) -> bool:
    """True iff *sigma* passes the current-mode threshold.

    Steady state: σ ≥ SIGMA_MIN (0.43).
    Bootstrap:    σ ≥ BOOTSTRAP_SIGMA_MIN (0.33).

    A σ that's numerically ~0 (all-equal rewards) is always rejected —
    even in bootstrap — because the resulting GRPO advantage is exactly
    zero (no signal).
    """
    if sigma < 1e-8:
        return False
    return sigma >= (BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN)


@dataclass(frozen=True)
class GroupKey:
    """Identity of a rollout group.

    Two completions belong to the same group iff they share both the
    miner that produced them and the task they answer.
    """

    miner_id: str
    task_id: str

    def as_str(self) -> str:
        return f"{self.miner_id}:{self.task_id}"


@dataclass(frozen=True)
class GroupVerdict:
    """Zone-filter outcome for one rollout group."""

    miner_id: str
    task_id: str
    n: int
    mean_reward: float
    sigma: float
    in_zone: bool

    def as_dict(self) -> dict:
        return {
            "miner_id": self.miner_id,
            "task_id": self.task_id,
            "n": self.n,
            "mean_reward": self.mean_reward,
            "sigma": self.sigma,
            "in_zone": self.in_zone,
        }


# Hard-fail reasons that are "environment said this rollout is wrong" — the
# rollout is still protocol-valid; its reward is just 0. These count toward
# the group's σ because GRPO needs both correct AND wrong rollouts within a
# group to compute (r − μ) / σ.
#
# All other hard-fails (signature, proof, tokens, prompt-binding, schema)
# are *protocol violations* — an adversarial or buggy miner. Those get
# dropped because they'd corrupt the group with untrusted rewards.
ENVIRONMENT_HARD_FAIL_REASONS = frozenset(
    {
        "environment_failed_evaluation",  # wrong answer / format fail — still a valid 0-reward rollout
        "reward_contract_violation",      # miner's declared reward disagreed with ours — still 0
        "reward_missing",                 # no reward reported — treat as 0
    }
)


def filter_groups(
    verdicts: Iterable[Mapping],
    *,
    bootstrap: bool = False,
    reward_key: str = "correctness",
    only_accepted: bool = False,
) -> dict[GroupKey, GroupVerdict]:
    """Group verdicts by (miner_id, task_id) and compute zone decisions.

    Parameters
    ----------
    verdicts:
        Iterable of verdict payloads (dicts with ``payload.producer_id`` or
        ``payload.miner_id``, ``payload.task_id``, ``payload.correctness``,
        ``payload.accepted``). Shape matches what ``service.validate_window``
        produces.
    bootstrap:
        If True, use ``BOOTSTRAP_SIGMA_MIN`` instead of ``SIGMA_MIN``.
    reward_key:
        Which float field in the verdict payload to use as the reward signal.
        For MATH we use ``correctness`` (0.0 or 1.0).
    only_accepted:
        Default False — for GRPO we WANT wrong-answer rollouts to count
        toward the group's σ (that's where the within-group contrast comes
        from). Protocol-invalid rollouts (bad signature, bad proof, bad
        tokens, prompt-binding mismatch, schema) are always dropped;
        environment-level hard-fails (``environment_failed_evaluation``,
        ``reward_contract_violation``, ``reward_missing``) are kept
        because those rollouts are protocol-valid and just carry reward=0.

        Setting True reverts to the stricter "pipeline-accepted only"
        behaviour — useful for weight scoring, not for GRPO group σ.
    """
    buckets: dict[GroupKey, list[float]] = defaultdict(list)
    for verdict in verdicts:
        payload = verdict.get("payload", verdict)
        accepted = bool(payload.get("accepted", False))
        hard_fail = payload.get("hard_fail_reason")
        # Drop protocol-invalid rollouts. Keep environment-level fails.
        if only_accepted and not accepted:
            continue
        if not accepted and hard_fail is not None and hard_fail not in ENVIRONMENT_HARD_FAIL_REASONS:
            continue
        miner_id = str(
            payload.get("miner_id")
            or payload.get("producer_id")
            or verdict.get("producer_id")
            or ""
        )
        task_id = str(payload.get("task_id") or "")
        if not miner_id or not task_id:
            continue
        try:
            reward = float(payload.get(reward_key, 0.0))
        except (TypeError, ValueError):
            reward = 0.0
        buckets[GroupKey(miner_id=miner_id, task_id=task_id)].append(reward)

    out: dict[GroupKey, GroupVerdict] = {}
    for key, rewards in buckets.items():
        n = len(rewards)
        if n == 0:
            continue
        sigma = rewards_std(rewards)
        out[key] = GroupVerdict(
            miner_id=key.miner_id,
            task_id=key.task_id,
            n=n,
            mean_reward=sum(rewards) / n,
            sigma=sigma,
            in_zone=is_in_zone(sigma, bootstrap=bootstrap),
        )
    return out


def zone_summary(
    groups: Mapping[GroupKey, GroupVerdict],
    *,
    bootstrap: bool = False,
) -> dict:
    """Compact summary suitable for scorecard / rollout_bundle embedding."""
    in_zone = sum(1 for g in groups.values() if g.in_zone)
    total = len(groups)
    return {
        "sigma_min": BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN,
        "bootstrap": bootstrap,
        "total_groups": total,
        "in_zone_groups": in_zone,
        "out_of_zone_groups": total - in_zone,
        "groups": [g.as_dict() | {"key": k.as_str()} for k, g in groups.items()],
    }


__all__ = [
    "BOOTSTRAP_SIGMA_MIN",
    "DEFAULT_M_ROLLOUTS",
    "GroupKey",
    "GroupVerdict",
    "SIGMA_MIN",
    "filter_groups",
    "is_in_zone",
    "rewards_std",
    "zone_summary",
]
