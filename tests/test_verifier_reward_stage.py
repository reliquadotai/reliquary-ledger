"""Tests for stage 7 (reward) — currently passive until miners claim rewards."""

from __future__ import annotations

from reliquary_inference.validator.validators.base import RejectReason, StageContext
from reliquary_inference.validator.validators.reward import RewardStage


def _context(payload_claim=None, env_reward=None, tolerance: float | None = None) -> StageContext:
    extras: dict = {}
    if env_reward is not None:
        extras["environment_reward"] = env_reward
    if tolerance is not None:
        extras["reward_tolerance"] = tolerance
    payload = {}
    if payload_claim is not None:
        payload["claimed_reward"] = payload_claim
    return StageContext(
        completion={"payload": payload},
        task_batch={},
        seen_nonces=set(),
        extras=extras,
    )


def test_no_claimed_reward_passes_silently() -> None:
    stage = RewardStage()
    result = stage.check(_context(payload_claim=None))
    assert result.passed is True
    assert result.metadata["status"] == "no_reward_claim"


def test_claimed_reward_without_env_reward_passes() -> None:
    stage = RewardStage()
    result = stage.check(_context(payload_claim=0.5, env_reward=None))
    assert result.passed is True
    assert result.metadata["status"] == "env_reward_unavailable"


def test_matching_claimed_and_env_reward_passes() -> None:
    stage = RewardStage()
    result = stage.check(_context(payload_claim=0.75, env_reward=0.75))
    assert result.passed is True
    assert result.metadata["delta"] == 0.0


def test_divergent_claimed_and_env_reward_rejects() -> None:
    stage = RewardStage()
    result = stage.check(_context(payload_claim=0.9, env_reward=0.1, tolerance=1e-3))
    assert result.passed is False
    assert result.reason is RejectReason.REWARD_CONTRACT_VIOLATION


def test_unparseable_values_rejects() -> None:
    stage = RewardStage()
    result = stage.check(_context(payload_claim="high", env_reward=0.5))
    assert result.passed is False
    assert result.reason is RejectReason.REWARD_CONTRACT_VIOLATION
