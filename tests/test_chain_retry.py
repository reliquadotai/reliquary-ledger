"""Tests for the bounded exponential-backoff retry helper.

Spec reference: private/reliquary-plan/notes/spec-chain-adapter.md
acceptance tests 1-5.
"""

from __future__ import annotations

import random

import pytest

from reliquary_inference.chain.retry import RetryPolicy, compute_delay_seconds, retry_with_backoff


def test_policy_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_policy_rejects_invalid_jitter() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(jitter_ratio=1.5)


def test_policy_rejects_negative_delays() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_seconds=-1.0)


def test_zero_jitter_produces_geometric_schedule() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=64.0, jitter_ratio=0.0)
    delays = [compute_delay_seconds(i, policy) for i in range(4)]
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_max_delay_caps_exponential_growth() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=5.0, jitter_ratio=0.0)
    delays = [compute_delay_seconds(i, policy) for i in range(6)]
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(base_delay_seconds=1.0, multiplier=1.0, max_delay_seconds=1.0, jitter_ratio=0.25)
    rng = random.Random(42)
    for _ in range(50):
        delay = compute_delay_seconds(0, policy, rng=rng)
        assert 0.75 <= delay <= 1.25


def test_success_short_circuits() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    result = retry_with_backoff(
        fn,
        policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=sleeps.append,
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_all_fail_reraises_last_exception() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ValueError(f"attempt {calls['n']} failed")

    with pytest.raises(ValueError, match="attempt 3 failed"):
        retry_with_backoff(
            fn,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
            sleep=sleeps.append,
        )

    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_single_attempt_policy_does_not_sleep() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        retry_with_backoff(
            fn,
            policy=RetryPolicy(max_attempts=1, base_delay_seconds=1.0),
            sleep=sleeps.append,
        )
    assert calls["n"] == 1
    assert sleeps == []


def test_deterministic_with_seeded_rng() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=16.0, jitter_ratio=0.25)
    rng_a = random.Random(1234)
    rng_b = random.Random(1234)
    seq_a = [compute_delay_seconds(i, policy, rng=rng_a) for i in range(5)]
    seq_b = [compute_delay_seconds(i, policy, rng=rng_b) for i in range(5)]
    assert seq_a == seq_b
