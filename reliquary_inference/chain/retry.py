"""Bounded exponential-backoff retry helper with jittered delays.

Deliberately chain-agnostic: takes a callable, not a subtensor object.
Used by :func:`set_weights_with_retry` and :func:`commit_policy_metadata`
but can wrap any idempotent chain call.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for bounded exponential-backoff retries."""

    max_attempts: int = 4
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 16.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("delay seconds must be non-negative")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if not (0 <= self.jitter_ratio <= 1):
            raise ValueError("jitter_ratio must be in [0, 1]")


def compute_delay_seconds(
    attempt_index: int,
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
) -> float:
    """Return the sleep delay before the ``attempt_index``-th retry (0-based).

    The k-th retry (after the k-th failure) waits ``min(base * mult^k, max)``
    seconds, times a uniform jitter factor in ``[1-j, 1+j]``.
    """
    rng = rng or random
    raw = policy.base_delay_seconds * (policy.multiplier ** attempt_index)
    capped = min(raw, policy.max_delay_seconds)
    if policy.jitter_ratio == 0:
        return max(0.0, capped)
    low = 1.0 - policy.jitter_ratio
    high = 1.0 + policy.jitter_ratio
    return max(0.0, capped * rng.uniform(low, high))


def retry_with_backoff(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    rng: random.Random | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn`` up to ``policy.max_attempts`` times, sleeping between retries.

    Re-raises the last exception on final failure.
    """
    policy = policy or RetryPolicy()
    attempt = 0
    last_exc: BaseException | None = None
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry all exceptions
            last_exc = exc
            if attempt >= policy.max_attempts:
                break
            delay = compute_delay_seconds(attempt - 1, policy, rng=rng)
            sleep(delay)
    assert last_exc is not None
    raise last_exc
