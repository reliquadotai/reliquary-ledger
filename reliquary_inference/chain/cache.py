"""TTL-bound metagraph cache.

Validators read the cached snapshot until its age exceeds ``ttl_seconds``;
stale reads trigger a refresh at the next window boundary. Keeps the
subtensor websocket quiet under high-QPS scenarios and provides a
graceful degraded-mode pointer when refresh fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetagraphCache:
    ttl_seconds: float
    _snapshot: Any = field(default=None, init=False, repr=False)
    _fetched_at: float | None = field(default=None, init=False, repr=False)

    def set(self, snapshot: Any, now: float | None = None) -> None:
        self._snapshot = snapshot
        self._fetched_at = now if now is not None else time.time()

    def snapshot(self) -> Any:
        if self._snapshot is None:
            raise ValueError("metagraph cache is empty")
        return self._snapshot

    def is_stale(self, now: float | None = None) -> bool:
        if self._fetched_at is None:
            return True
        delta = (now if now is not None else time.time()) - self._fetched_at
        return delta > self.ttl_seconds

    def age_seconds(self, now: float | None = None) -> float:
        if self._fetched_at is None:
            return float("inf")
        return (now if now is not None else time.time()) - self._fetched_at

    def clear(self) -> None:
        self._snapshot = None
        self._fetched_at = None
