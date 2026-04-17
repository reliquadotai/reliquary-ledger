"""Tests for the TTL-bound metagraph cache.

Spec reference: private/reliquary-plan/notes/spec-chain-adapter.md
acceptance tests 9-10.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary_inference.chain.cache import MetagraphCache


def test_empty_cache_is_stale() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    assert cache.is_stale(now=0.0) is True
    assert cache.age_seconds(now=0.0) == float("inf")


def test_snapshot_raises_when_empty() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    with pytest.raises(ValueError):
        cache.snapshot()


def test_set_then_snapshot_returns_same_object() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    meta = SimpleNamespace(hotkeys=["a", "b"], uids=[1, 2])
    cache.set(meta, now=100.0)
    assert cache.snapshot() is meta


def test_fresh_cache_is_not_stale() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    cache.set(object(), now=100.0)
    assert cache.is_stale(now=130.0) is False
    assert cache.age_seconds(now=130.0) == 30.0


def test_ttl_elapsed_cache_is_stale() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    cache.set(object(), now=100.0)
    assert cache.is_stale(now=200.0) is True


def test_clear_reverts_to_empty_state() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    cache.set(object(), now=100.0)
    cache.clear()
    assert cache.is_stale(now=100.0) is True
    with pytest.raises(ValueError):
        cache.snapshot()


def test_replacing_snapshot_resets_fetched_at() -> None:
    cache = MetagraphCache(ttl_seconds=60.0)
    cache.set(object(), now=100.0)
    cache.set(object(), now=200.0)
    assert cache.age_seconds(now=201.0) == 1.0
