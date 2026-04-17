"""Tests for BittensorChainAdapter.set_weights_with_retry + commit_policy_metadata.

Spec reference: private/reliquary-plan/notes/spec-chain-adapter.md
acceptance tests 6-8, 11-12.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from reliquary_inference.chain.adapter import (
    BittensorChainAdapter,
    PolicyCommitResult,
    WeightSubmissionResult,
)
from reliquary_inference.chain.retry import RetryPolicy


class _Wallet:
    def __init__(self, *, name: str, hotkey: str, path: str) -> None:
        self.name = name
        self.hotkey = hotkey
        self.path = path


def _adapter() -> BittensorChainAdapter:
    return BittensorChainAdapter(
        network="test",
        netuid=81,
        wallet_name="w",
        hotkey_name="h",
        wallet_path="/tmp",
        use_drand=False,
    )


def _install_dummy_wallet(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "bittensor_wallet",
        types.SimpleNamespace(Wallet=_Wallet),
    )


def test_set_weights_returns_no_matching_hotkeys(monkeypatch) -> None:
    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "get_metagraph",
        lambda: SimpleNamespace(hotkeys=["known"], uids=[0]),
    )

    result = adapter.set_weights_with_retry(
        window_id=1,
        weights={"unknown": 0.5},
    )

    assert isinstance(result, WeightSubmissionResult)
    assert result.success is False
    assert result.attempts == 0
    assert result.last_error == "no_matching_hotkeys"
    assert result.uids == []


def test_set_weights_succeeds_after_transient_failures(monkeypatch) -> None:
    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "get_metagraph",
        lambda: SimpleNamespace(hotkeys=["miner"], uids=[7]),
    )

    calls = {"n": 0}

    def _flaky_subtensor(callback):
        subtensor = SimpleNamespace(
            set_weights=lambda **_kw: _raise_then_ok()
        )
        return callback(subtensor)

    def _raise_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate-limited")
        return True

    monkeypatch.setattr(adapter, "_with_subtensor", _flaky_subtensor)

    sleeps: list[float] = []
    result = adapter.set_weights_with_retry(
        window_id=2,
        weights={"miner": 0.5},
        retry_policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=sleeps.append,
    )

    assert result.success is True
    assert result.attempts == 3
    assert result.uids == [7]
    assert len(sleeps) == 2


def test_set_weights_reports_failure_after_all_retries(monkeypatch) -> None:
    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "get_metagraph",
        lambda: SimpleNamespace(hotkeys=["miner"], uids=[7]),
    )

    def _always_fail(callback):
        subtensor = SimpleNamespace(set_weights=lambda **_kw: _raise())
        return callback(subtensor)

    def _raise():
        raise RuntimeError("chain busy")

    monkeypatch.setattr(adapter, "_with_subtensor", _always_fail)

    result = adapter.set_weights_with_retry(
        window_id=3,
        weights={"miner": 0.5},
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=lambda _s: None,
    )

    assert result.success is False
    assert result.attempts == 3
    assert result.last_error is not None
    assert "chain busy" in result.last_error


def test_set_weights_uses_metagraph_cache_when_fresh(monkeypatch) -> None:
    from reliquary_inference.chain.cache import MetagraphCache

    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()

    cache = MetagraphCache(ttl_seconds=120.0)
    cache.set(SimpleNamespace(hotkeys=["miner"], uids=[42]), now=100.0)

    live_calls = {"n": 0}

    def _live_metagraph():
        live_calls["n"] += 1
        return SimpleNamespace(hotkeys=[], uids=[])

    monkeypatch.setattr(adapter, "get_metagraph", _live_metagraph)

    def _subtensor_ok(callback):
        return callback(SimpleNamespace(set_weights=lambda **_kw: True))

    monkeypatch.setattr(adapter, "_with_subtensor", _subtensor_ok)

    monkeypatch.setattr(
        "reliquary_inference.chain.cache.time.time", lambda: 150.0
    )
    result = adapter.set_weights_with_retry(
        window_id=5,
        weights={"miner": 0.75},
        metagraph_cache=cache,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=lambda _s: None,
    )

    assert result.success is True
    assert result.uids == [42]
    assert live_calls["n"] == 0


def test_commit_policy_metadata_success(monkeypatch) -> None:
    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()

    captured = {}

    def _capture(callback):
        subtensor = SimpleNamespace(commit=lambda wallet, netuid, data: captured.setdefault("data", data) or True)
        return callback(subtensor)

    monkeypatch.setattr(adapter, "_with_subtensor", _capture)

    result = adapter.commit_policy_metadata(
        policy_version="v1",
        metadata_hash="0xabc",
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=lambda _s: None,
    )

    assert isinstance(result, PolicyCommitResult)
    assert result.success is True
    assert result.commitment_key == "reliquary_policy_v1"
    assert result.commitment_hash == "0xabc"
    assert captured["data"] == "reliquary_policy_v1=0xabc"


def test_commit_policy_metadata_failure_propagates_error(monkeypatch) -> None:
    _install_dummy_wallet(monkeypatch)
    adapter = _adapter()

    def _always_fail(callback):
        return callback(SimpleNamespace(commit=lambda **_kw: (_ for _ in ()).throw(RuntimeError("nack"))))

    monkeypatch.setattr(adapter, "_with_subtensor", _always_fail)

    result = adapter.commit_policy_metadata(
        policy_version="v1",
        metadata_hash="0xabc",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.0, jitter_ratio=0.0, max_delay_seconds=0.0),
        sleep=lambda _s: None,
    )

    assert result.success is False
    assert result.attempts == 2
    assert "nack" in result.last_error
