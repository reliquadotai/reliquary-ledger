"""Tests for :mod:`reliquary_inference.shared.policy_applier`.

Uses a fake MiningEngine whose ``.model`` is a tiny torch.nn.Linear so
we can verify tensor-level in-place mutation without loading a real LLM.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.shared.policy_applier import (
    APPLIER_METRIC_APPLIED,
    APPLIER_METRIC_REJECTED,
    APPLIER_METRIC_SKIPPED_UNKNOWN,
    ReloadingPolicyApplier,
    bundle_aware_delta_loader,
)
from reliquary_inference.shared.policy_consumer import LoadedDelta


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _FakeEngine:
    """Minimal MiningEngine-shaped object — just a .model attribute."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 2, bias=False),  # parameter name: "0.weight"
            torch.nn.Linear(2, 2, bias=False),  # parameter name: "1.weight"
        )


@dataclass
class _Shard:
    tensor_name: str
    data_bytes: bytes
    scale: float
    shape: tuple[int, ...]
    payload_sha256: str = ""


@dataclass
class _Bundle:
    shards: list[_Shard]
    merkle_root_hex: str = "a" * 64


def _make_shard(tensor_name: str, delta_float: torch.Tensor) -> _Shard:
    """Quantize a fp32 delta tensor into an int8 shard (symmetric, same as
    reliquary.training.delta_checkpoints._quantize)."""
    import numpy as np

    arr = delta_float.detach().cpu().float().numpy().ravel()
    amax = float(np.abs(arr).max()) if arr.size else 0.0
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
    data_bytes = q.tobytes()
    return _Shard(
        tensor_name=tensor_name,
        data_bytes=data_bytes,
        scale=scale,
        shape=tuple(delta_float.shape),
        payload_sha256=hashlib.sha256(data_bytes).hexdigest(),
    )


# --------------------------------------------------------------------------
# Applier happy path
# --------------------------------------------------------------------------


def test_apply_updates_named_parameter_in_place():
    engine = _FakeEngine()
    p0_before = engine.model[0].weight.data.clone()

    delta_tensor = torch.full_like(p0_before, 0.01)
    shard = _make_shard("0.weight", delta_tensor)
    bundle = _Bundle(shards=[shard])
    loaded = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"",
        shard_digests=(shard.payload_sha256,),
        extra={"bundle": bundle},
    )

    applier = ReloadingPolicyApplier(engine)
    applier(loaded)

    p0_after = engine.model[0].weight.data
    diff = (p0_after - p0_before).abs().max().item()
    # Delta was ~0.01, int8 quantization rounds to ~scale resolution.
    assert 0.005 < diff < 0.02, f"expected ~0.01 change, got {diff}"
    assert applier.metrics_counters[APPLIER_METRIC_APPLIED] == 1
    assert applier.metrics_counters[APPLIER_METRIC_SKIPPED_UNKNOWN] == 0


def test_apply_skips_tensor_not_in_model():
    """A ragged bundle (has tensor names the model doesn't) doesn't crash."""
    engine = _FakeEngine()

    delta = torch.full((2, 2), 0.01)
    shard_unknown = _make_shard("nonexistent.weight", delta)
    shard_known = _make_shard("1.weight", torch.full_like(engine.model[1].weight, 0.02))

    bundle = _Bundle(shards=[shard_unknown, shard_known])
    loaded = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"",
        shard_digests=(shard_unknown.payload_sha256, shard_known.payload_sha256),
        extra={"bundle": bundle},
    )

    p1_before = engine.model[1].weight.data.clone()
    applier = ReloadingPolicyApplier(engine)
    applier(loaded)

    # Known shard applied.
    assert (engine.model[1].weight.data - p1_before).abs().max().item() > 0
    # Unknown skipped.
    assert applier.metrics_counters[APPLIER_METRIC_SKIPPED_UNKNOWN] == 1
    assert applier.metrics_counters[APPLIER_METRIC_APPLIED] == 1


# --------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------


def test_missing_bundle_in_extra_raises():
    engine = _FakeEngine()
    loaded = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"",
        shard_digests=(),
        extra={},  # no "bundle" key
    )
    applier = ReloadingPolicyApplier(engine)
    with pytest.raises(RuntimeError, match="bundle"):
        applier(loaded)
    assert applier.metrics_counters[APPLIER_METRIC_REJECTED] == 1


def test_shape_mismatch_raises_before_any_mutation():
    engine = _FakeEngine()
    p0_before = engine.model[0].weight.data.clone()

    # Intentionally wrong shape
    bad_delta = torch.ones(5, 5)
    bad_shard = _make_shard("0.weight", bad_delta)

    good_delta = torch.full_like(engine.model[1].weight, 0.01)
    good_shard = _make_shard("1.weight", good_delta)
    p1_before = engine.model[1].weight.data.clone()

    bundle = _Bundle(shards=[bad_shard, good_shard])
    loaded = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"",
        shard_digests=(bad_shard.payload_sha256, good_shard.payload_sha256),
        extra={"bundle": bundle},
    )

    applier = ReloadingPolicyApplier(engine)
    with pytest.raises(ValueError, match="shape mismatch"):
        applier(loaded)

    # No tensor was mutated because pre-flight catches shape errors
    # before any `.add_()` runs.
    assert torch.equal(engine.model[0].weight.data, p0_before)
    assert torch.equal(engine.model[1].weight.data, p1_before)
    assert applier.metrics_counters[APPLIER_METRIC_REJECTED] == 1
    assert applier.metrics_counters[APPLIER_METRIC_APPLIED] == 0


# --------------------------------------------------------------------------
# bundle_aware_delta_loader
# --------------------------------------------------------------------------


def test_bundle_aware_delta_loader_attaches_bundle():
    """Returned LoadedDelta has .extra['bundle'] populated."""

    shard = _Shard(
        tensor_name="x",
        data_bytes=b"\x00" * 8,
        scale=1.0,
        shape=(2, 4),
        payload_sha256=hashlib.sha256(b"\x00" * 8).hexdigest(),
    )
    bundle = _Bundle(shards=[shard], merkle_root_hex="b" * 64)

    def fake_fetch(*, run_id, window_id, backend):
        assert run_id == "r1"
        assert window_id == 42
        return bundle

    class _FakeBackend:
        def get(self, key):
            return b"manifest-bytes"
        def put(self, k, d): ...
        def list(self, p): return []
        def delete(self, k): ...

    loader = bundle_aware_delta_loader(fake_fetch, lambda: _FakeBackend())
    loaded = loader(
        run_id="r1",
        window_id=42,
        expected_merkle_root_hex="b" * 64,
        backend=_FakeBackend(),
    )
    assert loaded.merkle_root_hex == "b" * 64
    assert loaded.extra["bundle"] is bundle
    assert loaded.raw_manifest_bytes == b"manifest-bytes"


def test_bundle_aware_delta_loader_falls_back_when_backend_invalid():
    """If backend handed in doesn't expose put/get/list/delete, use factory."""

    shard = _Shard(tensor_name="x", data_bytes=b"\x00", scale=1.0, shape=(1,))
    bundle = _Bundle(shards=[shard])

    def fake_fetch(*, run_id, window_id, backend):
        # Assert the factory backend was used, not the one passed in.
        assert isinstance(backend, _FactoryBackend)
        return bundle

    class _IncompleteBackend:
        def get(self, key):
            return b"manifest"
        # missing put / list / delete

    class _FactoryBackend:
        def get(self, key):
            return b"factory-manifest"
        def put(self, k, d): ...
        def list(self, p): return []
        def delete(self, k): ...

    loader = bundle_aware_delta_loader(fake_fetch, lambda: _FactoryBackend())
    loaded = loader(
        run_id="r",
        window_id=1,
        expected_merkle_root_hex="a" * 64,
        backend=_IncompleteBackend(),
    )
    assert loaded.raw_manifest_bytes == b"factory-manifest"
