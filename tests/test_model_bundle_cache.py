"""Tests for the bundle-cache + mutation helpers in shared/modeling.py.

The validator path hot-swaps its model by mutating cached bundles in place
(rather than reloading the 6GB model from disk per completion). These
tests verify the cache gives identity on repeat loads and the
apply_delta_to_cached_bundles helper mutates every cached bundle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.shared import modeling
from reliquary_inference.shared.modeling import (
    apply_delta_to_cached_bundles,
    cached_bundles,
    clear_bundle_cache,
    load_model_bundle,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_bundle_cache()
    yield
    clear_bundle_cache()


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_toy_bundle_is_cached():
    b1 = load_model_bundle("toy://unit-test", device="cpu")
    b2 = load_model_bundle("toy://unit-test", device="cpu")
    # Same dict object returned on second call.
    assert b1 is b2
    assert cached_bundles() == [b1]


def test_different_refs_cache_separately():
    a = load_model_bundle("toy://a", device="cpu")
    b = load_model_bundle("toy://b", device="cpu")
    assert a is not b
    assert len(cached_bundles()) == 2


def test_clear_bundle_cache_drops_all():
    load_model_bundle("toy://x", device="cpu")
    load_model_bundle("toy://y", device="cpu")
    assert len(cached_bundles()) == 2
    clear_bundle_cache()
    assert cached_bundles() == []


# ---------------------------------------------------------------------------
# Delta application via global cache
# ---------------------------------------------------------------------------


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


def _make_int8_shard(tensor_name, delta_float):
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


def test_apply_delta_to_cached_bundles_mutates_toy_model():
    """Inject a fake model into the cache + verify in-place mutation."""
    # Build a synthetic bundle whose model is a tiny nn.Module with a
    # known parameter name. Load_model_bundle's toy path gives us an
    # opaque ToyModel; let's instead manually register a cached bundle
    # with a real nn.Linear so we can observe weights.
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2, bias=False)  # "weight" is the param name
    fake_bundle = {
        "model": model,
        "tokenizer": object(),
        "device": "cpu",
        "model_ref": "synthetic://test",
    }
    modeling._BUNDLE_CACHE[("synthetic://test", "cpu", False, "auto", False)] = fake_bundle

    pre = model.weight.data.clone()

    delta = torch.full_like(model.weight, 0.01)
    shard = _make_int8_shard("weight", delta)
    bundle = _Bundle(shards=[shard])

    mutated = apply_delta_to_cached_bundles(bundle)
    assert mutated == 1
    diff = (model.weight.data - pre).abs().max().item()
    # Delta was ~0.01; int8 quantization resolution ~0.002 at this scale.
    assert 0.005 < diff < 0.02


def test_apply_delta_to_multiple_cached_bundles():
    torch.manual_seed(0)
    model_a = torch.nn.Linear(4, 2, bias=False)
    model_b = torch.nn.Linear(4, 2, bias=False)
    for name, m in [("synthetic://a", model_a), ("synthetic://b", model_b)]:
        modeling._BUNDLE_CACHE[(name, "cpu", False, "auto", False)] = {
            "model": m,
            "tokenizer": object(),
            "device": "cpu",
            "model_ref": name,
        }

    pre_a = model_a.weight.data.clone()
    pre_b = model_b.weight.data.clone()

    delta = torch.full_like(model_a.weight, 0.01)
    shard = _make_int8_shard("weight", delta)
    bundle = _Bundle(shards=[shard])

    mutated = apply_delta_to_cached_bundles(bundle)
    assert mutated == 2
    assert (model_a.weight.data - pre_a).abs().max().item() > 0.005
    assert (model_b.weight.data - pre_b).abs().max().item() > 0.005


def test_apply_delta_with_empty_cache_is_noop():
    mutated = apply_delta_to_cached_bundles(_Bundle(shards=[]))
    assert mutated == 0


def test_unknown_tensor_skipped_when_applying_to_cached_bundle():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2, bias=False)
    modeling._BUNDLE_CACHE[("synthetic://s", "cpu", False, "auto", False)] = {
        "model": model,
        "tokenizer": object(),
        "device": "cpu",
        "model_ref": "synthetic://s",
    }
    pre = model.weight.data.clone()

    delta = torch.full((2, 2), 0.01)
    shard = _make_int8_shard("no_such_tensor", delta)
    bundle = _Bundle(shards=[shard])

    mutated = apply_delta_to_cached_bundles(bundle)
    # One bundle mutated (applier ran once), but the shard's tensor didn't
    # match any model param — weights unchanged.
    assert mutated == 1
    assert torch.equal(model.weight.data, pre)
