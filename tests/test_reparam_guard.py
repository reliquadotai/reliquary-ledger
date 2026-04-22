"""Tests for the reparam-trick sanity guards.

Covers the three gates (finite, magnitude floor, layer ratio) against
honest-looking deltas and the classic RMSNorm×Linear reparam attack.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from reliquary_inference.shared.reparam_guard import (
    PROJ_MIN_MEAN_ABS,
    ReparamGuardResult,
    check_layer_scale_ratio,
    check_projection_magnitude,
    check_tensor_finite,
    guard_delta_shards,
)


@dataclass
class _Shard:
    """Test stand-in for DeltaShard — carries just the fields the guard reads."""

    tensor_name: str
    data_bytes: list[float]


# ---------------------------------------------------------------------------
# check_tensor_finite
# ---------------------------------------------------------------------------


def test_finite_check_passes_clean_data():
    assert check_tensor_finite("w", [0.01, -0.02, 0.001]) is None


def test_finite_check_catches_nan():
    res = check_tensor_finite("w", [0.01, float("nan"), 0.001])
    assert res is not None
    assert res.reason == "non_finite_values"
    assert res.offending_tensor == "w"


def test_finite_check_catches_inf():
    res = check_tensor_finite("w", [0.01, float("inf"), 0.001])
    assert res is not None
    assert res.reason == "non_finite_values"


# ---------------------------------------------------------------------------
# check_projection_magnitude
# ---------------------------------------------------------------------------


def test_magnitude_floor_passes_normal_tensor():
    assert check_projection_magnitude("w", mean_abs=0.005) is None


def test_magnitude_floor_rejects_tiny_tensor():
    res = check_projection_magnitude("w", mean_abs=1e-8)
    assert res is not None
    assert res.reason == "projection_magnitude_below_floor"
    assert res.observed_value == 1e-8
    assert res.threshold == PROJ_MIN_MEAN_ABS


def test_magnitude_floor_accepts_exactly_threshold():
    assert check_projection_magnitude("w", mean_abs=PROJ_MIN_MEAN_ABS) is None


def test_magnitude_floor_custom_threshold():
    # Tensor would pass the default 1e-4 but fails a stricter 1e-2 threshold.
    res = check_projection_magnitude("w", mean_abs=1e-3, threshold=1e-2)
    assert res is not None
    assert res.threshold == 1e-2


# ---------------------------------------------------------------------------
# check_layer_scale_ratio
# ---------------------------------------------------------------------------


def test_ratio_check_passes_balanced_layer():
    means = {"q": 0.01, "k": 0.015, "v": 0.012}
    assert check_layer_scale_ratio("L0", means) is None


def test_ratio_check_rejects_reparam_attack():
    # Classic RMSNorm×Linear exploit: γ scaled to α, W scaled to 1/α
    # with α = 1e4. Mean-|w| ratio = 1e8.
    means = {"norm.weight": 1e4, "self_attn.q_proj.weight": 1e-4}
    res = check_layer_scale_ratio("L0", means)
    assert res is not None
    assert res.reason.startswith("layer_scale_ratio_exceeded")
    assert res.observed_value == pytest.approx(1e8)


def test_ratio_check_ignores_single_tensor():
    # Can't compute a ratio from one value.
    assert check_layer_scale_ratio("L0", {"q": 0.01}) is None


def test_ratio_check_custom_threshold():
    means = {"q": 100.0, "k": 0.01}
    # 1e4 ratio exceeds a 1e3 threshold.
    res = check_layer_scale_ratio("L0", means, threshold=1e3)
    assert res is not None


# ---------------------------------------------------------------------------
# guard_delta_shards — end-to-end
# ---------------------------------------------------------------------------


def test_guard_passes_honest_delta():
    shards = [
        _Shard("model.layers.0.self_attn.q_proj.weight", [0.002, -0.003, 0.001]),
        _Shard("model.layers.0.self_attn.k_proj.weight", [0.002, -0.002, 0.003]),
        _Shard("model.layers.1.self_attn.q_proj.weight", [0.001, 0.002, -0.001]),
    ]
    result = guard_delta_shards(shards)
    assert result.ok is True


def test_guard_rejects_nan_shard():
    shards = [
        _Shard("model.layers.0.self_attn.q_proj.weight", [0.002, float("nan"), 0.001]),
    ]
    result = guard_delta_shards(shards)
    assert result.ok is False
    assert result.reason == "non_finite_values"


def test_guard_rejects_dead_shard():
    shards = [
        _Shard("model.layers.0.self_attn.q_proj.weight", [1e-9, 1e-9]),
    ]
    result = guard_delta_shards(shards)
    assert result.ok is False
    assert "projection_magnitude_below_floor" in result.reason


def test_guard_rejects_reparam_layer():
    # One tensor in layer 0 got scaled up by 1e6, another scaled down —
    # classic signature of the RMSNorm×Linear exploit.
    shards = [
        _Shard("model.layers.0.norm.weight", [1e3, 2e3, 1.5e3]),
        _Shard("model.layers.0.self_attn.q_proj.weight", [5e-3, -3e-3, 4e-3]),
    ]
    result = guard_delta_shards(shards)
    assert result.ok is False
    assert "layer_scale_ratio_exceeded" in result.reason


def test_guard_ignores_unknown_tensor_format():
    """Shards without recognised payload shape are skipped, not fail-closed."""
    class WeirdShard:
        tensor_name = "model.layers.0.foo.weight"
        data_bytes = object()  # can't compute mean-abs from this

    result = guard_delta_shards([WeirdShard()])
    assert result.ok is True  # skipped, not rejected


def test_guard_tolerates_missing_fields():
    class Empty:
        pass

    result = guard_delta_shards([Empty()])
    assert result.ok is True


def test_guard_result_as_dict_serializes():
    res = ReparamGuardResult(
        ok=False, reason="x", offending_tensor="t", observed_value=1.0, threshold=0.1,
    )
    d = res.as_dict()
    assert d == {
        "ok": False,
        "reason": "x",
        "offending_tensor": "t",
        "observed_value": 1.0,
        "threshold": 0.1,
    }
