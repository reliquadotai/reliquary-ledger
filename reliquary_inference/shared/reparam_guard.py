"""Reparameterization-trick sanity guards for policy deltas.

**Context.** RMSNorm followed by a Linear layer has a 1-D rescale symmetry:
``f(x; γ, W) = f(x; α·γ, W/α)`` for any α > 0. The forward pass is
identical, but gradient magnitudes scale as ``|W|`` (for γ) and ``|γ|``
(for W), so one SGD step at a reasonable learning rate blows up the model.

An attacker who can publish a policy delta (e.g. by compromising the
Forge trainer or stealing the policy-authority HMAC) could ship a
checkpoint that passes every content-hash check but is effectively
untrainable: as soon as a miner runs its first mining cycle or the
validator applies the delta to its cached model, outputs diverge
wildly. The attack surface is narrower for us than for Teutonic-style
checkpoint-upload subnets (HMAC signature already blocks unsigned
deltas), but defense-in-depth is cheap.

These guards are **static** — they only read the delta's tensors, no
forward pass, no model load. They catch the classic RMSNorm×Linear
exploit by checking:

  1. **Projection magnitude floor.** Every target tensor's mean |w|
     must exceed ``PROJ_MIN_MEAN_ABS``. A tensor rescaled to ~0 is
     either numerically dead or part of a reparam pair with another
     tensor scaled to ~∞.

  2. **Per-layer scale ratio.** For a consistent layer grouping
     (tensors sharing the first ``model.layers.N`` prefix), the max
     and min mean-|w| across tensors in the layer must be within
     ``LAYER_SCALE_RATIO_MAX``. Honest deltas keep the ratio small;
     reparam attacks create huge internal imbalance.

  3. **Shard finite-value guard.** NaN / inf in any delta tensor is
     an immediate reject — no honest training produces those.

Ported from Teutonic's eval_torch.trainability_probe / validator.py
RMSNorm guards (see private/reliquary-plan/notes/competitive-teutonic.md)
but simplified: static tensor inspection instead of live SGD probe,
since we don't accept arbitrary checkpoints from miners — only
signed deltas from our Forge trainer — and the attack surface is
narrower here. The probe-based version is a follow-up if we ever
open up miner-submitted checkpoints.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


# Defaults match the Teutonic values that caught live adversarial models
# in their mainnet deployment; can be overridden via env for tuning.
PROJ_MIN_MEAN_ABS = float(
    os.environ.get("RELIQUARY_INFERENCE_REPARAM_PROJ_MIN_MEAN_ABS", "1e-4")
)
LAYER_SCALE_RATIO_MAX = float(
    os.environ.get("RELIQUARY_INFERENCE_REPARAM_LAYER_SCALE_RATIO_MAX", "1e5")
)

_LAYER_PREFIX_RE = re.compile(r"^(model\.layers\.\d+)\.")


@dataclass(frozen=True)
class ReparamGuardResult:
    """Outcome of a reparam sanity check over a delta's target tensors."""

    ok: bool
    reason: str = ""
    offending_tensor: str = ""
    observed_value: float = 0.0
    threshold: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "offending_tensor": self.offending_tensor,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
        }


def check_tensor_finite(name: str, data: Any) -> ReparamGuardResult | None:
    """Reject NaN / inf in the delta. Returns None if OK."""
    try:
        import torch  # local import so the module imports on pure-python hosts
        if isinstance(data, torch.Tensor):
            if not bool(torch.isfinite(data).all()):
                return ReparamGuardResult(
                    ok=False,
                    reason="non_finite_values",
                    offending_tensor=name,
                )
            return None
    except ImportError:
        pass
    # Fallback for numpy arrays / python lists used in tests:
    try:
        flat = list(data) if not isinstance(data, (int, float)) else [data]
        for v in flat:
            if isinstance(v, (list, tuple)):
                result = check_tensor_finite(name, v)
                if result is not None:
                    return result
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                return ReparamGuardResult(
                    ok=False,
                    reason="non_finite_values",
                    offending_tensor=name,
                )
    except (TypeError, ValueError):
        pass
    return None


def check_projection_magnitude(
    name: str,
    mean_abs: float,
    *,
    threshold: float = PROJ_MIN_MEAN_ABS,
) -> ReparamGuardResult | None:
    """Reject tensors whose mean |w| fell below the magnitude floor.

    Under RMSNorm×Linear reparameterization, one of the paired tensors
    ends up scaled to ~0 and the other to ~∞. Either half trips this
    check (the ~∞ half trips the layer-scale ratio check in the
    companion function).
    """
    if mean_abs < threshold:
        return ReparamGuardResult(
            ok=False,
            reason="projection_magnitude_below_floor",
            offending_tensor=name,
            observed_value=mean_abs,
            threshold=threshold,
        )
    return None


def check_layer_scale_ratio(
    layer_prefix: str,
    tensor_means: dict[str, float],
    *,
    threshold: float = LAYER_SCALE_RATIO_MAX,
) -> ReparamGuardResult | None:
    """Within a single transformer layer, max(mean_|w|) / min(mean_|w|)
    must stay under ``threshold``. Honest deltas keep this ratio small
    (typical < 100); reparam attacks push it to 1e6+.
    """
    vals = [v for v in tensor_means.values() if v > 0]
    if len(vals) < 2:
        return None
    hi = max(vals)
    lo = min(vals)
    ratio = hi / lo if lo > 0 else float("inf")
    if ratio > threshold:
        # Identify the pair that drove the ratio for the operator log.
        hi_name = max(tensor_means, key=lambda k: tensor_means[k])
        lo_name = min(tensor_means, key=lambda k: tensor_means[k])
        return ReparamGuardResult(
            ok=False,
            reason=f"layer_scale_ratio_exceeded:{layer_prefix}[{hi_name}/{lo_name}]",
            offending_tensor=hi_name,
            observed_value=ratio,
            threshold=threshold,
        )
    return None


def guard_delta_shards(
    shards: Iterable[Any],
    *,
    name_attr: str = "tensor_name",
    data_attr: str = "data_bytes",
    proj_threshold: float = PROJ_MIN_MEAN_ABS,
    layer_ratio_threshold: float = LAYER_SCALE_RATIO_MAX,
) -> ReparamGuardResult:
    """Run all three guards over a shard iterable.

    Each shard must expose:
      - a tensor name at ``name_attr`` (default ``tensor_name``)
      - a tensor-like object at ``data_attr`` that we can compute
        mean-abs over (torch.Tensor OR numpy array OR raw bytes we
        reinterpret as float32)

    Unrecognised shard formats are ignored (return OK rather than
    fail-closed) — this module is defense-in-depth and should never
    block a legitimate delta on a format quirk.

    Returns a single ``ReparamGuardResult``: ``ok=True`` iff every
    shard passes every check.
    """
    by_layer: dict[str, dict[str, float]] = {}
    for shard in shards:
        name = str(getattr(shard, name_attr, None) or "")
        data = getattr(shard, data_attr, None)
        if not name or data is None:
            continue
        fin = check_tensor_finite(name, data)
        if fin is not None:
            return fin
        mean_abs = _compute_mean_abs(data)
        if mean_abs is None:
            continue
        mag = check_projection_magnitude(name, mean_abs, threshold=proj_threshold)
        if mag is not None:
            return mag
        m = _LAYER_PREFIX_RE.match(name)
        if m:
            by_layer.setdefault(m.group(1), {})[name] = mean_abs

    for prefix, means in by_layer.items():
        ratio_res = check_layer_scale_ratio(
            prefix, means, threshold=layer_ratio_threshold
        )
        if ratio_res is not None:
            return ratio_res

    return ReparamGuardResult(ok=True)


def _compute_mean_abs(data: Any) -> float | None:
    """Compute mean-abs over a torch.Tensor / numpy array / raw bytes.

    Returns None when we can't interpret the payload — the caller
    treats that as "no signal" rather than fail-closed.
    """
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return float(data.detach().abs().mean().item())
    except ImportError:
        pass
    try:
        # numpy array
        import numpy as np
        if hasattr(data, "dtype") and hasattr(data, "shape"):
            arr = np.asarray(data).astype(np.float32).ravel()
            if arr.size == 0:
                return 0.0
            return float(np.abs(arr).mean())
    except ImportError:
        pass
    if isinstance(data, (bytes, bytearray, memoryview)):
        # Raw float32 bytes (our DeltaBundle shard format).
        try:
            import numpy as np
            buf = np.frombuffer(bytes(data), dtype=np.float32)
            if buf.size == 0:
                return 0.0
            return float(np.abs(buf).mean())
        except ImportError:
            return None
    if isinstance(data, (list, tuple)):
        flat = [float(v) for v in data if isinstance(v, (int, float))]
        if not flat:
            return 0.0
        return sum(abs(v) for v in flat) / len(flat)
    return None


__all__ = [
    "LAYER_SCALE_RATIO_MAX",
    "PROJ_MIN_MEAN_ABS",
    "ReparamGuardResult",
    "check_layer_scale_ratio",
    "check_projection_magnitude",
    "check_tensor_finite",
    "guard_delta_shards",
]
