"""Hidden-state sketch verifier for GPU/framework-agnostic proofs.

Key innovations:
1. Top-K selection: Focus on important activations (stable)
2. Logarithmic bucketing: Coarse quantization reduces sensitivity
3. Sketch verification: Random linear projection for cryptographic binding

Security: ~10^-167 forgery probability across K=32 challenged positions.
"""

from __future__ import annotations

import logging
import math
import os as _os

import torch

from ..constants import (
    PRIME_Q,
    PROOF_COEFF_RANGE,
    PROOF_NUM_BUCKETS,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
    PROOF_TOPK,
)

logger = logging.getLogger(__name__)


def log_magnitude_bucket(value: float, num_buckets: int = PROOF_NUM_BUCKETS) -> int:
    """Map activation to logarithmic magnitude bucket with sign preservation.

    Args:
        value: Activation value to bucket
        num_buckets: Number of buckets per sign (default: 8)

    Returns:
        Signed bucket index in [-num_buckets+1, 0, num_buckets-1]
    """
    if math.isnan(value):
        logger.warning(
            "NaN value encountered in hidden state. Treating as zero bucket."
        )
        return 0

    if math.isinf(value):
        logger.warning(
            "Infinity value encountered in hidden state. Clamping to maximum bucket."
        )
        return num_buckets - 1 if value > 0 else -(num_buckets - 1)

    abs_val = abs(value)

    if abs_val < 1e-6:
        return 0

    log_val = math.log2(abs_val + 1.0)
    scale_factor = num_buckets / 10.0
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))

    return bucket if value >= 0 else -bucket


def log_magnitude_bucket_vectorized(
    values: torch.Tensor,
    num_buckets: int = PROOF_NUM_BUCKETS,
) -> torch.Tensor:
    """Vectorized log-magnitude bucketing, bit-identical to the scalar version.

    Uses float64 arithmetic to match the scalar path.
    """
    abs_vals = values.abs().to(torch.float64)
    scale_factor = num_buckets / 10.0

    log_vals = torch.log2(abs_vals + 1.0)
    raw_buckets = (log_vals * scale_factor).to(torch.int64)
    raw_buckets = torch.clamp(raw_buckets, min=0, max=num_buckets - 1)

    sign_positive = values >= 0
    buckets = torch.where(sign_positive, raw_buckets, -raw_buckets)

    # Edge cases (applied in priority order)
    zero = torch.zeros_like(buckets)
    deadzone_mask = abs_vals < 1e-6
    buckets = torch.where(deadzone_mask, zero, buckets)

    nan_mask = torch.isnan(values)
    buckets = torch.where(nan_mask, zero, buckets)

    inf_mask = torch.isinf(values)
    if inf_mask.any():
        pos_inf = torch.tensor(num_buckets - 1, dtype=torch.int64, device=values.device)
        neg_inf = torch.tensor(-(num_buckets - 1), dtype=torch.int64, device=values.device)
        inf_buckets = torch.where(values > 0, pos_inf, neg_inf)
        buckets = torch.where(inf_mask, inf_buckets, buckets)

    return buckets


def adaptive_sketch_tolerance(position: int, sequence_length: int) -> int:
    """Compute position-dependent sketch tolerance.

    tolerance = base + growth * sqrt(position)
    """
    return int(PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH * math.sqrt(position))


class SketchProofVerifier:
    """Sketch-based verifier for framework-agnostic hidden state proofs."""

    def __init__(
        self,
        hidden_dim: int,
        topk: int = PROOF_TOPK,
        num_buckets: int = PROOF_NUM_BUCKETS,
        r_coeff_range: int = PROOF_COEFF_RANGE,
    ):
        self.hidden_dim = hidden_dim
        self.topk = max(1, min(topk, hidden_dim))
        self.num_buckets = num_buckets
        self.r_coeff_range = r_coeff_range

    def generate_r_vec(self, randomness_hex: str) -> torch.Tensor:
        """Generate small bounded coefficient vector from randomness.

        Returns:
            Tensor of shape [topk] with int8 coefficients in [-R, R]
        """
        from .crypto import RNG_LABEL, prf

        clean_hex = randomness_hex.strip().replace("0x", "").replace("0X", "")
        if len(clean_hex) % 2 != 0:
            clean_hex = "0" + clean_hex

        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=2 * self.topk,
        )

        import numpy as np

        int16_vals = np.frombuffer(raw, dtype=">i2")[: self.topk]
        coeffs = (np.abs(int16_vals) % (2 * self.r_coeff_range + 1)) - self.r_coeff_range

        return torch.from_numpy(coeffs.astype(np.int8))

    def create_commitment(self, hidden_state: torch.Tensor, r_vec: torch.Tensor) -> dict:
        """Create commitment for a single token position.

        Emits ``{"sketch": int, "hidden_norm": float}``. The
        ``hidden_norm`` field lets the validator's
        ``verify_commitment`` cross-check against its own
        recomputation — closes the dim-downsizing attack vector
        flagged in the 2026-04-29 security audit (#3). Backwards
        compatible: validators that don't enforce the bound simply
        ignore the field.
        """
        abs_hidden = torch.abs(hidden_state)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices = topk_result.indices

        indices, _ = torch.sort(indices)
        values = hidden_state[indices]

        buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in values],
            dtype=torch.int8,
            device=values.device,
        )

        sketch = torch.dot(
            buckets.to(torch.float32),
            r_vec.to(device=buckets.device, dtype=torch.float32),
        ).to(torch.int64)
        sketch_val = int(sketch.item()) % PRIME_Q
        hidden_norm = float(hidden_state.float().norm().item())

        return {"sketch": sketch_val, "hidden_norm": hidden_norm}

    def create_commitments_batch(self, h_layer: torch.Tensor, r_vec: torch.Tensor) -> list[dict]:
        """Create commitments for all positions at once (vectorized).

        Produces bit-identical results to calling create_commitment() in a loop.
        """
        seq_len = h_layer.size(0)

        abs_h = h_layer.abs()
        _, topk_indices = torch.topk(abs_h, k=self.topk, dim=1)
        del abs_h

        topk_indices, _ = torch.sort(topk_indices, dim=1)
        signed_values = torch.gather(h_layer, dim=1, index=topk_indices)

        buckets = log_magnitude_bucket_vectorized(signed_values, self.num_buckets)
        del signed_values

        buckets_f = buckets.to(torch.float32)
        r_vec_f = r_vec.to(torch.float32).to(buckets_f.device)
        sketches = (buckets_f @ r_vec_f).to(torch.int64)
        del buckets, buckets_f

        sketches_list = sketches.tolist()
        sketch_vals = [s % PRIME_Q for s in sketches_list]

        # Per-position hidden_norm — see create_commitment docstring
        # for rationale. Vectorised: one norm per row of h_layer.
        hidden_norms = h_layer.float().norm(dim=1).tolist()

        return [
            {"sketch": sketch_vals[pos], "hidden_norm": float(hidden_norms[pos])}
            for pos in range(seq_len)
        ]

    def verify_commitment(
        self,
        validator_hidden: torch.Tensor,
        miner_commitment: dict,
        r_vec: torch.Tensor,
        sequence_length: int,
        position: int,
    ) -> tuple[bool, dict]:
        """Verify commitment using sketch check."""
        tolerance = adaptive_sketch_tolerance(position, sequence_length)

        abs_hidden = torch.abs(validator_hidden)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices, _ = torch.sort(topk_result.indices)
        validator_values = validator_hidden[indices]

        validator_buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in validator_values],
            dtype=torch.int8,
            device=validator_values.device,
        )

        validator_sketch = torch.dot(
            validator_buckets.to(torch.float32),
            r_vec.to(device=validator_buckets.device, dtype=torch.float32),
        ).to(torch.int64)
        validator_sketch_val = int(validator_sketch.item()) % PRIME_Q

        miner_sketch_val = miner_commitment["sketch"]
        sketch_diff = abs(validator_sketch_val - miner_sketch_val)
        mod_diff = min(sketch_diff, PRIME_Q - sketch_diff)
        is_valid = mod_diff <= tolerance

        # Hidden-norm cross-check (security audit #3): if the miner
        # included ``hidden_norm`` in the commitment, compare it
        # against the validator's recomputation. Catches a class of
        # cheating where an attacker submits sketches consistent with
        # a smaller hidden_dim (e.g. via adapter compression) but
        # whose underlying activation magnitudes don't match what
        # this layer would emit. Env-gated default-off so testnet
        # miners can roll out the new commitment shape before the
        # validator starts enforcing.
        validator_hidden_norm = float(validator_hidden.float().norm().item())
        miner_hidden_norm: float | None = None
        hidden_norm_valid: bool = True
        hidden_norm_diff_rel: float | None = None
        try:
            raw_norm = miner_commitment.get("hidden_norm")  # type: ignore[union-attr]
            if raw_norm is not None:
                miner_hidden_norm = float(raw_norm)
        except Exception:
            miner_hidden_norm = None
        if miner_hidden_norm is not None:
            denom = max(abs(validator_hidden_norm), 1e-9)
            hidden_norm_diff_rel = abs(validator_hidden_norm - miner_hidden_norm) / denom
            tol_rel = float(
                _os.environ.get("RELIQUARY_INFERENCE_HIDDEN_NORM_TOL_REL", "0.05")
            )
            enforce = _os.environ.get(
                "RELIQUARY_INFERENCE_ENFORCE_HIDDEN_NORM_BOUNDS", ""
            ).lower() in {"1", "true", "yes", "on"}
            hidden_norm_valid = hidden_norm_diff_rel <= tol_rel
            if enforce and not hidden_norm_valid:
                is_valid = False

        diagnostics = {
            "sketch_diff": mod_diff,
            "sketch_valid": (mod_diff <= tolerance),
            "sketch_tolerance": tolerance,
            "overall_valid": is_valid,
            "validator_sketch": validator_sketch_val,
            "miner_sketch": miner_sketch_val,
            "position": position,
            "validator_hidden_norm": validator_hidden_norm,
            "miner_hidden_norm": miner_hidden_norm,
            "hidden_norm_diff_rel": hidden_norm_diff_rel,
            "hidden_norm_valid": hidden_norm_valid,
        }

        if not is_valid:
            sample_vals = validator_values[:5].tolist() if len(validator_values) >= 5 else validator_values.tolist()
            sample_buckets = validator_buckets[:5].tolist() if len(validator_buckets) >= 5 else validator_buckets.tolist()
            logger.warning(
                "[verify_commitment] SKETCH MISMATCH: position=%d | "
                "validator_sketch=%d | miner_sketch=%d | diff=%d | tolerance=%d | "
                "sample_values=%s | sample_buckets=%s | hidden_norm=%.4f",
                position, validator_sketch_val, miner_sketch_val, mod_diff, tolerance,
                [f"{v:.4f}" for v in sample_vals], sample_buckets,
                float(validator_hidden.norm().item()),
            )

        return is_valid, diagnostics
