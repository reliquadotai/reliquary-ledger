"""Torch-based determinism tests for the proof sketch computation.

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md invariants 1, 3, 4.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.protocol.sketch_verifier import (
    SketchProofVerifier,
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
)

HIDDEN_DIM = 256
SEED_RANDOMNESS = "00000000000000000000000000000000000000000000000000000000deadbeef"


def _make_hidden_state(seed: int, hidden_dim: int = HIDDEN_DIM) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(hidden_dim, generator=generator, dtype=torch.float32)


def _make_verifier() -> SketchProofVerifier:
    return SketchProofVerifier(hidden_dim=HIDDEN_DIM)


def test_create_commitment_is_bit_exact_over_ten_repeats() -> None:
    verifier = _make_verifier()
    hidden = _make_hidden_state(seed=42)
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)

    sketches = [verifier.create_commitment(hidden, r_vec)["sketch"] for _ in range(10)]
    assert len(set(sketches)) == 1, f"determinism broken across repeats: {sketches}"


def test_scalar_and_vectorized_bucketing_agree() -> None:
    torch.manual_seed(7)
    values = torch.randn(512, dtype=torch.float32) * 3.0
    vectorized = log_magnitude_bucket_vectorized(values)
    scalar = torch.tensor(
        [log_magnitude_bucket(v.item()) for v in values],
        dtype=torch.int64,
    )
    assert torch.equal(vectorized, scalar), "vectorized bucketing diverges from scalar path"


def test_bucket_handles_nan_inf_and_deadzone() -> None:
    assert log_magnitude_bucket(float("nan")) == 0
    assert log_magnitude_bucket(float("inf")) == 7
    assert log_magnitude_bucket(float("-inf")) == -7
    assert log_magnitude_bucket(5e-7) == 0
    assert log_magnitude_bucket(-5e-7) == 0


def test_bucket_preserves_sign() -> None:
    positive_bucket = log_magnitude_bucket(3.5)
    negative_bucket = log_magnitude_bucket(-3.5)
    assert positive_bucket > 0
    assert negative_bucket < 0
    assert abs(positive_bucket) == abs(negative_bucket)


def test_bucket_monotonic_in_magnitude_per_sign() -> None:
    magnitudes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    positive_buckets = [log_magnitude_bucket(m) for m in magnitudes]
    for earlier, later in zip(positive_buckets, positive_buckets[1:]):
        assert earlier <= later


def test_batch_matches_per_position_loop_bit_exact() -> None:
    verifier = _make_verifier()
    torch.manual_seed(11)
    seq_len = 8
    h_layer = torch.randn(seq_len, HIDDEN_DIM, dtype=torch.float32)
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)

    batch_commitments = verifier.create_commitments_batch(h_layer, r_vec)
    loop_commitments = [
        verifier.create_commitment(h_layer[pos], r_vec) for pos in range(seq_len)
    ]

    assert len(batch_commitments) == seq_len
    for pos in range(seq_len):
        assert batch_commitments[pos]["sketch"] == loop_commitments[pos]["sketch"], (
            f"position {pos}: batch={batch_commitments[pos]} loop={loop_commitments[pos]}"
        )


def test_r_vec_is_deterministic_given_same_randomness() -> None:
    verifier = _make_verifier()
    r1 = verifier.generate_r_vec(SEED_RANDOMNESS)
    r2 = verifier.generate_r_vec(SEED_RANDOMNESS)
    assert torch.equal(r1, r2)


def test_r_vec_differs_for_different_randomness() -> None:
    verifier = _make_verifier()
    r1 = verifier.generate_r_vec(SEED_RANDOMNESS)
    r2 = verifier.generate_r_vec("00" + SEED_RANDOMNESS[2:-2] + "ff")
    assert not torch.equal(r1, r2)


def test_sketch_always_in_modular_range() -> None:
    from reliquary_inference.protocol.constants import PRIME_Q

    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)

    for seed in range(5):
        hidden = _make_hidden_state(seed=seed)
        sketch = verifier.create_commitment(hidden, r_vec)["sketch"]
        assert 0 <= sketch < PRIME_Q
