"""Tests characterizing the sketch layer under mismatched inputs.

The sketch layer's primary guarantee is **deterministic, position-dependent
bit-exact replay** of a known hidden state. Detection of hidden-state
tampering, cross-prompt replay, and model substitution is delegated to the
full 9-stage verifier pipeline (01_TIER1_PRD.md Epic 2) which composes the
sketch check with token binding, termination validation, logprob replay,
and distribution validation.

These tests codify **what sketch alone provides**:
  - honest roundtrip → bit-exact sketch match → acceptance at zero diff;
  - position-dependent tolerance function that grows as sqrt(position);
  - acceptance behavior at far-future positions where tolerance has grown.

They intentionally DO NOT assert single-stage rejection of arbitrary
hidden-state tampering. Observed per-position sketch variance under random
r_vec coefficients is O(2000), so tolerance 6000 absorbs typical random
shifts. Stronger rejection semantics ride in higher stages.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.protocol.constants import CHALLENGE_K
from reliquary_inference.protocol.sketch_verifier import SketchProofVerifier

HIDDEN_DIM = 256
SEED_RANDOMNESS = "00000000000000000000000000000000000000000000000000000000deadbeef"


def _make_hidden_state(seed: int, hidden_dim: int = HIDDEN_DIM, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(hidden_dim, generator=generator, dtype=torch.float32) * scale


def _make_rollout(base_seed: int, seq_len: int = CHALLENGE_K, scale: float = 1.0) -> torch.Tensor:
    return torch.stack(
        [_make_hidden_state(seed=base_seed + i, scale=scale) for i in range(seq_len)]
    )


def _make_verifier() -> SketchProofVerifier:
    return SketchProofVerifier(hidden_dim=HIDDEN_DIM)


def test_honest_rollout_accepts_all_challenge_positions() -> None:
    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)
    h_layer = _make_rollout(base_seed=100)

    commits = verifier.create_commitments_batch(h_layer, r_vec)
    for pos in range(CHALLENGE_K):
        is_valid, diagnostics = verifier.verify_commitment(
            h_layer[pos],
            commits[pos],
            r_vec,
            sequence_length=CHALLENGE_K,
            position=pos,
        )
        assert is_valid, (pos, diagnostics)


def test_per_position_honest_has_zero_sketch_diff() -> None:
    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)
    h_layer = _make_rollout(base_seed=200, scale=1.0)

    commits = verifier.create_commitments_batch(h_layer, r_vec)
    for pos in range(CHALLENGE_K):
        _, diagnostics = verifier.verify_commitment(
            h_layer[pos],
            commits[pos],
            r_vec,
            sequence_length=CHALLENGE_K,
            position=pos,
        )
        assert diagnostics["sketch_diff"] == 0, (pos, diagnostics)


def test_tolerance_grows_with_position_per_spec() -> None:
    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)
    hidden = _make_hidden_state(seed=700)
    commitment = verifier.create_commitment(hidden, r_vec)

    _, diag_zero = verifier.verify_commitment(
        hidden, commitment, r_vec, sequence_length=5000, position=0
    )
    _, diag_4096 = verifier.verify_commitment(
        hidden, commitment, r_vec, sequence_length=5000, position=4096
    )
    assert diag_zero["sketch_tolerance"] < diag_4096["sketch_tolerance"]
    assert diag_zero["sketch_tolerance"] == 6000
    assert diag_4096["sketch_tolerance"] == 6000 + int(5.0 * (4096 ** 0.5))


def test_far_future_position_accepts_honest_commit() -> None:
    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)

    hidden = _make_hidden_state(seed=900, scale=2.0)
    commit = verifier.create_commitment(hidden, r_vec)
    is_valid, diagnostics = verifier.verify_commitment(
        hidden, commit, r_vec, sequence_length=32_768, position=32_767
    )
    assert is_valid, diagnostics


def test_diagnostics_contain_spec_required_fields() -> None:
    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)
    hidden = _make_hidden_state(seed=1000)
    commit = verifier.create_commitment(hidden, r_vec)
    _, diag = verifier.verify_commitment(
        hidden, commit, r_vec, sequence_length=1, position=0
    )
    for key in (
        "sketch_diff",
        "sketch_valid",
        "sketch_tolerance",
        "overall_valid",
        "validator_sketch",
        "miner_sketch",
        "position",
    ):
        assert key in diag, f"missing key in verify_commitment diagnostics: {key}"


def test_miner_and_validator_sketch_in_modular_range() -> None:
    """Both reported sketches must lie in [0, PRIME_Q)."""
    from reliquary_inference.protocol.constants import PRIME_Q

    verifier = _make_verifier()
    r_vec = verifier.generate_r_vec(SEED_RANDOMNESS)
    for seed in range(8):
        hidden = _make_hidden_state(seed=seed, scale=1.0 + seed)
        commit = verifier.create_commitment(hidden, r_vec)
        _, diag = verifier.verify_commitment(
            hidden, commit, r_vec, sequence_length=1, position=0
        )
        assert 0 <= diag["miner_sketch"] < PRIME_Q, diag
        assert 0 <= diag["validator_sketch"] < PRIME_Q, diag
