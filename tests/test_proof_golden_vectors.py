"""Golden-vector regression test for the proof sketch.

Locks in bit-exact sketch outputs for a fixed set of ``(seed, scale, r_vec)``
inputs. Any future change that alters the sketch computation (bucketing,
dot-product precision, modular reduction order, or r_vec derivation) will
flip at least one of these vectors and fail loudly.

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md
acceptance test 1 (determinism) and invariant 9 (version binding).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.protocol.constants import LEDGER_PROOF_VERSION
from reliquary_inference.protocol.sketch_verifier import SketchProofVerifier

HIDDEN_DIM = 128
RANDOMNESS_HEX = "00000000000000000000000000000000000000000000000000000000deadbeef"


GOLDEN_SINGLE_POSITION = {
    (0, 1.0): 2147483323,
    (0, 2.5): 2147483319,
    (0, 8.0): 2147482675,
    (1, 1.0): 2147483143,
    (1, 2.5): 2147483064,
    (1, 8.0): 2147482135,
    (7, 1.0): 194,
    (7, 2.5): 439,
    (7, 8.0): 716,
    (42, 1.0): 2147483631,
    (42, 2.5): 2147483634,
    (42, 8.0): 2147483550,
    (100, 1.0): 2147483361,
    (100, 2.5): 2147483303,
    (100, 8.0): 2147482789,
}


GOLDEN_BATCH_ROLLOUT = [
    2147483559,
    2147483144,
    145,
    2147483186,
    2147483175,
    894,
    1216,
    2147483319,
]


def _make_hidden_state(seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(HIDDEN_DIM, generator=generator, dtype=torch.float32) * scale


def _verifier() -> SketchProofVerifier:
    return SketchProofVerifier(hidden_dim=HIDDEN_DIM)


def test_protocol_version_matches_golden_fixture() -> None:
    """If the protocol version changes, these golden vectors must be regenerated."""
    assert LEDGER_PROOF_VERSION == "v5", (
        f"Protocol version changed to {LEDGER_PROOF_VERSION!r}; regenerate "
        "GOLDEN_SINGLE_POSITION and GOLDEN_BATCH_ROLLOUT against the new algorithm."
    )


@pytest.mark.parametrize("seed,scale,expected_sketch", [
    (seed, scale, sketch) for (seed, scale), sketch in GOLDEN_SINGLE_POSITION.items()
])
def test_single_position_sketch_matches_golden_vector(
    seed: int, scale: float, expected_sketch: int
) -> None:
    verifier = _verifier()
    r_vec = verifier.generate_r_vec(RANDOMNESS_HEX)
    hidden = _make_hidden_state(seed=seed, scale=scale)
    sketch = verifier.create_commitment(hidden, r_vec)["sketch"]
    assert sketch == expected_sketch, (
        f"sketch regression: seed={seed}, scale={scale}, got {sketch}, expected {expected_sketch}"
    )


def test_batch_rollout_matches_golden_vector() -> None:
    verifier = _verifier()
    r_vec = verifier.generate_r_vec(RANDOMNESS_HEX)
    generator = torch.Generator()
    generator.manual_seed(1234)
    h_layer = torch.randn(8, HIDDEN_DIM, generator=generator, dtype=torch.float32) * 3.0

    sketches = [c["sketch"] for c in verifier.create_commitments_batch(h_layer, r_vec)]
    assert sketches == GOLDEN_BATCH_ROLLOUT, (
        f"batch sketch regression: got {sketches}, expected {GOLDEN_BATCH_ROLLOUT}"
    )


def test_r_vec_first_eight_coefficients_are_frozen() -> None:
    """Locks in the PRF-derived r_vec. Any change to crypto.prf, RNG_LABEL, or
    the generate_r_vec algorithm will flip this test."""
    verifier = _verifier()
    r_vec = verifier.generate_r_vec(RANDOMNESS_HEX)
    expected_prefix = [64, 99, -85, 94, 121, 30, -122, -35]
    assert r_vec[:8].tolist() == expected_prefix, (
        f"r_vec PRF regression: got {r_vec[:8].tolist()}, expected {expected_prefix}"
    )
