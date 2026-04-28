"""Acceptance tests for the sqrt-scaled sketch tolerance function.

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md, acceptance test 4.
"""

from __future__ import annotations

import math

import pytest

from reliquary_inference.protocol.constants import (
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
)
from reliquary_inference.protocol.sketch_verifier import adaptive_sketch_tolerance


# Tolerance base tightened from 6000 to 1000 on 2026-04-28 after empirical
# calibration on staging2 RTX 6000B Blackwell (n=24, honest=0 floor). The
# formula and growth coefficient are unchanged; only the base shifts.
@pytest.mark.parametrize(
    "position, expected",
    [
        (0, 1000),
        (256, 1000 + int(5.0 * math.sqrt(256))),
        (1024, 1000 + int(5.0 * math.sqrt(1024))),
        (4096, 1000 + int(5.0 * math.sqrt(4096))),
    ],
)
def test_tolerance_matches_spec_formula(position: int, expected: int) -> None:
    assert adaptive_sketch_tolerance(position, sequence_length=position + 1) == expected


def test_tolerance_at_origin_equals_base() -> None:
    assert adaptive_sketch_tolerance(0, sequence_length=1) == PROOF_SKETCH_TOLERANCE_BASE


def test_tolerance_is_monotonically_non_decreasing() -> None:
    positions = [0, 1, 4, 16, 64, 256, 1024, 4096, 8192]
    tolerances = [adaptive_sketch_tolerance(p, sequence_length=p + 1) for p in positions]
    for earlier, later in zip(tolerances, tolerances[1:]):
        assert earlier <= later


def test_tolerance_growth_rate_matches_spec_coefficient() -> None:
    pos = 10_000
    t = adaptive_sketch_tolerance(pos, sequence_length=pos + 1)
    expected = int(PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH * math.sqrt(pos))
    assert t == expected
