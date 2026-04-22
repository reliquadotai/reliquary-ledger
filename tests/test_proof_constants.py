"""Acceptance tests for Reliquary Ledger proof protocol constants.

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md invariants 1-9.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reliquary_inference import constants as root_constants
from reliquary_inference.protocol import constants as protocol_constants

_PROTOCOL_CONSTANTS_PATH = Path(protocol_constants.__file__).resolve()


def test_protocol_constants_are_immutable_by_construction() -> None:
    source = _PROTOCOL_CONSTANTS_PATH.read_text()
    for banned in ("os.getenv", "os.environ", "getenv("):
        assert banned not in source, (
            f"protocol/constants.py contains {banned!r}; "
            "protocol constants are mathematical parameters, not runtime knobs"
        )


def test_ledger_proof_version_is_v1() -> None:
    assert protocol_constants.LEDGER_PROOF_VERSION == "v1"
    assert root_constants.PROOF_VERSION == "v1"
    assert root_constants.LEDGER_PROOF_VERSION == "v1"


def test_prime_q_is_mersenne_m31() -> None:
    assert protocol_constants.PRIME_Q == (1 << 31) - 1
    assert protocol_constants.PRIME_Q == 2_147_483_647


def test_challenge_and_sketch_shape() -> None:
    assert protocol_constants.CHALLENGE_K == 32
    assert protocol_constants.PROOF_TOPK == 16
    assert protocol_constants.PROOF_NUM_BUCKETS == 8
    assert protocol_constants.PROOF_COEFF_RANGE == 127


def test_tolerance_parameters() -> None:
    assert protocol_constants.PROOF_SKETCH_TOLERANCE_BASE == 6000
    assert protocol_constants.PROOF_SKETCH_TOLERANCE_GROWTH == pytest.approx(5.0)


def test_attention_kernel_is_flash_attention_2() -> None:
    assert protocol_constants.ATTN_IMPLEMENTATION == "flash_attention_2"


def test_layer_index_is_last_layer() -> None:
    assert protocol_constants.LAYER_INDEX == -1


def test_root_constants_reexport_matches_protocol() -> None:
    for name in protocol_constants._PROTOCOL_CONSTANTS:
        assert getattr(root_constants, name) == getattr(protocol_constants, name), (
            f"root constants.{name} diverges from protocol/constants.{name}"
        )
