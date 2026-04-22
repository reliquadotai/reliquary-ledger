"""Drift-alignment tests against the shared ``reliquary-protocol`` package.

If this breaks, either a Ledger constant drifted from the shared authoritative
value without bumping reliquary_protocol.VERSION, or the shared package bumped
and this subnet didn't follow. Either way, the fix is to align both sides and
bump the shared version together — never relax this test.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

reliquary_protocol = pytest.importorskip(
    "reliquary_protocol",
    reason="shared reliquary-protocol package not installed",
)

from reliquary_inference import constants as root_constants
from reliquary_inference.protocol import constants as subnet_constants
from reliquary_inference.utils.json_io import sha256_json as subnet_sha256_json
from reliquary_inference.utils.json_io import (
    stable_json_dumps as subnet_stable_json_dumps,
)

# ---------------------------------------------------------------------------
# Canonicalization alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"a": 1},
        {"z": 1, "a": 2, "m": 3},
        {"nested": {"z": 1, "a": 2}},
        {"unicode": "héllo"},
        {"nums": [1, 2, 3], "flag": True, "zero": 0},
    ],
)
def test_stable_json_dumps_matches_shared(payload: dict) -> None:
    assert subnet_stable_json_dumps(payload) == reliquary_protocol.stable_json_dumps(
        payload
    )


def test_sha256_json_matches_shared_for_verdict_envelope() -> None:
    payload = {
        "accepted": True,
        "completion_id": "c-1",
        "miner_hotkey": "m-1",
        "window_id": 42,
        "scores": {"coherence": 0.9, "reward": 1.0},
    }
    assert subnet_sha256_json(payload) == reliquary_protocol.sha256_json(payload)


# ---------------------------------------------------------------------------
# Proof-layer constants
# ---------------------------------------------------------------------------


def test_prime_q_matches_shared() -> None:
    assert subnet_constants.PRIME_Q == reliquary_protocol.PRIME_Q


def test_challenge_k_matches_shared() -> None:
    assert subnet_constants.CHALLENGE_K == reliquary_protocol.CHALLENGE_K


def test_proof_topk_matches_shared() -> None:
    assert subnet_constants.PROOF_TOPK == reliquary_protocol.PROOF_TOPK


def test_proof_num_buckets_matches_shared() -> None:
    assert subnet_constants.PROOF_NUM_BUCKETS == reliquary_protocol.PROOF_NUM_BUCKETS


def test_proof_coeff_range_matches_shared() -> None:
    assert subnet_constants.PROOF_COEFF_RANGE == reliquary_protocol.PROOF_COEFF_RANGE


def test_proof_sketch_tolerance_matches_shared() -> None:
    assert (
        subnet_constants.PROOF_SKETCH_TOLERANCE_BASE
        == reliquary_protocol.PROOF_SKETCH_TOLERANCE_BASE
    )
    assert (
        subnet_constants.PROOF_SKETCH_TOLERANCE_GROWTH
        == reliquary_protocol.PROOF_SKETCH_TOLERANCE_GROWTH
    )


def test_layer_index_matches_shared() -> None:
    assert subnet_constants.LAYER_INDEX == reliquary_protocol.LAYER_INDEX


def test_attn_implementation_matches_shared() -> None:
    assert (
        subnet_constants.ATTN_IMPLEMENTATION
        == reliquary_protocol.ATTN_IMPLEMENTATION
    )


def test_ledger_proof_version_matches_shared() -> None:
    assert (
        subnet_constants.LEDGER_PROOF_VERSION
        == reliquary_protocol.LEDGER_PROOF_VERSION
    )


# ---------------------------------------------------------------------------
# Mesh consensus + copycat + distribution
# ---------------------------------------------------------------------------


def test_mesh_consensus_values_match_shared() -> None:
    assert (
        subnet_constants.MESH_STAKE_CAP_FRACTION
        == reliquary_protocol.MESH_STAKE_CAP_FRACTION
    )
    assert (
        subnet_constants.MESH_MIN_QUORUM_STAKE_FRACTION
        == reliquary_protocol.MESH_MIN_QUORUM_STAKE_FRACTION
    )
    assert (
        subnet_constants.MESH_OUTLIER_THRESHOLD
        == reliquary_protocol.MESH_OUTLIER_THRESHOLD
    )
    assert (
        subnet_constants.MESH_OUTLIER_RATE_GATE
        == reliquary_protocol.MESH_OUTLIER_RATE_GATE
    )


def test_copycat_values_match_shared() -> None:
    assert (
        subnet_constants.COPYCAT_WINDOW_THRESHOLD
        == reliquary_protocol.COPYCAT_WINDOW_THRESHOLD
    )
    assert (
        subnet_constants.COPYCAT_INTERVAL_THRESHOLD
        == reliquary_protocol.COPYCAT_INTERVAL_THRESHOLD
    )
    assert (
        subnet_constants.COPYCAT_INTERVAL_LENGTH
        == reliquary_protocol.COPYCAT_INTERVAL_LENGTH
    )
    assert (
        subnet_constants.COPYCAT_GATE_DURATION_WINDOWS
        == reliquary_protocol.COPYCAT_GATE_DURATION_WINDOWS
    )


def test_logprob_and_distribution_values_match_shared() -> None:
    assert (
        subnet_constants.LOGPROB_DRIFT_THRESHOLD
        == reliquary_protocol.LOGPROB_DRIFT_THRESHOLD
    )
    assert (
        subnet_constants.LOGPROB_DRIFT_QUORUM
        == reliquary_protocol.LOGPROB_DRIFT_QUORUM
    )
    assert (
        subnet_constants.DISTRIBUTION_RATIO_BAND_HIGH
        == reliquary_protocol.DISTRIBUTION_RATIO_BAND_HIGH
    )
    assert (
        subnet_constants.DISTRIBUTION_RATIO_BAND_LOW
        == reliquary_protocol.DISTRIBUTION_RATIO_BAND_LOW
    )
    assert (
        subnet_constants.DISTRIBUTION_MIN_POSITIONS
        == reliquary_protocol.DISTRIBUTION_MIN_POSITIONS
    )


def test_rng_labels_match_shared() -> None:
    # The subnet's root constants module exposes RNG_LABEL with same shape.
    assert reliquary_protocol.RNG_LABEL["sketch"] == b"sketch"
    assert reliquary_protocol.RNG_LABEL["open"] == b"open"
    assert reliquary_protocol.RNG_LABEL["task"] == b"task"
    if hasattr(root_constants, "RNG_LABEL"):
        for key, value in reliquary_protocol.RNG_LABEL.items():
            assert root_constants.RNG_LABEL.get(key) == value


# ---------------------------------------------------------------------------
# Signature primitive alignment
# ---------------------------------------------------------------------------


def test_hmac_sign_matches_stdlib() -> None:
    expected = hmac.new(b"secret", b"data", hashlib.sha256).hexdigest()
    assert reliquary_protocol.hmac_sign(b"data", "secret") == expected


def test_escape_label_value_matches_subnet_mesh_observability() -> None:
    """The subnet's mesh_observability module escapes labels identically.
    This test pins that same bytes come out of the shared helper."""
    raw = 'node"hostile\\name'
    assert reliquary_protocol.escape_label_value(raw) == 'node\\"hostile\\\\name'


def test_artifact_type_enum_covers_inference_wire_vocab() -> None:
    expected = {"verdict", "completion", "scorecard", "window_manifest"}
    assert expected <= set(reliquary_protocol.ArtifactTypes)


# ---------------------------------------------------------------------------
# Version presence
# ---------------------------------------------------------------------------


def test_shared_package_exposes_version_string() -> None:
    assert isinstance(reliquary_protocol.VERSION, str)
    assert reliquary_protocol.VERSION.count(".") >= 2  # semver-shaped
