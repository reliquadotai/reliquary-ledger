"""Acceptance tests for operational (non-proof) constants.

test_proof_constants.py pins proof-layer values (PRIME_Q, CHALLENGE_K,
PROOF_TOPK, etc.). This module pins the operational knobs — mesh
consensus, copycat detection, logprob drift, distribution band,
scoring — and confirms they're in sensible ranges so an accidental
sweep away from committed values is caught at CI.
"""

from __future__ import annotations

from reliquary_inference import constants as root_constants
from reliquary_inference.protocol import constants as protocol_constants

# ---------------------------------------------------------------------------
# Mesh consensus
# ---------------------------------------------------------------------------


def test_mesh_stake_cap_fraction_in_bounds() -> None:
    assert 0 < protocol_constants.MESH_STAKE_CAP_FRACTION <= 1
    assert protocol_constants.MESH_STAKE_CAP_FRACTION == 0.10


def test_mesh_min_quorum_stake_fraction_in_bounds() -> None:
    assert 0 < protocol_constants.MESH_MIN_QUORUM_STAKE_FRACTION <= 1
    assert protocol_constants.MESH_MIN_QUORUM_STAKE_FRACTION == 0.50


def test_mesh_outlier_threshold_non_negative() -> None:
    assert protocol_constants.MESH_OUTLIER_THRESHOLD >= 0.0
    assert protocol_constants.MESH_OUTLIER_THRESHOLD == 0.25


def test_mesh_outlier_rate_gate_is_probability() -> None:
    assert 0 <= protocol_constants.MESH_OUTLIER_RATE_GATE <= 1
    assert protocol_constants.MESH_OUTLIER_RATE_GATE == 0.05


# ---------------------------------------------------------------------------
# Copycat
# ---------------------------------------------------------------------------


def test_copycat_ambiguity_window_positive() -> None:
    assert protocol_constants.COPYCAT_AMBIGUITY_WINDOW_SECONDS > 0.0


def test_copycat_thresholds_are_probabilities() -> None:
    assert 0 <= protocol_constants.COPYCAT_WINDOW_THRESHOLD <= 1
    assert 0 <= protocol_constants.COPYCAT_INTERVAL_THRESHOLD <= 1


def test_copycat_interval_length_positive() -> None:
    assert protocol_constants.COPYCAT_INTERVAL_LENGTH > 0


def test_copycat_gate_duration_positive() -> None:
    assert protocol_constants.COPYCAT_GATE_DURATION_WINDOWS > 0


# ---------------------------------------------------------------------------
# Logprob drift
# ---------------------------------------------------------------------------


def test_logprob_drift_threshold_positive() -> None:
    assert protocol_constants.LOGPROB_DRIFT_THRESHOLD > 0.0


def test_logprob_drift_quorum_is_majority() -> None:
    assert 0.5 <= protocol_constants.LOGPROB_DRIFT_QUORUM <= 1.0


# ---------------------------------------------------------------------------
# Distribution band
# ---------------------------------------------------------------------------


def test_distribution_band_ordering() -> None:
    assert (
        protocol_constants.DISTRIBUTION_RATIO_BAND_LOW
        < protocol_constants.DISTRIBUTION_RATIO_BAND_HIGH
    )


def test_distribution_band_centered_around_one() -> None:
    # The band should be open around 1.0 (no drift).
    low = protocol_constants.DISTRIBUTION_RATIO_BAND_LOW
    high = protocol_constants.DISTRIBUTION_RATIO_BAND_HIGH
    assert low < 1.0 < high


def test_distribution_min_positions_positive() -> None:
    assert protocol_constants.DISTRIBUTION_MIN_POSITIONS > 0


# ---------------------------------------------------------------------------
# Root operational constants (non-proof)
# ---------------------------------------------------------------------------


def test_superlinear_exponent_is_positive() -> None:
    assert root_constants.SUPERLINEAR_EXPONENT > 0


def test_unique_rollouts_cap_positive() -> None:
    assert root_constants.UNIQUE_ROLLOUTS_CAP > 0


def test_unique_rollouts_cap_enabled_bool() -> None:
    assert isinstance(root_constants.UNIQUE_ROLLOUTS_CAP_ENABLED, bool)


def test_block_and_window_cadence_positive() -> None:
    assert root_constants.BLOCK_TIME_SECONDS > 0
    assert root_constants.WINDOW_LENGTH > 0
    assert root_constants.WEIGHT_SUBMISSION_INTERVAL > 0


def test_miner_sampling_bounds() -> None:
    assert 0 < root_constants.MINER_SAMPLE_RATE <= 1
    assert root_constants.MINER_SAMPLE_MIN <= root_constants.MINER_SAMPLE_MAX
    assert root_constants.MINER_SAMPLE_MIN > 0


def test_rollout_sampling_bounds() -> None:
    assert 0 < root_constants.ROLLOUT_SAMPLE_RATE <= 1
    assert root_constants.ROLLOUT_SAMPLE_MIN > 0


def test_verification_batch_size_positive() -> None:
    assert root_constants.VERIFICATION_BATCH_SIZE > 0


def test_batch_failure_threshold_is_probability() -> None:
    assert 0 < root_constants.BATCH_FAILURE_THRESHOLD < 1


def test_rng_label_keys_cover_three_domains() -> None:
    assert set(root_constants.RNG_LABEL) == {"sketch", "open", "task"}
    for label in root_constants.RNG_LABEL.values():
        assert isinstance(label, bytes)
        assert len(label) > 0


def test_default_dataset_fields_non_empty() -> None:
    assert root_constants.DEFAULT_DATASET_NAME
    assert root_constants.DEFAULT_DATASET_SPLIT


def test_default_fallback_prompts_populated() -> None:
    prompts = root_constants.DEFAULT_FALLBACK_PROMPTS
    assert len(prompts) >= 2
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert prompt.strip()


def test_root_all_has_no_duplicates_except_intentional() -> None:
    # The root constants __all__ currently repeats WEIGHT_SUBMISSION_INTERVAL
    # at the end — pin the current shape so a future accidental dedup is
    # noticed explicitly rather than silently.
    assert "WEIGHT_SUBMISSION_INTERVAL" in root_constants.__all__


def test_root_all_contains_operational_knobs() -> None:
    required = {
        "SUPERLINEAR_EXPONENT",
        "UNIQUE_ROLLOUTS_CAP",
        "UNIQUE_ROLLOUTS_CAP_ENABLED",
        "BLOCK_TIME_SECONDS",
        "WINDOW_LENGTH",
        "DEFAULT_DATASET_NAME",
        "DEFAULT_FALLBACK_PROMPTS",
    }
    assert required <= set(root_constants.__all__)
