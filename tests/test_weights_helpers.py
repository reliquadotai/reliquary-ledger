"""Expanded tests for reliquary_inference.validator.weights.compute_weights.

test_scoring.py covers only a single positive-case happy path; this
module pins the full contract (normalization, cap behavior, zero
floors, empty input, Sybil-resistance exponent semantics).
"""

from __future__ import annotations

import pytest

from reliquary_inference.constants import (
    SUPERLINEAR_EXPONENT,
    UNIQUE_ROLLOUTS_CAP,
)
from reliquary_inference.validator.weights import compute_weights


def test_compute_weights_empty_input_returns_empty() -> None:
    assert compute_weights({}) == {}


def test_compute_weights_sum_to_one_when_any_positive() -> None:
    weights = compute_weights(
        {
            "miner-A": {"unique": 5, "valid": 5},
            "miner-B": {"unique": 2, "valid": 2},
        }
    )
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0


def test_compute_weights_zero_unique_is_zero_weight() -> None:
    weights = compute_weights(
        {
            "miner-A": {"unique": 0, "valid": 10},
            "miner-B": {"unique": 5, "valid": 5},
        }
    )
    assert weights["miner-A"] == 0.0
    assert weights["miner-B"] == 1.0


def test_compute_weights_zero_valid_is_zero_weight() -> None:
    weights = compute_weights(
        {
            "miner-A": {"unique": 5, "valid": 0},
            "miner-B": {"unique": 5, "valid": 5},
        }
    )
    assert weights["miner-A"] == 0.0
    assert weights["miner-B"] == 1.0


def test_compute_weights_all_zeros_returns_zero_for_every_miner() -> None:
    miners = {
        "miner-A": {"unique": 0, "valid": 0},
        "miner-B": {"unique": 0, "valid": 0},
    }
    weights = compute_weights(miners)
    assert weights == {"miner-A": 0.0, "miner-B": 0.0}


def test_compute_weights_missing_fields_default_to_zero() -> None:
    weights = compute_weights({"miner-A": {}})
    assert weights == {"miner-A": 0.0}


def test_compute_weights_single_miner_gets_full_mass() -> None:
    weights = compute_weights({"miner-only": {"unique": 7, "valid": 7}})
    assert weights == {"miner-only": 1.0}


def test_compute_weights_superlinear_exponent_concentrates_mass() -> None:
    miners = {
        "miner-big": {"unique": 10, "valid": 10},
        "miner-small": {"unique": 5, "valid": 5},
    }
    linear = compute_weights(miners, superlinear_exponent=1.0)
    cubed = compute_weights(miners, superlinear_exponent=3.0)
    assert cubed["miner-big"] > linear["miner-big"]


def test_compute_weights_applies_unique_cap_when_enabled() -> None:
    miners = {
        "miner-capped": {"unique": UNIQUE_ROLLOUTS_CAP + 100, "valid": UNIQUE_ROLLOUTS_CAP + 100},
        "miner-ceiling": {"unique": UNIQUE_ROLLOUTS_CAP, "valid": UNIQUE_ROLLOUTS_CAP},
    }
    weights = compute_weights(miners, cap_enabled=True)
    assert weights["miner-capped"] == pytest.approx(weights["miner-ceiling"])


def test_compute_weights_cap_disabled_preserves_raw_count() -> None:
    miners = {
        "miner-big": {"unique": UNIQUE_ROLLOUTS_CAP + 100, "valid": 1},
        "miner-small": {"unique": UNIQUE_ROLLOUTS_CAP, "valid": 1},
    }
    weights = compute_weights(miners, cap_enabled=False)
    assert weights["miner-big"] > weights["miner-small"]


def test_compute_weights_uses_default_superlinear_exponent() -> None:
    # Default exponent is SUPERLINEAR_EXPONENT; swapping to 1.0 must change the shape.
    miners = {
        "miner-A": {"unique": 10, "valid": 10},
        "miner-B": {"unique": 3, "valid": 3},
    }
    default_weights = compute_weights(miners)
    linear_weights = compute_weights(miners, superlinear_exponent=1.0)
    if SUPERLINEAR_EXPONENT != 1.0:
        assert default_weights["miner-A"] != pytest.approx(linear_weights["miner-A"])


def test_compute_weights_weights_are_non_negative() -> None:
    weights = compute_weights(
        {
            "miner-A": {"unique": 5, "valid": 5},
            "miner-B": {"unique": 0, "valid": 5},
        }
    )
    for hotkey, weight in weights.items():
        assert weight >= 0.0, f"{hotkey}: {weight}"


def test_compute_weights_tied_miners_get_equal_weight() -> None:
    weights = compute_weights(
        {
            "miner-A": {"unique": 7, "valid": 7},
            "miner-B": {"unique": 7, "valid": 7},
        }
    )
    assert weights["miner-A"] == pytest.approx(weights["miner-B"])
    assert weights["miner-A"] == pytest.approx(0.5)


def test_compute_weights_many_miners_normalization() -> None:
    miners = {f"miner-{i}": {"unique": i + 1, "valid": i + 1} for i in range(8)}
    weights = compute_weights(miners)
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
