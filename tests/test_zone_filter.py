"""Tests for the DAPO/GRPO zone filter."""

from __future__ import annotations

import math

import pytest

from reliquary_inference.validator.zone_filter import (
    BOOTSTRAP_SIGMA_MIN,
    SIGMA_MIN,
    GroupKey,
    filter_groups,
    is_in_zone,
    rewards_std,
    zone_summary,
)


# ---------------------------------------------------------------------------
# rewards_std
# ---------------------------------------------------------------------------


def test_rewards_std_empty_returns_zero():
    assert rewards_std([]) == 0.0


def test_rewards_std_singleton_returns_zero():
    assert rewards_std([0.7]) == 0.0


def test_rewards_std_population_formula():
    # Population std of [0, 0, 1, 1] = sqrt(0.25) = 0.5.
    assert rewards_std([0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_rewards_std_all_equal_is_zero():
    assert rewards_std([1.0] * 8) == 0.0
    assert rewards_std([0.0] * 8) == 0.0


def test_rewards_std_bernoulli_k_equals_2_out_of_8():
    # σ of [1,1,0,0,0,0,0,0] = sqrt(p(1-p)) with p=0.25 => ≈ 0.4330
    sigma = rewards_std([1, 1, 0, 0, 0, 0, 0, 0])
    assert sigma == pytest.approx(math.sqrt(0.25 * 0.75), abs=1e-6)
    assert sigma >= SIGMA_MIN  # exactly at the boundary


def test_rewards_std_bernoulli_k_equals_1_out_of_8():
    # σ of [1,0,0,0,0,0,0,0] with p=0.125 => ≈ 0.3307
    sigma = rewards_std([1, 0, 0, 0, 0, 0, 0, 0])
    assert sigma == pytest.approx(math.sqrt(0.125 * 0.875), abs=1e-6)
    # Above bootstrap but below steady-state.
    assert sigma < SIGMA_MIN
    assert sigma >= BOOTSTRAP_SIGMA_MIN


# ---------------------------------------------------------------------------
# is_in_zone
# ---------------------------------------------------------------------------


def test_is_in_zone_steady_state_boundary():
    assert is_in_zone(SIGMA_MIN) is True
    assert is_in_zone(SIGMA_MIN - 1e-4) is False


def test_is_in_zone_zero_always_out():
    assert is_in_zone(0.0) is False
    assert is_in_zone(0.0, bootstrap=True) is False


def test_is_in_zone_bootstrap_loosens_threshold():
    # σ ≈ 0.35 is above bootstrap but below steady-state.
    assert is_in_zone(0.35, bootstrap=False) is False
    assert is_in_zone(0.35, bootstrap=True) is True


def test_is_in_zone_very_small_below_threshold():
    # Numerical noise at 1e-10 should be treated as "all equal" → out.
    assert is_in_zone(1e-10) is False


# ---------------------------------------------------------------------------
# filter_groups
# ---------------------------------------------------------------------------


def _verdict(miner_id: str, task_id: str, correctness: float, accepted: bool = True) -> dict:
    return {
        "payload": {
            "miner_id": miner_id,
            "task_id": task_id,
            "correctness": correctness,
            "accepted": accepted,
        },
    }


def test_filter_groups_bins_by_miner_and_task():
    verdicts = [
        _verdict("A", "t1", 1.0),
        _verdict("A", "t1", 0.0),
        _verdict("A", "t2", 1.0),
        _verdict("B", "t1", 0.0),
    ]
    groups = filter_groups(verdicts)
    assert set(groups.keys()) == {
        GroupKey("A", "t1"),
        GroupKey("A", "t2"),
        GroupKey("B", "t1"),
    }
    assert groups[GroupKey("A", "t1")].n == 2
    assert groups[GroupKey("A", "t2")].n == 1
    assert groups[GroupKey("B", "t1")].n == 1


def test_filter_groups_zone_split_binary_rewards():
    # Group "in-zone": k=4/8 → σ = 0.5 → passes.
    in_zone = [_verdict("A", "t1", float(i < 4)) for i in range(8)]
    # Group "out-of-zone": k=0/8 → σ = 0 → fails.
    out_of_zone = [_verdict("B", "t2", 0.0) for _ in range(8)]
    # Group "boundary": k=2/8 → σ ≈ 0.433 → passes (≥ 0.43).
    boundary = [_verdict("C", "t3", float(i < 2)) for i in range(8)]

    groups = filter_groups(in_zone + out_of_zone + boundary)
    assert groups[GroupKey("A", "t1")].in_zone is True
    assert groups[GroupKey("B", "t2")].in_zone is False
    assert groups[GroupKey("C", "t3")].in_zone is True


def test_filter_groups_bootstrap_accepts_looser():
    # k=1/8 → σ ≈ 0.331 — fails steady-state, passes bootstrap.
    verdicts = [_verdict("A", "t1", float(i < 1)) for i in range(8)]
    assert filter_groups(verdicts)[GroupKey("A", "t1")].in_zone is False
    assert filter_groups(verdicts, bootstrap=True)[GroupKey("A", "t1")].in_zone is True


def test_filter_groups_drops_rejected_by_default():
    # 4 accepted (k=4/8), 4 rejected (would be k=8/8 if included).
    verdicts = (
        [_verdict("A", "t1", 1.0, accepted=True) for _ in range(4)]
        + [_verdict("A", "t1", 0.0, accepted=True) for _ in range(4)]
        + [_verdict("A", "t1", 1.0, accepted=False) for _ in range(4)]
    )
    group = filter_groups(verdicts)[GroupKey("A", "t1")]
    assert group.n == 8
    assert group.mean_reward == 0.5


def test_filter_groups_includes_rejected_when_flag_off():
    verdicts = [
        _verdict("A", "t1", 1.0, accepted=True),
        _verdict("A", "t1", 0.0, accepted=False),
    ]
    group = filter_groups(verdicts, only_accepted=False)[GroupKey("A", "t1")]
    assert group.n == 2


def test_filter_groups_missing_ids_skipped():
    verdicts = [
        {"payload": {"task_id": "t1", "correctness": 1.0, "accepted": True}},  # no miner_id
        {"payload": {"miner_id": "A", "correctness": 1.0, "accepted": True}},  # no task_id
        {"payload": {"miner_id": "A", "task_id": "t1", "correctness": 1.0, "accepted": True}},
    ]
    groups = filter_groups(verdicts)
    assert len(groups) == 1
    assert GroupKey("A", "t1") in groups


def test_filter_groups_reads_producer_id_fallback():
    """Some verdict payloads carry producer_id instead of miner_id."""
    verdicts = [
        {
            "payload": {"task_id": "t1", "correctness": 1.0, "accepted": True},
            "producer_id": "A",
        },
    ]
    groups = filter_groups(verdicts)
    assert GroupKey("A", "t1") in groups


# ---------------------------------------------------------------------------
# zone_summary
# ---------------------------------------------------------------------------


def test_zone_summary_shape_and_counts():
    groups = filter_groups(
        [_verdict("A", "t1", float(i < 4)) for i in range(8)]
        + [_verdict("B", "t2", 0.0) for _ in range(8)]
    )
    s = zone_summary(groups)
    assert s["sigma_min"] == SIGMA_MIN
    assert s["bootstrap"] is False
    assert s["total_groups"] == 2
    assert s["in_zone_groups"] == 1
    assert s["out_of_zone_groups"] == 1
    assert len(s["groups"]) == 2
    keys = {g["key"] for g in s["groups"]}
    assert keys == {"A:t1", "B:t2"}


def test_zone_summary_bootstrap_flag():
    s = zone_summary({}, bootstrap=True)
    assert s["sigma_min"] == BOOTSTRAP_SIGMA_MIN
    assert s["bootstrap"] is True
