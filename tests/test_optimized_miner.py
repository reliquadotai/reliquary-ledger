"""Tests for the optimized miner reference.

Pure-python tests for the prompt-selection + scoring + local-σ-gate
logic that don't require GPUs or real models. The forward-pass-based
``score_prompt`` is exercised through dependency injection: tests
inject a fake scorer so the selection logic can be verified
independently.
"""

from __future__ import annotations

from typing import Any

import pytest

from reliquary_inference.miner.optimized_engine import (
    OptimizedMiningEngine,
    _candidate_prompt_text,
    _candidate_task_id,
    _normalize_entropy_to_unit_interval,
)


class _StubEngine:
    """Stand-in exposing just the methods select_prompts needs; lets
    us test the selection logic without instantiating MiningEngine
    (which loads a real model).
    """

    def __init__(self, scorer):
        self._scorer = scorer

    score_prompt = property(lambda self: self._scorer)
    select_prompts = OptimizedMiningEngine.select_prompts


def _engine_with_scores(score_by_text):
    return _StubEngine(scorer=lambda text: score_by_text.get(text, 0.0))


def _candidate(task_id: int, prompt: str) -> dict[str, Any]:
    return {"task_id": task_id, "prompt": prompt}


def sigma_for(rewards: list[float]) -> float:
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    return var ** 0.5


# ────────  _normalize_entropy_to_unit_interval  ────────


def test_normalize_entropy_below_floor_returns_zero() -> None:
    assert _normalize_entropy_to_unit_interval(0.5, floor=2.0, ceil=10.0) == 0.0
    assert _normalize_entropy_to_unit_interval(2.0, floor=2.0, ceil=10.0) == 0.0


def test_normalize_entropy_above_ceil_returns_one() -> None:
    assert _normalize_entropy_to_unit_interval(15.0, floor=2.0, ceil=10.0) == 1.0
    assert _normalize_entropy_to_unit_interval(10.0, floor=2.0, ceil=10.0) == 1.0


def test_normalize_entropy_in_band_is_linear() -> None:
    assert _normalize_entropy_to_unit_interval(6.0, floor=2.0, ceil=10.0) == pytest.approx(0.5)
    assert _normalize_entropy_to_unit_interval(4.0, floor=2.0, ceil=10.0) == pytest.approx(0.25)


def test_normalize_entropy_handles_degenerate_band() -> None:
    assert _normalize_entropy_to_unit_interval(5.0, floor=10.0, ceil=10.0) == 0.0
    assert _normalize_entropy_to_unit_interval(5.0, floor=10.0, ceil=2.0) == 0.0


# ────────  _candidate_task_id / _candidate_prompt_text  ────────


def test_candidate_task_id_pulls_task_id_first() -> None:
    assert _candidate_task_id({"task_id": 42, "id": 99}) == 42


def test_candidate_task_id_falls_back_to_id() -> None:
    assert _candidate_task_id({"id": 99}) == 99


def test_candidate_task_id_falls_back_to_prompt_idx() -> None:
    assert _candidate_task_id({"prompt_idx": 7}) == 7


def test_candidate_task_id_uses_python_id_as_last_resort() -> None:
    candidate = {"prompt": "no id here"}
    out = _candidate_task_id(candidate)
    assert isinstance(out, int)


def test_candidate_prompt_text_pulls_prompt_first() -> None:
    assert _candidate_prompt_text({"prompt": "P", "text": "T"}) == "P"


def test_candidate_prompt_text_falls_back_to_text_then_question() -> None:
    assert _candidate_prompt_text({"text": "T"}) == "T"
    assert _candidate_prompt_text({"question": "Q"}) == "Q"


def test_candidate_prompt_text_returns_empty_when_nothing_found() -> None:
    assert _candidate_prompt_text({"unrelated": 5}) == ""


# ────────  select_prompts  ────────


def test_select_prompts_empty_pool_returns_empty() -> None:
    eng = _engine_with_scores({})
    assert OptimizedMiningEngine.select_prompts(eng, [], n=5) == []


def test_select_prompts_n_zero_returns_empty() -> None:
    eng = _engine_with_scores({"a": 0.9})
    assert OptimizedMiningEngine.select_prompts(
        eng, [_candidate(0, "a")], n=0
    ) == []


def test_select_prompts_returns_top_n_by_score() -> None:
    pool = [
        _candidate(0, "low"),
        _candidate(1, "mid"),
        _candidate(2, "high"),
    ]
    scores = {"low": 0.1, "mid": 0.5, "high": 0.9}
    eng = _engine_with_scores(scores)
    out = OptimizedMiningEngine.select_prompts(eng, pool, n=2)
    assert [c["task_id"] for c in out] == [2, 1]


def test_select_prompts_drops_cooldown_task_ids() -> None:
    pool = [
        _candidate(0, "high"),
        _candidate(1, "low"),
        _candidate(2, "mid"),
    ]
    scores = {"high": 0.9, "mid": 0.5, "low": 0.1}
    eng = _engine_with_scores(scores)
    out = OptimizedMiningEngine.select_prompts(
        eng, pool, n=2, cooldown_task_ids={0}
    )
    assert [c["task_id"] for c in out] == [2, 1]


def test_select_prompts_returns_fewer_when_pool_smaller_than_n() -> None:
    pool = [_candidate(0, "p"), _candidate(1, "q")]
    eng = _engine_with_scores({"p": 0.5, "q": 0.6})
    out = OptimizedMiningEngine.select_prompts(eng, pool, n=10)
    assert len(out) == 2


def test_select_prompts_is_deterministic_across_calls() -> None:
    pool = [_candidate(i, f"p{i}") for i in range(20)]
    scores = {f"p{i}": float(i % 5) / 4.0 for i in range(20)}
    eng = _engine_with_scores(scores)
    a = OptimizedMiningEngine.select_prompts(eng, pool, n=8)
    b = OptimizedMiningEngine.select_prompts(eng, pool, n=8)
    assert [c["task_id"] for c in a] == [c["task_id"] for c in b]


def test_select_prompts_stable_tiebreak_by_arrival_order() -> None:
    pool = [_candidate(i, f"same_{i}") for i in range(5)]
    eng = _engine_with_scores({f"same_{i}": 0.5 for i in range(5)})
    out = OptimizedMiningEngine.select_prompts(eng, pool, n=3)
    assert [c["task_id"] for c in out] == [0, 1, 2]


def test_select_prompts_returns_empty_when_all_cooled_down() -> None:
    pool = [_candidate(i, f"p{i}") for i in range(5)]
    eng = _engine_with_scores({f"p{i}": 0.5 for i in range(5)})
    out = OptimizedMiningEngine.select_prompts(
        eng, pool, n=3, cooldown_task_ids={0, 1, 2, 3, 4}
    )
    assert out == []


# ────────  estimate_in_zone (static)  ────────


def test_estimate_in_zone_empty_rewards_returns_zero_and_false() -> None:
    sigma, in_zone = OptimizedMiningEngine.estimate_in_zone([], sigma_min=0.43)
    assert sigma == 0.0
    assert in_zone is False


def test_estimate_in_zone_uniform_rewards_returns_zero_sigma() -> None:
    sigma, in_zone = OptimizedMiningEngine.estimate_in_zone(
        [1.0, 1.0, 1.0, 1.0], sigma_min=0.43
    )
    assert sigma == 0.0
    assert in_zone is False


def test_estimate_in_zone_binary_split_8_rollouts_4_correct() -> None:
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    sigma, in_zone = OptimizedMiningEngine.estimate_in_zone(rewards, sigma_min=0.43)
    assert sigma == pytest.approx(0.5)
    assert in_zone is True


def test_estimate_in_zone_binary_split_8_rollouts_2_correct_above_threshold() -> None:
    rewards = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sigma, in_zone = OptimizedMiningEngine.estimate_in_zone(rewards, sigma_min=0.43)
    assert 0.43 < sigma < 0.44
    assert in_zone is True


def test_estimate_in_zone_threshold_is_inclusive() -> None:
    rewards = [0.5, 0.5, 1.0, 0.0]
    sigma_at_exact = sigma_for(rewards)
    sigma, in_zone = OptimizedMiningEngine.estimate_in_zone(
        rewards, sigma_min=sigma_at_exact
    )
    assert in_zone is True
