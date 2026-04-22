"""Tests for the Hendrycks MATH task source (adapted from romain13190/reliquary).

Covers the boxed-answer extractor, LaTeX normalization, reward scoring,
and the TaskSource-protocol adapter's deterministic window batching.
Doesn't require the full 12.5k-row HF dataset — all tests use synthetic
problem dicts OR a fake env.
"""

from __future__ import annotations

import pytest

from reliquary_inference.dataset.task_sources.math_env import (
    _last_boxed_only_string,
    _normalize_answer,
    _strip_boxed_wrapper,
    compute_math_reward,
    evaluate_math_trace,
)


# ---------------------------------------------------------------------------
# Balanced-brace extractor
# ---------------------------------------------------------------------------


def test_last_boxed_simple():
    assert _last_boxed_only_string(r"answer is \boxed{42}") == r"\boxed{42}"


def test_last_boxed_nested():
    s = r"answer is \boxed{\frac{1}{2}}"
    assert _last_boxed_only_string(s) == r"\boxed{\frac{1}{2}}"


def test_last_boxed_picks_last():
    s = r"first guess \boxed{10} then corrected \boxed{20}"
    assert _last_boxed_only_string(s) == r"\boxed{20}"


def test_last_boxed_fbox_fallback():
    assert _last_boxed_only_string(r"ans \fbox{7}") == r"\fbox{7}"


def test_last_boxed_none_on_missing():
    assert _last_boxed_only_string("no boxed expression here") is None


def test_last_boxed_none_on_unterminated():
    # Opening brace never closes.
    assert _last_boxed_only_string(r"\boxed{42") is None


# ---------------------------------------------------------------------------
# Strip boxed wrapper
# ---------------------------------------------------------------------------


def test_strip_boxed_wrapper_removes_shell():
    assert _strip_boxed_wrapper(r"\boxed{42}") == "42"
    assert _strip_boxed_wrapper(r"\fbox{hi}") == "hi"


def test_strip_boxed_wrapper_passthrough_if_not_boxed():
    assert _strip_boxed_wrapper("just text") == "just text"


# ---------------------------------------------------------------------------
# LaTeX normalization
# ---------------------------------------------------------------------------


def test_normalize_dfrac_equals_frac():
    assert _normalize_answer(r"\dfrac{1}{2}") == _normalize_answer(r"\frac{1}{2}")


def test_normalize_strips_left_right():
    assert _normalize_answer(r"\left(1,2\right)") == _normalize_answer("(1,2)")


def test_normalize_strips_text_wrapper():
    assert _normalize_answer(r"7\text{ feet}") == "7feet"


def test_normalize_collapses_whitespace():
    assert _normalize_answer(r"1 / 2") == "1/2"


def test_normalize_strips_trailing_period():
    assert _normalize_answer("42.") == "42"


# ---------------------------------------------------------------------------
# compute_math_reward
# ---------------------------------------------------------------------------


def test_reward_correct_integer():
    problem = {"ground_truth": "42"}
    completion = r"Thinking... the answer is \boxed{42}"
    assert compute_math_reward(problem, completion) == 1.0


def test_reward_correct_fraction_with_dfrac_vs_frac():
    problem = {"ground_truth": r"\frac{1}{2}"}
    completion = r"So \boxed{\dfrac{1}{2}}"
    assert compute_math_reward(problem, completion) == 1.0


def test_reward_wrong_integer():
    problem = {"ground_truth": "42"}
    assert compute_math_reward(problem, r"\boxed{43}") == 0.0


def test_reward_missing_boxed():
    problem = {"ground_truth": "42"}
    assert compute_math_reward(problem, "the answer is 42") == 0.0


def test_reward_never_raises_on_garbage():
    problem = {"ground_truth": "42"}
    assert compute_math_reward(problem, None) == 0.0  # type: ignore[arg-type]
    assert compute_math_reward(problem, "\\boxed{") == 0.0


def test_reward_empty_ground_truth_scores_zero():
    # Defensive: if ground_truth extraction failed upstream, never match on empty.
    assert compute_math_reward({"ground_truth": ""}, r"\boxed{}") == 0.0


# ---------------------------------------------------------------------------
# evaluate_math_trace
# ---------------------------------------------------------------------------


def test_evaluate_accepted():
    task = {"reference_answer": "42"}
    r = evaluate_math_trace(task, r"Reasoning: ... \boxed{42}")
    assert r["accepted"] is True
    assert r["correctness_or_judge"] == 1.0
    assert r["format_ok"] is True
    assert r["final_answer"] == "42"


def test_evaluate_wrong_answer():
    task = {"reference_answer": "42"}
    r = evaluate_math_trace(task, r"\boxed{43}")
    assert r["accepted"] is False
    assert r["correctness_or_judge"] == 0.0
    assert r["format_ok"] is True
    assert r["format_reason"] == "wrong_answer"


def test_evaluate_missing_boxed():
    task = {"reference_answer": "42"}
    r = evaluate_math_trace(task, "answer is 42 but no box")
    assert r["accepted"] is False
    assert r["format_ok"] is False
    assert r["format_reason"] == "missing_boxed_answer"


# ---------------------------------------------------------------------------
# MathTasksSource adapter — deterministic windowing
# ---------------------------------------------------------------------------


class _FakeEnv:
    name = "fake_math"

    def __init__(self, size: int = 64):
        self._size = size

    def __len__(self):
        return self._size

    def get_problem(self, index: int):
        import hashlib
        idx = index % self._size
        prompt = f"What is {idx} + {idx}?"
        return {
            "prompt": prompt,
            "ground_truth": str(2 * idx),
            "id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "dataset_index": idx,
        }

    def compute_reward(self, problem, completion):
        from reliquary_inference.dataset.task_sources.math_env import compute_math_reward
        return compute_math_reward(problem, completion)


def test_math_task_source_deterministic_per_window():
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    # Inject fake env to avoid requiring HF download in tests
    src._env = _FakeEnv(size=128)

    ctx = {
        "window_id": 6957000,
        "public_randomness": "deadbeefcafebabe",
        "model_ref": "toy://test",
    }
    batch1 = src.build_window_batch(ctx, count=8)
    batch2 = src.build_window_batch(ctx, count=8)
    assert [t["task_id"] for t in batch1["tasks"]] == [t["task_id"] for t in batch2["tasks"]]
    assert batch1["task_source"] == "math"
    assert len(batch1["tasks"]) == 8


def test_math_task_source_different_window_different_problems():
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    src._env = _FakeEnv(size=128)

    ctx1 = {"window_id": 6957000, "public_randomness": "aaaa" * 8, "model_ref": "toy://t"}
    ctx2 = {"window_id": 6957030, "public_randomness": "bbbb" * 8, "model_ref": "toy://t"}

    b1 = src.build_window_batch(ctx1, count=8)
    b2 = src.build_window_batch(ctx2, count=8)
    idx1 = {t["dataset_index"] for t in b1["tasks"]}
    idx2 = {t["dataset_index"] for t in b2["tasks"]}
    # Shouldn't be identical sets — different randomness seed picks different problems.
    assert idx1 != idx2


def test_math_task_source_cooldown_skipped():
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    src._env = _FakeEnv(size=32)
    ctx = {
        "window_id": 6957000,
        "public_randomness": "aaaa" * 8,
        "model_ref": "toy://t",
    }
    # First pick, no cooldown:
    baseline = src.build_window_batch(ctx, count=8)
    baseline_idxs = {t["dataset_index"] for t in baseline["tasks"]}

    # Second pick with those indices in cooldown: none of them should recur.
    ctx_cool = dict(ctx, cooldown_indices=list(baseline_idxs))
    excluded = src.build_window_batch(ctx_cool, count=8)
    excluded_idxs = {t["dataset_index"] for t in excluded["tasks"]}
    assert excluded_idxs.isdisjoint(baseline_idxs)


def test_math_task_source_tasks_have_boxed_format_hint():
    """The prompt the miner sees should request a \\boxed{...} answer —
    that's what evaluate_math_trace scores against."""
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    src._env = _FakeEnv(size=16)
    ctx = {"window_id": 1, "public_randomness": "f" * 16, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=3)
    for task in batch["tasks"]:
        assert "boxed" in task["prompt"].lower()


def test_math_task_source_evaluate_completion_accepts_correct():
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    src._env = _FakeEnv(size=16)
    task = {
        "task_id": "math-1-0",
        "reference_answer": "42",
        "contamination_policy": {"forbidden_overlap_tags": []},
    }
    completion = {"payload": {"completion_text": r"computation ... \boxed{42}"}}
    r = src.evaluate_completion(completion, task)
    assert r["accepted"] is True
    assert r["evaluation"]["correctness_or_judge"] == 1.0


def test_math_task_source_evaluate_completion_rejects_wrong():
    from reliquary_inference.dataset.task_sources import MathTasksSource

    src = MathTasksSource()
    src._env = _FakeEnv(size=16)
    task = {
        "task_id": "math-1-0",
        "reference_answer": "42",
        "contamination_policy": {"forbidden_overlap_tags": []},
    }
    completion = {"payload": {"completion_text": r"\boxed{99}"}}
    r = src.evaluate_completion(completion, task)
    assert r["accepted"] is False
    assert r["evaluation"]["final_answer"] == "99"


def test_math_task_source_registered_in_build():
    from reliquary_inference.dataset.task_sources import build_task_source, MathTasksSource

    src = build_task_source("math")
    assert isinstance(src, MathTasksSource)


def test_math_task_source_unknown_id_raises():
    from reliquary_inference.dataset.task_sources import build_task_source

    with pytest.raises(ValueError, match="Unsupported task source"):
        build_task_source("nonexistent_xyz")
