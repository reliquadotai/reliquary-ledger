"""Tests for the GSM8K task source + MixedTasksSource weighted mixing.

Uses a synthetic fake env (no HF download) — we're exercising the
task-source adapter + mix dispatch logic, not the dataset itself.
"""

from __future__ import annotations

import hashlib

import pytest

from reliquary_inference.dataset.task_sources import (
    GSM8KTasksSource,
    MathTasksSource,
    MixedTasksSource,
    _WeightedSource,
    build_task_source,
)
from reliquary_inference.dataset.task_sources.gsm8k_env import (
    extract_gsm8k_answer,
)


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------


def test_extract_gsm8k_answer_simple():
    assert extract_gsm8k_answer("Blah ... #### 72") == "72"


def test_extract_gsm8k_answer_strips_commas():
    assert extract_gsm8k_answer("Natalia had ... #### 1,234") == "1234"


def test_extract_gsm8k_answer_negative():
    assert extract_gsm8k_answer("... #### -5") == "-5"


def test_extract_gsm8k_answer_decimal():
    assert extract_gsm8k_answer("... #### 3.14") == "3.14"


def test_extract_gsm8k_answer_missing_marker():
    assert extract_gsm8k_answer("no marker here") == ""


def test_extract_gsm8k_answer_empty():
    assert extract_gsm8k_answer("") == ""


# ---------------------------------------------------------------------------
# GSM8K task source — fake env
# ---------------------------------------------------------------------------


class _FakeGSM8KEnv:
    """GSM8K-shaped fake env (identity `_resolve`, `Level 1` default)."""

    name = "fake_gsm8k"

    def __init__(self, size: int = 64, with_answer: bool = True) -> None:
        self._size = size
        self._with_answer = with_answer

    def __len__(self) -> int:
        return self._size

    def _resolve(self, index: int) -> int:
        return int(index) % self._size

    def get_problem(self, index: int) -> dict:
        idx = self._resolve(index)
        question = f"Sarah had {idx} apples. How many remain after eating one?"
        return {
            "prompt": question,
            "ground_truth": str(idx - 1) if self._with_answer else "",
            "id": hashlib.sha256(question.encode()).hexdigest()[:16],
            "dataset_index": idx,
            "level": "Level 1",
            "subject": "Arithmetic Word Problems",
        }

    def compute_reward(self, problem, completion):
        return 0.0


def test_gsm8k_task_source_registered_in_build():
    src = build_task_source("gsm8k")
    assert isinstance(src, GSM8KTasksSource)


def test_gsm8k_source_produces_tasks_with_gsm8k_source_tag():
    src = GSM8KTasksSource()
    src._env = _FakeGSM8KEnv(size=32)
    ctx = {
        "window_id": 6957000,
        "public_randomness": "deadbeef" * 2,
        "model_ref": "toy://t",
    }
    batch = src.build_window_batch(ctx, count=4)
    assert batch["task_source"] == "gsm8k"
    assert len(batch["tasks"]) == 4
    for task in batch["tasks"]:
        assert task["task_family"] == "gsm8k"
        assert task["task_id"].startswith("gsm8k-")
        assert any(t == "source:gsm8k" for t in task["tags"])
        assert "boxed" in task["prompt"].lower()


def test_gsm8k_source_deterministic_per_window():
    src1 = GSM8KTasksSource()
    src1._env = _FakeGSM8KEnv(size=128)
    src2 = GSM8KTasksSource()
    src2._env = _FakeGSM8KEnv(size=128)
    ctx = {
        "window_id": 6957000,
        "public_randomness": "deadbeefcafebabe",
        "model_ref": "toy://t",
    }
    b1 = src1.build_window_batch(ctx, count=8)
    b2 = src2.build_window_batch(ctx, count=8)
    assert [t["task_id"] for t in b1["tasks"]] == [t["task_id"] for t in b2["tasks"]]


def test_gsm8k_source_skips_rows_with_no_extracted_answer():
    """Rows whose ``####`` tail didn't parse land in the pool with an
    empty ground_truth — drop them so the validator doesn't score
    against a blank reference (which never matches)."""
    src = GSM8KTasksSource()
    src._env = _FakeGSM8KEnv(size=32, with_answer=False)
    ctx = {"window_id": 1, "public_randomness": "aaaa" * 8, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=8)
    assert batch["tasks"] == []


def test_gsm8k_source_evaluate_completion_accepts_correct():
    """End-to-end: 5 apples, eat one → 4. Boxed 4 ⇒ accepted."""
    src = GSM8KTasksSource()
    src._env = _FakeGSM8KEnv(size=16)
    task = {
        "task_id": "gsm8k-1-5",
        "reference_answer": "4",
        "contamination_policy": {"forbidden_overlap_tags": []},
    }
    completion = {"payload": {"completion_text": r"so ... \boxed{4}"}}
    r = src.evaluate_completion(completion, task)
    assert r["accepted"] is True
    assert r["evaluation"]["correctness_or_judge"] == 1.0


# ---------------------------------------------------------------------------
# MixedTasksSource
# ---------------------------------------------------------------------------


class _FakeMathEnv:
    """Drop-in for MATHEnvironment in test — no HF + no level filter."""

    name = "fake_math_for_mix"

    def __init__(self, size: int = 64):
        self._size = size

    def __len__(self):
        return self._size

    def _resolve(self, index: int) -> int:
        return int(index) % self._size

    def get_problem(self, index: int):
        idx = self._resolve(index)
        prompt = f"MATH problem #{idx}"
        return {
            "prompt": prompt,
            "ground_truth": str(idx),
            "id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "dataset_index": idx,
            "level": "Level 1",
        }

    def compute_reward(self, problem, completion):
        return 0.0


def _build_fake_mix(math_weight: float, gsm_weight: float):
    math_src = MathTasksSource(exclude_holdout=False)
    math_src._env = _FakeMathEnv(size=128)
    gsm_src = GSM8KTasksSource()
    gsm_src._env = _FakeGSM8KEnv(size=128)
    return MixedTasksSource(
        sources=(
            _WeightedSource(source=math_src, weight=math_weight),
            _WeightedSource(source=gsm_src, weight=gsm_weight),
        ),
    )


def test_mixed_source_returns_combined_count():
    src = _build_fake_mix(math_weight=1.0, gsm_weight=1.0)
    ctx = {"window_id": 1, "public_randomness": "deadbeef" * 2, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=12)
    assert batch["task_source"] == "mixed"
    assert len(batch["tasks"]) == 12


def test_mixed_source_proportional_allocation():
    """With weights 2:1 and count=12, expect ~8 math + 4 gsm."""
    src = _build_fake_mix(math_weight=2.0, gsm_weight=1.0)
    ctx = {"window_id": 1, "public_randomness": "deadbeef" * 2, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=12)
    families = [t["task_family"] for t in batch["tasks"]]
    math_count = families.count("math")
    gsm_count = families.count("gsm8k")
    assert math_count == 8
    assert gsm_count == 4


def test_mixed_source_is_deterministic_per_window():
    src1 = _build_fake_mix(1.0, 1.0)
    src2 = _build_fake_mix(1.0, 1.0)
    ctx = {"window_id": 1, "public_randomness": "cafebabe" * 2, "model_ref": "toy://t"}
    b1 = src1.build_window_batch(ctx, count=10)
    b2 = src2.build_window_batch(ctx, count=10)
    assert [t["task_id"] for t in b1["tasks"]] == [t["task_id"] for t in b2["tasks"]]


def test_mixed_source_shuffles_across_sub_sources():
    """Task order in the output batch must not be "all math, then all
    gsm" — the shuffle step interleaves so per-window sampling doesn't
    surface source-specific degradation as a positional artifact."""
    src = _build_fake_mix(1.0, 1.0)
    ctx = {"window_id": 1, "public_randomness": "abcdef12" * 2, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=16)
    families = [t["task_family"] for t in batch["tasks"]]
    # Not fully sorted by family — i.e. at least one adjacent pair
    # changes family.
    adjacent_changes = sum(
        1 for a, b in zip(families, families[1:]) if a != b
    )
    assert adjacent_changes >= 1


def test_mixed_source_empty_raises():
    empty = MixedTasksSource(sources=())
    with pytest.raises(ValueError, match="no constituent sources"):
        empty.build_window_batch(
            {"window_id": 1, "public_randomness": "f" * 16, "model_ref": "toy://t"},
            count=4,
        )


def test_mixed_source_zero_weights_raises():
    math_src = MathTasksSource(exclude_holdout=False)
    math_src._env = _FakeMathEnv(size=16)
    src = MixedTasksSource(
        sources=(_WeightedSource(source=math_src, weight=0.0),)
    )
    with pytest.raises(ValueError, match="weights must be positive"):
        src.build_window_batch(
            {"window_id": 1, "public_randomness": "f" * 16, "model_ref": "toy://t"},
            count=4,
        )


def test_mixed_source_verify_and_evaluate_dispatch_by_task_family():
    src = _build_fake_mix(1.0, 1.0)
    ctx = {"window_id": 1, "public_randomness": "12345678" * 2, "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=4)
    # Find one task from each family; evaluate with a plausible correct
    # completion and confirm the mixed dispatch returns the right
    # sub-source's evaluation.
    by_family: dict[str, dict] = {}
    for t in batch["tasks"]:
        by_family.setdefault(t["task_family"], t)
    for family, task in by_family.items():
        fake_completion = {
            "payload": {
                "task_id": task["task_id"],
                "completion_text": rf"so \boxed{{{task['reference_answer']}}}",
                "contamination_tags": [],
            }
        }
        result = src.evaluate_completion(fake_completion, task)
        assert result["accepted"] is True, f"family={family} should accept correct boxed"


def test_build_task_source_mixed_requires_mix_kwarg():
    with pytest.raises(ValueError, match="requires a non-empty `mix` kwarg"):
        build_task_source("mixed")


def test_build_task_source_mixed_rejects_nested_mix():
    with pytest.raises(ValueError, match="nested 'mixed' sources"):
        build_task_source("mixed", mix=[("mixed", 1.0)])


def test_build_task_source_mixed_constructs_weighted_source():
    src = build_task_source(
        "mixed", mix=[("gsm8k", 1.0), ("gsm8k", 2.0)]
    )
    assert isinstance(src, MixedTasksSource)
    assert len(src.sources) == 2
    assert src.sources[0].weight == 1.0
    assert src.sources[1].weight == 2.0
