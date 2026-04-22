"""Tests for :mod:`reliquary_inference.dataset.math_holdout`.

Covers:

- Holdout indices are deterministic + within dataset bounds + the
  expected size.
- Holdout task ids are stable sha256-derived strings (no HF fetch
  needed for the derivation — we stub the dataset cache).
- :class:`MathTasksSource.build_window_batch` never returns a task
  whose raw dataset index is in the holdout (the anti-leakage guard).
- The guard can be turned off via ``exclude_holdout=False`` for unit
  tests that don't care about eval provenance.
"""

from __future__ import annotations

import pytest

from reliquary_inference.dataset import math_holdout
from reliquary_inference.dataset.math_holdout import (
    HOLDOUT_DATASET,
    HOLDOUT_DATASET_SIZE,
    holdout_indices,
    holdout_task_ids,
)
from reliquary_inference.dataset.task_sources import MathTasksSource
from reliquary_protocol import (
    DEFAULT_HOLDOUT_SEED,
    DEFAULT_HOLDOUT_SIZE,
    derive_eval_holdout_indices,
)

# ---------------------------------------------------------------------------
# holdout_indices
# ---------------------------------------------------------------------------


def test_holdout_indices_matches_protocol_derivation():
    """The Ledger derivation must match the Forge derivation byte-for-byte
    through the shared protocol helper — otherwise the miner skips one
    set and the evaluator runs a different set, and the EvalBundle is
    meaningless."""
    local = holdout_indices()
    canonical = frozenset(
        derive_eval_holdout_indices(
            seed=DEFAULT_HOLDOUT_SEED,
            dataset_size=HOLDOUT_DATASET_SIZE,
            holdout_size=DEFAULT_HOLDOUT_SIZE,
        )
    )
    assert local == canonical


def test_holdout_indices_size_and_range():
    idx = holdout_indices()
    assert len(idx) == DEFAULT_HOLDOUT_SIZE
    assert all(0 <= i < HOLDOUT_DATASET_SIZE for i in idx)


def test_holdout_indices_is_cached():
    a = holdout_indices()
    b = holdout_indices()
    # Same object (via the protocol's internal cache) OR equal sets —
    # either way, two calls must agree.
    assert a == b


def test_holdout_dataset_identity_is_pinned():
    # Tripwire: changing this silently would invalidate every published
    # eval bundle. Deliberate change requires a holdout label version bump.
    assert HOLDOUT_DATASET == "qwedsacf/competition_math"
    assert HOLDOUT_DATASET_SIZE == 12500


# ---------------------------------------------------------------------------
# holdout_task_ids
# ---------------------------------------------------------------------------


class _StubDataset:
    """Small synthetic MATH-shaped dataset for holdout task-id tests."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict:
        return self._rows[int(index)]


@pytest.fixture
def stubbed_dataset(monkeypatch):
    """Inject a 12500-row stub into :class:`_HoldoutProblemCache` so the
    holdout id path doesn't touch HF. Each row's ``problem`` is
    deterministically derived from its index so test assertions can
    recompute the expected sha256."""
    rows = [
        {
            "problem": f"Synthetic MATH problem #{i}",
            "solution": f"... \\boxed{{{i}}}",
            "level": f"Level {((i % 5) + 1)}",
            "type": ["Algebra", "Geometry", "Number Theory", "Probability", "Calculus"][i % 5],
        }
        for i in range(HOLDOUT_DATASET_SIZE)
    ]
    stub = _StubDataset(rows)
    monkeypatch.setattr(math_holdout._HoldoutProblemCache, "_dataset", stub)
    return stub


def test_holdout_task_ids_are_sha256_truncated(stubbed_dataset):
    import hashlib

    ids = holdout_task_ids()
    # Every id must be a 16-char hex string (the sha256 truncation).
    for tid in ids:
        assert len(tid) == 16
        assert all(c in "0123456789abcdef" for c in tid)

    # Every id must correspond to a problem from the holdout slice —
    # recompute from the stub rows and ensure set equality.
    expected = {
        hashlib.sha256(f"Synthetic MATH problem #{i}".encode()).hexdigest()[:16]
        for i in holdout_indices()
    }
    assert set(ids) == expected


def test_holdout_task_ids_are_sorted(stubbed_dataset):
    ids = holdout_task_ids()
    assert ids == sorted(ids)


def test_load_holdout_problems_shape(stubbed_dataset):
    from reliquary_inference.dataset.math_holdout import load_holdout_problems

    problems = load_holdout_problems()
    assert len(problems) == DEFAULT_HOLDOUT_SIZE
    for p in problems:
        assert set(p.keys()) >= {
            "prompt",
            "reference_answer",
            "problem_id",
            "dataset_index",
            "level",
            "subject",
        }
        assert p["dataset_index"] in holdout_indices()
        assert p["prompt"].startswith("Synthetic MATH problem #")
        assert p["reference_answer"] == str(p["dataset_index"])


# ---------------------------------------------------------------------------
# MathTasksSource sampling guard
# ---------------------------------------------------------------------------


class _HoldoutAwareFakeEnv:
    """Fake env that exposes ``_resolve`` identity mapping + size ==
    HOLDOUT_DATASET_SIZE so the holdout exclusion is exercised against
    the real 500-index set.

    Using size==HOLDOUT_DATASET_SIZE (12500) matches the production
    scale so any off-by-one in the filtered-index vs raw-index
    distinction surfaces in the test."""

    name = "fake_holdout_aware"

    def __init__(self, size: int = HOLDOUT_DATASET_SIZE):
        self._size = size

    def __len__(self):
        return self._size

    def _resolve(self, index: int) -> int:
        return int(index) % self._size

    def get_problem(self, index: int):
        import hashlib

        idx = int(index) % self._size
        prompt = f"Problem #{idx}"
        return {
            "prompt": prompt,
            "ground_truth": str(idx),
            "id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "dataset_index": idx,
            "level": "Level 1",
        }

    def compute_reward(self, problem, completion):
        return 0.0


def test_holdout_indices_are_never_sampled_by_miner():
    """Core anti-leakage test — the miner must NEVER produce a task
    whose raw dataset index is in the holdout."""
    src = MathTasksSource()
    src._env = _HoldoutAwareFakeEnv(size=HOLDOUT_DATASET_SIZE)
    holdout = holdout_indices()

    # Probe many windows — each picks 8 tasks; none should be in holdout.
    # ``build_window_batch`` reads the FIRST 8 hex chars of public_randomness
    # as its RNG seed, so vary those (not just the low-order bits).
    for window_id in range(100):
        seed_material = f"{window_id:08x}ffffffff"
        ctx = {
            "window_id": window_id,
            "public_randomness": seed_material,
            "model_ref": "toy://t",
        }
        batch = src.build_window_batch(ctx, count=8)
        for task in batch["tasks"]:
            assert task["dataset_index"] not in holdout


def test_exclude_holdout_false_disables_guard():
    """Explicitly opting out of the guard must let the miner sample the
    full range, including holdout indices. (Used only for unit tests
    that want to exercise full-range behaviour — production always
    runs with the guard ON.)"""
    src = MathTasksSource(exclude_holdout=False)
    src._env = _HoldoutAwareFakeEnv(size=HOLDOUT_DATASET_SIZE)

    # With enough probes, we must eventually sample an index that lives
    # in the holdout set — proves the guard is the only reason the
    # guarded path never produces them. The seed derivation reads the
    # FIRST 8 hex chars, so vary those to get distinct RNGs.
    holdout = holdout_indices()
    sampled: set[int] = set()
    for window_id in range(2000):
        ctx = {
            "window_id": window_id,
            "public_randomness": f"{window_id:08x}ffffffff",
            "model_ref": "toy://t",
        }
        batch = src.build_window_batch(ctx, count=8)
        for task in batch["tasks"]:
            sampled.add(int(task["dataset_index"]))
        if sampled & holdout:
            break
    assert sampled & holdout, (
        "with exclude_holdout=False the miner must be able to sample "
        "holdout indices; if this never triggers, the guard path is "
        "being taken incorrectly"
    )


def test_guard_handles_envs_without_resolve_method():
    """Unfiltered / fake envs without ``_resolve`` must still work —
    the guard falls through with the raw index in that case."""

    class _NoResolveEnv:
        def __init__(self):
            self._size = 64

        def __len__(self):
            return self._size

        def get_problem(self, index: int):
            import hashlib

            idx = int(index) % self._size
            return {
                "prompt": f"P{idx}",
                "ground_truth": str(idx),
                "id": hashlib.sha256(f"P{idx}".encode()).hexdigest()[:16],
                "dataset_index": idx,
            }

        def compute_reward(self, problem, completion):
            return 0.0

    src = MathTasksSource()
    src._env = _NoResolveEnv()
    ctx = {"window_id": 1, "public_randomness": "deadbeef", "model_ref": "toy://t"}
    batch = src.build_window_batch(ctx, count=4)
    assert len(batch["tasks"]) == 4
