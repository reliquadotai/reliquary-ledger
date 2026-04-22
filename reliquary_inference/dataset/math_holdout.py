"""Deterministic held-out MATH slice for eval.

The Reliquary eval harness (``reliquary.eval.math_harness`` on the
Forge side) runs the current policy against a fixed held-out slice of
Hendrycks MATH and publishes a signed :class:`~reliquary_protocol.EvalBundle`
tagged with the policy hash + accuracy + per-level / per-subject breakdown.

For the eval to mean anything, the held-out problems must NEVER appear
in the miner's sampling pool — otherwise the policy has trained on the
holdout directly. This module provides:

- :func:`holdout_indices` — the deterministic set of MATH dataset
  indices reserved for eval. Derived via
  :func:`reliquary_protocol.derive_eval_holdout_indices` so the Forge
  side picks up byte-identical indices without any shared state.
- :func:`holdout_task_ids` — the stable per-problem id list (sha256 of
  the problem text) that goes into every published EvalBundle. Third
  parties can walk from the public seed constants → problem_ids →
  recomputed accuracy independently.
- :func:`load_holdout_problems` — materialize the held-out problems
  with ``prompt`` / ``reference_answer`` / ``level`` / ``subject`` for
  the evaluator to run.

The miner sampling path in :mod:`.task_sources` consumes
:func:`holdout_indices` to skip these dataset rows when drawing a
window batch.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Optional

from reliquary_protocol import (
    DEFAULT_HOLDOUT_SEED,
    DEFAULT_HOLDOUT_SIZE,
    HOLDOUT_LABEL_VERSION,
    derive_eval_holdout_indices,
)

#: The Hendrycks MATH mirror used by Reliquary. Pinned here so the
#: EvalBundle can record which dataset identity the holdout was derived
#: against. Changing this requires a holdout label_version bump.
HOLDOUT_DATASET: str = "qwedsacf/competition_math"

#: Number of Hendrycks MATH problems in the canonical train split.
#: Hard-coded so the holdout derivation doesn't require network access
#: at module import time. Verified via test against the live dataset.
HOLDOUT_DATASET_SIZE: int = 12500


def holdout_indices(
    *,
    dataset_size: int = HOLDOUT_DATASET_SIZE,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: str = DEFAULT_HOLDOUT_SEED,
) -> frozenset[int]:
    """Return the reserved dataset-index set for the eval holdout.

    Wraps :func:`reliquary_protocol.derive_eval_holdout_indices` so the
    Ledger sampling guard and the Forge evaluator derive byte-identical
    sets. Result is frozen for use as a fast O(1) exclusion check in
    :class:`.task_sources.MathTasksSource.build_window_batch`.
    """
    return frozenset(
        derive_eval_holdout_indices(
            seed=seed,
            dataset_size=dataset_size,
            holdout_size=holdout_size,
            label_version=HOLDOUT_LABEL_VERSION,
        )
    )


def problem_id_for_index(index: int) -> str:
    """Return the stable ``problem_id`` for a dataset row.

    Uses the same sha256-of-question truncation as
    :meth:`MATHEnvironment.get_problem` so the Forge EvalBundle's
    ``holdout_task_ids`` list matches Ledger's internal problem id
    naming exactly. Pulls the problem text lazily via
    :class:`_HoldoutProblemCache` to avoid loading the full dataset
    when the caller only needs indices.
    """
    question = _HoldoutProblemCache.question_at(int(index))
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def holdout_task_ids(
    *,
    dataset_size: int = HOLDOUT_DATASET_SIZE,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: str = DEFAULT_HOLDOUT_SEED,
) -> list[str]:
    """Return the stable problem-id list for the holdout slice.

    These ids are what every published :class:`EvalBundle` records as
    ``holdout_task_ids`` so third parties can verify (a) which exact
    problems were evaluated and (b) the same set was evaluated every
    cycle. Sorted lexicographically to match the canonicalization rule
    in :func:`reliquary_protocol.build_eval_bundle`.
    """
    indices = sorted(
        holdout_indices(
            dataset_size=dataset_size, holdout_size=holdout_size, seed=seed
        )
    )
    return sorted({problem_id_for_index(i) for i in indices})


def load_holdout_problems(
    *,
    dataset_size: int = HOLDOUT_DATASET_SIZE,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: str = DEFAULT_HOLDOUT_SEED,
) -> list[dict[str, Any]]:
    """Materialize the held-out problems with ground truth + metadata.

    Each returned dict mirrors :meth:`MATHEnvironment.get_problem` so
    downstream scoring through :func:`evaluate_math_trace` works
    unchanged:

    ``{ "prompt", "reference_answer", "problem_id", "dataset_index",
        "level", "subject" }``

    Subject is parsed from the dataset's ``type`` column and falls back
    to ``""`` if unavailable. Loads the full Hendrycks MATH set on first
    call and caches it at module level.
    """
    from .task_sources.math_env import _last_boxed_only_string, _strip_boxed_wrapper

    _HoldoutProblemCache.ensure_loaded()
    dataset = _HoldoutProblemCache._dataset
    assert dataset is not None  # _ensure_loaded() raises otherwise
    indices = sorted(
        holdout_indices(
            dataset_size=dataset_size, holdout_size=holdout_size, seed=seed
        )
    )
    problems: list[dict[str, Any]] = []
    for idx in indices:
        row = dataset[int(idx)]
        question: str = row["problem"]
        solution: str = row["solution"]
        boxed = _last_boxed_only_string(solution)
        gt_str = _strip_boxed_wrapper(boxed) if boxed else ""
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        problems.append(
            {
                "prompt": question,
                "reference_answer": gt_str,
                "problem_id": problem_id,
                "dataset_index": int(idx),
                "level": str(row.get("level", "") or ""),
                "subject": str(row.get("type", "") or ""),
            }
        )
    return problems


class _HoldoutProblemCache:
    """Module-level cache of the Hendrycks MATH dataset rows.

    Intentionally separate from :class:`MATHEnvironment`'s cache so
    Forge's eval path can load the dataset without instantiating a
    full miner-side task source. Lazy-loaded on first use.
    """

    _dataset: ClassVar[Optional[Any]] = None

    @classmethod
    def ensure_loaded(cls) -> None:
        if cls._dataset is None:
            import datasets as hf_datasets  # type: ignore[import-not-found]

            cls._dataset = hf_datasets.load_dataset(
                HOLDOUT_DATASET, split="train"
            )

    @classmethod
    def question_at(cls, index: int) -> str:
        cls.ensure_loaded()
        row = cls._dataset[int(index)]  # type: ignore[index]
        return str(row["problem"])


__all__ = [
    "HOLDOUT_DATASET",
    "HOLDOUT_DATASET_SIZE",
    "holdout_indices",
    "holdout_task_ids",
    "load_holdout_problems",
    "problem_id_for_index",
]
