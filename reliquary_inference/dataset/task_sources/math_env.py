"""Hendrycks MATH environment — adapted from romain13190/reliquary (MIT).

Drops in a real reasoning benchmark (12 500 problems, 5 difficulty levels,
7 subjects) to replace the synthetic arithmetic generator. Miners generate
solutions, validators extract the last ``\\boxed{...}`` answer + LaTeX-
normalize, compare to the problem's ground truth.

Why MATH specifically:
  - Binary reward {0, 1} with an objective scoring rule → zone filter
    (σ ≥ 0.43) maps to "2-6 correct out of 8" directly, no reward-scale
    tuning needed.
  - Verifier-friendly: no LLM judge, no reward model. Just string-match
    after a short list of LaTeX canonicalizations. Cross-GPU determinism
    is preserved.
  - Real variance on Qwen2.5-3B: the 3B model solves a meaningful but
    non-trivial fraction, so rollout groups genuinely split 2-6 correct,
    which is exactly where GRPO pulls gradient signal.

Licensing:
  - Romain13190/reliquary is MIT (same license as this repo).
  - Origin of the extract/normalize logic: Hendrycks et al., "Measuring
    Mathematical Problem Solving" (MATH dataset), NeurIPS 2021.
  - HF mirror: ``qwedsacf/competition_math`` (the original
    ``hendrycks/competition_math`` was unpublished from the Hub).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, ClassVar, Optional

# ---------------------------------------------------------------------------
# Balanced-brace extraction of the last \boxed{...} / \fbox{...}
# ---------------------------------------------------------------------------


def _last_boxed_only_string(text: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) substring.

    Walks braces to handle nested expressions like ``\\boxed{\\frac{1}{2}}``.
    A regex can't match balanced braces; this scan can.
    Returns None if no balanced wrapper is found.
    """
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    open_idx = text.index("{", idx)
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[idx : j + 1]
    return None


def _strip_boxed_wrapper(s: str) -> str:
    """If ``s`` starts with ``\\boxed{`` / ``\\fbox{`` and ends with ``}``, return inner."""
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return s


# ---------------------------------------------------------------------------
# Answer normalization — conservative LaTeX simplification for equality
# ---------------------------------------------------------------------------

_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\{([^}]*)\}")


def _normalize_answer(s: str) -> str:
    """Conservative LaTeX normalization for equality comparison.

    String-level only (no CAS): the rules below cover transforms that
    actually occur in Hendrycks MATH ground truths without changing
    mathematical meaning.
    """
    if s is None:
        return ""
    # Drop LaTeX spacing macros first so downstream rules see clean text.
    for macro in (r"\!", r"\,", r"\ ", r"\;", r"\:"):
        s = s.replace(macro, "")
    # Drop \left / \right size modifiers — presentational only.
    s = s.replace(r"\left", "").replace(r"\right", "")
    # Canonicalize fraction macros.
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # Strip \text{...} and \mbox{...} wrappers (keep inner content).
    s = _TEXT_RE.sub(r"\1", s)
    s = _MBOX_RE.sub(r"\1", s)
    # Strip math-mode delimiters.
    s = s.replace(r"\$", "").replace("$", "")
    # Strip trailing period / whitespace.
    s = s.strip().rstrip(".").strip()
    # Collapse whitespace (MATH answers should be whitespace-insensitive).
    s = re.sub(r"\s+", "", s)
    return s


def compute_math_reward(problem: dict, completion: str) -> float:
    """Score a MATH completion. Returns 1.0 on exact-after-normalization match, else 0.0.

    Never raises on malformed input.
    """
    try:
        boxed = _last_boxed_only_string(completion)
        if boxed is None:
            return 0.0
        candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
        gt_raw = str(problem.get("ground_truth", ""))
        gt = _normalize_answer(_strip_boxed_wrapper(gt_raw))
        return 1.0 if candidate == gt and gt != "" else 0.0
    except Exception:
        return 0.0


def evaluate_math_trace(task: dict[str, Any], completion: str) -> dict[str, Any]:
    """Full evaluation matching the shape of ``evaluate_reasoning_trace``.

    Returns a dict with:
      - accepted: bool (reward > 0)
      - correctness_or_judge: float in [0, 1]
      - policy_compliance: float (1.0 for compliant format)
      - format_ok: bool (true iff at least one boxed expression was found)
      - format_reason: str ("ok" | "missing_boxed_answer" | "wrong_answer")
      - final_answer: str (extracted boxed content after normalization, or "")
    """
    boxed = _last_boxed_only_string(completion or "")
    if boxed is None:
        return {
            "accepted": False,
            "correctness_or_judge": 0.0,
            "policy_compliance": 0.0,
            "format_ok": False,
            "format_reason": "missing_boxed_answer",
            "final_answer": "",
        }
    candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
    gt_raw = str(task.get("reference_answer", ""))
    gt = _normalize_answer(_strip_boxed_wrapper(gt_raw))
    is_correct = candidate == gt and gt != ""
    return {
        "accepted": is_correct,
        "correctness_or_judge": 1.0 if is_correct else 0.0,
        "policy_compliance": 1.0,
        "format_ok": True,
        "format_reason": "ok" if is_correct else "wrong_answer",
        "final_answer": candidate,
    }


# ---------------------------------------------------------------------------
# Environment — Hendrycks MATH via the qwedsacf mirror
# ---------------------------------------------------------------------------


class MATHEnvironment:
    """Environment backed by the full Hendrycks MATH set (12 500 problems).

    Ground truths are extracted once from the ``solution`` field by taking
    the content of the last ``\\boxed{...}``; completions are scored with
    the same extraction against the completion text.

    The dataset is lazy-loaded on first instantiation + cached at module
    level so multiple task-source instances in the same process share one
    HF download.

    ``max_level`` (optional, default None = no filter) restricts the
    sampling pool to problems at or below a difficulty level. Levels are
    the Hendrycks 1..5 scale (1 = easiest). When a new fleet is
    bootstrapping with a weak base model, ``max_level=2`` lets the
    policy actually solve enough problems to produce non-zero group σ.
    """

    name: str = "math"

    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self, max_level: Optional[int] = None) -> None:
        if MATHEnvironment._dataset_cache is None:
            import datasets as hf_datasets
            MATHEnvironment._dataset_cache = hf_datasets.load_dataset(
                "qwedsacf/competition_math", split="train"
            )
        ds = MATHEnvironment._dataset_cache
        # Pre-filter indices by level if requested. The dataset's ``level``
        # column is "Level 1".."Level 5" plus a handful of "Level ?" rows;
        # parse the integer and drop unparseable rows from the filtered pool.
        if max_level is not None:
            def _parse_level(s: str) -> Optional[int]:
                try:
                    return int(s.split()[-1])
                except (ValueError, AttributeError, IndexError):
                    return None
            self._filter_indices: Optional[list[int]] = [
                i for i, lvl in enumerate(ds["level"])
                if (_parse_level(lvl) is not None and _parse_level(lvl) <= int(max_level))
            ]
            if not self._filter_indices:
                raise ValueError(
                    f"MATH max_level={max_level} matched zero problems; "
                    "check the dataset's `level` column values."
                )
        else:
            self._filter_indices = None
        self._dataset = ds
        self.max_level = max_level

    def __len__(self) -> int:
        if self._filter_indices is not None:
            return len(self._filter_indices)
        return len(self._dataset)

    def _resolve(self, index: int) -> int:
        """Map an external index (0..len) to the underlying dataset row.

        When ``max_level`` is set we maintain a filtered view; external
        indices are 0..len(filter_indices) and internally map to the
        actual dataset rows that passed the filter.
        """
        if self._filter_indices is not None:
            return self._filter_indices[index % len(self._filter_indices)]
        return index % len(self._dataset)

    def get_problem(self, index: int) -> dict:
        """Return problem at ``index`` (wraps modulo len(self))."""
        idx = self._resolve(index)
        row = self._dataset[idx]
        question: str = row["problem"]
        solution: str = row["solution"]
        boxed = _last_boxed_only_string(solution)
        gt_str = _strip_boxed_wrapper(boxed) if boxed else ""
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        level = row.get("level", "")
        return {
            "prompt": question,
            "ground_truth": gt_str,
            "id": problem_id,
            "dataset_index": idx,
            "level": level,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        return compute_math_reward(problem, completion)


__all__ = [
    "MATHEnvironment",
    "compute_math_reward",
    "evaluate_math_trace",
]
