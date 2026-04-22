"""GSM8K environment — grade-school word problems as a companion to MATH.

Purpose:
  Doubles the size of the deterministic task pool (MATH 12,500 → MATH
  12,500 + GSM8K 7,473). Hendrycks MATH alone cycles through the
  Level 1-2 bootstrap pool in ~2 days of continuous mining; adding
  GSM8K (easier word-problem style, also numeric-answer-friendly)
  extends the sustainable-mining horizon and gives the policy broader
  reasoning exposure for the same GRPO update path.

Dataset:
  ``openai/gsm8k`` (``main`` config, ``train`` split), 7,473 problems.
  Ground-truth answers live after a ``####`` separator in the
  ``answer`` field (e.g. ``"... #### 72"``).

Scoring:
  Miners produce a ``\\boxed{...}`` completion per the shared MATH
  system prompt. The validator's ``evaluate_math_trace`` already
  reads the miner's last boxed expression and compares against the
  task's ``reference_answer``. We extract the numeric answer from the
  GSM8K ``#### X`` tail and store it as the ``reference_answer`` — no
  verifier changes required.

Licensing:
  GSM8K is MIT-licensed (OpenAI 2021). Dataset is accessed via the HF
  Hub mirror.
"""

from __future__ import annotations

import hashlib
import re
from typing import ClassVar, Optional

# Matches the GSM8K solution tail: "... #### 42" / "... #### 42.5" / etc.
_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-+]?\d+(?:[.,]\d+)*)\s*$")


def extract_gsm8k_answer(answer_text: str) -> str:
    """Return the numeric final answer from a GSM8K ``answer`` field.

    GSM8K solutions end with ``#### <number>``. We strip that block,
    normalize commas, and return the bare number string. Returns
    ``""`` if no ``####`` tail is found so upstream code can drop the
    problem from the pool rather than score against a blank reference.
    """
    if not answer_text:
        return ""
    match = _GSM8K_ANSWER_RE.search(answer_text.strip())
    if not match:
        return ""
    # Drop comma thousands-separators (e.g. "1,234" → "1234") so the
    # validator's string-match tolerates the canonical form.
    return match.group(1).replace(",", "")


class GSM8KEnvironment:
    """Environment backed by the full GSM8K train set (7,473 problems).

    Mirrors :class:`MATHEnvironment` — lazy-loaded HF dataset cached at
    class level, ``get_problem(index)`` wraps modulo ``len(self)``,
    ``_resolve`` is an identity mapping (no difficulty filtering since
    GSM8K is uniformly ~Level 1-2 equivalent to MATH).

    Difficulty filtering is unsupported: GSM8K has no level column, and
    the problems span a narrow difficulty band (roughly MATH Level 1-2).
    Callers that want hard math filter ``math`` with ``max_level``
    instead, and optionally mix in GSM8K via :class:`MixedTasksSource`
    for baseline word-problem exposure.
    """

    name: str = "gsm8k"

    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        if GSM8KEnvironment._dataset_cache is None:
            import datasets as hf_datasets  # type: ignore[import-not-found]

            # ``openai/gsm8k`` requires specifying the "main" config.
            GSM8KEnvironment._dataset_cache = hf_datasets.load_dataset(
                "openai/gsm8k", "main", split="train"
            )
        self._dataset = GSM8KEnvironment._dataset_cache

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def _resolve(self, index: int) -> int:
        """Identity mapping — GSM8K has no filter layer."""
        return int(index) % len(self._dataset)  # type: ignore[arg-type]

    def get_problem(self, index: int) -> dict:
        """Return problem at ``index`` (wraps modulo ``len(self)``)."""
        idx = self._resolve(index)
        row = self._dataset[idx]  # type: ignore[index]
        question: str = row["question"]
        answer_field: str = row["answer"]
        numeric_answer = extract_gsm8k_answer(answer_field)
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        return {
            "prompt": question,
            "ground_truth": numeric_answer,
            "id": problem_id,
            "dataset_index": idx,
            # GSM8K has no explicit level; tag everything as "Level 1"
            # (grade-school) so the EvalBundle's per-level breakdown
            # has a consistent bucket for GSM8K problems.
            "level": "Level 1",
            # Subject bucket: GSM8K is entirely word-problem arithmetic.
            "subject": "Arithmetic Word Problems",
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        """Delegate to the shared MATH scorer — boxed-answer extraction
        plus LaTeX normalization handles both datasets identically."""
        from .math_env import compute_math_reward

        return compute_math_reward(problem, completion)


__all__ = [
    "GSM8KEnvironment",
    "extract_gsm8k_answer",
]
