from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...shared.modeling import load_tokenizer_for_model
from ..loader import (
    deterministic_indices,
    get_prompt_by_index,
    load_dataset_cached,
    prompt_hash,
)
from ..reasoning import (
    evaluate_reasoning_trace,
    generate_reasoning_tasks,
    render_reasoning_conversation,
)


class TaskSource(Protocol):
    source_id: str

    def build_window_batch(self, window_context: dict[str, Any], count: int) -> dict[str, Any]:
        ...

    def verify_task_binding(
        self,
        completion: dict[str, Any],
        task_batch: dict[str, Any],
        tokenizer: Any,
    ) -> tuple[bool, dict[str, Any]]:
        ...

    def evaluate_completion(self, completion: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class DatasetPromptsSource:
    source_id: str = "dataset_prompts"

    def build_window_batch(self, window_context: dict[str, Any], count: int) -> dict[str, Any]:
        dataset = load_dataset_cached(
            dataset_name=str(window_context.get("dataset_name", "")) or "karpathy/climbmix-400b-shuffle",
            split=str(window_context.get("dataset_split", "train")),
        )
        seed_material = f"{window_context['window_id']}|{window_context['public_randomness']}|dataset"
        indices = deterministic_indices(seed_material=seed_material, dataset_size=len(dataset), count=count)
        tasks: list[dict[str, Any]] = []
        for order, dataset_index in enumerate(indices):
            prompt = get_prompt_by_index(dataset, dataset_index)
            if prompt is None:
                continue
            tasks.append(
                {
                    "task_id": f"dataset-{window_context['window_id']}-{dataset_index}",
                    "dataset_index": dataset_index,
                    "order_index": order,
                    "prompt": prompt,
                    "prompt_hash": prompt_hash(prompt),
                    "tags": [f"dataset_index:{dataset_index}"],
                    "verification_mode": "prompt_prefix",
                }
            )
        return {
            "task_source": self.source_id,
            "window_id": window_context["window_id"],
            "public_randomness": window_context["public_randomness"],
            "model_ref": window_context["model_ref"],
            "tasks": tasks,
        }

    def verify_task_binding(
        self,
        completion: dict[str, Any],
        task_batch: dict[str, Any],
        tokenizer: Any,
    ) -> tuple[bool, dict[str, Any]]:
        task = _task_lookup(task_batch).get(completion["payload"]["task_id"])
        if task is None:
            return False, {"reason": "unknown_task_id"}
        expected_tokens = tokenizer.encode(task["prompt"], add_special_tokens=False)
        tokens = completion["payload"]["tokens"]
        prompt_length = completion["payload"]["prompt_length"]
        ok = (
            completion["payload"]["task_source"] == self.source_id
            and completion["payload"].get("prompt_hash") == task["prompt_hash"]
            and prompt_length == len(expected_tokens)
            and tokens[:prompt_length] == expected_tokens
        )
        return ok, {"reason": "ok" if ok else "prompt_prefix_mismatch", "task": task}

    def evaluate_completion(self, completion: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, "reason": "ok", "task": task}


@dataclass
class ReasoningTasksSource:
    source_id: str = "reasoning_tasks"

    def build_window_batch(self, window_context: dict[str, Any], count: int) -> dict[str, Any]:
        seed = int(str(window_context["public_randomness"])[:8], 16)
        model_ref = str(window_context["model_ref"])
        tokenizer = None
        if not model_ref.startswith("toy://"):
            try:
                tokenizer = load_tokenizer_for_model(model_ref)
            except Exception:
                tokenizer = None
        tasks = generate_reasoning_tasks(count=count, seed=seed, split="train", task_family=self.source_id)
        normalized = []
        for task in tasks:
            rendered_prompt = task["prompt"]
            if not model_ref.startswith("toy://"):
                rendered_prompt = render_reasoning_conversation(
                    task["prompt"],
                    tokenizer=tokenizer,
                    add_generation_prompt=True,
                )
            normalized.append(
                {
                    **task,
                    "source_prompt": task["prompt"],
                    "prompt": rendered_prompt,
                    "prompt_hash": prompt_hash(rendered_prompt),
                    "verification_mode": "exact_final_answer",
                }
            )
        return {
            "task_source": self.source_id,
            "window_id": window_context["window_id"],
            "public_randomness": window_context["public_randomness"],
            "model_ref": window_context["model_ref"],
            "tasks": normalized,
        }

    def verify_task_binding(
        self,
        completion: dict[str, Any],
        task_batch: dict[str, Any],
        tokenizer: Any,
    ) -> tuple[bool, dict[str, Any]]:
        task = _task_lookup(task_batch).get(completion["payload"]["task_id"])
        if task is None:
            return False, {"reason": "unknown_task_id"}
        expected_tokens = tokenizer.encode(task["prompt"], add_special_tokens=False)
        tokens = completion["payload"]["tokens"]
        prompt_length = completion["payload"]["prompt_length"]
        ok = (
            completion["payload"]["task_source"] == self.source_id
            and completion["payload"].get("prompt_hash") == task["prompt_hash"]
            and prompt_length == len(expected_tokens)
            and tokens[:prompt_length] == expected_tokens
        )
        return ok, {"reason": "ok" if ok else "prompt_hash_mismatch", "task": task}

    def evaluate_completion(self, completion: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        contamination_tags = set(completion["payload"].get("contamination_tags", []))
        forbidden = set(task.get("contamination_policy", {}).get("forbidden_overlap_tags", []))
        if contamination_tags & forbidden:
            return {
                "accepted": False,
                "reason": "contamination_detected",
                "task": task,
                "evaluation": {},
            }
        evaluation = evaluate_reasoning_trace(task, completion["payload"]["completion_text"])
        return {
            "accepted": bool(evaluation["accepted"]),
            "reason": "ok" if evaluation["accepted"] else evaluation["format_reason"],
            "task": task,
            "evaluation": evaluation,
        }


def _task_lookup(task_batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in task_batch["payload"]["tasks"]}


# --- Hendrycks MATH source (real reasoning benchmark, replaces synthetic arithmetic) ---


MATH_SYSTEM_PROMPT = (
    "You are a mathematics expert. Solve the problem step by step. "
    "Show your reasoning, then give the final answer enclosed in "
    "\\boxed{...}. The reward depends on the boxed answer matching "
    "the ground truth after LaTeX normalization, so only the content "
    "inside the FINAL \\boxed{...} is graded."
)


def _render_math_conversation(problem_text: str, tokenizer: Any) -> str:
    """Wrap a MATH problem in the model's chat template with a system prompt
    that prescribes the ``\\boxed{...}`` answer format the validator scores on.

    Falls back to a plain prompt if the tokenizer has no chat template.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem_text}\n\nSolution:"
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": problem_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem_text}\n\nSolution:"


@dataclass
class MathTasksSource:
    """Hendrycks MATH environment exposed as a Reliquary TaskSource.

    Task selection is deterministic per (window_id, public_randomness):
    both miner and every validator derive the same problem-index set
    independently. Optional ``cooldown_indices`` arg lets the caller
    exclude recently-trained prompts (curriculum diversity).

    Reward verification uses ``evaluate_math_trace`` — boxed-answer
    extraction + LaTeX-normalized exact match, no LLM judge.
    """

    source_id: str = "math"
    max_level: int | None = None
    _env: Any = None  # lazy-loaded MATHEnvironment

    def _environment(self):
        if self._env is None:
            from .math_env import MATHEnvironment
            self._env = MATHEnvironment(max_level=self.max_level)
        return self._env

    def build_window_batch(self, window_context: dict[str, Any], count: int) -> dict[str, Any]:
        import random

        env = self._environment()
        seed = int(str(window_context["public_randomness"])[:8], 16)
        rng = random.Random(seed)
        model_ref = str(window_context["model_ref"])
        tokenizer = None
        if not model_ref.startswith("toy://"):
            try:
                tokenizer = load_tokenizer_for_model(model_ref)
            except Exception:
                tokenizer = None

        cooldown: set[int] = set(window_context.get("cooldown_indices") or [])
        n_problems = len(env)
        picked: set[int] = set()
        tasks: list[dict[str, Any]] = []
        attempts = 0
        while len(tasks) < count and attempts < count * 200:
            attempts += 1
            idx = rng.randrange(n_problems)
            if idx in picked or idx in cooldown:
                continue
            picked.add(idx)
            problem = env.get_problem(idx)
            rendered = _render_math_conversation(problem["prompt"], tokenizer)
            tasks.append(
                {
                    "task_id": f"math-{window_context['window_id']}-{idx}",
                    "task_family": self.source_id,
                    "split": "train",
                    "seed": seed,
                    "dataset_index": idx,
                    "prompt": rendered,
                    "source_prompt": problem["prompt"],
                    "prompt_hash": prompt_hash(rendered),
                    "reference_answer": problem["ground_truth"],
                    "problem_id": problem["id"],
                    "difficulty": 0.70,
                    "template_id": "math-boxed-v1",
                    "evaluation_policy": {"mode": "exact_final_answer"},
                    "contamination_policy": {"forbidden_overlap_tags": ["benchmark_holdout"]},
                    "tags": [f"problem_id:{problem['id']}", "split:train", "source:hendrycks_math"],
                    "verification_mode": "exact_final_answer",
                }
            )
        return {
            "task_source": self.source_id,
            "window_id": window_context["window_id"],
            "public_randomness": window_context["public_randomness"],
            "model_ref": window_context["model_ref"],
            "tasks": tasks,
        }

    def verify_task_binding(
        self,
        completion: dict[str, Any],
        task_batch: dict[str, Any],
        tokenizer: Any,
    ) -> tuple[bool, dict[str, Any]]:
        task = _task_lookup(task_batch).get(completion["payload"]["task_id"])
        if task is None:
            return False, {"reason": "unknown_task_id"}
        expected_tokens = tokenizer.encode(task["prompt"], add_special_tokens=False)
        tokens = completion["payload"]["tokens"]
        prompt_length = completion["payload"]["prompt_length"]
        ok = (
            completion["payload"]["task_source"] == self.source_id
            and completion["payload"].get("prompt_hash") == task["prompt_hash"]
            and prompt_length == len(expected_tokens)
            and tokens[:prompt_length] == expected_tokens
        )
        return ok, {"reason": "ok" if ok else "prompt_hash_mismatch", "task": task}

    def evaluate_completion(self, completion: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        from .math_env import evaluate_math_trace

        contamination_tags = set(completion["payload"].get("contamination_tags", []))
        forbidden = set(task.get("contamination_policy", {}).get("forbidden_overlap_tags", []))
        if contamination_tags & forbidden:
            return {
                "accepted": False,
                "reason": "contamination_detected",
                "task": task,
                "evaluation": {},
            }
        evaluation = evaluate_math_trace(task, completion["payload"]["completion_text"])
        return {
            "accepted": bool(evaluation["accepted"]),
            "reason": "ok" if evaluation["accepted"] else evaluation["format_reason"],
            "task": task,
            "evaluation": evaluation,
        }


def build_task_source(source_id: str, *, max_level: int | None = None) -> TaskSource:
    if source_id == "dataset_prompts":
        return DatasetPromptsSource()
    if source_id == "reasoning_tasks":
        return ReasoningTasksSource()
    if source_id == "math":
        return MathTasksSource(max_level=max_level)
    raise ValueError(f"Unsupported task source: {source_id}")
