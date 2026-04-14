from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..loader import deterministic_indices, get_prompt_by_index, load_dataset_cached, prompt_hash
from ..reasoning import evaluate_reasoning_trace, generate_reasoning_tasks, render_reasoning_conversation
from ...shared.modeling import load_tokenizer_for_model


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


def build_task_source(source_id: str) -> TaskSource:
    if source_id == "dataset_prompts":
        return DatasetPromptsSource()
    if source_id == "reasoning_tasks":
        return ReasoningTasksSource()
    raise ValueError(f"Unsupported task source: {source_id}")
