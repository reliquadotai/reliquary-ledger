"""Stage 6: environment-specific semantic evaluation.

Delegates to the task source's ``evaluate_completion`` which scores the
completion under the environment's rules (arithmetic correctness, reasoning
trace acceptance, SAT assignment validity, etc.).
"""

from __future__ import annotations

from typing import Any

from .base import RejectReason, StageContext, StageResult, accept, reject


class EnvironmentStage:
    name: str = "environment"

    def check(self, context: StageContext) -> StageResult:
        from ...dataset.task_sources import build_task_source

        task_source_name = context.payload.get("task_source")
        try:
            task_source = build_task_source(task_source_name)
        except Exception as exc:
            return reject(
                self.name,
                RejectReason.ENVIRONMENT_UNKNOWN,
                {"task_source": task_source_name, "exc": type(exc).__name__, "detail": str(exc)[:160]},
            )

        task = context.extras.get("task")
        if task is None:
            summary = context.extras.get("task_binding_summary")
            if isinstance(summary, dict):
                task = summary.get("task")
        if task is None:
            return reject(
                self.name,
                RejectReason.ENVIRONMENT_UNKNOWN,
                {"reason": "task_unavailable_from_prompt_stage"},
            )

        semantic_result = task_source.evaluate_completion(context.completion, task)
        context.extras["semantic_result"] = semantic_result
        context.extras["semantic_evaluation"] = (
            semantic_result.get("evaluation", {}) if isinstance(semantic_result, dict) else {}
        )

        if not isinstance(semantic_result, dict):
            return reject(
                self.name,
                RejectReason.ENVIRONMENT_FAILED_EVALUATION,
                {"reason": "non_dict_result"},
            )

        if not semantic_result.get("accepted", False):
            context.extras["environment_reject_reason"] = semantic_result.get("reason")
            return reject(
                self.name,
                RejectReason.ENVIRONMENT_FAILED_EVALUATION,
                {"reason": semantic_result.get("reason"), "evaluation": semantic_result.get("evaluation", {})},
            )

        return accept(
            self.name,
            {"evaluation": semantic_result.get("evaluation", {})},
        )
