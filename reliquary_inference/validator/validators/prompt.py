"""Stage 3: prompt binding.

Confirms the miner's tokens encode the task/prompt advertised in the window's
task batch. Delegates to the task source's own ``verify_task_binding`` which
is responsible for the hash-binding semantics of each environment family.
"""

from __future__ import annotations

from typing import Any

from .base import RejectReason, StageContext, StageResult, accept, reject


class PromptStage:
    name: str = "prompt"

    def check(self, context: StageContext) -> StageResult:
        from ...dataset.task_sources import build_task_source

        task_source_name = context.payload.get("task_source")
        try:
            task_source = build_task_source(task_source_name)
        except Exception as exc:
            return reject(
                self.name,
                RejectReason.PROMPT_SOURCE_UNKNOWN,
                {"task_source": task_source_name, "exc": type(exc).__name__, "detail": str(exc)[:160]},
            )

        tokenizer = context.tokenizer
        ok, summary = task_source.verify_task_binding(
            context.completion, context.task_batch, tokenizer
        )
        if not ok:
            return reject(
                self.name,
                RejectReason.PROMPT_BINDING_MISMATCH,
                {"summary": _safe_dict(summary)},
            )

        context.extras["task_binding_summary"] = summary
        context.extras["task"] = summary.get("task") if isinstance(summary, dict) else None
        return accept(self.name, {"summary": _safe_dict(summary)})


def _safe_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items() if isinstance(k, str)}
    return {"value": str(obj)[:160]}
