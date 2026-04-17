"""Nine-stage verifier pipeline.

Spec reference: private/reliquary-plan/notes/spec-nine-stage-verifier.md.

Usage:

    context = StageContext(completion=..., task_batch=..., seen_nonces=..., model=..., tokenizer=...)
    verdict = run_pipeline(default_stages(), context)
    if verdict.accepted: ...
    else: log_rejection(verdict.stage_failed, verdict.reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .validators.base import (
    RejectReason,
    StageContext,
    StageResult,
    VerifierStage,
)
from .validators.distribution import DistributionStage
from .validators.environment import EnvironmentStage
from .validators.logprob import LogprobStage
from .validators.prompt import PromptStage
from .validators.proof import ProofStage
from .validators.reward import RewardStage
from .validators.schema import SchemaStage
from .validators.termination import TerminationStage
from .validators.tokens import TokensStage


@dataclass
class VerdictResult:
    """Final outcome of running the pipeline on a single completion."""

    accepted: bool
    stage_failed: str | None = None
    reason: RejectReason | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    soft_flags: list[StageResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class StagePolicy:
    """Knobs for the operator: which stages are enabled, soft threshold."""

    enabled_stages: set[str] | None = None
    soft_fail_threshold: float = 0.51

    def is_enabled(self, name: str) -> bool:
        if self.enabled_stages is None:
            return True
        return name in self.enabled_stages


def default_stages() -> list[VerifierStage]:
    """Canonical ordered stages. Do not reorder in deployments — fix bugs
    upstream instead."""
    return [
        SchemaStage(),
        TokensStage(),
        PromptStage(),
        ProofStage(),
        TerminationStage(),
        EnvironmentStage(),
        RewardStage(),
        LogprobStage(),
        DistributionStage(),
    ]


def run_pipeline(
    stages: Sequence[VerifierStage],
    context: StageContext,
    policy: StagePolicy | None = None,
) -> VerdictResult:
    """Execute ``stages`` in order against ``context``.

    Hard failures short-circuit; soft failures (distribution stage) are
    appended to ``soft_flags`` but do not halt execution.
    """
    policy = policy or StagePolicy()
    stage_results: list[StageResult] = []
    soft_flags: list[StageResult] = []

    for stage in stages:
        if not policy.is_enabled(stage.name):
            continue
        try:
            result = stage.check(context)
        except Exception as exc:  # pragma: no cover — defensive
            result = StageResult(
                stage=stage.name,
                passed=False,
                soft_fail=False,
                reason=RejectReason.PIPELINE_STAGE_EXCEPTION,
                metadata={"exc": type(exc).__name__, "detail": str(exc)[:240]},
            )
            stage_results.append(result)
            return VerdictResult(
                accepted=False,
                stage_failed=stage.name,
                reason=result.reason,
                stage_results=stage_results,
                soft_flags=soft_flags,
                metadata=_summary(context),
            )

        stage_results.append(result)

        if result.passed:
            continue

        if result.soft_fail:
            soft_flags.append(result)
            continue

        return VerdictResult(
            accepted=False,
            stage_failed=stage.name,
            reason=result.reason,
            stage_results=stage_results,
            soft_flags=soft_flags,
            metadata=_summary(context),
        )

    return VerdictResult(
        accepted=True,
        stage_failed=None,
        reason=None,
        stage_results=stage_results,
        soft_flags=soft_flags,
        metadata=_summary(context),
    )


def _summary(context: StageContext) -> dict:
    return {
        "producer_id": context.producer_id,
        "checked_positions": context.extras.get("checked_positions"),
        "passed_positions": context.extras.get("passed_positions"),
        "task_source": context.payload.get("task_source"),
    }
