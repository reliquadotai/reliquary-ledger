"""Shared primitives for the nine-stage verifier pipeline.

Spec reference: private/reliquary-plan/notes/spec-nine-stage-verifier.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RejectReason(str, Enum):
    """Canonical enum of every reason the pipeline may reject a completion.

    Free-text reasons are forbidden downstream; this enum is the sole vocabulary
    emitted to metrics, audit artifacts, and miner feedback channels.
    """

    # Stage 1: schema
    SCHEMA_MISSING_FIELD = "schema_missing_field"
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    SCHEMA_DUPLICATE_NONCE = "schema_duplicate_nonce"

    # Stage 2: tokens
    TOKENS_OUT_OF_VOCAB = "tokens_out_of_vocab"
    TOKENS_LENGTH_EXCEEDED = "tokens_length_exceeded"

    # Stage 3: prompt binding
    PROMPT_BINDING_MISMATCH = "prompt_binding_mismatch"
    PROMPT_SOURCE_UNKNOWN = "prompt_source_unknown"

    # Stage 4: proof
    PROOF_SKETCH_MISMATCH = "proof_sketch_mismatch"
    PROOF_NO_POSITIONS_CHECKED = "proof_no_positions_checked"

    # Stage 5: termination
    TERMINATION_NO_EOS = "termination_no_eos"
    TERMINATION_OVERFLOW = "termination_overflow"

    # Stage 6: environment
    ENVIRONMENT_FAILED_EVALUATION = "environment_failed_evaluation"
    ENVIRONMENT_UNKNOWN = "environment_unknown"

    # Stage 7: reward
    REWARD_CONTRACT_VIOLATION = "reward_contract_violation"
    REWARD_MISSING = "reward_missing"

    # Stage 8: logprob
    LOGPROB_DRIFT_EXCEEDED = "logprob_drift_exceeded"
    LOGPROB_MISSING = "logprob_missing"

    # Stage 9: distribution (soft)
    DISTRIBUTION_MEDIAN_OUT_OF_BAND = "distribution_median_out_of_band"

    # Pre-pipeline / orthogonal
    SIGNATURE_INVALID = "signature_invalid"

    # Pipeline meta
    PIPELINE_STAGE_EXCEPTION = "pipeline_stage_exception"
    PIPELINE_MODEL_NOT_LOADED = "pipeline_model_not_loaded"


# Canonical stage names matching the order in the PRD.
STAGE_ORDER: tuple[str, ...] = (
    "schema",
    "tokens",
    "prompt",
    "proof",
    "termination",
    "environment",
    "reward",
    "logprob",
    "distribution",
)


@dataclass
class StageResult:
    """Outcome of a single stage's check."""

    stage: str
    passed: bool
    soft_fail: bool = False
    reason: RejectReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed and self.reason is not None:
            raise ValueError(
                f"stage={self.stage}: passed=True cannot coexist with reason={self.reason}"
            )
        if not self.passed and not self.soft_fail and self.reason is None:
            raise ValueError(
                f"stage={self.stage}: hard-fail StageResult must carry a reason"
            )


@dataclass
class StageContext:
    """Immutable-ish input bundle handed to each stage.

    Stages may read any field. The only mutation permitted is appending nonces
    to ``seen_nonces``; all other state flows via StageResult.metadata.
    """

    completion: dict[str, Any]
    task_batch: dict[str, Any]
    seen_nonces: set[tuple[str, int]]
    model: Any = None
    tokenizer: Any = None
    randomness: str = ""
    signing_secret: bytes | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> dict[str, Any]:
        return self.completion.get("payload", {})

    @property
    def producer_id(self) -> str:
        return str(self.completion.get("producer_id", ""))


class VerifierStage(Protocol):
    """Each stage is any object with a stage name and a ``check`` method."""

    name: str

    def check(self, context: StageContext) -> StageResult: ...


def accept(stage: str, metadata: dict[str, Any] | None = None) -> StageResult:
    """Helper: canonical accepted StageResult."""
    return StageResult(stage=stage, passed=True, metadata=dict(metadata or {}))


def reject(
    stage: str,
    reason: RejectReason,
    metadata: dict[str, Any] | None = None,
) -> StageResult:
    """Helper: canonical hard-fail StageResult."""
    return StageResult(
        stage=stage,
        passed=False,
        soft_fail=False,
        reason=reason,
        metadata=dict(metadata or {}),
    )


def soft_flag(
    stage: str,
    reason: RejectReason,
    metadata: dict[str, Any] | None = None,
) -> StageResult:
    """Helper: canonical soft-fail StageResult (pipeline continues)."""
    return StageResult(
        stage=stage,
        passed=False,
        soft_fail=True,
        reason=reason,
        metadata=dict(metadata or {}),
    )
