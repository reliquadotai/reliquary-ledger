"""Tests for the verifier stage base primitives.

Spec reference: private/reliquary-plan/notes/spec-nine-stage-verifier.md invariants 1-7.
"""

from __future__ import annotations

import pytest

from reliquary_inference.validator.validators.base import (
    STAGE_ORDER,
    RejectReason,
    StageContext,
    StageResult,
    accept,
    reject,
    soft_flag,
)


def test_stage_order_contains_all_nine_stages() -> None:
    assert STAGE_ORDER == (
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
    assert len(STAGE_ORDER) == 9


def test_reject_reason_enum_is_exhaustive_over_nine_stages() -> None:
    stage_prefixes = {
        "schema",
        "tokens",
        "prompt",
        "proof",
        "termination",
        "environment",
        "reward",
        "logprob",
        "distribution",
    }
    found = {member.value.split("_", 1)[0] for member in RejectReason}
    assert stage_prefixes.issubset(found), (
        f"stages missing a reject reason: {stage_prefixes - found}"
    )


def test_accept_helper_produces_pass_result() -> None:
    result = accept("schema", {"k": "v"})
    assert result.passed is True
    assert result.soft_fail is False
    assert result.reason is None
    assert result.metadata == {"k": "v"}
    assert result.stage == "schema"


def test_reject_helper_produces_hard_fail_with_reason() -> None:
    result = reject("tokens", RejectReason.TOKENS_OUT_OF_VOCAB, {"vocab_size": 32000})
    assert result.passed is False
    assert result.soft_fail is False
    assert result.reason is RejectReason.TOKENS_OUT_OF_VOCAB
    assert result.metadata == {"vocab_size": 32000}


def test_soft_flag_helper_marks_soft_fail() -> None:
    result = soft_flag("distribution", RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND, {"median_w": 1.35})
    assert result.passed is False
    assert result.soft_fail is True
    assert result.reason is RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND


def test_stage_result_post_init_rejects_passed_with_reason() -> None:
    with pytest.raises(ValueError):
        StageResult(
            stage="schema",
            passed=True,
            reason=RejectReason.SCHEMA_VERSION_MISMATCH,
        )


def test_stage_result_post_init_requires_reason_on_hard_fail() -> None:
    with pytest.raises(ValueError):
        StageResult(stage="tokens", passed=False, soft_fail=False, reason=None)


def test_stage_result_allows_soft_fail_without_reason() -> None:
    result = StageResult(stage="distribution", passed=False, soft_fail=True, reason=None)
    assert result.soft_fail is True
    assert result.reason is None


def test_stage_context_payload_and_producer_id_accessors() -> None:
    ctx = StageContext(
        completion={"producer_id": "miner-42", "payload": {"proof_version": "v1"}},
        task_batch={},
        seen_nonces=set(),
    )
    assert ctx.producer_id == "miner-42"
    assert ctx.payload == {"proof_version": "v1"}


def test_stage_context_handles_missing_completion_fields() -> None:
    ctx = StageContext(completion={}, task_batch={}, seen_nonces=set())
    assert ctx.producer_id == ""
    assert ctx.payload == {}
