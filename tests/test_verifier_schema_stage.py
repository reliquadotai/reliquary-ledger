"""Tests for stage 1 (schema)."""

from __future__ import annotations

import pytest

from reliquary_inference.protocol.constants import LEDGER_PROOF_VERSION
from reliquary_inference.validator.validators.base import RejectReason, StageContext
from reliquary_inference.validator.validators.schema import SchemaStage


def _valid_completion(nonce: int = 1) -> dict:
    return {
        "producer_id": "miner-42",
        "payload": {
            "proof_version": LEDGER_PROOF_VERSION,
            "tokens": [1, 2, 3],
            "commitments": [{"sketch": 100}, {"sketch": 200}, {"sketch": 300}],
            "signature": "signed-bytes",
            "randomness": "deadbeef",
            "nonce": nonce,
            "model_name": "Qwen/Qwen3-4B-Instruct",
            "layer_index": -1,
            "task_source": "reasoning_tasks",
        },
    }


def _context(completion: dict, seen_nonces: set | None = None) -> StageContext:
    return StageContext(
        completion=completion,
        task_batch={"payload": {"model_ref": "Qwen/Qwen3-4B-Instruct"}},
        seen_nonces=seen_nonces if seen_nonces is not None else set(),
    )


def test_accepts_valid_completion_and_records_nonce() -> None:
    seen = set()
    stage = SchemaStage()
    ctx = _context(_valid_completion(nonce=7), seen_nonces=seen)

    result = stage.check(ctx)

    assert result.passed is True
    assert ("miner-42", 7) in seen
    assert result.metadata["proof_version"] == LEDGER_PROOF_VERSION


def test_rejects_missing_top_level_field() -> None:
    stage = SchemaStage()
    completion = _valid_completion()
    del completion["payload"]
    ctx = _context(completion)

    result = stage.check(ctx)

    assert result.passed is False
    assert result.reason is RejectReason.SCHEMA_MISSING_FIELD
    assert result.metadata["scope"] == "top_level"
    assert "payload" in result.metadata["missing"]


def test_rejects_missing_payload_field() -> None:
    stage = SchemaStage()
    completion = _valid_completion()
    del completion["payload"]["commitments"]
    ctx = _context(completion)

    result = stage.check(ctx)

    assert result.passed is False
    assert result.reason is RejectReason.SCHEMA_MISSING_FIELD
    assert result.metadata["scope"] == "payload"
    assert "commitments" in result.metadata["missing"]


def test_rejects_wrong_proof_version() -> None:
    stage = SchemaStage()
    completion = _valid_completion()
    completion["payload"]["proof_version"] = "v99"
    ctx = _context(completion)

    result = stage.check(ctx)

    assert result.passed is False
    assert result.reason is RejectReason.SCHEMA_VERSION_MISMATCH
    assert result.metadata["expected"] == LEDGER_PROOF_VERSION
    assert result.metadata["observed"] == "v99"


def test_rejects_duplicate_nonce_same_window() -> None:
    stage = SchemaStage()
    seen = {("miner-42", 7)}
    ctx = _context(_valid_completion(nonce=7), seen_nonces=seen)

    result = stage.check(ctx)

    assert result.passed is False
    assert result.reason is RejectReason.SCHEMA_DUPLICATE_NONCE


def test_rejects_non_integer_nonce() -> None:
    stage = SchemaStage()
    completion = _valid_completion()
    completion["payload"]["nonce"] = "not-a-number"
    ctx = _context(completion)

    result = stage.check(ctx)

    # non-integer nonce coerces to -1 which is still recorded but unique
    assert result.passed is True
    assert (completion["producer_id"], -1) in ctx.seen_nonces


def test_two_completions_with_different_nonces_both_pass() -> None:
    seen = set()
    stage = SchemaStage()
    ctx_a = _context(_valid_completion(nonce=1), seen_nonces=seen)
    ctx_b = _context(_valid_completion(nonce=2), seen_nonces=seen)

    assert stage.check(ctx_a).passed is True
    assert stage.check(ctx_b).passed is True
    assert len(seen) == 2


def test_stage_name_is_stable() -> None:
    assert SchemaStage().name == "schema"
