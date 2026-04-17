"""Tests for stage 5 (termination)."""

from __future__ import annotations

from types import SimpleNamespace

from reliquary_inference.validator.validators.base import RejectReason, StageContext
from reliquary_inference.validator.validators.termination import TerminationStage


def _context_with_tokens(
    tokens: list[int],
    *,
    eos_token_id: int | None = 0,
    max_position_embeddings: int | None = 1024,
    stop_sequences: list[list[int]] | None = None,
) -> StageContext:
    config = SimpleNamespace()
    if max_position_embeddings is not None:
        config.max_position_embeddings = max_position_embeddings
    model = SimpleNamespace(config=config)
    tokenizer = SimpleNamespace(eos_token_id=eos_token_id) if eos_token_id is not None else SimpleNamespace()

    extras: dict = {}
    if stop_sequences is not None:
        extras["stop_sequences"] = stop_sequences

    return StageContext(
        completion={"payload": {"tokens": tokens}},
        task_batch={},
        seen_nonces=set(),
        model=model,
        tokenizer=tokenizer,
        extras=extras,
    )


def test_accepts_completion_ending_with_eos_token() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens([1, 2, 3, 0], eos_token_id=0)

    result = stage.check(ctx)
    assert result.passed is True
    assert result.metadata["ended_with"] == "eos"


def test_accepts_completion_hitting_max_length_exactly() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens([1] * 32, eos_token_id=99, max_position_embeddings=32)

    result = stage.check(ctx)
    assert result.passed is True
    assert result.metadata["ended_with"] == "max_length"


def test_rejects_completion_exceeding_max_length() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens([1] * 33, eos_token_id=99, max_position_embeddings=32)

    result = stage.check(ctx)
    assert result.passed is False
    assert result.reason is RejectReason.TERMINATION_OVERFLOW


def test_rejects_no_eos_mid_context() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens([1, 2, 3, 4, 5], eos_token_id=0, max_position_embeddings=1024)

    result = stage.check(ctx)
    assert result.passed is False
    assert result.reason is RejectReason.TERMINATION_NO_EOS


def test_rejects_empty_tokens() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens([], eos_token_id=0)

    result = stage.check(ctx)
    assert result.passed is False
    assert result.reason is RejectReason.TERMINATION_NO_EOS


def test_accepts_completion_ending_with_documented_stop_sequence() -> None:
    stage = TerminationStage()
    ctx = _context_with_tokens(
        [1, 2, 3, 99, 100],
        eos_token_id=0,
        max_position_embeddings=1024,
        stop_sequences=[[99, 100]],
    )

    result = stage.check(ctx)
    assert result.passed is True
    assert result.metadata["ended_with"] == "stop_sequence"


def test_accepts_multiple_eos_variants_from_tokenizer_and_config() -> None:
    stage = TerminationStage()
    config = SimpleNamespace(max_position_embeddings=1024, eot_token_id=5)
    model = SimpleNamespace(config=config)
    tokenizer = SimpleNamespace(eos_token_id=2)
    ctx = StageContext(
        completion={"payload": {"tokens": [1, 2, 3, 5]}},
        task_batch={},
        seen_nonces=set(),
        model=model,
        tokenizer=tokenizer,
    )

    result = stage.check(ctx)
    assert result.passed is True
    assert result.metadata["ended_with"] == "eos"


def test_stage_name_is_stable() -> None:
    assert TerminationStage().name == "termination"
