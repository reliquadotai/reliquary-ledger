"""Integration tests for the upgraded logprob + distribution stages."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.validator.sampler_replay import SamplingParams, replay_logprob
from reliquary_inference.validator.validators.base import RejectReason, StageContext
from reliquary_inference.validator.validators.distribution import DistributionStage
from reliquary_inference.validator.validators.logprob import LogprobStage


def _logits_sequence(seed: int, n: int, vocab: int = 64) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(vocab, generator=gen, dtype=torch.float32) * 2.0 for _ in range(n)]


def _honest_payload(
    *,
    prompt_length: int = 4,
    n_generated: int = 16,
    seed: int = 42,
    vocab: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[dict, list[torch.Tensor]]:
    logits = _logits_sequence(seed=seed, n=n_generated, vocab=vocab)
    params = SamplingParams(temperature=temperature, top_p=top_p)
    tokens = list(range(prompt_length))
    logprobs: list[float] = []
    for i, l in enumerate(logits):
        chosen = int(torch.argmax(l).item())
        prior = tokens.copy()
        logprobs.append(replay_logprob(l, params, chosen, prior))
        tokens.append(chosen)
    payload = {
        "tokens": tokens,
        "logprobs": logprobs,
        "prompt_length": prompt_length,
        "sampling_params": {"temperature": temperature, "top_p": top_p, "repetition_penalty": 1.0},
        "logits_commitment": {"merkle_root_hex": "deadbeef", "chunk_size": 8, "num_chunks": 2},
    }
    return payload, logits


def _context(payload: dict, cached_logits: list[torch.Tensor]) -> StageContext:
    return StageContext(
        completion={"producer_id": "miner-1", "payload": payload},
        task_batch={},
        seen_nonces=set(),
        extras={"cached_logits": cached_logits},
    )


def test_logprob_stage_accepts_honest_completion() -> None:
    payload, logits = _honest_payload()
    result = LogprobStage().check(_context(payload, logits))
    assert result.passed is True, result.metadata
    assert result.metadata["fraction_passing"] == 1.0


def test_distribution_stage_accepts_honest_completion() -> None:
    payload, logits = _honest_payload()
    result = DistributionStage().check(_context(payload, logits))
    assert result.passed is True, result.metadata
    assert 0.9 <= result.metadata["median_ratio"] <= 1.1


def test_logprob_stage_hard_fails_when_logprobs_are_tampered() -> None:
    payload, logits = _honest_payload()
    payload["logprobs"] = [-5.0] * len(payload["logprobs"])
    result = LogprobStage().check(_context(payload, logits))
    assert result.passed is False
    assert result.reason is RejectReason.LOGPROB_DRIFT_EXCEEDED


def test_distribution_stage_soft_flags_tampered_tokens() -> None:
    payload, logits = _honest_payload()
    # Replace every chosen token with a low-probability one (last index of vocab).
    pl = payload["prompt_length"]
    tampered_tokens = payload["tokens"][:pl] + [logits[0].shape[0] - 1] * len(logits)
    payload["tokens"] = tampered_tokens
    result = DistributionStage().check(_context(payload, logits))
    assert result.passed is False
    assert result.soft_fail is True
    assert result.reason is RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND


def test_logprob_stage_passes_when_logprobs_absent() -> None:
    payload, logits = _honest_payload()
    del payload["logprobs"]
    result = LogprobStage().check(_context(payload, logits))
    assert result.passed is True
    assert result.metadata["status"] == "logprob_unavailable"


def test_distribution_stage_passes_when_commitment_absent() -> None:
    payload, logits = _honest_payload()
    del payload["logits_commitment"]
    result = DistributionStage().check(_context(payload, logits))
    assert result.passed is True
    assert result.metadata["status"] == "logit_cache_unavailable"


def test_distribution_stage_passes_with_insufficient_positions() -> None:
    payload, logits = _honest_payload(n_generated=4)
    result = DistributionStage().check(_context(payload, logits))
    assert result.passed is True
    assert result.metadata["status"] == "insufficient_positions"


def test_logprob_stage_rejects_length_mismatch() -> None:
    payload, logits = _honest_payload()
    payload["logprobs"] = payload["logprobs"][:-2]
    result = LogprobStage().check(_context(payload, logits))
    assert result.passed is False
    assert result.reason is RejectReason.LOGPROB_DRIFT_EXCEEDED


def test_logprob_stage_passes_when_cached_logits_missing() -> None:
    payload, _ = _honest_payload()
    ctx = StageContext(
        completion={"producer_id": "miner-1", "payload": payload},
        task_batch={},
        seen_nonces=set(),
    )
    result = LogprobStage().check(ctx)
    assert result.passed is True
    assert result.metadata["status"] == "logit_cache_not_loaded"
