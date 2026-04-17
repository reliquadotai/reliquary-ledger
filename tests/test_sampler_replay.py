"""Sampler replay parity + behavior tests.

Spec reference: private/reliquary-plan/notes/spec-distribution-validator.md.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.validator.sampler_replay import (
    SamplingParams,
    median_importance_ratio,
    replay_logprob,
    replay_probability,
)


def _logits(seed: int, vocab: int = 64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, generator=gen, dtype=torch.float32) * 2.0


def test_sampling_params_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.0, top_p=0.9)


def test_sampling_params_rejects_invalid_top_p() -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.7, top_p=1.5)
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.7, top_p=0.0)


def test_sampling_params_rejects_below_one_repetition_penalty() -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.7, top_p=0.9, repetition_penalty=0.95)


def test_replay_probability_returns_zero_for_out_of_vocab_token() -> None:
    logits = _logits(seed=1)
    params = SamplingParams(temperature=0.7, top_p=0.9)
    assert replay_probability(logits, params, chosen_token=-1, prior_tokens=[]) == 0.0
    assert replay_probability(logits, params, chosen_token=10_000, prior_tokens=[]) == 0.0


def test_replay_probability_sums_over_nucleus_approximately_one() -> None:
    """Nucleus probs should sum to 1 after renormalization."""
    logits = _logits(seed=2)
    params = SamplingParams(temperature=0.7, top_p=0.9)
    vocab = logits.shape[0]
    total = sum(
        replay_probability(logits, params, chosen_token=i, prior_tokens=[])
        for i in range(vocab)
    )
    assert math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4)


def test_replay_probability_returns_zero_for_token_outside_top_p() -> None:
    logits = torch.tensor([10.0, 0.0, -10.0], dtype=torch.float32)
    params = SamplingParams(temperature=1.0, top_p=0.5)
    assert replay_probability(logits, params, chosen_token=2, prior_tokens=[]) == 0.0
    assert replay_probability(logits, params, chosen_token=0, prior_tokens=[]) > 0.9


def test_replay_probability_determinism() -> None:
    logits = _logits(seed=3)
    params = SamplingParams(temperature=0.7, top_p=0.9, repetition_penalty=1.05)
    values = [
        replay_probability(logits, params, chosen_token=5, prior_tokens=[1, 2, 3])
        for _ in range(10)
    ]
    assert len(set(values)) == 1


def test_repetition_penalty_reduces_prior_token_probability() -> None:
    logits = torch.tensor([5.0, 1.0, -1.0], dtype=torch.float32)
    params_no_rep = SamplingParams(temperature=1.0, top_p=1.0, repetition_penalty=1.0)
    params_rep = SamplingParams(temperature=1.0, top_p=1.0, repetition_penalty=2.0)
    p_no_rep = replay_probability(logits, params_no_rep, chosen_token=0, prior_tokens=[0])
    p_rep = replay_probability(logits, params_rep, chosen_token=0, prior_tokens=[0])
    assert p_rep < p_no_rep


def test_replay_matches_hf_logits_processor_pipeline() -> None:
    """Parity check against HuggingFace's canonical LogitsProcessorList."""
    transformers = pytest.importorskip("transformers")
    from transformers import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    vocab = 128
    torch.manual_seed(42)
    raw_logits = torch.randn(vocab, dtype=torch.float32)
    prior_tokens = [3, 7, 13]
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.1

    input_ids = torch.tensor([prior_tokens], dtype=torch.long)
    batched = raw_logits.unsqueeze(0).clone()
    processors = LogitsProcessorList(
        [
            RepetitionPenaltyLogitsProcessor(repetition_penalty),
            TemperatureLogitsWarper(temperature),
            TopPLogitsWarper(top_p),
        ]
    )
    hf_filtered = processors(input_ids, batched)
    hf_probs = torch.softmax(hf_filtered[0], dim=-1)

    params = SamplingParams(
        temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty
    )
    for token in range(vocab):
        ours = replay_probability(raw_logits, params, token, prior_tokens)
        theirs = float(hf_probs[token].item())
        assert math.isclose(ours, theirs, rel_tol=1e-4, abs_tol=1e-6), (
            f"token {token}: ours={ours} vs HF={theirs}"
        )


def test_replay_logprob_matches_log_of_replay_probability() -> None:
    logits = _logits(seed=5)
    params = SamplingParams(temperature=0.7, top_p=0.9)
    for token in (0, 3, 10, 42):
        p = replay_probability(logits, params, token, prior_tokens=[])
        lp = replay_logprob(logits, params, token, prior_tokens=[])
        if p > 0:
            assert math.isclose(lp, math.log(p), rel_tol=1e-5)
        else:
            assert lp < -30.0


def test_median_importance_ratio_honest_centered_on_one() -> None:
    # If replay and miner probs match closely, median ratio ≈ 1.0
    replay = [0.1, 0.2, 0.3, 0.4, 0.5]
    miner = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert median_importance_ratio(replay, miner) == pytest.approx(1.0)


def test_median_importance_ratio_detects_tampering() -> None:
    # Miner over-reports probabilities → median ratio drops below 1.
    replay = [0.1, 0.1, 0.1, 0.1, 0.1]
    miner = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert median_importance_ratio(replay, miner) == pytest.approx(0.2)


def test_median_importance_ratio_handles_zero_miner_probability() -> None:
    replay = [0.5]
    miner = [0.0]
    assert median_importance_ratio(replay, miner) > 1e6


def test_median_importance_ratio_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        median_importance_ratio([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        median_importance_ratio([], [])
