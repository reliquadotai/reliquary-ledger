"""HuggingFace sampler replay for stages 8 (logprob) and 9 (distribution).

Re-runs the RepetitionPenalty → Temperature → TopP pipeline on a single
position's logits and returns the probability the sampler would have
assigned to the claimed token. Matches HF's ``LogitsProcessorList`` output
at float32 epsilon for use in independent importance-sampling checks.

This is deliberately stateless and chain-free: the stage pulls committed
logits + params from the miner's artifact, calls :func:`replay_probability`,
and compares against the miner's claimed logprob.

Spec: private/reliquary-plan/notes/spec-distribution-validator.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive (got {self.temperature!r})")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1] (got {self.top_p!r})")
        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0 (got {self.repetition_penalty!r}); "
                "values below 1.0 amplify repetition and are not supported"
            )


def replay_probability(
    logits,
    params: SamplingParams,
    chosen_token: int,
    prior_tokens,
) -> float:
    """Return the probability the sampler would have assigned to ``chosen_token``.

    Arguments:
        logits: 1-D torch.Tensor of shape [vocab_size], float32 preferred.
        params: SamplingParams (temperature, top_p, repetition_penalty).
        chosen_token: int token id the miner claims was sampled.
        prior_tokens: iterable of ints (prompt + previously-generated) used for
            the repetition-penalty computation.

    Returns 0.0 if ``chosen_token`` is outside the top-p nucleus. Otherwise
    returns the softmax probability at that index after the full pipeline.
    """
    import torch

    if logits.dim() != 1:
        raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
    vocab = logits.shape[0]
    if not (0 <= chosen_token < vocab):
        return 0.0

    logits = logits.detach().clone().to(torch.float32)

    if params.repetition_penalty != 1.0 and prior_tokens:
        prior_set = sorted({int(t) for t in prior_tokens if 0 <= int(t) < vocab})
        if prior_set:
            idx = torch.tensor(prior_set, device=logits.device, dtype=torch.long)
            prior_logits = logits[idx]
            penalized = torch.where(
                prior_logits > 0,
                prior_logits / params.repetition_penalty,
                prior_logits * params.repetition_penalty,
            )
            logits[idx] = penalized

    logits = logits / params.temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > params.top_p
    if mask.any():
        first_excluded = int(mask.to(torch.int64).argmax().item())
    else:
        first_excluded = vocab
    keep = first_excluded + 1 if first_excluded < vocab else vocab
    keep_indices = sorted_indices[:keep]
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[keep_indices] = True

    if not bool(keep_mask[chosen_token].item()):
        return 0.0

    neg_inf = torch.full_like(logits, fill_value=float("-inf"))
    filtered = torch.where(keep_mask, logits, neg_inf)
    probs = torch.softmax(filtered, dim=-1)
    return float(probs[chosen_token].item())


def replay_logprob(
    logits,
    params: SamplingParams,
    chosen_token: int,
    prior_tokens,
    eps: float = 1e-30,
) -> float:
    """Log-probability equivalent of :func:`replay_probability`.

    Returns ``math.log(eps)`` for tokens outside the nucleus so the caller can
    operate entirely in log-space without division-by-zero surprises.
    """
    p = replay_probability(logits, params, chosen_token, prior_tokens)
    return math.log(max(p, eps))


def median_importance_ratio(
    replay_probs: list[float],
    miner_probs: list[float],
    eps: float = 1e-12,
) -> float:
    """Median of ``p_replay / p_miner`` across positions.

    Both input lists must be the same length. Positions with `miner_probs[i] <= 0`
    contribute `replay / eps` (large number) so they pull the median outside
    any sensible band — reflecting that a miner claiming a zero-probability
    token is unambiguously wrong.
    """
    if len(replay_probs) != len(miner_probs):
        raise ValueError("replay_probs and miner_probs must have equal length")
    if not replay_probs:
        raise ValueError("need at least one position")
    ratios = [
        r / max(m, eps)
        for r, m in zip(replay_probs, miner_probs)
    ]
    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)
    if n % 2 == 1:
        return ratios_sorted[n // 2]
    return 0.5 * (ratios_sorted[n // 2 - 1] + ratios_sorted[n // 2])
