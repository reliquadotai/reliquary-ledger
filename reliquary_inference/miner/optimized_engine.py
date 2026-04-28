"""Optimized mining engine — frontier-σ prompt picker + cooldown-aware.

Reference implementation of the miner competitive surface. The base
``MiningEngine`` mines uniform-random prompts from the window's task
batch. This subclass demonstrates how a competitive miner picks
prompts to maximize expected acceptance + maximize σ at the validator's
zone filter (so the rollouts are training-eligible, not just
verifier-passing).

Competitive surfaces this engine optimizes:

1. **Frontier-σ prompt selection.** A lightweight predictor scores
   each candidate prompt by how likely the M=8 rollouts will land in
   the σ ≥ 0.43 zone. Score is the next-token entropy of the model's
   forward pass on the prompt (one cheap forward per candidate, no
   generation). High entropy → uncertain policy → more rollout
   diversity → more likely in-zone. Picking prompts the model is
   uncertain on **also** maximizes the gradient signal that flows
   back to the trainer through the closed-loop bridge — so the
   miner is rewarded for picking prompts that move the policy.

2. **Cooldown awareness.** Prompts the validator has already batched
   in the last ``BATCH_PROMPT_COOLDOWN_WINDOWS`` are dropped before
   ranking, so we never waste a generation on a guaranteed-rejected
   submission.

3. **Early-submit posture.** The base engine submits as soon as the
   batch is built; we keep that. The validator's ``signed_round``
   FIFO ordering means submitting in the first half of the window
   maximizes our slot-grab probability.

4. **Local σ gate.** After generating M rollouts and scoring the
   reward locally, the engine optionally drops the submission if
   the actual measured σ is below the validator's threshold. This
   trades throughput for acceptance rate; useful when network
   bandwidth or chain-side rate-limiting is the constraint.

Wired in via env: ``RELIQUARY_INFERENCE_MINER_OPTIMIZED=1`` switches
the CLI ``run-miner`` path to instantiate ``OptimizedMiningEngine``
instead of the base ``MiningEngine``.

Honest framing: this is a **reference**, not the production-optimal
miner. A truly maximal miner would maintain a per-prompt difficulty
prior, replay sampled rollouts cheaply on a smaller draft model,
and adapt its frontier-σ band to the validator's measured filter.
We expose enough hooks here that a downstream operator can swap in
those refinements without forking the engine.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_base():
    """Lazy-import MiningEngine so the helper symbols in this module
    (selection, scoring, estimate_in_zone) are importable from a CPU
    box without transformers installed.
    """
    from .engine import MiningEngine

    return MiningEngine


class _OptimizedMiningEngineMeta(type):
    """Metaclass that resolves MiningEngine as the base lazily on first
    access. Lets tests import the class symbol without paying the
    transformers import cost; only constructing an instance triggers
    the heavy chain.
    """

    def __new__(mcs, name, bases, namespace):
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._base_resolved = False

    def __call__(cls, *args, **kwargs):
        if not cls._base_resolved:
            base = _import_base()
            cls.__bases__ = (base,) + cls.__bases__
            cls._base_resolved = True
        return super().__call__(*args, **kwargs)


class OptimizedMiningEngine(metaclass=_OptimizedMiningEngineMeta):
    """Competitive-miner reference. Subclasses ``MiningEngine`` and adds:

    - ``score_prompt(text)`` — fast in-zone prediction (next-token entropy).
    - ``select_prompts(candidates, n, cooldown_task_ids)`` — picks
      ``n`` prompts most likely to land in zone, dropping cooldowns.
    - ``estimate_in_zone(rewards)`` — local σ gate before submit.

    The base class's ``generate_m_completions`` + GRAIL commitment
    flow is reused unchanged. Only the prompt-selection layer changes.
    """

    # Tuning knobs. Override via __init__ args or subclass.
    DEFAULT_ENTROPY_FLOOR_NATS = 2.0
    DEFAULT_ENTROPY_CEIL_NATS = 10.0
    DEFAULT_LOCAL_SIGMA_GATE = 0.0  # 0 = no gate; raise to skip out-of-zone groups

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        entropy_floor: float = DEFAULT_ENTROPY_FLOOR_NATS,
        entropy_ceil: float = DEFAULT_ENTROPY_CEIL_NATS,
        local_sigma_gate: float = DEFAULT_LOCAL_SIGMA_GATE,
    ) -> None:
        # The metaclass injected MiningEngine as a base on first call,
        # so this super().__init__ wires up bundle/model/tokenizer/etc.
        super().__init__(cfg=cfg)  # type: ignore[misc]
        self.entropy_floor = float(entropy_floor)
        self.entropy_ceil = float(entropy_ceil)
        self.local_sigma_gate = float(local_sigma_gate)
        logger.info(
            "OptimizedMiningEngine ready: entropy_band=[%.2f, %.2f] nats, "
            "local_sigma_gate=%.3f",
            self.entropy_floor,
            self.entropy_ceil,
            self.local_sigma_gate,
        )

    # ────────  Frontier-σ scoring  ────────

    def score_prompt(self, text: str) -> float:
        """Estimate `P(M rollouts land in zone)` for this prompt.

        One forward pass on the prompt (no generation). The next-token
        distribution's Shannon entropy is the proxy:
        - High entropy → policy is uncertain → 8 rollouts will diverge
          → high σ → likely in zone.
        - Low entropy → policy is confident → 8 rollouts will agree
          → low σ → likely out of zone.

        Returns a score in [0, 1], where 1 = ideal frontier. The
        mapping is a piecewise linear normalization of entropy in
        nats over ``[entropy_floor, entropy_ceil]``.
        """
        import torch  # lazy

        device = self.bundle["device"]
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(tokens, use_cache=False)
            logits = outputs.logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Shannon entropy in nats.
        entropy = -(probs * log_probs).sum().item()
        return _normalize_entropy_to_unit_interval(
            entropy, floor=self.entropy_floor, ceil=self.entropy_ceil
        )

    def select_prompts(
        self,
        candidates: list[dict[str, Any]],
        n: int,
        cooldown_task_ids: set[int] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Pick ``n`` prompts most likely to land in zone.

        Drops cooldown candidates first, scores the rest, returns the
        top-``n`` by descending score. Determinism: same candidate
        list (in the same order) + same model + same cooldown set →
        same output, suitable for being part of the proof binding.
        """
        if n <= 0 or not candidates:
            return []
        cooldown_set = set(cooldown_task_ids or ())
        available = [
            c for c in candidates if _candidate_task_id(c) not in cooldown_set
        ]
        if not available:
            return []
        scored = [(self.score_prompt(_candidate_prompt_text(c)), idx, c)
                  for idx, c in enumerate(available)]
        # Sort by (-score, original_index) for stable, deterministic ordering.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, _, c in scored[:n]]

    # ────────  Local σ gate (post-generation)  ────────

    @staticmethod
    def estimate_in_zone(
        rewards: list[float],
        sigma_min: float,
    ) -> tuple[float, bool]:
        """Compute σ from a rollout group's rewards and check the gate.

        Mirrors ``zone_filter.rewards_std`` (population std, divide by N).
        Returns ``(sigma, in_zone)``.
        """
        if not rewards:
            return 0.0, False
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = var ** 0.5
        return sigma, sigma >= sigma_min


def _candidate_task_id(candidate: dict[str, Any]) -> Any:
    """Pull the task id from various task-batch entry shapes.

    Uses ``is not None`` rather than truthiness so ``task_id=0`` is
    a valid id (not falsy-coerced through to the next fallback).
    """
    for key in ("task_id", "id", "prompt_idx"):
        value = candidate.get(key)
        if value is not None:
            return value
    return id(candidate)


def _candidate_prompt_text(candidate: dict[str, Any]) -> str:
    """Pull the prompt text. Falls back to empty string only as a
    last-resort default; that prompt will score 0 entropy and be
    de-prioritized naturally."""
    return str(
        candidate.get("prompt")
        or candidate.get("text")
        or candidate.get("question")
        or ""
    )


def _normalize_entropy_to_unit_interval(
    entropy_nats: float, *, floor: float, ceil: float
) -> float:
    """Piecewise-linear map from entropy → [0, 1] frontier-σ score.

    Below ``floor``: 0.0 (policy too confident, rollouts will agree).
    Above ``ceil``: 1.0 (policy too uncertain, rewards will be noise).
    The ceiling bound matters less than the floor in practice — most
    real prompts on Qwen-class 3B/4B models sit in [3, 8] nats on
    MATH-style queries.
    """
    if ceil <= floor:
        return 0.0
    if entropy_nats <= floor:
        return 0.0
    if entropy_nats >= ceil:
        return 1.0
    return (entropy_nats - floor) / (ceil - floor)


__all__ = [
    "OptimizedMiningEngine",
]
