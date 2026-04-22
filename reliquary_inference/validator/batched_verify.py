"""Batched forward pass for proof verification.

The per-completion ProofStage runs a separate forward pass for each
rollout — when a miner submits M=8 rollouts on the same prompt (the
GRPO group), that's 8 serial forward passes on the validator. With the
miner now batching its generate() call (see MiningEngine.generate_m_completions),
the validator becomes the bottleneck.

This module pre-computes hidden states for a group of M completions in
ONE batched forward pass and stores them per-completion in
``context.extras["cached_hidden_states"]``. The existing ProofStage
picks up the cached tensor if present and skips its own forward pass
entirely.

Key invariants preserved:
  - Sequences are padded to the group max length with pad_token_id.
    Attention mask zeros out the pad tokens so the model doesn't
    attend to them.
  - Each completion's hidden_states slice is exactly
    ``[:real_seq_len, :]`` — trailing pad positions are never exposed,
    so the commitment verify sees the same per-position hidden states
    it would have seen in the per-completion forward.
  - Gracefully falls back to per-completion forward on any error
    (model shape mismatch, CUDA OOM on oversized batch, etc.).
"""

from __future__ import annotations

from typing import Any


def compute_cached_hidden_states(
    *,
    completions: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    layer_index: int,
) -> dict[str, Any]:
    """Run ONE batched forward over all ``completions`` and return
    ``{completion_id: hidden_states_tensor}``.

    Failure modes (any of these returns {} so the caller falls back):
      - Empty completions list.
      - Model / tokenizer not loaded.
      - Tokens missing / malformed.
      - CUDA OOM on the batched shape (we'd rather serialize than crash).

    Never raises — returns an empty dict on any exception.
    """
    if not completions:
        return {}
    try:
        import torch

        from ..protocol.constants import LAYER_INDEX as DEFAULT_LAYER_INDEX
        from ..shared.forward import forward_single_layer

        assert model is not None, "model must be loaded"
        assert tokenizer is not None, "tokenizer must be loaded"
        layer = DEFAULT_LAYER_INDEX if layer_index is None else int(layer_index)

        # Collect (completion_id, tokens) pairs; drop any with no tokens.
        entries: list[tuple[str, list[int]]] = []
        for comp in completions:
            payload = comp.get("payload", {})
            tokens = payload.get("tokens") or []
            comp_id = str(comp.get("artifact_id") or "")
            if not comp_id or not tokens:
                continue
            entries.append((comp_id, list(tokens)))
        if not entries:
            return {}

        device = next(model.parameters()).device
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        max_len = max(len(t) for _, t in entries)
        batch_tokens: list[list[int]] = []
        batch_masks: list[list[int]] = []
        for _cid, toks in entries:
            pad_n = max_len - len(toks)
            batch_tokens.append(list(toks) + [int(pad_id)] * pad_n)
            batch_masks.append([1] * len(toks) + [0] * pad_n)

        input_ids = torch.tensor(batch_tokens, device=device)
        attention_mask = torch.tensor(batch_masks, device=device)
        with torch.no_grad():
            hidden_states, _ = forward_single_layer(
                model, input_ids, attention_mask, layer,
            )

        # Per-completion slice: real_len only, so downstream commitment
        # verify sees identical hidden states to the per-completion path.
        cached: dict[str, Any] = {}
        for i, (cid, toks) in enumerate(entries):
            cached[cid] = hidden_states[i, : len(toks), :].contiguous()
        return cached
    except Exception:
        # Any failure → empty dict, ProofStage falls back to its own forward.
        return {}


def group_completions_for_batched_forward(
    all_completions: list[dict[str, Any]],
    *,
    max_batch_size: int = 8,
) -> list[list[dict[str, Any]]]:
    """Split completions into groups that share (miner, window) and are
    sized under ``max_batch_size`` so one forward pass fits in GPU memory.

    Each returned group is a list of completion dicts from the same miner
    on the same window_id. Larger-than-max_batch_size groups are split.
    """
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for comp in all_completions:
        miner = str(comp.get("producer_id") or "")
        try:
            window_id = int(comp.get("window_id") or comp["payload"].get("window_id") or 0)
        except (KeyError, TypeError, ValueError):
            window_id = 0
        by_key.setdefault((miner, window_id), []).append(comp)

    groups: list[list[dict[str, Any]]] = []
    for _key, bucket in by_key.items():
        # Stable-sort by sample_index so per-group ordering is deterministic.
        bucket.sort(key=lambda c: int(c.get("payload", {}).get("sample_index", 0)))
        for i in range(0, len(bucket), max_batch_size):
            groups.append(bucket[i : i + max_batch_size])
    return groups


__all__ = [
    "compute_cached_hidden_states",
    "group_completions_for_batched_forward",
]
