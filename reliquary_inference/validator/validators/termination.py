"""Stage 5: termination check.

A completion must end with an EOS token, stop at a documented stop sequence,
or hit the model's maximum context length. Unterminated completions that are
shorter than max length are rejected — they either indicate truncation bugs
or miners that emit partial output without a clean stop.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept, reject


class TerminationStage:
    name: str = "termination"

    def check(self, context: StageContext) -> StageResult:
        payload = context.payload
        tokens = payload.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            return reject(
                self.name,
                RejectReason.TERMINATION_NO_EOS,
                {"reason": "empty_tokens"},
            )

        eos_ids = _collect_eos_ids(context)
        stop_sequences = _collect_stop_sequences(context)
        max_length = _resolve_max_length(context)

        last_token = int(tokens[-1])
        if last_token in eos_ids:
            return accept(
                self.name,
                {"ended_with": "eos", "last_token": last_token},
            )

        if stop_sequences and _ends_with_stop(tokens, stop_sequences):
            return accept(
                self.name,
                {"ended_with": "stop_sequence"},
            )

        if max_length is not None and len(tokens) >= max_length:
            if len(tokens) > max_length:
                return reject(
                    self.name,
                    RejectReason.TERMINATION_OVERFLOW,
                    {"length": len(tokens), "max_length": max_length},
                )
            return accept(
                self.name,
                {"ended_with": "max_length", "length": len(tokens)},
            )

        return reject(
            self.name,
            RejectReason.TERMINATION_NO_EOS,
            {
                "last_token": last_token,
                "eos_ids": sorted(eos_ids),
                "length": len(tokens),
                "max_length": max_length,
            },
        )


def _collect_eos_ids(context: StageContext) -> set[int]:
    ids: set[int] = set()
    for source in (context.tokenizer, getattr(context.model, "config", None)):
        if source is None:
            continue
        for attr in ("eos_token_id", "eot_token_id", "pad_token_id"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, list):
                ids.update(int(v) for v in value if isinstance(v, int))
    declared = context.extras.get("eos_token_ids")
    if isinstance(declared, (list, tuple, set)):
        ids.update(int(v) for v in declared)
    return ids


def _collect_stop_sequences(context: StageContext) -> list[list[int]]:
    raw = context.extras.get("stop_sequences")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[list[int]] = []
    for seq in raw:
        if isinstance(seq, (list, tuple)) and all(isinstance(v, int) for v in seq):
            out.append(list(seq))
    return out


def _ends_with_stop(tokens: list[int], stop_sequences: list[list[int]]) -> bool:
    return any(
        seq and len(tokens) >= len(seq) and tokens[-len(seq):] == seq
        for seq in stop_sequences
    )


def _resolve_max_length(context: StageContext) -> int | None:
    config = getattr(context.model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "model_max_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    override = context.extras.get("max_length")
    if isinstance(override, int) and override > 0:
        return override
    return None
