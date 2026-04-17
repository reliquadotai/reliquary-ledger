"""Stage 2: token validity.

Delegates to :func:`reliquary_inference.protocol.tokens.verify_tokens` for
vocab-bound and length-bound checks. Isolating the call behind a stage
wrapper means future token-level checks (special-token rules, byte-pair
decoder sanity) land here without touching the pipeline plumbing.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept, reject


class TokensStage:
    name: str = "tokens"

    def check(self, context: StageContext) -> StageResult:
        from ...protocol.tokens import verify_tokens
        from ...shared.hf_compat import resolve_max_context_length, resolve_vocab_size

        tokens = context.payload.get("tokens", [])
        model_config = getattr(context.model, "config", None)

        if model_config is None:
            return accept(self.name, {"status": "model_config_unavailable"})

        vocab_size = resolve_vocab_size(model_config)
        if vocab_size is not None:
            bad_ids = [
                t for t in tokens
                if not isinstance(t, int) or t < 0 or t >= vocab_size
            ]
            if bad_ids:
                return reject(
                    self.name,
                    RejectReason.TOKENS_OUT_OF_VOCAB,
                    {"vocab_size": vocab_size, "offending_ids": bad_ids[:10]},
                )

        max_length = resolve_max_context_length(model_config)
        if isinstance(max_length, int) and len(tokens) > max_length:
            return reject(
                self.name,
                RejectReason.TOKENS_LENGTH_EXCEEDED,
                {"length": len(tokens), "max_length": max_length},
            )

        if not verify_tokens(tokens, model_config):
            return reject(
                self.name,
                RejectReason.TOKENS_OUT_OF_VOCAB,
                {"reason": "verify_tokens_false"},
            )

        return accept(self.name, {"length": len(tokens), "vocab_size": vocab_size})
