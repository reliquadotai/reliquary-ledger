"""Fail-loud enforcement of Flash Attention 2 for proof-critical model loads."""

from __future__ import annotations

from typing import Any

from ..protocol.constants import ATTN_IMPLEMENTATION


class FlashAttentionRequiredError(RuntimeError):
    """Raised when a model loaded for proof generation or verification is not
    using the network-wide attention kernel."""


def require_flash_attention_2(model: Any) -> None:
    """Assert that ``model`` is configured with the network-wide attention kernel.

    The proof protocol's cross-framework tolerance is calibrated against a
    specific attention implementation. Any deviation silently widens the
    acceptance window and invites false positives or admits crafted forgeries.
    """
    observed = _observe_attn_implementation(model)
    if observed != ATTN_IMPLEMENTATION:
        raise FlashAttentionRequiredError(
            f"Model attention implementation is {observed!r}; "
            f"proof protocol requires {ATTN_IMPLEMENTATION!r}. "
            "Rebuild the environment with flash-attn installed and pass "
            "`attn_implementation=\"flash_attention_2\"` to `from_pretrained`."
        )


def _observe_attn_implementation(model: Any) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("_attn_implementation", "attn_implementation"):
        value = getattr(config, attr, None)
        if value is not None:
            return str(value)
    return None
