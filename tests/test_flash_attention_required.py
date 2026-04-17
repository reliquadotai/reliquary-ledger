"""Acceptance tests for the Flash Attention 2 fail-loud guard.

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md, acceptance test 9.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary_inference.protocol.constants import ATTN_IMPLEMENTATION
from reliquary_inference.shared.flash_attention import (
    FlashAttentionRequiredError,
    require_flash_attention_2,
)


def _mock_model(attn_impl: str | None = None, attr_name: str = "_attn_implementation") -> object:
    config = SimpleNamespace()
    if attn_impl is not None:
        setattr(config, attr_name, attn_impl)
    return SimpleNamespace(config=config)


def test_accepts_model_with_flash_attention_2_flag() -> None:
    model = _mock_model(ATTN_IMPLEMENTATION)
    require_flash_attention_2(model)


def test_accepts_model_with_public_attn_implementation_attribute() -> None:
    model = _mock_model(ATTN_IMPLEMENTATION, attr_name="attn_implementation")
    require_flash_attention_2(model)


def test_rejects_model_with_eager_attention() -> None:
    model = _mock_model("eager")
    with pytest.raises(FlashAttentionRequiredError) as excinfo:
        require_flash_attention_2(model)
    assert "eager" in str(excinfo.value)
    assert ATTN_IMPLEMENTATION in str(excinfo.value)


def test_rejects_model_with_sdpa_attention() -> None:
    model = _mock_model("sdpa")
    with pytest.raises(FlashAttentionRequiredError):
        require_flash_attention_2(model)


def test_rejects_model_without_any_attn_implementation() -> None:
    model = _mock_model(None)
    with pytest.raises(FlashAttentionRequiredError):
        require_flash_attention_2(model)


def test_rejects_model_without_config() -> None:
    model = SimpleNamespace()
    with pytest.raises(FlashAttentionRequiredError):
        require_flash_attention_2(model)


def test_error_message_tells_caller_how_to_fix() -> None:
    model = _mock_model("eager")
    with pytest.raises(FlashAttentionRequiredError) as excinfo:
        require_flash_attention_2(model)
    msg = str(excinfo.value)
    assert "flash_attention_2" in msg
    assert "from_pretrained" in msg
