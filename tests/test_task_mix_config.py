"""Tests for ``RELIQUARY_INFERENCE_TASK_MIX`` env-var parsing.

The mix parser is the bridge between operator config (a one-line env
var) and the kwarg shape ``build_task_source`` requires for
``"mixed"``. A coordinated cutover across the mesh depends on every
validator + miner deriving the same mix from the same env var, so
the parsing rules are pinned by these tests.
"""

from __future__ import annotations

import pytest

from reliquary_inference.config import _env_task_mix


def _set_env(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("RELIQUARY_INFERENCE_TASK_MIX", raising=False)
    else:
        monkeypatch.setenv("RELIQUARY_INFERENCE_TASK_MIX", value)


def test_unset_returns_none(monkeypatch):
    _set_env(monkeypatch, None)
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") is None


def test_empty_string_returns_none(monkeypatch):
    _set_env(monkeypatch, "")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") is None


def test_whitespace_only_returns_none(monkeypatch):
    _set_env(monkeypatch, "   ")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") is None


def test_canonical_two_source_mix(monkeypatch):
    _set_env(monkeypatch, "math:2,gsm8k:1")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") == [
        ("math", 2.0),
        ("gsm8k", 1.0),
    ]


def test_float_weights(monkeypatch):
    _set_env(monkeypatch, "math:1.5,gsm8k:0.5")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") == [
        ("math", 1.5),
        ("gsm8k", 0.5),
    ]


def test_internal_whitespace_tolerated(monkeypatch):
    _set_env(monkeypatch, "  math : 2 , gsm8k : 1  ")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") == [
        ("math", 2.0),
        ("gsm8k", 1.0),
    ]


def test_trailing_comma_dropped(monkeypatch):
    _set_env(monkeypatch, "math:1,gsm8k:1,")
    assert _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX") == [
        ("math", 1.0),
        ("gsm8k", 1.0),
    ]


def test_missing_colon_raises(monkeypatch):
    _set_env(monkeypatch, "math:1,gsm8k")
    with pytest.raises(ValueError, match="source_id:weight"):
        _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX")


def test_non_numeric_weight_raises(monkeypatch):
    _set_env(monkeypatch, "math:abc")
    with pytest.raises(ValueError, match="not a number"):
        _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX")


def test_empty_source_id_raises(monkeypatch):
    _set_env(monkeypatch, ":1")
    with pytest.raises(ValueError, match="empty source_id"):
        _env_task_mix("RELIQUARY_INFERENCE_TASK_MIX")
