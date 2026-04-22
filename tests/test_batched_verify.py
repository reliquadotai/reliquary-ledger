"""Tests for reliquary_inference.validator.batched_verify.

These tests exercise the grouping + graceful-fallback contract without
actually loading a model — a fake model / tokenizer stand-in covers the
batched forward path, and the grouping helper is pure Python.
"""

from __future__ import annotations

from reliquary_inference.validator.batched_verify import (
    compute_cached_hidden_states,
    group_completions_for_batched_forward,
)


def _comp(miner: str, window_id: int, sample_index: int, tokens: list[int], artifact_id: str = "") -> dict:
    return {
        "artifact_id": artifact_id or f"{miner}-{window_id}-{sample_index}",
        "producer_id": miner,
        "window_id": window_id,
        "payload": {
            "sample_index": sample_index,
            "window_id": window_id,
            "tokens": tokens,
        },
    }


# ---------------------------------------------------------------------------
# group_completions_for_batched_forward
# ---------------------------------------------------------------------------


def test_group_by_miner_and_window():
    comps = [
        _comp("A", 100, 0, [1, 2, 3]),
        _comp("A", 100, 1, [4, 5, 6]),
        _comp("B", 100, 0, [7, 8]),
        _comp("A", 200, 0, [9, 10]),
    ]
    groups = group_completions_for_batched_forward(comps, max_batch_size=8)
    # 3 unique (miner, window) keys → 3 groups
    assert len(groups) == 3
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1, 2]


def test_group_respects_max_batch_size():
    comps = [_comp("A", 100, i, [1, 2, 3]) for i in range(17)]
    groups = group_completions_for_batched_forward(comps, max_batch_size=8)
    # 17 completions → 8 + 8 + 1
    assert [len(g) for g in groups] == [8, 8, 1]


def test_group_ordering_by_sample_index():
    comps = [
        _comp("A", 100, 3, [1]),
        _comp("A", 100, 1, [1]),
        _comp("A", 100, 2, [1]),
        _comp("A", 100, 0, [1]),
    ]
    groups = group_completions_for_batched_forward(comps)
    assert len(groups) == 1
    indices = [c["payload"]["sample_index"] for c in groups[0]]
    assert indices == [0, 1, 2, 3]


def test_group_handles_missing_window_id():
    bad = {
        "artifact_id": "x",
        "producer_id": "A",
        "payload": {"sample_index": 0, "tokens": [1, 2]},
    }
    groups = group_completions_for_batched_forward([bad])
    assert len(groups) == 1
    assert groups[0][0]["artifact_id"] == "x"


# ---------------------------------------------------------------------------
# compute_cached_hidden_states — graceful fallback
# ---------------------------------------------------------------------------


def test_compute_cached_empty_input_returns_empty():
    assert compute_cached_hidden_states(
        completions=[],
        model=None,
        tokenizer=None,
        layer_index=-1,
    ) == {}


def test_compute_cached_no_tokens_returns_empty():
    comp = {"artifact_id": "c1", "payload": {"tokens": []}}
    assert compute_cached_hidden_states(
        completions=[comp],
        model=None,
        tokenizer=None,
        layer_index=-1,
    ) == {}


def test_compute_cached_null_model_returns_empty():
    """Model=None triggers the AssertionError path → empty dict."""
    comp = _comp("A", 100, 0, [1, 2, 3])
    result = compute_cached_hidden_states(
        completions=[comp],
        model=None,
        tokenizer=None,
        layer_index=-1,
    )
    assert result == {}
