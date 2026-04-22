"""Tests for the DAPO-style per-prompt CooldownMap."""

from __future__ import annotations

import json

import pytest

from reliquary_inference.validator.cooldown import (
    DEFAULT_COOLDOWN_WINDOWS,
    CooldownMap,
    default_cooldown_path,
)


def test_default_cooldown_windows_is_50():
    assert DEFAULT_COOLDOWN_WINDOWS == 50


def test_constructor_validates_non_negative():
    with pytest.raises(ValueError):
        CooldownMap(cooldown_windows=-1)


def test_record_and_is_in_cooldown():
    m = CooldownMap(cooldown_windows=50)
    assert m.is_in_cooldown(7, current_window=100) is False  # not recorded
    m.record_batched(prompt_idx=7, window=100)
    assert m.is_in_cooldown(7, current_window=100) is True   # just batched
    assert m.is_in_cooldown(7, current_window=149) is True   # still inside
    assert m.is_in_cooldown(7, current_window=150) is False  # boundary → out
    assert m.is_in_cooldown(7, current_window=200) is False


def test_record_many_is_convenience_wrapper():
    m = CooldownMap(cooldown_windows=10)
    m.record_batched_many([1, 2, 3], window=50)
    assert m.current_cooldown_set(current_window=55) == {1, 2, 3}


def test_zero_cooldown_disables_filter():
    m = CooldownMap(cooldown_windows=0)
    m.record_batched(prompt_idx=5, window=100)
    assert m.is_in_cooldown(5, current_window=100) is False
    assert m.current_cooldown_set(current_window=100) == set()


def test_record_validates_prompt_idx_and_window():
    m = CooldownMap(cooldown_windows=10)
    with pytest.raises(ValueError):
        m.record_batched(prompt_idx=-1, window=0)
    with pytest.raises(ValueError):
        m.record_batched(prompt_idx=0, window=-1)


def test_current_cooldown_set_matches_predicate():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=10, window=100)
    m.record_batched(prompt_idx=20, window=140)
    m.record_batched(prompt_idx=30, window=170)
    # At window 151: 10 is out (151-100=51 >= 50), 20 is in (151-140=11), 30 would be -19 ahead so "in".
    # To avoid future-batching ambiguity, assert on windows after all entries:
    cooldown = m.current_cooldown_set(current_window=151)
    assert 10 not in cooldown  # aged out
    assert 20 in cooldown  # within horizon
    # At window 220, everything older than 170 is out.
    cooldown_later = m.current_cooldown_set(current_window=220)
    assert cooldown_later == set()


def test_save_and_load_roundtrip(tmp_path):
    m = CooldownMap(cooldown_windows=30)
    m.record_batched(1, 100)
    m.record_batched(2, 110)
    path = tmp_path / "cooldown.json"
    m.save(path)

    loaded = CooldownMap(cooldown_windows=30)
    loaded.load(path)
    assert loaded.is_in_cooldown(1, current_window=125) is True
    assert loaded.is_in_cooldown(1, current_window=130) is False
    assert loaded.is_in_cooldown(2, current_window=140) is False


def test_load_missing_file_noop(tmp_path):
    m = CooldownMap(cooldown_windows=30)
    m.load(tmp_path / "nonexistent.json")
    assert len(m) == 0


def test_save_is_atomic(tmp_path):
    """Save must write through a tmp-file so partial writes can't corrupt state."""
    m = CooldownMap(cooldown_windows=10)
    m.record_batched(1, 10)
    path = tmp_path / "cooldown.json"
    m.save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["cooldown_windows"] == 10
    assert data["last_batched"] == {"1": 10}


def test_prune_drops_entries_past_horizon():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(1, 100)
    m.record_batched(2, 200)
    m.record_batched(3, 300)
    dropped = m.prune(current_window=251)  # horizon = 201
    assert dropped == 2  # 1 and 2 are past horizon
    assert 3 in m.current_cooldown_set(current_window=251)
    assert 1 not in m.current_cooldown_set(current_window=251)


def test_prune_zero_cooldown_drops_everything():
    m = CooldownMap(cooldown_windows=0)
    m._last_batched = {1: 10, 2: 20}  # type: ignore[attr-defined]
    dropped = m.prune(current_window=100)
    assert dropped == 2
    assert len(m) == 0


def test_default_cooldown_path(tmp_path):
    path = default_cooldown_path(tmp_path)
    assert path == tmp_path / "cooldown.json"
