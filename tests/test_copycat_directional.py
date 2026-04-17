"""Acceptance tests for directional copycat attribution.

Spec reference: private/reliquary-plan/notes/spec-copycat-directional.md.
"""

from __future__ import annotations

from reliquary_inference.protocol.constants import (
    COPYCAT_AMBIGUITY_WINDOW_SECONDS,
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_WINDOW_THRESHOLD,
)
from reliquary_inference.validator.copycat import (
    CopycatHistory,
    Submission,
    detect_copycats,
    detect_index_copycats,
    hash_completion,
)


def _sub(hotkey: str, index: int, text: str, t: float | None) -> Submission:
    return Submission(hotkey=hotkey, index=index, content_hash=hash_completion(text), upload_time=t)


def test_directional_attribution_rejects_later_uploader() -> None:
    verdict = detect_copycats(
        [
            _sub("a", 10, "Answer: 7", t=1.0),
            _sub("b", 10, "Answer: 7", t=5.0),
        ],
        window_id=0,
    )
    assert verdict.rejected_indices == {"b": {10}}
    assert any(e.kind == "index" and e.original_miner == "a" and e.copycat_miners == ["b"]
               for e in verdict.audit_entries)


def test_ambiguous_window_rejects_neither() -> None:
    verdict = detect_copycats(
        [
            _sub("a", 10, "Answer: 7", t=1.0),
            _sub("b", 10, "Answer: 7", t=1.0 + COPYCAT_AMBIGUITY_WINDOW_SECONDS - 0.1),
        ],
        window_id=0,
    )
    assert verdict.rejected_indices == {}
    assert ("a", "b", 10) in verdict.ambiguous_pairs
    audit = next(e for e in verdict.audit_entries if e.kind == "index")
    assert audit.original_miner is None
    assert audit.note == "ambiguous_window"


def test_timestamp_unavailable_rejects_neither() -> None:
    verdict = detect_copycats(
        [
            _sub("a", 10, "Answer: 7", t=None),
            _sub("b", 10, "Answer: 7", t=5.0),
        ],
        window_id=0,
    )
    assert verdict.rejected_indices == {}
    audit = next(e for e in verdict.audit_entries if e.kind == "index")
    assert audit.note == "timestamp_unavailable"


def test_three_way_contest_blames_two_later() -> None:
    verdict = detect_copycats(
        [
            _sub("a", 10, "Answer: 7", t=1.0),
            _sub("b", 10, "Answer: 7", t=10.0),
            _sub("c", 10, "Answer: 7", t=20.0),
        ],
        window_id=0,
    )
    assert verdict.rejected_indices == {"b": {10}, "c": {10}}


def test_content_hash_attribution_detects_cross_index_copy() -> None:
    verdict = detect_copycats(
        [
            _sub("a", 10, "Answer: 42", t=1.0),
            _sub("b", 20, "Answer: 42", t=5.0),
        ],
        window_id=0,
    )
    shared_hash = hash_completion("Answer: 42")
    assert verdict.rejected_content_hashes.get("b") == {shared_hash}
    assert verdict.rejected_indices == {}


def test_window_threshold_flags_miner() -> None:
    submissions = [_sub("a", i, f"answer-{i}", t=1.0) for i in range(100)]
    submissions += [_sub("b", i, f"answer-{i}", t=5.0) for i in range(10)]
    verdict = detect_copycats(submissions, window_id=0)
    assert "b" in verdict.flagged_miners
    assert verdict.overlap_ratios_per_window["b"] > COPYCAT_WINDOW_THRESHOLD


def test_interval_gating_requires_two_consecutive_flags() -> None:
    history = CopycatHistory()

    subs = (
        [_sub("a", i, f"r-{i}", t=1.0) for i in range(100)]
        + [_sub("b", i, f"r-{i}", t=5.0) for i in range(5)]
    )

    v0 = detect_copycats(subs, window_id=0, history=history)
    assert history.interval_ratio("b") > COPYCAT_INTERVAL_THRESHOLD
    assert "b" in v0.flagged_miners
    assert "b" not in v0.gated_miners, "first consecutive interval flag must not gate"

    v1 = detect_copycats(subs, window_id=1, history=history)
    assert "b" in v1.gated_miners, "second consecutive interval flag must gate"


def test_no_history_means_no_gating_signal() -> None:
    subs = (
        [_sub("a", i, f"x-{i}", t=1.0) for i in range(10)]
        + [_sub("b", i, f"x-{i}", t=5.0) for i in range(5)]
    )
    v = detect_copycats(subs, window_id=0, history=None)
    assert v.gated_miners == set()


def test_idempotent_detect_produces_identical_verdict() -> None:
    subs = [
        _sub("a", 10, "Answer: 7", t=1.0),
        _sub("b", 10, "Answer: 7", t=5.0),
        _sub("c", 11, "Answer: 8", t=2.0),
    ]
    v1 = detect_copycats(subs, window_id=0)
    v2 = detect_copycats(subs, window_id=0)
    assert v1.rejected_indices == v2.rejected_indices
    assert v1.rejected_content_hashes == v2.rejected_content_hashes
    assert [e.contested_value for e in v1.audit_entries] == [
        e.contested_value for e in v2.audit_entries
    ]


def test_self_duplicates_do_not_inflate_audit() -> None:
    subs = [
        _sub("a", 10, "Answer: 7", t=1.0),
        _sub("a", 10, "Answer: 7", t=1.0),
        _sub("b", 10, "Answer: 7", t=5.0),
    ]
    verdict = detect_copycats(subs, window_id=0)
    assert verdict.rejected_indices == {"b": {10}}
    assert sum(1 for e in verdict.audit_entries if e.kind == "index" and e.contested_value == "10") == 1


def test_empty_input_returns_empty_verdict() -> None:
    verdict = detect_copycats([], window_id=0)
    assert verdict.rejected_indices == {}
    assert verdict.audit_entries == []


def test_legacy_detect_index_copycats_still_works() -> None:
    rejected = detect_index_copycats(
        {
            "a": {"indices": {10, 11}, "upload_time": 1.0},
            "b": {"indices": {11, 12}, "upload_time": 5.0},
        }
    )
    assert rejected == {"b": {11}}


def test_legacy_detect_ambiguous_timestamps_do_not_reject() -> None:
    rejected = detect_index_copycats(
        {
            "a": {"indices": {10}, "upload_time": 1.0},
            "b": {"indices": {10}, "upload_time": 1.5},
        }
    )
    assert rejected == {}
