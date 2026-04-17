"""Directional copycat detection with timestamp-based attribution.

Extends the prior index-only detector with:
  - explicit ambiguity window where near-simultaneous uploaders are not blamed;
  - content-hash copycat detection (independent of claimed dataset index);
  - rolling per-miner overlap-ratio tracking with hysteresis gating;
  - structured audit entries suitable for cross-validator consensus.

Spec: private/reliquary-plan/notes/spec-copycat-directional.md.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from ..protocol.constants import (
    COPYCAT_AMBIGUITY_WINDOW_SECONDS,
    COPYCAT_GATE_DURATION_WINDOWS,
    COPYCAT_INTERVAL_LENGTH,
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_WINDOW_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    """One accepted completion from one miner in the current window."""

    hotkey: str
    index: int
    content_hash: str
    upload_time: float | None


@dataclass
class AuditEntry:
    kind: str
    contested_value: str
    original_miner: str | None
    copycat_miners: list[str]
    upload_timestamps: dict[str, float | None]
    note: str = ""


@dataclass
class CopycatVerdict:
    rejected_indices: dict[str, set[int]] = field(default_factory=dict)
    rejected_content_hashes: dict[str, set[str]] = field(default_factory=dict)
    ambiguous_pairs: list[tuple[str, str, int | str]] = field(default_factory=list)
    overlap_ratios_per_window: dict[str, float] = field(default_factory=dict)
    flagged_miners: set[str] = field(default_factory=set)
    gated_miners: set[str] = field(default_factory=set)
    audit_entries: list[AuditEntry] = field(default_factory=list)


class CopycatHistory:
    """Rolling per-miner overlap-ratio state across the last N windows.

    Maintains a small deque of ``(window_id, overlap_ratio)`` per miner, plus a
    sticky gating flag that persists for ``COPYCAT_GATE_DURATION_WINDOWS``
    windows once the miner crosses the interval threshold for two
    consecutive intervals.
    """

    def __init__(self, interval_length: int = COPYCAT_INTERVAL_LENGTH) -> None:
        self.interval_length = interval_length
        self._ratios: dict[str, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self.interval_length * 2)
        )
        self._prev_interval_flag: dict[str, bool] = {}
        self._gate_until_window: dict[str, int] = {}

    def record_window(self, window_id: int, overlap_ratios: dict[str, float]) -> None:
        for miner, ratio in overlap_ratios.items():
            self._ratios[miner].append((window_id, ratio))

    def interval_ratio(self, miner: str) -> float:
        entries = self._ratios.get(miner)
        if not entries:
            return 0.0
        recent = list(entries)[-self.interval_length:]
        if not recent:
            return 0.0
        return statistics.fmean(r for _, r in recent)

    def evaluate_gating(self, miner: str, window_id: int) -> bool:
        current_flag = self.interval_ratio(miner) > COPYCAT_INTERVAL_THRESHOLD
        prev_flag = self._prev_interval_flag.get(miner, False)
        self._prev_interval_flag[miner] = current_flag

        if current_flag and prev_flag:
            self._gate_until_window[miner] = window_id + COPYCAT_GATE_DURATION_WINDOWS

        gate_until = self._gate_until_window.get(miner)
        if gate_until is not None and window_id < gate_until:
            return True
        return False


def _sha_of(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_completion(text: str) -> str:
    """Canonical content hash for a completion."""
    return _sha_of(text)


def _attribute_group(
    kind: str,
    contested_value: str,
    miners: list[tuple[str, float | None]],
) -> tuple[list[str], AuditEntry, bool]:
    times = [t for _, t in miners]
    timestamps_map = {h: t for h, t in miners}

    if any(t is None for t in times):
        entry = AuditEntry(
            kind=kind,
            contested_value=contested_value,
            original_miner=None,
            copycat_miners=[],
            upload_timestamps=timestamps_map,
            note="timestamp_unavailable",
        )
        return [], entry, False

    t_min = min(times)  # type: ignore[type-var]
    t_max = max(times)  # type: ignore[type-var]

    if (t_max - t_min) <= COPYCAT_AMBIGUITY_WINDOW_SECONDS:
        entry = AuditEntry(
            kind=kind,
            contested_value=contested_value,
            original_miner=None,
            copycat_miners=[],
            upload_timestamps=timestamps_map,
            note="ambiguous_window",
        )
        return [], entry, True

    earliest = min(miners, key=lambda m: m[1] if m[1] is not None else float("inf"))[0]
    copycats = sorted({h for h, _ in miners if h != earliest})
    entry = AuditEntry(
        kind=kind,
        contested_value=contested_value,
        original_miner=earliest,
        copycat_miners=copycats,
        upload_timestamps=timestamps_map,
    )
    return copycats, entry, False


def detect_copycats(
    submissions: Iterable[Submission],
    *,
    window_id: int,
    history: CopycatHistory | None = None,
) -> CopycatVerdict:
    """Run directional attribution over the current window's submissions."""
    subs = list(submissions)
    verdict = CopycatVerdict()
    if len(subs) < 2:
        return verdict

    index_groups: dict[int, dict[str, float | None]] = defaultdict(dict)
    hash_groups: dict[str, dict[str, float | None]] = defaultdict(dict)
    per_miner_submitted: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for sub in subs:
        if sub.hotkey not in index_groups[sub.index] or (
            sub.upload_time is not None and index_groups[sub.index].get(sub.hotkey) is None
        ):
            index_groups[sub.index][sub.hotkey] = sub.upload_time
        if sub.hotkey not in hash_groups[sub.content_hash] or (
            sub.upload_time is not None and hash_groups[sub.content_hash].get(sub.hotkey) is None
        ):
            hash_groups[sub.content_hash][sub.hotkey] = sub.upload_time
        per_miner_submitted[sub.hotkey].add(("index", str(sub.index)))
        per_miner_submitted[sub.hotkey].add(("hash", sub.content_hash))

    rejected_indices: dict[str, set[int]] = defaultdict(set)
    for idx, miners_map in sorted(index_groups.items()):
        if len(miners_map) < 2:
            continue
        miners = sorted(miners_map.items())
        rejected, entry, ambiguous = _attribute_group("index", str(idx), miners)
        verdict.audit_entries.append(entry)
        for hotkey in rejected:
            rejected_indices[hotkey].add(idx)
            logger.warning(
                "copycat: hotkey=%s rejected index=%d (original=%s)",
                hotkey, idx, entry.original_miner,
            )
        if ambiguous:
            for i, (h_a, _) in enumerate(miners):
                for h_b, _ in miners[i + 1:]:
                    verdict.ambiguous_pairs.append((h_a, h_b, idx))

    rejected_hashes: dict[str, set[str]] = defaultdict(set)
    for digest, miners_map in sorted(hash_groups.items()):
        if len(miners_map) < 2:
            continue
        miners = sorted(miners_map.items())
        rejected, entry, ambiguous = _attribute_group("content_hash", digest, miners)
        verdict.audit_entries.append(entry)
        for hotkey in rejected:
            rejected_hashes[hotkey].add(digest)
            logger.warning(
                "copycat: hotkey=%s rejected content_hash=%s (original=%s)",
                hotkey, digest[:12], entry.original_miner,
            )
        if ambiguous:
            for i, (h_a, _) in enumerate(miners):
                for h_b, _ in miners[i + 1:]:
                    verdict.ambiguous_pairs.append((h_a, h_b, digest))

    verdict.rejected_indices = {k: v for k, v in rejected_indices.items()}
    verdict.rejected_content_hashes = {k: v for k, v in rejected_hashes.items()}

    for miner, submitted in per_miner_submitted.items():
        rejected_items = len(rejected_indices.get(miner, set())) + len(
            rejected_hashes.get(miner, set())
        )
        total = len(submitted)
        ratio = rejected_items / total if total else 0.0
        verdict.overlap_ratios_per_window[miner] = ratio
        if ratio > COPYCAT_WINDOW_THRESHOLD:
            verdict.flagged_miners.add(miner)

    if history is not None:
        history.record_window(window_id, verdict.overlap_ratios_per_window)
        for miner in per_miner_submitted.keys():
            interval_ratio = history.interval_ratio(miner)
            if interval_ratio > COPYCAT_INTERVAL_THRESHOLD:
                verdict.flagged_miners.add(miner)
            if history.evaluate_gating(miner, window_id):
                verdict.gated_miners.add(miner)

    verdict.audit_entries.sort(key=lambda e: (e.kind, e.contested_value))
    return verdict


def detect_index_copycats(
    submissions: dict[str, dict],
) -> dict[str, set[int]]:
    """Backward-compatible adapter for legacy callers.

    Maps the legacy ``{hotkey: {"indices": set[int], "upload_time": float}}``
    shape onto the new :class:`Submission` / :func:`detect_copycats` API with
    synthetic per-hotkey content hashes so the legacy test surface keeps
    passing while deployments migrate to the structured input.
    """
    subs: list[Submission] = []
    for hotkey, body in submissions.items():
        upload_time = body.get("upload_time")
        for idx in body.get("indices", set()):
            subs.append(
                Submission(
                    hotkey=hotkey,
                    index=int(idx),
                    content_hash=f"legacy::{hotkey}::{idx}",
                    upload_time=upload_time,
                )
            )
    verdict = detect_copycats(subs, window_id=0)
    return {k: set(v) for k, v in verdict.rejected_indices.items() if v}
