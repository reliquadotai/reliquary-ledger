"""Copycat detection via dataset index deduplication.

When two miners submit rollouts for the same dataset index, only the
miner who uploaded first (by S3 LastModified timestamp) keeps credit.
The later uploader's overlapping indices are rejected.

With 553M dataset rows and ~hundreds of miners each picking random
indices, collisions are extremely rare and almost certainly copying.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def detect_index_copycats(
    submissions: dict[str, dict],
) -> dict[str, set[int]]:
    """Detect and reject duplicate dataset indices across miners.

    For each index submitted by multiple miners, the earliest uploader
    (lowest upload_time) keeps it. All later uploaders have that index
    rejected. If timestamps are equal or either is None, no one is
    rejected for that index (benefit of the doubt).

    Args:
        submissions: {hotkey: {"indices": set[int], "upload_time": float | None}}

    Returns:
        {hotkey: set of rejected indices} — only hotkeys with rejections appear.
    """
    if len(submissions) < 2:
        return {}

    # Build reverse map: index → list of (hotkey, upload_time)
    index_to_miners: defaultdict[int, list[tuple[str, float | None]]] = defaultdict(list)
    for hotkey, sub in submissions.items():
        upload_time = sub.get("upload_time")
        for idx in sub.get("indices", set()):
            index_to_miners[idx].append((hotkey, upload_time))

    rejected: defaultdict[str, set[int]] = defaultdict(set)

    for idx, miners in index_to_miners.items():
        if len(miners) < 2:
            continue

        # Check if all timestamps are available and not equal
        times = [t for _, t in miners if t is not None]
        if len(times) < len(miners):
            logger.warning(
                "Index %d shared by %d miners but timestamp unavailable, skipping",
                idx, len(miners),
            )
            continue

        unique_times = set(times)
        if len(unique_times) == 1:
            continue

        # Find the earliest uploader
        earliest_hotkey = min(miners, key=lambda m: m[1] if m[1] is not None else float("inf"))[0]

        # Reject the index for everyone except the earliest
        for hotkey, _ in miners:
            if hotkey != earliest_hotkey:
                rejected[hotkey].add(idx)
                logger.warning(
                    "Copycat index %d: %s rejected (earlier uploader: %s)",
                    idx, hotkey, earliest_hotkey,
                )

    return dict(rejected)
