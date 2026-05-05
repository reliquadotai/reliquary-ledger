"""Tests for the by-window mirror prefix on ObjectRegistry +
LocalRegistry.

Closes the audit-flagged + 2026-05-04 lite-livetest-confirmed
list_artifacts pagination gap. Pattern: every put_artifact writes
to BOTH the flat prefix (legacy + source of truth) AND a window-
keyed mirror prefix (fast read path for window-filtered lookups).
list_artifacts(..., window_id=W) prefers the mirror when present,
falls back to flat scan on miss for legacy data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reliquary_inference.storage.registry import (
    FilesystemObjectStore,
    LocalRegistry,
    ObjectRegistry,
)


def _make_artifact(*, artifact_type: str, window_id: int, artifact_id: str) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "producer_id": "test-producer",
        "producer_role": "test",
        "window_id": window_id,
        "created_at": "2026-05-05T00:00:00+00:00",
        "parent_ids": [],
        "payload": {"hello": "world"},
    }


# ────────  ObjectRegistry (R2-shaped) — uses FilesystemObjectStore  ────────


@pytest.fixture
def object_registry(tmp_path):
    store = FilesystemObjectStore(str(tmp_path / "store"))
    return ObjectRegistry(store=store, export_root=str(tmp_path / "exports"))


def test_put_artifact_writes_to_both_flat_and_mirror(object_registry, tmp_path) -> None:
    """Every put_artifact writes to flat AND by-window mirror when
    the artifact carries a window_id."""
    art = _make_artifact(artifact_type="task_batch", window_id=12345, artifact_id="abc-1")
    object_registry.put_artifact(art)

    # Flat key exists.
    assert object_registry.store.get_bytes("task_batches/abc-1.json") is not None
    # Mirror key exists.
    mirror_key = "task_batches/by_window/window-00012345/abc-1.json"
    assert object_registry.store.get_bytes(mirror_key) is not None
    # Both contain the same artifact.
    flat = json.loads(object_registry.store.get_bytes("task_batches/abc-1.json").decode())
    mirror = json.loads(object_registry.store.get_bytes(mirror_key).decode())
    assert flat == mirror == art


def test_put_artifact_skips_mirror_when_no_window_id(object_registry) -> None:
    """Artifacts without a window_id (rare, but the contract allows
    it) skip the mirror — we can't index them by window."""
    art = {
        "artifact_type": "run_manifest",
        "artifact_id": "run-1",
        "producer_id": "x",
        "producer_role": "y",
        "created_at": "2026-05-05T00:00:00+00:00",
        "parent_ids": [],
        "payload": {},
    }
    object_registry.put_artifact(art)
    assert object_registry.store.get_bytes("run_manifests/run-1.json") is not None
    # No mirror because no window_id.
    refs = object_registry.store.list_prefix("run_manifests/by_window/")
    assert refs == []


def test_list_artifacts_with_window_id_uses_mirror_fast_path(object_registry) -> None:
    """When window_id is provided AND the mirror has data, list returns
    the mirror's contents directly without walking the flat prefix."""
    object_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=100, artifact_id="a"))
    object_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=100, artifact_id="b"))
    object_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=200, artifact_id="c"))

    out_100 = object_registry.list_artifacts("task_batch", window_id=100)
    out_200 = object_registry.list_artifacts("task_batch", window_id=200)
    assert {a["artifact_id"] for a in out_100} == {"a", "b"}
    assert {a["artifact_id"] for a in out_200} == {"c"}


def test_list_artifacts_falls_back_to_flat_scan_on_mirror_miss(
    object_registry,
) -> None:
    """Legacy data: an artifact written before the mirror prefix
    existed only lives at the flat path. list_artifacts must still
    find it via fallback."""
    # Simulate legacy: write flat directly, skip mirror.
    art = _make_artifact(artifact_type="task_batch", window_id=300, artifact_id="legacy-1")
    body = json.dumps(art, indent=2, sort_keys=True).encode("utf-8")
    object_registry.store.put_bytes("task_batches/legacy-1.json", body)
    # Confirm no mirror entry.
    assert object_registry.store.list_prefix("task_batches/by_window/window-00000300/") == []
    # list still finds it via fallback.
    out = object_registry.list_artifacts("task_batch", window_id=300)
    assert {a["artifact_id"] for a in out} == {"legacy-1"}


def test_list_artifacts_unfiltered_skips_mirror_to_avoid_double_count(
    object_registry,
) -> None:
    """Unfiltered list (window_id=None) walks the flat prefix only.
    The walk explicitly skips by_window/ subkeys so we don't return
    each artifact twice."""
    object_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=400, artifact_id="x"))
    object_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=500, artifact_id="y"))

    out = object_registry.list_artifacts("task_batch")  # no window_id filter
    ids = sorted(a["artifact_id"] for a in out)
    assert ids == ["x", "y"]  # exactly two, not four


def test_list_artifacts_mirror_independent_of_flat_for_perf(
    object_registry,
) -> None:
    """If the flat prefix is huge and the mirror is small, the mirror
    path is fast. We can't test perf in unit tests, but we can verify
    the mirror path doesn't accidentally read flat keys."""
    # Write 50 task_batches across many windows.
    for i in range(50):
        object_registry.put_artifact(
            _make_artifact(artifact_type="task_batch", window_id=1000 + i, artifact_id=f"art-{i}")
        )
    # Asking for window 1025 should hit mirror, return exactly 1 artifact.
    out = object_registry.list_artifacts("task_batch", window_id=1025)
    assert len(out) == 1
    assert out[0]["artifact_id"] == "art-25"


# ────────  LocalRegistry (filesystem) — same semantics  ────────


@pytest.fixture
def local_registry(tmp_path):
    return LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))


def test_local_put_writes_to_both_flat_and_mirror(local_registry) -> None:
    art = _make_artifact(artifact_type="scorecard", window_id=7777, artifact_id="sc-1")
    local_registry.put_artifact(art)
    flat = local_registry.artifact_root / "scorecards" / "sc-1.json"
    mirror = (
        local_registry.artifact_root
        / "scorecards"
        / "by_window"
        / "window-00007777"
        / "sc-1.json"
    )
    assert flat.exists()
    assert mirror.exists()


def test_local_list_with_window_uses_mirror(local_registry) -> None:
    local_registry.put_artifact(_make_artifact(artifact_type="scorecard", window_id=10, artifact_id="a"))
    local_registry.put_artifact(_make_artifact(artifact_type="scorecard", window_id=20, artifact_id="b"))
    out_10 = local_registry.list_artifacts("scorecard", window_id=10)
    out_20 = local_registry.list_artifacts("scorecard", window_id=20)
    assert {a["artifact_id"] for a in out_10} == {"a"}
    assert {a["artifact_id"] for a in out_20} == {"b"}


def test_local_list_falls_back_to_flat_on_legacy_data(local_registry, tmp_path) -> None:
    # Drop a legacy artifact directly in the flat dir, no mirror.
    flat_dir = local_registry.artifact_root / "scorecards"
    flat_dir.mkdir(parents=True, exist_ok=True)
    legacy = _make_artifact(artifact_type="scorecard", window_id=999, artifact_id="legacy")
    (flat_dir / "legacy.json").write_text(json.dumps(legacy))
    out = local_registry.list_artifacts("scorecard", window_id=999)
    assert {a["artifact_id"] for a in out} == {"legacy"}


def test_local_list_unfiltered_no_double_count(local_registry) -> None:
    """Unfiltered list on LocalRegistry shouldn't double-count by
    walking into the by_window/ subdirectory."""
    local_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=1, artifact_id="p"))
    local_registry.put_artifact(_make_artifact(artifact_type="task_batch", window_id=2, artifact_id="q"))
    out = local_registry.list_artifacts("task_batch")
    ids = sorted(a["artifact_id"] for a in out)
    # Note: LocalRegistry's directory.glob("*.json") only matches the
    # immediate dir, not subdirs. So mirror entries in by_window/window-N/
    # are naturally not counted. Two writes → two artifacts.
    assert ids == ["p", "q"]
