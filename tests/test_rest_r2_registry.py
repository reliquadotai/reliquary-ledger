"""Tests for :class:`RestR2ObjectStore` + :class:`RestR2Registry`.

The shim translates the ``ObjectRegistry`` store surface (``put_bytes`` /
``get_bytes`` / ``list_prefix``) onto
:class:`reliquary_protocol.storage.R2ObjectBackend`'s contract
(``put`` / ``get`` / ``list``). Tests use a fake R2 backend so we verify
wiring without touching real R2.
"""

from __future__ import annotations

import pytest

from reliquary_inference.storage.registry import (
    ObjectRegistry,
    RestR2ObjectStore,
)


class _FakeR2Backend:
    """Minimal stand-in for ``reliquary_protocol.storage.R2ObjectBackend``."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = bytes(data)

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self._store if k.startswith(prefix))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


# ---------------------------------------------------------------------------
# RestR2ObjectStore surface
# ---------------------------------------------------------------------------


def test_put_and_get_round_trip():
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    ref = store.put_bytes("a/b/c.json", b"hello")
    assert ref == {"backend": "r2_rest", "key": "a/b/c.json"}
    assert store.get_bytes("a/b/c.json") == b"hello"


def test_get_bytes_raises_file_not_found_when_missing():
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    with pytest.raises(FileNotFoundError, match="no-such-key"):
        store.get_bytes("no-such-key")


def test_list_prefix_returns_ref_shape():
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    store.put_bytes("artifacts/task_batches/a.json", b"1")
    store.put_bytes("artifacts/task_batches/b.json", b"2")
    store.put_bytes("artifacts/completions/c.json", b"3")
    refs = store.list_prefix("artifacts/task_batches")
    assert [r["key"] for r in refs] == [
        "artifacts/task_batches/a.json",
        "artifacts/task_batches/b.json",
    ]
    assert all(r["backend"] == "r2_rest" for r in refs)


def test_list_prefix_empty_when_no_match():
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    store.put_bytes("other/key.json", b"x")
    assert store.list_prefix("artifacts/") == []


# ---------------------------------------------------------------------------
# ObjectRegistry interop — verify the shim works with the real registry
# ---------------------------------------------------------------------------


def test_object_registry_put_and_get_artifact(tmp_path):
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    registry = ObjectRegistry(store, export_root=str(tmp_path / "exports"))
    artifact = {
        "artifact_type": "task_batch",
        "artifact_id": "abc123",
        "window_id": 42,
        "payload": {"task_count": 4},
    }
    key = registry.put_artifact(artifact)
    assert key == "task_batches/abc123.json"
    got = registry.get_artifact("task_batch", "abc123")
    assert got == artifact


def test_object_registry_completion_bundle_round_trip(tmp_path):
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    registry = ObjectRegistry(store, export_root=str(tmp_path / "exports"))
    completions = [{"completion_id": "c1"}, {"completion_id": "c2"}]
    ref = registry.write_completion_bundle(
        window_id=100, miner_id="miner-a", completions=completions
    )
    assert ref["key"] == "completion_bundles/window-00000100/miner-a.json.gz"
    refs = registry.list_completion_bundles(window_id=100)
    assert len(refs) == 1
    assert refs[0]["miner_id"] == "miner-a"
    round_tripped = registry.read_completion_bundle(refs[0])
    assert round_tripped == completions


def test_object_registry_list_artifacts_filters_by_window(tmp_path):
    backend = _FakeR2Backend()
    store = RestR2ObjectStore(backend=backend)
    registry = ObjectRegistry(store, export_root=str(tmp_path / "exports"))
    registry.put_artifact(
        {"artifact_type": "task_batch", "artifact_id": "a", "window_id": 1}
    )
    registry.put_artifact(
        {"artifact_type": "task_batch", "artifact_id": "b", "window_id": 2}
    )
    arts = registry.list_artifacts("task_batch", window_id=1)
    assert [a["artifact_id"] for a in arts] == ["a"]
