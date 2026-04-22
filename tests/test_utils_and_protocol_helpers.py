"""Unit tests for reliquary_inference utility + protocol artifact helpers.

Covers the pure-Python pieces that aren't tied to torch/transformers:
- ``reliquary_inference.utils.json_io`` canonical serialization helpers.
- ``reliquary_inference.protocol.artifacts`` (artifact_directory_name,
  make_artifact).

Token helpers live in a sibling module ``test_protocol_tokens.py``
because ``protocol.tokens`` imports from ``shared.hf_compat`` which
pulls in ``transformers`` — that suite skips cleanly when transformers
is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reliquary_inference.protocol.artifacts import (
    ARTIFACT_DIRECTORIES,
    artifact_directory_name,
    make_artifact,
)
from reliquary_inference.utils.json_io import (
    read_json,
    sha256_json,
    stable_json_dumps,
    write_json,
)

# ---------------------------------------------------------------------------
# utils.json_io
# ---------------------------------------------------------------------------


def test_stable_json_dumps_sorts_keys() -> None:
    assert stable_json_dumps({"z": 1, "a": 2}) == '{"a":2,"z":1}'


def test_stable_json_dumps_compact_separators() -> None:
    out = stable_json_dumps({"a": [1, 2]})
    assert ", " not in out
    assert ": " not in out


def test_stable_json_dumps_ensures_ascii_escapes() -> None:
    assert "\\u00e9" in stable_json_dumps({"g": "héllo"})


def test_stable_json_dumps_insertion_order_invariant() -> None:
    a = stable_json_dumps({"a": 1, "b": 2, "c": 3})
    b = stable_json_dumps({"c": 3, "b": 2, "a": 1})
    assert a == b


def test_sha256_json_hex_length_and_alphabet() -> None:
    digest = sha256_json({"k": "v"})
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_sha256_json_matches_manual_sha_over_canonical_bytes() -> None:
    payload = {"a": 1, "b": 2}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert sha256_json(payload) == expected


def test_write_json_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.json"
    write_json(target, {"ok": True})
    assert target.is_file()


def test_write_read_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "trip.json"
    payload = {"n": {"x": [1, 2, 3], "flag": False}}
    write_json(target, payload)
    assert read_json(target) == payload


def test_read_json_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "not-there.json")


def test_stable_json_dumps_rejects_non_json_types() -> None:
    with pytest.raises(TypeError):
        stable_json_dumps({"bytes": b"raw"})


def test_write_json_rejects_non_json_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json(tmp_path / "out.json", {"set": {1, 2, 3}})


# ---------------------------------------------------------------------------
# protocol.artifacts
# ---------------------------------------------------------------------------


def test_artifact_directory_name_known_types() -> None:
    assert artifact_directory_name("verdict") == "verdicts"
    assert artifact_directory_name("completion") == "completions"
    assert artifact_directory_name("task_batch") == "task_batches"


def test_artifact_directory_name_fallback_pluralizes() -> None:
    assert artifact_directory_name("widget") == "widgets"


def test_artifact_directory_name_covers_registered() -> None:
    for artifact_type, expected in ARTIFACT_DIRECTORIES.items():
        assert artifact_directory_name(artifact_type) == expected


def test_make_artifact_default_shape() -> None:
    artifact = make_artifact(
        artifact_type="verdict",
        producer_id="validator-A",
        producer_role="validator",
        window_id=7,
        payload={"ok": True},
    )
    assert artifact["artifact_type"] == "verdict"
    assert artifact["producer_id"] == "validator-A"
    assert artifact["producer_role"] == "validator"
    assert artifact["window_id"] == 7
    assert artifact["parent_ids"] == []


def test_make_artifact_id_is_sha256_hex() -> None:
    artifact = make_artifact(
        artifact_type="verdict",
        producer_id="validator-A",
        producer_role="validator",
        window_id=0,
        payload={"x": 1},
    )
    assert len(artifact["artifact_id"]) == 64
    assert all(ch in "0123456789abcdef" for ch in artifact["artifact_id"])


def test_make_artifact_id_is_deterministic_with_created_at() -> None:
    kwargs = dict(
        artifact_type="verdict",
        producer_id="validator-A",
        producer_role="validator",
        window_id=5,
        payload={"value": 42},
        created_at="2026-04-18T00:00:00+00:00",
    )
    a = make_artifact(**kwargs)
    b = make_artifact(**kwargs)
    assert a["artifact_id"] == b["artifact_id"]


def test_make_artifact_parent_ids_copied() -> None:
    parents = ["p-1", "p-2"]
    artifact = make_artifact(
        artifact_type="verdict",
        producer_id="validator-A",
        producer_role="validator",
        window_id=0,
        payload={},
        parent_ids=parents,
    )
    parents.append("p-3")
    assert artifact["parent_ids"] == ["p-1", "p-2"]


def test_make_artifact_different_payload_different_id() -> None:
    kwargs = dict(
        artifact_type="verdict",
        producer_id="validator-A",
        producer_role="validator",
        window_id=0,
        created_at="2026-04-18T00:00:00+00:00",
    )
    a = make_artifact(payload={"x": 1}, **kwargs)
    b = make_artifact(payload={"x": 2}, **kwargs)
    assert a["artifact_id"] != b["artifact_id"]
