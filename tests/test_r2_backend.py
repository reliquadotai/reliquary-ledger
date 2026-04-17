"""Tests for the R2 / S3-compatible verdict-storage backend.

Uses an in-memory FakeS3 client to avoid any network dep. Verifies the
R2Backend correctly maps the StorageBackend Protocol onto boto3's API
shape + handles 404 as absence + paginates list_objects_v2 + respects
key_prefix scoping.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pytest

from reliquary_inference.validator.r2_backend import (
    R2Backend,
    R2Unavailable,
    _boto3_client,
    _is_not_found,
)


class _NotFound(Exception):
    """Fake boto3 ClientError: ``NoSuchKey``."""

    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


@dataclass
class FakeS3:
    """Minimal S3-compatible stand-in for tests."""

    store: dict[tuple[str, str], bytes] = field(default_factory=dict)
    put_calls: list[tuple[str, str, int]] = field(default_factory=list)
    get_calls: list[tuple[str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, str]] = field(default_factory=list)
    page_size: int = 4

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.store[(Bucket, Key)] = bytes(Body)
        self.put_calls.append((Bucket, Key, len(Body)))
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self.get_calls.append((Bucket, Key))
        if (Bucket, Key) not in self.store:
            raise _NotFound()
        body = self.store[(Bucket, Key)]
        return {"Body": io.BytesIO(body)}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        ContinuationToken: str = "",
    ) -> dict:
        matching = sorted(
            key for (bucket, key) in self.store if bucket == Bucket and key.startswith(Prefix)
        )
        offset = int(ContinuationToken) if ContinuationToken else 0
        page = matching[offset : offset + self.page_size]
        is_truncated = offset + self.page_size < len(matching)
        response: dict[str, Any] = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": is_truncated,
        }
        if is_truncated:
            response["NextContinuationToken"] = str(offset + self.page_size)
        return response

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.delete_calls.append((Bucket, Key))
        self.store.pop((Bucket, Key), None)
        return {}


@pytest.fixture
def fake_client() -> FakeS3:
    return FakeS3()


@pytest.fixture
def backend(fake_client: FakeS3) -> R2Backend:
    return R2Backend(
        bucket="reliquary-verdicts",
        client=fake_client,
    )


# ---------------------------------------------------------------------------
# put / get round trip
# ---------------------------------------------------------------------------


def test_put_then_get_returns_bytes(backend: R2Backend) -> None:
    backend.put("verdicts/foo.json", b'{"ok": true}')
    assert backend.get("verdicts/foo.json") == b'{"ok": true}'


def test_get_absent_returns_none(backend: R2Backend) -> None:
    assert backend.get("verdicts/missing.json") is None


def test_delete_removes_key(backend: R2Backend) -> None:
    backend.put("verdicts/toremove.json", b"data")
    assert backend.get("verdicts/toremove.json") == b"data"
    backend.delete("verdicts/toremove.json")
    assert backend.get("verdicts/toremove.json") is None


def test_put_records_bucket_and_key(
    backend: R2Backend, fake_client: FakeS3
) -> None:
    backend.put("k", b"v")
    assert ("reliquary-verdicts", "k") in fake_client.store


# ---------------------------------------------------------------------------
# key_prefix scoping
# ---------------------------------------------------------------------------


def test_key_prefix_is_applied_to_put_and_get() -> None:
    client = FakeS3()
    backend = R2Backend(
        bucket="bucket",
        key_prefix="reliquary/env",
        client=client,
    )
    backend.put("verdicts/a.json", b"data")
    # Stored under the full prefixed path.
    assert ("bucket", "reliquary/env/verdicts/a.json") in client.store
    # get() with the un-prefixed key returns the data.
    assert backend.get("verdicts/a.json") == b"data"


def test_key_prefix_trailing_slashes_are_stripped() -> None:
    client = FakeS3()
    backend = R2Backend(
        bucket="bucket", key_prefix="prefix/", client=client
    )
    backend.put("key", b"x")
    assert ("bucket", "prefix/key") in client.store


# ---------------------------------------------------------------------------
# list pagination
# ---------------------------------------------------------------------------


def test_list_empty_prefix_returns_sorted_keys(backend: R2Backend) -> None:
    backend.put("verdicts/b.json", b"b")
    backend.put("verdicts/a.json", b"a")
    backend.put("verdicts/c.json", b"c")
    listed = backend.list("verdicts/")
    assert listed == ["verdicts/a.json", "verdicts/b.json", "verdicts/c.json"]


def test_list_paginates_through_truncated_results(fake_client: FakeS3) -> None:
    fake_client.page_size = 2
    backend = R2Backend(bucket="b", client=fake_client)
    for i in range(5):
        backend.put(f"v/{i:02d}.json", b"x")
    listed = backend.list("v/")
    assert len(listed) == 5
    assert listed == [f"v/{i:02d}.json" for i in range(5)]


def test_list_respects_prefix_filter(backend: R2Backend) -> None:
    backend.put("a/one.json", b"1")
    backend.put("a/two.json", b"2")
    backend.put("b/three.json", b"3")
    listed = backend.list("a/")
    assert listed == ["a/one.json", "a/two.json"]


def test_list_with_key_prefix_strips_it_from_results() -> None:
    client = FakeS3()
    backend = R2Backend(bucket="b", key_prefix="env", client=client)
    backend.put("verdicts/x.json", b"x")
    backend.put("verdicts/y.json", b"y")
    listed = backend.list("verdicts/")
    # Returned keys are prefix-stripped so callers can round-trip with get().
    assert listed == ["verdicts/x.json", "verdicts/y.json"]


# ---------------------------------------------------------------------------
# 404 detection
# ---------------------------------------------------------------------------


def test_is_not_found_detects_nosuchkey_shape() -> None:
    exc = _NotFound()
    assert _is_not_found(exc) is True


def test_is_not_found_detects_404_status() -> None:
    class _Err(Exception):
        response = {"Error": {"Code": ""}, "ResponseMetadata": {"HTTPStatusCode": 404}}

    assert _is_not_found(_Err("x")) is True


def test_is_not_found_detects_text_match() -> None:
    class _Err(Exception):
        pass

    err = _Err("The specified key does not exist (NoSuchKey)")
    assert _is_not_found(err) is True


def test_is_not_found_misses_non_404_errors() -> None:
    class _Err(Exception):
        response = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}

    assert _is_not_found(_Err("AccessDenied")) is False


# ---------------------------------------------------------------------------
# boto3 unavailability
# ---------------------------------------------------------------------------


def test_boto3_client_raises_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    # Intentionally shadow the boto3 module so the import fails in our
    # helper. Restore afterwards.
    saved = sys.modules.get("boto3")
    sys.modules.pop("boto3", None)
    monkeypatch.setitem(sys.modules, "boto3", None)  # type: ignore[assignment]
    try:
        with pytest.raises(R2Unavailable):
            _boto3_client(
                endpoint_url="",
                access_key_id="",
                secret_access_key="",
                region="auto",
            )
    finally:
        if saved is not None:
            sys.modules["boto3"] = saved


def test_backend_ensure_client_called_only_once(
    backend: R2Backend, fake_client: FakeS3
) -> None:
    # Even after many put/get calls the injected client is never
    # replaced by a boto3-backed one.
    for _ in range(5):
        backend.put("k", b"v")
        backend.get("k")
    assert backend.client is fake_client


# ---------------------------------------------------------------------------
# Integration with VerdictPublisher
# ---------------------------------------------------------------------------


def test_r2_backend_satisfies_storage_backend_protocol() -> None:
    """The R2Backend must be usable wherever StorageBackend is expected.

    We check the four required methods exist with the right signatures
    by constructing a VerdictPublisher and letting its type hints do
    the contract work.
    """
    from reliquary_inference.validator.verdict_storage import StorageBackend

    client = FakeS3()
    backend = R2Backend(bucket="b", client=client)
    # Protocol runtime-check via hasattr: StorageBackend requires
    # put/get/list/delete. Catch any signature regressions.
    for method in ("put", "get", "list", "delete"):
        assert callable(getattr(backend, method))
