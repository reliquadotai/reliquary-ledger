"""Cloudflare R2 / S3-compatible backend for VerdictPublisher.

Implements the ``StorageBackend`` Protocol from ``verdict_storage.py``
using boto3's S3 client (R2 speaks S3-compatible API). Operators
provide bucket + prefix + credentials + endpoint_url; everything else
is routed through the same publish/fetch code that drives the local
backend.

Dep-guarded: boto3 is not a hard install; ``R2Backend`` import raises
``R2Unavailable`` if the module can't be found. Tests inject a fake
client to avoid any network dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class R2Unavailable(RuntimeError):
    """Raised when boto3 is required but not installed."""


class _S3Client(Protocol):
    """Subset of the boto3 S3 client API we depend on."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Any: ...

    def get_object(self, *, Bucket: str, Key: str) -> Any: ...

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", ContinuationToken: str = "") -> Any: ...

    def delete_object(self, *, Bucket: str, Key: str) -> Any: ...


def _boto3_client(
    *,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
) -> _S3Client:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise R2Unavailable(
            "boto3 is not installed; install with `pip install reliquary-inference[r2]` "
            "or pass a client explicitly to R2Backend"
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=access_key_id or None,
        aws_secret_access_key=secret_access_key or None,
        region_name=region or None,
    )


@dataclass
class R2Backend:
    """S3-compatible StorageBackend for verdict publication.

    Constructor does NOT make a network call — a missing bucket or
    bad credentials surface on the first put/get. Tests pass
    ``client=FakeS3()`` to avoid any network dependency.
    """

    bucket: str
    endpoint_url: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "auto"
    key_prefix: str = ""
    client: _S3Client | None = None
    _initialized: bool = field(default=False, init=False, repr=False)

    def _ensure_client(self) -> _S3Client:
        if self.client is None:
            self.client = _boto3_client(
                endpoint_url=self.endpoint_url,
                access_key_id=self.access_key_id,
                secret_access_key=self.secret_access_key,
                region=self.region,
            )
        return self.client

    def _full_key(self, key: str) -> str:
        if not self.key_prefix:
            return key
        return f"{self.key_prefix.rstrip('/')}/{key.lstrip('/')}"

    # ------------------------------------------------------------------
    # StorageBackend Protocol
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        client = self._ensure_client()
        client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def get(self, key: str) -> bytes | None:
        client = self._ensure_client()
        full_key = self._full_key(key)
        try:
            response = client.get_object(Bucket=self.bucket, Key=full_key)
        except Exception as exc:
            # boto3 raises ClientError on 404 / 403. We only treat 404 as
            # absence; other errors propagate so the caller can retry.
            if _is_not_found(exc):
                return None
            raise
        body = response.get("Body")
        if body is None:
            return b""
        return body.read()

    def list(self, prefix: str) -> list[str]:
        client = self._ensure_client()
        full_prefix = self._full_key(prefix) if prefix else self.key_prefix
        keys: list[str] = []
        continuation: str | None = None
        strip_prefix = f"{self.key_prefix.rstrip('/')}/" if self.key_prefix else ""
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": full_prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for entry in response.get("Contents") or []:
                stored_key = str(entry["Key"])
                if strip_prefix and stored_key.startswith(strip_prefix):
                    stored_key = stored_key[len(strip_prefix):]
                keys.append(stored_key)
            if response.get("IsTruncated") and response.get("NextContinuationToken"):
                continuation = response["NextContinuationToken"]
                continue
            break
        return sorted(keys)

    def delete(self, key: str) -> None:
        client = self._ensure_client()
        client.delete_object(Bucket=self.bucket, Key=self._full_key(key))


def _is_not_found(exc: Exception) -> bool:
    """Detect S3 404 across boto3 versions without hard-depending on botocore."""
    response_error = getattr(exc, "response", None)
    if isinstance(response_error, dict):
        code = str(response_error.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return True
    status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get(
        "HTTPStatusCode"
    ) if hasattr(exc, "response") else None
    if status == 404:
        return True
    text = str(exc).lower()
    return "nosuchkey" in text or "not found" in text


__all__ = [
    "R2Backend",
    "R2Unavailable",
]
