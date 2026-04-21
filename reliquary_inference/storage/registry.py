from __future__ import annotations

import asyncio
import gzip
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..protocol.artifacts import artifact_directory_name
from ..utils.json_io import read_json, write_json


def _gzip_json_bytes(data: Any) -> bytes:
    return gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _gunzip_json_bytes(data: bytes) -> Any:
    return json.loads(gzip.decompress(data).decode("utf-8"))


class RegistryBase(ABC):
    @abstractmethod
    def put_artifact(self, artifact: dict[str, Any]) -> Path | str:
        raise NotImplementedError

    @abstractmethod
    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(self, artifact_type: str, *, window_id: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def write_completion_bundle(self, *, window_id: int, miner_id: str, completions: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_completion_bundles(self, *, window_id: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def read_completion_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def write_verdict_bundle(self, *, window_id: int, validator_id: str, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def read_verdict_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def run_dir(self, run_id: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def put_blob(self, *, key: str, data: bytes) -> Path | str | dict[str, Any]:
        raise NotImplementedError

    def predicted_completion_bundle_ref(self, *, window_id: int, miner_id: str) -> dict[str, Any]:
        return {"backend": "unknown", "window_id": window_id, "miner_id": miner_id}


class LocalRegistry(RegistryBase):
    def __init__(self, artifact_root: str, export_root: str) -> None:
        self.artifact_root = Path(artifact_root)
        self.export_root = Path(export_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_type: str, artifact_id: str) -> Path:
        return self.artifact_root / artifact_directory_name(artifact_type) / f"{artifact_id}.json"

    def put_artifact(self, artifact: dict[str, Any]) -> Path:
        path = self._artifact_path(artifact["artifact_type"], artifact["artifact_id"])
        write_json(path, artifact)
        return path

    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any]:
        return read_json(self._artifact_path(artifact_type, artifact_id))

    def list_artifacts(self, artifact_type: str, *, window_id: int | None = None) -> list[dict[str, Any]]:
        directory = self.artifact_root / artifact_directory_name(artifact_type)
        if not directory.exists():
            return []
        artifacts = [read_json(path) for path in sorted(directory.glob("*.json"))]
        if window_id is not None:
            artifacts = [artifact for artifact in artifacts if int(artifact["window_id"]) == int(window_id)]
        return artifacts

    def write_completion_bundle(self, *, window_id: int, miner_id: str, completions: list[dict[str, Any]]) -> dict[str, Any]:
        path = self.artifact_root / "completion_bundles" / f"window-{window_id:08d}" / f"{miner_id}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_gzip_json_bytes(completions))
        stat = path.stat()
        return {"backend": "local", "path": str(path), "miner_id": miner_id, "uploaded_at": stat.st_mtime}

    def predicted_completion_bundle_ref(self, *, window_id: int, miner_id: str) -> dict[str, Any]:
        path = self.artifact_root / "completion_bundles" / f"window-{window_id:08d}" / f"{miner_id}.json.gz"
        return {"backend": "local", "path": str(path), "miner_id": miner_id}

    def list_completion_bundles(self, *, window_id: int) -> list[dict[str, Any]]:
        directory = self.artifact_root / "completion_bundles" / f"window-{window_id:08d}"
        if not directory.exists():
            return []
        refs = []
        for path in sorted(directory.glob("*.json.gz")):
            refs.append(
                {
                    "backend": "local",
                    "path": str(path),
                    "miner_id": path.name.removesuffix(".json.gz"),
                    "uploaded_at": path.stat().st_mtime,
                }
            )
        return refs

    def read_completion_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        return _gunzip_json_bytes(Path(ref["path"]).read_bytes())

    def write_verdict_bundle(self, *, window_id: int, validator_id: str, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        path = self.artifact_root / "verdict_bundles" / f"window-{window_id:08d}" / f"{validator_id}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_gzip_json_bytes(verdicts))
        stat = path.stat()
        return {"backend": "local", "path": str(path), "validator_id": validator_id, "uploaded_at": stat.st_mtime}

    def read_verdict_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        return _gunzip_json_bytes(Path(ref["path"]).read_bytes())

    def run_dir(self, run_id: str) -> Path:
        path = self.export_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put_blob(self, *, key: str, data: bytes) -> Path:
        path = self.export_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


class FilesystemObjectStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"backend": "filesystem_object_store", "key": key, "uploaded_at": path.stat().st_mtime}

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def list_prefix(self, prefix: str) -> list[dict[str, Any]]:
        directory = self.root / prefix
        if not directory.exists():
            return []
        refs = []
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                refs.append({"backend": "filesystem_object_store", "key": str(path.relative_to(self.root)), "uploaded_at": path.stat().st_mtime})
        return refs


class R2ObjectStore:
    def __init__(self, *, bucket: str, endpoint_url: str, access_key_id: str, secret_access_key: str, region_name: str = "auto") -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

    async def _put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        ) as client:
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return {"backend": "r2", "key": key}

    async def _get_bytes(self, key: str) -> bytes:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        ) as client:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()

    async def _list_prefix(self, prefix: str) -> list[dict[str, Any]]:
        from aiobotocore.session import get_session

        session = get_session()
        refs: list[dict[str, Any]] = []
        async with session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        ) as client:
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "Prefix": prefix,
                }
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                response = await client.list_objects_v2(**kwargs)
                for entry in response.get("Contents", []):
                    refs.append(
                        {
                            "backend": "r2",
                            "key": str(entry["Key"]),
                            "uploaded_at": entry["LastModified"].timestamp(),
                        }
                    )
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")
        return refs

    def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        return asyncio.run(self._put_bytes(key, data))

    def get_bytes(self, key: str) -> bytes:
        return asyncio.run(self._get_bytes(key))

    def list_prefix(self, prefix: str) -> list[dict[str, Any]]:
        return asyncio.run(self._list_prefix(prefix))


class ObjectRegistry(RegistryBase):
    def __init__(self, store: FilesystemObjectStore | R2ObjectStore, export_root: str) -> None:
        self.store = store
        self.export_root = Path(export_root)
        self.export_root.mkdir(parents=True, exist_ok=True)

    def _artifact_key(self, artifact_type: str, artifact_id: str) -> str:
        return f"{artifact_directory_name(artifact_type)}/{artifact_id}.json"

    def put_artifact(self, artifact: dict[str, Any]) -> str:
        key = self._artifact_key(artifact["artifact_type"], artifact["artifact_id"])
        self.store.put_bytes(key, json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8"))
        return key

    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any]:
        raw = self.store.get_bytes(self._artifact_key(artifact_type, artifact_id))
        return json.loads(raw.decode("utf-8"))

    def list_artifacts(self, artifact_type: str, *, window_id: int | None = None) -> list[dict[str, Any]]:
        refs = self.store.list_prefix(artifact_directory_name(artifact_type))
        artifacts = [json.loads(self.store.get_bytes(ref["key"]).decode("utf-8")) for ref in refs]
        if window_id is not None:
            artifacts = [artifact for artifact in artifacts if int(artifact["window_id"]) == int(window_id)]
        return artifacts

    def write_completion_bundle(self, *, window_id: int, miner_id: str, completions: list[dict[str, Any]]) -> dict[str, Any]:
        key = f"completion_bundles/window-{window_id:08d}/{miner_id}.json.gz"
        ref = self.store.put_bytes(key, _gzip_json_bytes(completions))
        ref["miner_id"] = miner_id
        return ref

    def predicted_completion_bundle_ref(self, *, window_id: int, miner_id: str) -> dict[str, Any]:
        key = f"completion_bundles/window-{window_id:08d}/{miner_id}.json.gz"
        return {"backend": "object_store", "key": key, "miner_id": miner_id}

    def list_completion_bundles(self, *, window_id: int) -> list[dict[str, Any]]:
        refs = self.store.list_prefix(f"completion_bundles/window-{window_id:08d}")
        for ref in refs:
            ref["miner_id"] = Path(ref["key"]).name.removesuffix(".json.gz")
        return refs

    def read_completion_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        return _gunzip_json_bytes(self.store.get_bytes(ref["key"]))

    def write_verdict_bundle(self, *, window_id: int, validator_id: str, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        key = f"verdict_bundles/window-{window_id:08d}/{validator_id}.json.gz"
        ref = self.store.put_bytes(key, _gzip_json_bytes(verdicts))
        ref["validator_id"] = validator_id
        return ref

    def read_verdict_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        return _gunzip_json_bytes(self.store.get_bytes(ref["key"]))

    def run_dir(self, run_id: str) -> Path:
        path = self.export_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put_blob(self, *, key: str, data: bytes) -> dict[str, Any]:
        return self.store.put_bytes(key, data)


class R2Registry(ObjectRegistry):
    def __init__(self, *, artifact_root: str, export_root: str, bucket: str, endpoint_url: str, access_key_id: str, secret_access_key: str) -> None:
        super().__init__(
            R2ObjectStore(
                bucket=bucket,
                endpoint_url=endpoint_url,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
            ),
            export_root=export_root,
        )
        self.artifact_root = artifact_root


class RestR2ObjectStore:
    """R2 object store that uses Cloudflare's REST API instead of the S3 API.

    Satisfies the same ``put_bytes`` / ``get_bytes`` / ``list_prefix`` surface
    that :class:`ObjectRegistry` consumes, but reaches R2 through the shared
    :class:`reliquary_protocol.storage.R2ObjectBackend` — which authenticates
    with an account-level Cloudflare API token (``cfat_...``) rather than an
    S3-style access-key/secret pair.

    Why: in deployments where we already have the CF API token (for
    ``reliquary-protocol`` bridge work) but haven't provisioned separate
    S3-compatible R2 tokens in the CF dashboard, this backend lets the
    inference service write artifacts to R2 without touching boto3 or
    duplicating credential material.

    Caveat: Cloudflare's REST API edge-caches authenticated GETs for 4 h.
    Reliquary writes only to fresh content-addressable keys (window_id ×
    validator × completion_id, or artifact SHA-256), so the cache never
    bites production reads — but any admin / migration path that requires
    read-after-overwrite must use a fresh suffix instead.
    """

    def __init__(self, *, backend) -> None:  # backend is R2ObjectBackend
        self.backend = backend

    def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        self.backend.put(key, data)
        return {"backend": "r2_rest", "key": key}

    def get_bytes(self, key: str) -> bytes:
        result = self.backend.get(key)
        if result is None:
            raise FileNotFoundError(f"R2 object not found: {key}")
        return result

    def list_prefix(self, prefix: str) -> list[dict[str, Any]]:
        keys = self.backend.list(prefix)
        return [{"backend": "r2_rest", "key": k} for k in keys]


class RestR2Registry(ObjectRegistry):
    """:class:`ObjectRegistry` backed by :class:`RestR2ObjectStore`.

    Constructed from the four RELIQUARY_INFERENCE_R2_* env vars that
    match ``reliquary_protocol.storage.R2ObjectBackend``'s contract:
    account_id, bucket, cf_api_token, and optional public_url (for the
    r2.dev fast-path on public buckets).
    """

    def __init__(
        self,
        *,
        artifact_root: str,
        export_root: str,
        account_id: str,
        bucket: str,
        cf_api_token: str,
        public_url: str | None = None,
    ) -> None:
        from reliquary_protocol.storage import R2ObjectBackend

        backend = R2ObjectBackend(
            account_id=account_id,
            bucket=bucket,
            cf_api_token=cf_api_token,
            public_url=public_url or None,
        )
        super().__init__(
            RestR2ObjectStore(backend=backend),
            export_root=export_root,
        )
        self.artifact_root = artifact_root
