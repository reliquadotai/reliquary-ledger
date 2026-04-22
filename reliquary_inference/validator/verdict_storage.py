"""Verdict artifact signing, persistence, and cross-validator fetch.

Completes the remaining scope of Tier 2 Epic 1 after the pure-compute
aggregator shipped on 2026-04-17. A validator signs its per-completion
verdicts at window close, uploads them to a shared storage backend, and
peers fetch the window's verdicts for consensus. A local spool absorbs
transient backend failures so a flaky R2 connection does not drop the
window.

Auth-agnostic by design: the ``VerdictSigner`` and ``VerdictVerifier``
accept arbitrary callables. Production wires these to Bittensor hotkey
signing; unit tests wire a tiny HMAC.

Spec: ``private/reliquary-plan/notes/spec-verdict-storage.md`` +
``02_TIER2_PRD.md`` Epic 1 acceptance criteria (R2 upload + signature
verification + consensus).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .mesh import ValidatorIdentity, VerdictArtifact

logger = logging.getLogger(__name__)


VERDICT_STORAGE_VERSION: str = "v1"
VERDICT_CANONICAL_SEPARATORS: tuple[str, str] = (",", ":")
VERDICT_SCHEMA_REQUIRED_FIELDS: tuple[str, ...] = (
    "completion_id",
    "miner_hotkey",
    "window_id",
    "validator",
    "accepted",
    "scores",
    "signed_at",
    "schema_version",
)


class StorageBackend(Protocol):
    """Bytes-level key/value storage abstraction.

    ``put`` must be atomic — a partially-written object is never visible
    to a subsequent ``get`` — so the ``LocalFilesystemBackend``
    implementation writes to a temp file and renames.
    """

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...


class LocalFilesystemBackend:
    """Filesystem-backed storage. Keys are path-relative; writes are atomic.

    Suitable for unit tests and single-host validator deployments. For
    multi-host, compose a composite backend on top of an S3-compatible
    driver.
    """

    def __init__(self, root_dir: str | os.PathLike[str]) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        target = (self.root / key).resolve()
        root_resolved = self.root.resolve()
        if root_resolved not in target.parents and target != root_resolved:
            raise ValueError(f"key {key!r} escapes storage root")
        return target

    def put(self, key: str, data: bytes) -> None:
        final = self._resolve(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=str(final.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, final)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def get(self, key: str) -> bytes | None:
        target = self._resolve(key)
        if not target.exists():
            return None
        return target.read_bytes()

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists() or not base.is_dir():
            return []
        root = self.root.resolve()
        return sorted(
            str(p.resolve().relative_to(root))
            for p in base.rglob("*")
            if p.is_file()
        )

    def delete(self, key: str) -> None:
        target = self._resolve(key)
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def verdict_key(
    netuid: int,
    window_id: int,
    validator_hotkey: str,
    completion_id: str,
) -> str:
    """Canonical object key for a single verdict artifact."""
    safe_completion = completion_id.replace("/", "_")
    return f"verdicts/{netuid}/{window_id}/{validator_hotkey}/{safe_completion}.json"


def _canonicalize(artifact: VerdictArtifact) -> bytes:
    """Return the sorted-keys, compact-JSON canonical bytes of a verdict.

    Two logically-equal verdicts produce identical bytes regardless of
    dict insertion order, which is the property signatures rely on.
    """
    payload: dict[str, Any] = {
        "completion_id": artifact.completion_id,
        "miner_hotkey": artifact.miner_hotkey,
        "window_id": artifact.window_id,
        "validator": {
            "hotkey": artifact.validator.hotkey,
            "stake": artifact.validator.stake,
            "signer_id": artifact.validator.signer_id or artifact.validator.hotkey,
        },
        "accepted": artifact.accepted,
        "stage_failed": artifact.stage_failed,
        "reject_reason": artifact.reject_reason,
        "scores": {k: artifact.scores[k] for k in sorted(artifact.scores)},
        "signed_at": artifact.signed_at,
        "schema_version": VERDICT_STORAGE_VERSION,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=VERDICT_CANONICAL_SEPARATORS,
    ).encode("utf-8")


@dataclass(frozen=True)
class VerdictSigner:
    """Signs canonical verdict bytes. Never inspects the callable's key material."""

    signer_id: str
    sign: Callable[[bytes], str]


@dataclass(frozen=True)
class VerdictVerifier:
    """Verifies signatures against an expected-hotkey allowlist."""

    expected_hotkeys: dict[str, Any]
    verify: Callable[[str, bytes, str, Any], bool]
    """verify(signer_id, canonical_bytes, signature, public_key) -> bool."""


@dataclass(frozen=True)
class PublishOutcome:
    success: bool
    backend: str
    key: str | None
    attempts: int
    last_error: str | None = None


@dataclass
class InvalidArtifactReport:
    key: str
    reason: str
    signer_id: str | None = None
    detail: str = ""


@dataclass
class FetchResult:
    artifacts: list[VerdictArtifact] = field(default_factory=list)
    invalid: list[InvalidArtifactReport] = field(default_factory=list)
    window_id: int = 0
    netuid: int = 0


class VerdictPublisher:
    """Signs and publishes verdicts; spools on backend failure."""

    def __init__(
        self,
        backend: StorageBackend,
        signer: VerdictSigner,
        *,
        netuid: int,
        spool_dir: str | os.PathLike[str] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.signer = signer
        self.netuid = netuid
        self.max_attempts = max_attempts
        self.spool_dir = Path(spool_dir) if spool_dir else None
        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)

    def _envelope(self, artifact: VerdictArtifact) -> bytes:
        try:
            canonical = _canonicalize(artifact)
        except Exception as exc:
            raise RuntimeError(f"canonicalize failed: {type(exc).__name__}: {exc}") from exc
        try:
            signature = self.signer.sign(canonical)
        except Exception as exc:
            raise RuntimeError(f"sign failed: {type(exc).__name__}: {exc}") from exc
        envelope = {
            "payload_json": canonical.decode("utf-8"),
            "signature": signature,
            "signer_id": self.signer.signer_id,
        }
        return json.dumps(envelope, sort_keys=True, separators=VERDICT_CANONICAL_SEPARATORS).encode("utf-8")

    def publish(self, artifact: VerdictArtifact) -> PublishOutcome:
        key = verdict_key(
            self.netuid,
            artifact.window_id,
            artifact.validator.hotkey,
            artifact.completion_id,
        )
        try:
            envelope = self._envelope(artifact)
        except RuntimeError as exc:
            return PublishOutcome(
                success=False, backend="none", key=None, attempts=0, last_error=str(exc),
            )

        last_err: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.backend.put(key, envelope)
                return PublishOutcome(success=True, backend="remote", key=key, attempts=attempt)
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "verdict publish attempt %d/%d failed for %s: %s",
                    attempt, self.max_attempts, key, last_err,
                )

        if self.spool_dir is not None:
            spool_key = self.spool_dir / key.replace("/", "__")
            spool_key.parent.mkdir(parents=True, exist_ok=True)
            spool_key.write_bytes(envelope)
            return PublishOutcome(
                success=False, backend="spool", key=str(spool_key),
                attempts=self.max_attempts, last_error=last_err,
            )
        return PublishOutcome(
            success=False, backend="none", key=None,
            attempts=self.max_attempts, last_error=last_err,
        )

    def flush_spool(self) -> list[PublishOutcome]:
        """Attempt to upload every spooled envelope. Successful entries are deleted."""
        if self.spool_dir is None or not self.spool_dir.exists():
            return []
        outcomes: list[PublishOutcome] = []
        for entry in sorted(self.spool_dir.rglob("*")):
            if not entry.is_file():
                continue
            key = entry.name.replace("__", "/")
            envelope = entry.read_bytes()
            try:
                self.backend.put(key, envelope)
                entry.unlink()
                outcomes.append(PublishOutcome(success=True, backend="remote", key=key, attempts=1))
            except Exception as exc:  # noqa: BLE001
                outcomes.append(PublishOutcome(
                    success=False, backend="spool", key=str(entry),
                    attempts=1, last_error=f"{type(exc).__name__}: {exc}",
                ))
        return outcomes


class VerdictFetcher:
    """Fetches a window's verdicts and surfaces signature-valid ones."""

    def __init__(
        self,
        backend: StorageBackend,
        verifier: VerdictVerifier,
        *,
        netuid: int,
    ) -> None:
        self.backend = backend
        self.verifier = verifier
        self.netuid = netuid

    def fetch_window(self, window_id: int) -> FetchResult:
        prefix = f"verdicts/{self.netuid}/{window_id}/"
        keys = self.backend.list(prefix)
        result = FetchResult(window_id=window_id, netuid=self.netuid)

        for key in keys:
            envelope_bytes = self.backend.get(key)
            if envelope_bytes is None:
                result.invalid.append(InvalidArtifactReport(key=key, reason="missing"))
                continue

            try:
                envelope = json.loads(envelope_bytes.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="malformed_json",
                    detail=f"{type(exc).__name__}: {exc}",
                ))
                continue

            signer_id = envelope.get("signer_id")
            signature = envelope.get("signature")
            payload_str = envelope.get("payload_json")

            if not (isinstance(signer_id, str) and isinstance(signature, str) and isinstance(payload_str, str)):
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="malformed_json",
                    detail="envelope fields missing or wrong type",
                ))
                continue

            public_key = self.verifier.expected_hotkeys.get(signer_id)
            if public_key is None:
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="unknown_signer", signer_id=signer_id,
                ))
                continue

            canonical = payload_str.encode("utf-8")
            try:
                ok = self.verifier.verify(signer_id, canonical, signature, public_key)
            except Exception as exc:  # noqa: BLE001
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="bad_signature", signer_id=signer_id,
                    detail=f"{type(exc).__name__}: {exc}",
                ))
                continue
            if not ok:
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="bad_signature", signer_id=signer_id,
                ))
                continue

            try:
                payload = json.loads(payload_str)
            except Exception:  # noqa: BLE001
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="malformed_json", signer_id=signer_id,
                    detail="payload_json is not valid JSON",
                ))
                continue

            missing = [f for f in VERDICT_SCHEMA_REQUIRED_FIELDS if f not in payload]
            if missing:
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="schema_mismatch", signer_id=signer_id,
                    detail=f"missing fields: {missing}",
                ))
                continue

            if payload["window_id"] != window_id:
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="schema_mismatch", signer_id=signer_id,
                    detail=f"window_id mismatch: envelope for {window_id}, payload {payload['window_id']}",
                ))
                continue

            validator = payload["validator"]
            if validator.get("hotkey") != signer_id:
                result.invalid.append(InvalidArtifactReport(
                    key=key, reason="schema_mismatch", signer_id=signer_id,
                    detail="validator.hotkey != signer_id",
                ))
                continue

            artifact = VerdictArtifact(
                completion_id=payload["completion_id"],
                miner_hotkey=payload["miner_hotkey"],
                window_id=int(payload["window_id"]),
                validator=ValidatorIdentity(
                    hotkey=validator["hotkey"],
                    stake=float(validator.get("stake", 0.0)),
                    signer_id=validator.get("signer_id", signer_id),
                ),
                accepted=bool(payload["accepted"]),
                stage_failed=payload.get("stage_failed"),
                reject_reason=payload.get("reject_reason"),
                scores=dict(payload["scores"]),
                signed_at=float(payload["signed_at"]),
                signature=signature,
            )
            result.artifacts.append(artifact)

        return result
