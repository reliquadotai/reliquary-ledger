"""Rollout bundle producer — closes the Ledger → Forge bridge.

After a window's mesh aggregation completes, this module packages the
accepted rollouts (task batch + completions + mesh-aggregated verdicts
+ scorecard) into the canonical ``RolloutBundle`` shape that Forge's
``InferenceRegistryAdapter`` expects, signs the envelope, and publishes
to the shared ``StorageBackend``. Failure to reach the backend spools
locally and flushes on reconnect — same pattern as ``VerdictPublisher``.

Paired consumer: ``reliquary.inference_adapter.InferenceRegistryAdapter``
which already reads ``{manifest, task_batch, scorecard, completions,
verdicts}`` off the same backend.

Spec: ``private/reliquary-plan/notes/spec-closed-loop-bridge.md``
§4 (Ledger rollout bundle producer).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from reliquary_protocol import (
    BRIDGE_VERSION,
    BridgeSigner,
    BridgeVerifier,
    HmacBridgeSigner,
    HmacBridgeVerifier,
    ROLLOUT_BUNDLE_TYPE,
    RolloutBundle,
    build_rollout_bundle,
    envelope_from_dict,
    sign_envelope,
    verify_envelope,
)

from .mesh import MeshAggregationReport, VerdictArtifact
from .verdict_storage import StorageBackend


logger = logging.getLogger(__name__)


ROLLOUT_BUNDLE_METRIC_PUBLISHED = "reliquary_bridge_rollout_bundle_published_total"
ROLLOUT_BUNDLE_METRIC_SPOOLED = "reliquary_bridge_rollout_bundle_spooled_total"
ROLLOUT_BUNDLE_METRIC_REGRESSED = "reliquary_bridge_rollout_bundle_regression_total"


@dataclass(frozen=True)
class PublishRolloutOutcome:
    """Result of a publish call — successful or spooled."""

    success: bool
    key: str | None
    spooled_path: str | None
    attempts: int
    last_error: str | None = None


@dataclass(frozen=True)
class FetchRolloutResult:
    """Return value of :meth:`RolloutBundleFetcher.fetch`.

    ``bundles`` holds the verified bundles; ``invalid`` records any
    keys that failed signature, schema, or version checks — with a
    structured reason so the operator can triage.
    """

    bundles: list[RolloutBundle] = field(default_factory=list)
    invalid: list["InvalidBundleReport"] = field(default_factory=list)
    netuid: int = 0


@dataclass(frozen=True)
class InvalidBundleReport:
    key: str
    reason: str
    signer_id: str | None = None
    detail: str = ""


class RolloutBundlePublisher:
    """Builds, signs, and publishes a :class:`RolloutBundle`.

    Spool-and-retry semantics mirror ``VerdictPublisher``:
    - On backend failure, serialized envelope goes to ``spool_dir``.
    - ``flush_spool`` drains the spool sorted, re-uploading each.
    - The caller controls when ``flush_spool`` runs; typical cadence
      is "once per window close" in the validator orchestrator.
    """

    def __init__(
        self,
        backend: StorageBackend,
        signer: BridgeSigner,
        *,
        netuid: int,
        spool_dir: str | os.PathLike[str] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.signer = signer
        self.netuid = int(netuid)
        self.max_attempts = max(1, int(max_attempts))
        self.spool_dir = Path(spool_dir) if spool_dir is not None else None
        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._last_published_window: int | None = None
        self.metrics_counters: dict[str, int] = {
            ROLLOUT_BUNDLE_METRIC_PUBLISHED: 0,
            ROLLOUT_BUNDLE_METRIC_SPOOLED: 0,
            ROLLOUT_BUNDLE_METRIC_REGRESSED: 0,
        }

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def publish(
        self,
        *,
        window_id: int,
        mesh_report: MeshAggregationReport,
        manifest: Mapping[str, Any],
        task_batch: Mapping[str, Any],
        scorecard: Mapping[str, Any] | None,
        accepted_completions: Sequence[Mapping[str, Any]],
        verdict_artifacts: Sequence[VerdictArtifact] | None = None,
        window_range: tuple[int, int] | None = None,
        published_at: str | None = None,
    ) -> PublishRolloutOutcome:
        """Build + sign + publish a bundle for one closed window.

        Window monotonicity is enforced in-process: if ``window_id <=``
        the last successfully-published window, the call is rejected and
        the regression counter is incremented. Operators can reset by
        constructing a fresh publisher.
        """
        if (
            self._last_published_window is not None
            and int(window_id) <= self._last_published_window
        ):
            self.metrics_counters[ROLLOUT_BUNDLE_METRIC_REGRESSED] += 1
            return PublishRolloutOutcome(
                success=False,
                key=None,
                spooled_path=None,
                attempts=0,
                last_error=(
                    f"window {window_id} <= last_published {self._last_published_window}"
                ),
            )

        verdict_dicts = _verdicts_to_dicts(
            mesh_report=mesh_report,
            verdict_artifacts=verdict_artifacts or (),
        )
        accepted_completion_ids = _accepted_completion_ids(mesh_report)
        filtered_completions = [
            dict(c)
            for c in accepted_completions
            if str(c.get("completion_id", "")) in accepted_completion_ids
        ]

        bundle = build_rollout_bundle(
            netuid=self.netuid,
            window_id=int(window_id),
            window_range=window_range,
            manifest=manifest,
            task_batch=task_batch,
            scorecard=scorecard,
            completions=filtered_completions,
            verdicts=verdict_dicts,
            producer_hotkey=str(self.signer.signer_id),
            published_at=published_at,
        )
        key = bundle.storage_key()
        envelope = sign_envelope(ROLLOUT_BUNDLE_TYPE, bundle, self.signer)
        envelope_bytes = envelope.canonical_bytes()

        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.backend.put(key, envelope_bytes)
                self._last_published_window = int(window_id)
                self.metrics_counters[ROLLOUT_BUNDLE_METRIC_PUBLISHED] += 1
                logger.info(
                    "rollout_bundle.publish ok window=%d key=%s attempts=%d",
                    window_id,
                    key,
                    attempt,
                )
                return PublishRolloutOutcome(
                    success=True,
                    key=key,
                    spooled_path=None,
                    attempts=attempt,
                    last_error=None,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "rollout_bundle.publish failure window=%d attempt=%d err=%s",
                    window_id,
                    attempt,
                    last_error,
                )

        spooled_path = self._spool(key, envelope_bytes)
        self.metrics_counters[ROLLOUT_BUNDLE_METRIC_SPOOLED] += 1
        return PublishRolloutOutcome(
            success=False,
            key=key,
            spooled_path=spooled_path,
            attempts=self.max_attempts,
            last_error=last_error,
        )

    def flush_spool(self) -> list[PublishRolloutOutcome]:
        """Drain the spool by re-attempting each spooled envelope.

        Each spool entry is tried once per call (no internal retries);
        failures stay in the spool for the next flush. Drain order is
        filename-sorted — spool entries are prefixed with the canonical
        storage key so this mimics chronological replay.
        """
        if self.spool_dir is None:
            return []
        outcomes: list[PublishRolloutOutcome] = []
        for entry in sorted(self.spool_dir.iterdir()):
            if not entry.is_file():
                continue
            data = entry.read_bytes()
            try:
                envelope_dict = json.loads(data.decode("utf-8"))
            except Exception as exc:
                logger.error("spool entry %s is corrupt: %s", entry, exc)
                continue
            try:
                key = _key_from_spool_entry(entry.name)
            except ValueError as exc:
                logger.error("spool entry %s has unparseable name: %s", entry, exc)
                continue
            try:
                self.backend.put(key, data)
                entry.unlink(missing_ok=True)  # type: ignore[call-arg]
                self.metrics_counters[ROLLOUT_BUNDLE_METRIC_PUBLISHED] += 1
                outcomes.append(
                    PublishRolloutOutcome(
                        success=True,
                        key=key,
                        spooled_path=None,
                        attempts=1,
                    )
                )
            except Exception as exc:  # pragma: no cover - external failure
                outcomes.append(
                    PublishRolloutOutcome(
                        success=False,
                        key=key,
                        spooled_path=str(entry),
                        attempts=1,
                        last_error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return outcomes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spool(self, key: str, envelope_bytes: bytes) -> str | None:
        if self.spool_dir is None:
            return None
        spool_name = _spool_name(key)
        target = self.spool_dir / spool_name
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=str(self.spool_dir))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(envelope_bytes)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return str(target)


class RolloutBundleFetcher:
    """Verifies and returns every bundle under a window prefix.

    Forge callers can use this for smoke-testing the bridge without
    running the full ``InferenceRegistryAdapter`` pipeline. Production
    Forge still goes through ``InferenceRegistryAdapter`` which knows
    about cursor state, idempotency, and incremental pickup.
    """

    def __init__(
        self,
        backend: StorageBackend,
        verifier: BridgeVerifier,
        *,
        netuid: int,
    ) -> None:
        self.backend = backend
        self.verifier = verifier
        self.netuid = int(netuid)

    def fetch(self, prefix: str | None = None) -> FetchRolloutResult:
        """Return every valid bundle under ``prefix``.

        ``prefix`` defaults to ``rollouts/<netuid>/`` (all windows).
        """
        if prefix is None:
            prefix = f"rollouts/{self.netuid}/"
        result = FetchRolloutResult(netuid=self.netuid)
        for key in self.backend.list(prefix):
            data = self.backend.get(key)
            if data is None:
                result.invalid.append(
                    InvalidBundleReport(key=key, reason="missing_after_list")
                )
                continue
            try:
                envelope_dict = json.loads(data.decode("utf-8"))
            except Exception as exc:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key, reason="malformed_envelope", detail=str(exc)
                    )
                )
                continue
            try:
                envelope = envelope_from_dict(envelope_dict)
            except Exception as exc:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key, reason="envelope_schema", detail=str(exc)
                    )
                )
                continue
            if envelope.artifact_type != ROLLOUT_BUNDLE_TYPE:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key,
                        reason="wrong_artifact_type",
                        detail=envelope.artifact_type,
                        signer_id=envelope.signer_id,
                    )
                )
                continue
            if not verify_envelope(envelope, self.verifier):
                result.invalid.append(
                    InvalidBundleReport(
                        key=key,
                        reason="bad_signature",
                        signer_id=envelope.signer_id,
                    )
                )
                continue
            try:
                inner = RolloutBundle.from_dict(json.loads(envelope.payload_json))
            except Exception as exc:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key,
                        reason="bundle_schema",
                        detail=str(exc),
                        signer_id=envelope.signer_id,
                    )
                )
                continue
            if inner.netuid != self.netuid:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key,
                        reason="netuid_mismatch",
                        detail=f"expected {self.netuid} got {inner.netuid}",
                        signer_id=envelope.signer_id,
                    )
                )
                continue
            if inner.version != BRIDGE_VERSION:
                result.invalid.append(
                    InvalidBundleReport(
                        key=key,
                        reason="bridge_version_mismatch",
                        detail=f"expected {BRIDGE_VERSION} got {inner.version}",
                        signer_id=envelope.signer_id,
                    )
                )
                continue
            result.bundles.append(inner)
        # Stable order: by (window_id, producer_hotkey) to pin test assertions.
        result.bundles.sort(key=lambda b: (b.window_id, b.producer_hotkey))
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _accepted_completion_ids(report: MeshAggregationReport) -> set[str]:
    return {
        completion_id
        for completion_id, median in report.median_verdicts.items()
        if median.accepted and median.quorum_satisfied
    }


def _verdicts_to_dicts(
    *,
    mesh_report: MeshAggregationReport,
    verdict_artifacts: Sequence[VerdictArtifact],
) -> list[dict[str, Any]]:
    """Project the mesh-aggregated verdicts into plain dicts.

    Every accepted median verdict yields one dict. Any matching raw
    validator artifact under the same completion_id is also embedded
    so downstream Forge consumers have per-validator provenance even
    after aggregation (they can see how many validators participated,
    which were outliers, etc.).
    """
    by_completion: dict[str, list[VerdictArtifact]] = {}
    for art in verdict_artifacts:
        by_completion.setdefault(art.completion_id, []).append(art)

    out: list[dict[str, Any]] = []
    for completion_id, median in mesh_report.median_verdicts.items():
        if not median.accepted or not median.quorum_satisfied:
            continue
        per_validator = [
            {
                "validator_hotkey": art.validator.hotkey,
                "signer_id": art.validator.signer_id or art.validator.hotkey,
                "stake": float(art.validator.stake),
                "accepted": bool(art.accepted),
                "stage_failed": art.stage_failed,
                "reject_reason": art.reject_reason,
                "scores": {k: float(v) for k, v in art.scores.items()},
                "signed_at": float(art.signed_at),
            }
            for art in by_completion.get(completion_id, [])
        ]
        per_validator.sort(key=lambda d: str(d["validator_hotkey"]))
        out.append(
            {
                "completion_id": str(completion_id),
                "accepted": True,
                "median_scores": {k: float(v) for k, v in median.median_scores.items()},
                "acceptance_score": float(median.acceptance_score),
                "participating_validators": sorted(
                    str(h) for h in median.participating_validators
                ),
                "outlier_validators": sorted(
                    str(h) for h in median.outlier_validators
                ),
                "per_validator": per_validator,
            }
        )
    # Stable: by completion_id.
    out.sort(key=lambda d: str(d["completion_id"]))
    return out


def _spool_name(key: str) -> str:
    """Map a storage key to a flat spool-dir filename."""
    return key.replace("/", "__")


def _key_from_spool_entry(name: str) -> str:
    # Reverse of _spool_name. Raise if the entry isn't a mapped key.
    if "__" not in name:
        raise ValueError(f"spool entry {name!r} has no __ separator")
    return name.replace("__", "/")


def make_hmac_signer(hotkey: str, secret: str) -> HmacBridgeSigner:
    """Convenience constructor for a HMAC bridge signer.

    Exposed so callers that already have a hotkey → HMAC secret mapping
    don't need to import from ``reliquary_protocol`` directly.
    """
    return HmacBridgeSigner(signer_id=hotkey, secret=secret)


def make_hmac_verifier(secrets: Mapping[str, str]) -> HmacBridgeVerifier:
    """Convenience constructor for a HMAC bridge verifier."""
    return HmacBridgeVerifier(secrets=dict(secrets))


__all__ = [
    "FetchRolloutResult",
    "InvalidBundleReport",
    "PublishRolloutOutcome",
    "ROLLOUT_BUNDLE_METRIC_PUBLISHED",
    "ROLLOUT_BUNDLE_METRIC_REGRESSED",
    "ROLLOUT_BUNDLE_METRIC_SPOOLED",
    "RolloutBundleFetcher",
    "RolloutBundlePublisher",
    "make_hmac_signer",
    "make_hmac_verifier",
]
