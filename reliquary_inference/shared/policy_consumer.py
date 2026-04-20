"""Policy consumer — closes the Forge → Ledger bridge.

Polls the shared storage backend for fresh :class:`PolicyCommitment`
records published by the Forge ``policy_authority``, verifies the
signatures, resolves each commitment to its referenced
:class:`CheckpointAttestation` + delta bundle, runs a local smoke
check, and — on success — invokes an operator-supplied applier at
the next Ledger window boundary.

Design notes:

- Purely I/O + verification; *all* model code (delta application,
  vLLM reload, transformers ``from_pretrained``) is injected via the
  :class:`PolicyApplier` and :class:`DeltaLoader` Protocols. That
  keeps the consumer unit-testable with no torch dependency.
- Mid-window swap is forbidden by construction: the consumer returns
  a :class:`PolicySwapOutcome` marked ``ready`` but only invokes the
  applier when ``ledger_window >= effective_at_window``. The caller
  drives the window clock.
- Fail-before-mutate: any signature / merkle / smoke failure returns
  a rejected outcome before any state is mutated. The previously-
  applied policy stays live.
- Idempotent: polling with the same committed artifacts + the same
  ``current_policy_window`` is a no-op.

Spec: ``private/reliquary-plan/notes/spec-closed-loop-bridge.md``
§5 (Ledger policy consumer).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from reliquary_protocol import (
    BRIDGE_VERSION,
    BridgeVerifier,
    CHECKPOINT_ATTESTATION_TYPE,
    POLICY_COMMITMENT_TYPE,
    CheckpointAttestation,
    PolicyCommitment,
    envelope_from_dict,
    verify_envelope,
    verify_policy_commitment,
)

from ..validator.verdict_storage import StorageBackend


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counters (wired into Prometheus by the validator service)
# ---------------------------------------------------------------------------

POLICY_SWAP_METRIC_OK = "reliquary_bridge_policy_swap_total"
POLICY_SWAP_METRIC_REJECTED = "reliquary_bridge_policy_swap_rejected_total"
POLICY_SWAP_METRIC_SKIPPED = "reliquary_bridge_policy_swap_skipped_total"


# ---------------------------------------------------------------------------
# Injection surfaces
# ---------------------------------------------------------------------------


class DeltaLoader(Protocol):
    """Pulls a delta bundle's raw bytes + metadata for smoke + apply.

    The fetch logic for real delta bundles lives on the Forge side
    (``reliquary.training.checkpoint_storage.fetch_bundle``) which
    reads the manifest, fetches each shard, and re-verifies the
    per-shard sha256 + recomputed Merkle root. The consumer stays
    decoupled from that concrete pipeline so it can be swapped for
    a lighter pull in tests.
    """

    def __call__(
        self,
        *,
        run_id: str,
        window_id: int,
        expected_merkle_root_hex: str,
        backend: StorageBackend,
    ) -> "LoadedDelta":  # pragma: no cover - protocol
        ...


class SmokeRunner(Protocol):
    """Produces a deterministic hex digest over a loaded delta + base.

    Must be byte-identical on Forge (producer) and Ledger (consumer)
    for the same inputs — that's what the ``smoke_hash_hex`` check
    validates.
    """

    def __call__(self, delta: "LoadedDelta") -> str:  # pragma: no cover - protocol
        ...


class PolicyApplier(Protocol):
    """Applies a smoked-and-verified delta to the live policy.

    Implementation is caller-owned: prod wires a transformers / vLLM
    hot-swap here; tests wire a no-op that just records which deltas
    were applied in which order.
    """

    def __call__(self, delta: "LoadedDelta") -> None:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class LoadedDelta:
    """In-memory handle on a delta bundle ready to smoke + apply.

    ``raw_manifest_bytes`` + ``shard_digests`` let ``SmokeRunner``
    compute a deterministic hash without re-reading the backend.
    Production implementations may also carry tensor-level state
    (e.g. in-memory ``DeltaBundle``); tests just need the digests.
    """

    run_id: str
    window_id: int
    merkle_root_hex: str
    raw_manifest_bytes: bytes
    shard_digests: tuple[str, ...]
    extra: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicySwapOutcome:
    """Result of one :meth:`PolicyConsumer.poll_once` invocation.

    ``state`` is the terminal label — one of:

    - ``"applied"`` — delta was applied, ``current_policy_window`` advanced.
    - ``"ready"`` — commitment verified but ``effective_at_window`` is
      still in the future; caller should re-poll after the next window.
    - ``"idle"`` — no new commitment above ``current_policy_window``.
    - ``"rejected"`` — commitment found but failed verification; see
      ``reason``. Old policy stays live.
    """

    state: str
    reason: str | None = None
    commitment: PolicyCommitment | None = None
    attestation: CheckpointAttestation | None = None
    delta: LoadedDelta | None = None
    applied_at_ledger_window: int | None = None


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class PolicyConsumer:
    """Drain policy commitments + apply deltas at window boundary."""

    def __init__(
        self,
        *,
        backend: StorageBackend,
        verifier: BridgeVerifier,
        inference_netuid: int,
        training_netuid: int,
        delta_loader: DeltaLoader,
        smoke_runner: SmokeRunner,
        applier: PolicyApplier,
        current_policy_window: int = -1,
    ) -> None:
        self.backend = backend
        self.verifier = verifier
        self.inference_netuid = int(inference_netuid)
        self.training_netuid = int(training_netuid)
        self.delta_loader = delta_loader
        self.smoke_runner = smoke_runner
        self.applier = applier
        self.current_policy_window = int(current_policy_window)
        self._quarantine: list[str] = []
        self.metrics_counters: dict[str, int] = {
            POLICY_SWAP_METRIC_OK: 0,
            POLICY_SWAP_METRIC_REJECTED: 0,
            POLICY_SWAP_METRIC_SKIPPED: 0,
        }

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def poll_once(self, *, ledger_window: int) -> PolicySwapOutcome:
        """Poll the backend once; return an outcome.

        ``ledger_window`` is the current validator-facing window id;
        the consumer applies at most one delta whose
        ``effective_at_window <= ledger_window`` and which is strictly
        greater than any previously applied.
        """
        commitment = self._latest_applicable_commitment(ledger_window=ledger_window)
        if commitment is None:
            return PolicySwapOutcome(state="idle")
        if commitment.effective_at_window > ledger_window:
            return PolicySwapOutcome(
                state="ready",
                reason="effective_window_in_future",
                commitment=commitment,
            )

        # Fetch + verify attestation.
        attestation_or_err = self._load_and_verify_attestation(commitment)
        if isinstance(attestation_or_err, str):
            return self._rejected(
                reason=attestation_or_err, commitment=commitment, attestation=None
            )
        attestation = attestation_or_err

        # Fetch + smoke-check delta bundle.
        try:
            delta = self.delta_loader(
                run_id=attestation.checkpoint_run_id,
                window_id=attestation.checkpoint_window_id,
                expected_merkle_root_hex=attestation.merkle_root_hex,
                backend=self.backend,
            )
        except Exception as exc:
            return self._rejected(
                reason=f"delta_load_failed: {type(exc).__name__}: {exc}",
                commitment=commitment,
                attestation=attestation,
            )

        if delta.merkle_root_hex.lower() != attestation.merkle_root_hex.lower():
            return self._rejected(
                reason=(
                    f"merkle_mismatch: delta={delta.merkle_root_hex} "
                    f"attestation={attestation.merkle_root_hex}"
                ),
                commitment=commitment,
                attestation=attestation,
            )

        try:
            smoke_hex = self.smoke_runner(delta).lower()
        except Exception as exc:
            return self._rejected(
                reason=f"smoke_failed: {type(exc).__name__}: {exc}",
                commitment=commitment,
                attestation=attestation,
            )
        if smoke_hex != attestation.smoke_hash_hex.lower():
            return self._rejected(
                reason=(
                    f"smoke_hash_mismatch: computed={smoke_hex} "
                    f"claimed={attestation.smoke_hash_hex}"
                ),
                commitment=commitment,
                attestation=attestation,
            )

        # All checks passed — atomic apply.
        try:
            self.applier(delta)
        except Exception as exc:
            return self._rejected(
                reason=f"apply_failed: {type(exc).__name__}: {exc}",
                commitment=commitment,
                attestation=attestation,
            )

        self.current_policy_window = int(commitment.effective_at_window)
        self.metrics_counters[POLICY_SWAP_METRIC_OK] += 1
        logger.info(
            "policy_consumer.applied run_id=%s window=%d effective_at=%d",
            attestation.checkpoint_run_id,
            attestation.checkpoint_window_id,
            commitment.effective_at_window,
        )
        return PolicySwapOutcome(
            state="applied",
            commitment=commitment,
            attestation=attestation,
            delta=delta,
            applied_at_ledger_window=int(ledger_window),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _latest_applicable_commitment(
        self, *, ledger_window: int
    ) -> PolicyCommitment | None:
        prefix = f"commitments/{self.inference_netuid}/policy/"
        best: PolicyCommitment | None = None
        for key in self.backend.list(prefix):
            if key in self._quarantine:
                continue
            raw = self.backend.get(key)
            if raw is None:
                continue
            try:
                envelope_dict = json.loads(raw.decode("utf-8"))
            except Exception:
                self._quarantine.append(key)
                self.metrics_counters[POLICY_SWAP_METRIC_SKIPPED] += 1
                continue
            # The commitment is self-signed (not enveloped) — the
            # signature field is on the dataclass itself. We verify
            # via ``verify_policy_commitment`` below.
            try:
                commitment = PolicyCommitment.from_dict(envelope_dict)
            except Exception:
                self._quarantine.append(key)
                self.metrics_counters[POLICY_SWAP_METRIC_SKIPPED] += 1
                continue
            if commitment.version != BRIDGE_VERSION:
                self._quarantine.append(key)
                self.metrics_counters[POLICY_SWAP_METRIC_SKIPPED] += 1
                continue
            if commitment.inference_netuid != self.inference_netuid:
                continue
            if commitment.training_netuid != self.training_netuid:
                continue
            if commitment.effective_at_window <= self.current_policy_window:
                continue
            if not verify_policy_commitment(commitment, self.verifier):
                self._quarantine.append(key)
                self.metrics_counters[POLICY_SWAP_METRIC_SKIPPED] += 1
                continue
            if best is None or (
                commitment.effective_at_window > best.effective_at_window
                and commitment.effective_at_window <= ledger_window
            ):
                if commitment.effective_at_window <= ledger_window:
                    best = commitment
            elif (
                best is None
                and commitment.effective_at_window > self.current_policy_window
            ):
                # Future-dated commitment — surface as "ready" below.
                best = commitment
        # If nothing applicable but there's a future-dated commitment,
        # surface the earliest future one so the caller can schedule.
        if best is None:
            future_candidates: list[PolicyCommitment] = []
            for key in self.backend.list(prefix):
                if key in self._quarantine:
                    continue
                raw = self.backend.get(key)
                if raw is None:
                    continue
                try:
                    envelope_dict = json.loads(raw.decode("utf-8"))
                    commitment = PolicyCommitment.from_dict(envelope_dict)
                except Exception:
                    continue
                if commitment.effective_at_window <= self.current_policy_window:
                    continue
                if commitment.inference_netuid != self.inference_netuid:
                    continue
                if commitment.training_netuid != self.training_netuid:
                    continue
                if not verify_policy_commitment(commitment, self.verifier):
                    continue
                future_candidates.append(commitment)
            if future_candidates:
                future_candidates.sort(key=lambda c: c.effective_at_window)
                return future_candidates[0]
        return best

    def _load_and_verify_attestation(
        self, commitment: PolicyCommitment
    ) -> CheckpointAttestation | str:
        raw = self.backend.get(commitment.attestation_key)
        if raw is None:
            return f"attestation_missing: {commitment.attestation_key}"
        try:
            envelope_dict = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return f"attestation_envelope_malformed: {exc}"
        try:
            envelope = envelope_from_dict(envelope_dict)
        except Exception as exc:
            return f"attestation_envelope_schema: {exc}"
        if envelope.artifact_type != CHECKPOINT_ATTESTATION_TYPE:
            return f"attestation_wrong_type: {envelope.artifact_type}"
        if not verify_envelope(envelope, self.verifier):
            return "attestation_bad_signature"
        try:
            attestation = CheckpointAttestation.from_dict(
                json.loads(envelope.payload_json)
            )
        except Exception as exc:
            return f"attestation_schema: {exc}"
        if attestation.artifact_id() != commitment.attestation_id:
            return (
                f"attestation_id_mismatch: computed={attestation.artifact_id()} "
                f"claimed={commitment.attestation_id}"
            )
        if attestation.training_netuid != self.training_netuid:
            return f"attestation_netuid_mismatch: {attestation.training_netuid}"
        if attestation.inference_netuid != self.inference_netuid:
            return (
                f"attestation_inference_netuid_mismatch: "
                f"{attestation.inference_netuid}"
            )
        return attestation

    def _rejected(
        self,
        *,
        reason: str,
        commitment: PolicyCommitment | None,
        attestation: CheckpointAttestation | None,
    ) -> PolicySwapOutcome:
        self.metrics_counters[POLICY_SWAP_METRIC_REJECTED] += 1
        logger.warning("policy_consumer.rejected reason=%s", reason)
        if commitment is not None:
            self._quarantine.append(commitment.storage_key())
        return PolicySwapOutcome(
            state="rejected",
            reason=reason,
            commitment=commitment,
            attestation=attestation,
        )


# ---------------------------------------------------------------------------
# Default SmokeRunner (deterministic hash of shard digests)
# ---------------------------------------------------------------------------


def default_smoke_runner(delta: LoadedDelta) -> str:
    """Deterministic baseline ``SmokeRunner``.

    Produces ``sha256(manifest_bytes || concat(sorted shard digests))``.
    Matches :func:`reliquary.training.policy_attestation.compute_smoke_hash`
    byte-for-byte so a Forge-side precomputed smoke hash verifies
    against a Ledger-side re-computation without any chain-specific
    state.

    This is a corruption-in-transit signal, not an anti-forgery
    signal: a dishonest Forge that poisons both the delta AND the
    attestation's smoke hash would still match. Anti-forgery depends
    on the ``BridgeVerifier`` refusing commitments not signed by the
    policy_authority hotkey in the allowlist.
    """
    h = hashlib.sha256()
    h.update(delta.raw_manifest_bytes)
    for digest in sorted(delta.shard_digests):
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


__all__ = [
    "DeltaLoader",
    "LoadedDelta",
    "POLICY_SWAP_METRIC_OK",
    "POLICY_SWAP_METRIC_REJECTED",
    "POLICY_SWAP_METRIC_SKIPPED",
    "PolicyApplier",
    "PolicyConsumer",
    "PolicySwapOutcome",
    "SmokeRunner",
    "default_smoke_runner",
]
