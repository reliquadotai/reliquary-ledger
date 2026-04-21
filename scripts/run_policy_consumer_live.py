"""Ledger-side closed-loop verification.

Runs the PolicyConsumer against R2. Expected flow:

  1. Poll commitments/462/policy/ → find the PolicyCommitment we just
     published from staging2.
  2. Verify signature via HmacBridgeVerifier keyed by the shared secret.
  3. Fetch the referenced CheckpointAttestation blob, verify its
     envelope signature, recompute attestation_id and compare to the
     commitment's claim.
  4. Load the manifest + shard payload_sha256 digests (via a DeltaLoader
     that runs the R2 REST fetch).
  5. Run the smoke hash check — must match the attestation's
     smoke_hash_hex byte-for-byte.
  6. Call applier(delta) — in this demo, applier just records that
     it would have applied (no live model mutation; we're validating
     the bridge, not swapping the live miner out from under it).

Success means the full signature chain from commitment → attestation
→ delta → shard digests → smoke hash verifies end-to-end against real
R2 artifacts, with real Qwen2.5-3B weights underlying them.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

from reliquary_inference.shared.policy_consumer import (
    LoadedDelta,
    PolicyConsumer,
    default_smoke_runner,
)
from reliquary_inference.validator.rollout_bundle import make_hmac_verifier
from reliquary_protocol.storage import R2ObjectBackend


POLICY_AUTHORITY_HOTKEY = "reliquary-policy-authority-v1"
INFERENCE_NETUID = 462
TRAINING_NETUID = 462


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _r2_delta_loader(
    *,
    run_id: str,
    window_id: int,
    expected_merkle_root_hex: str,
    backend,
):
    """Fetch a delta bundle's manifest + shard digests from R2.

    PolicyConsumer calls this with backend as a kwarg; keep the signature
    keyword-only to match.
    """
    manifest_key = f"checkpoints/{run_id}/{window_id}/manifest.json"
    manifest_bytes = backend.get(manifest_key)
    if manifest_bytes is None:
        raise RuntimeError(f"missing manifest at {manifest_key}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    shard_digests = tuple(entry["payload_sha256"] for entry in manifest["shards"])
    log(f"  loaded manifest + {len(shard_digests)} shard digests")
    return LoadedDelta(
        run_id=run_id,
        window_id=window_id,
        merkle_root_hex=manifest["merkle_root_hex"],
        raw_manifest_bytes=manifest_bytes,
        shard_digests=shard_digests,
    )


class _R2StorageBackendShim:
    """Thin wrapper matching the StorageBackend Protocol that
    PolicyConsumer expects, backed by R2ObjectBackend.
    """

    def __init__(self, inner: R2ObjectBackend) -> None:
        self.inner = inner

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(key, data)

    def get(self, key: str) -> bytes | None:
        return self.inner.get(key)

    def list(self, prefix: str) -> list[str]:
        return self.inner.list(prefix)

    def delete(self, key: str) -> None:
        self.inner.delete(key)


class _RecordingApplier:
    def __init__(self) -> None:
        self.applied: list[LoadedDelta] = []

    def __call__(self, delta: LoadedDelta) -> None:
        # In this demo we DON'T mutate the live miner — just record the
        # event. Production policy_consumer calls a real model-swap
        # hook here.
        self.applied.append(delta)
        log(f"  applier called: would swap to run_id={delta.run_id} window={delta.window_id}")


def main() -> int:
    r2_token = open("/save/secrets/r2-cf-api-token").read().strip()
    signing_secret = open("/save/secrets/signing-secret").read().strip()

    r2 = R2ObjectBackend(
        account_id="d5332aea7e3780d0f2391a4e4f6ddfbc",
        bucket="reliquary",
        cf_api_token=r2_token,
        public_url="https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev",
    )
    backend = _R2StorageBackendShim(r2)

    verifier = make_hmac_verifier({POLICY_AUTHORITY_HOTKEY: signing_secret})

    applier = _RecordingApplier()

    def loader(**kwargs):
        return _r2_delta_loader(**kwargs)

    consumer = PolicyConsumer(
        backend=backend,
        verifier=verifier,
        inference_netuid=INFERENCE_NETUID,
        training_netuid=TRAINING_NETUID,
        delta_loader=loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )

    # The Forge cycle published a commitment with effective_at_ledger_window=6956700.
    # Poll with a ledger_window >= that to trigger the apply.
    log("polling policy_consumer (ledger_window=6956800 so effective window has already elapsed)")
    outcome = consumer.poll_once(ledger_window=6956800)

    log(f"outcome.state  = {outcome.state}")
    log(f"outcome.reason = {outcome.reason}")
    if outcome.commitment is not None:
        log(f"outcome.commitment.checkpoint_run_id = {outcome.commitment.checkpoint_run_id}")
        log(f"outcome.commitment.effective_at_window = {outcome.commitment.effective_at_window}")
    if outcome.attestation is not None:
        log(f"outcome.attestation.merkle_root = {outcome.attestation.merkle_root_hex}")
        log(f"outcome.attestation.smoke_hash  = {outcome.attestation.smoke_hash_hex}")
        log(f"outcome.attestation.consumed_windows = {outcome.attestation.consumed_rollout_windows}")
    if outcome.delta is not None:
        log(f"outcome.delta.run_id = {outcome.delta.run_id}")
        log(f"outcome.delta.merkle_root = {outcome.delta.merkle_root_hex}")
        log(f"outcome.delta.shard_digests = {len(outcome.delta.shard_digests)} shards")
    log(f"applier invocations: {len(applier.applied)}")
    log(f"current_policy_window: {consumer.current_policy_window}")

    return 0 if outcome.state == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
