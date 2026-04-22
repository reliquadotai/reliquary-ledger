"""Tests for :mod:`reliquary_inference.shared.policy_consumer`."""

from __future__ import annotations

import hashlib
import json

from reliquary_protocol import (
    CHECKPOINT_ATTESTATION_TYPE,
    CheckpointAttestation,
    PolicyCommitment,
    build_checkpoint_attestation,
    build_policy_commitment,
    sign_envelope,
)

from reliquary_inference.shared.policy_consumer import (
    POLICY_SWAP_METRIC_OK,
    POLICY_SWAP_METRIC_REJECTED,
    LoadedDelta,
    PolicyConsumer,
    default_smoke_runner,
)
from reliquary_inference.validator.rollout_bundle import (
    make_hmac_signer,
    make_hmac_verifier,
)
from reliquary_inference.validator.verdict_storage import LocalFilesystemBackend

AUTHORITY = "authority_fleet"
SECRET = "authority-fleet-secret"


def _signer():
    return make_hmac_signer(AUTHORITY, SECRET)


def _verifier():
    return make_hmac_verifier({AUTHORITY: SECRET})


def _publish_attestation_and_commitment(
    backend,
    *,
    run_id: str = "run-0001",
    checkpoint_window_id: int = 5,
    effective_at_window: int = 100,
    consumed_rollout_windows=(10, 11),
    smoke_hash_hex: str | None = None,
    merkle_root_hex: str = "a" * 64,
) -> tuple[CheckpointAttestation, PolicyCommitment, bytes]:
    """Publish an attestation + commitment + fake manifest blob.

    Returns (attestation, commitment, fake_manifest_bytes). The fake
    manifest bytes go at a deterministic path so a test-side DeltaLoader
    can find them.
    """
    manifest_bytes = json.dumps(
        {"run_id": run_id, "window_id": checkpoint_window_id}, sort_keys=True
    ).encode("utf-8")
    backend.put(f"checkpoints/{run_id}/{checkpoint_window_id}/manifest.json", manifest_bytes)
    shard_digests = (
        hashlib.sha256(b"shard_0").hexdigest(),
        hashlib.sha256(b"shard_1").hexdigest(),
    )
    # Stash fake shard payloads so a real loader would find them.
    backend.put(f"checkpoints/{run_id}/{checkpoint_window_id}/shards/a.bin", b"shard_0")
    backend.put(f"checkpoints/{run_id}/{checkpoint_window_id}/shards/b.bin", b"shard_1")

    if smoke_hash_hex is None:
        h = hashlib.sha256()
        h.update(manifest_bytes)
        for d in sorted(shard_digests):
            h.update(d.encode("utf-8"))
        smoke_hash_hex = h.hexdigest()

    attestation = build_checkpoint_attestation(
        training_netuid=3,
        inference_netuid=81,
        checkpoint_run_id=run_id,
        checkpoint_window_id=checkpoint_window_id,
        merkle_root_hex=merkle_root_hex,
        base_snapshot_window_id=0,
        consumed_rollout_windows=list(consumed_rollout_windows),
        consumed_rollout_keys=[
            f"rollouts/81/{w}-{w}/val_x.json" for w in consumed_rollout_windows
        ],
        policy_authority_hotkey=AUTHORITY,
        smoke_hash_hex=smoke_hash_hex,
    )
    attestation_env = sign_envelope(CHECKPOINT_ATTESTATION_TYPE, attestation, _signer())
    backend.put(
        attestation.storage_key(),
        attestation_env.canonical_bytes(),
    )
    commitment = build_policy_commitment(
        attestation=attestation,
        effective_at_window=effective_at_window,
        signer=_signer(),
    )
    backend.put(
        commitment.storage_key(),
        json.dumps(commitment.to_dict(), sort_keys=True).encode("utf-8"),
    )
    return attestation, commitment, manifest_bytes


def _ok_delta_loader(*, run_id, window_id, expected_merkle_root_hex, backend):
    """Test DeltaLoader: reads the manifest + canonical shard digests."""
    mf_bytes = backend.get(f"checkpoints/{run_id}/{window_id}/manifest.json")
    assert mf_bytes is not None, "fake manifest should exist in tests"
    shards = (
        hashlib.sha256(b"shard_0").hexdigest(),
        hashlib.sha256(b"shard_1").hexdigest(),
    )
    return LoadedDelta(
        run_id=run_id,
        window_id=window_id,
        merkle_root_hex=expected_merkle_root_hex,
        raw_manifest_bytes=mf_bytes,
        shard_digests=shards,
    )


class _Applier:
    def __init__(self):
        self.applied: list[LoadedDelta] = []

    def __call__(self, delta):
        self.applied.append(delta)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_poll_once_applies_ready_commitment(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(backend, effective_at_window=100)

    applier = _Applier()
    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "applied"
    assert outcome.applied_at_ledger_window == 101
    assert len(applier.applied) == 1
    assert consumer.current_policy_window == 100
    assert consumer.metrics_counters[POLICY_SWAP_METRIC_OK] == 1


def test_poll_once_returns_ready_when_future_window(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(backend, effective_at_window=200)

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=100)  # < effective
    assert outcome.state == "ready"
    assert outcome.reason == "effective_window_in_future"
    assert consumer.current_policy_window == -1  # unchanged


def test_poll_once_idempotent_after_apply(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(backend, effective_at_window=100)

    applier = _Applier()
    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )
    r1 = consumer.poll_once(ledger_window=101)
    r2 = consumer.poll_once(ledger_window=102)
    assert r1.state == "applied"
    assert r2.state == "idle"  # no newer commitment
    assert len(applier.applied) == 1


def test_poll_once_idle_when_backend_empty(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    assert consumer.poll_once(ledger_window=100).state == "idle"


def test_multiple_commitments_pick_latest(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(
        backend,
        run_id="run-a",
        checkpoint_window_id=5,
        effective_at_window=100,
    )
    _publish_attestation_and_commitment(
        backend,
        run_id="run-b",
        checkpoint_window_id=6,
        effective_at_window=150,
    )

    applier = _Applier()
    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )
    # Ledger window past both effective windows → pick the later one.
    outcome = consumer.poll_once(ledger_window=200)
    assert outcome.state == "applied"
    assert outcome.commitment.effective_at_window == 150
    assert outcome.attestation.checkpoint_run_id == "run-b"
    assert consumer.current_policy_window == 150


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_reject_on_bad_commitment_signature(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    attestation, commitment, _ = _publish_attestation_and_commitment(
        backend, effective_at_window=100
    )
    # Tamper commitment on disk.
    tampered = dict(commitment.to_dict())
    tampered["signature"] = "0" * 64
    backend.put(
        commitment.storage_key(),
        json.dumps(tampered, sort_keys=True).encode("utf-8"),
    )

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    # Bad-signature commitments are silently skipped (→ idle).
    assert outcome.state == "idle"


def test_reject_on_attestation_id_mismatch(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    attestation, commitment, _ = _publish_attestation_and_commitment(
        backend, effective_at_window=100
    )
    # Rewrite the attestation blob with a DIFFERENT attestation (same
    # signature keypair, but a different payload); commitment still
    # references the OLD attestation_id → mismatch.
    different = build_checkpoint_attestation(
        training_netuid=3,
        inference_netuid=81,
        checkpoint_run_id=attestation.checkpoint_run_id,
        checkpoint_window_id=attestation.checkpoint_window_id,
        merkle_root_hex="b" * 64,  # different
        base_snapshot_window_id=0,
        consumed_rollout_windows=[99],
        consumed_rollout_keys=["rollouts/81/99-99/val_x.json"],
        policy_authority_hotkey=AUTHORITY,
        smoke_hash_hex="c" * 64,
    )
    env = sign_envelope(CHECKPOINT_ATTESTATION_TYPE, different, _signer())
    backend.put(attestation.storage_key(), env.canonical_bytes())

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "rejected"
    assert "attestation_id_mismatch" in (outcome.reason or "")
    assert consumer.metrics_counters[POLICY_SWAP_METRIC_REJECTED] == 1


def test_reject_on_merkle_mismatch(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    attestation, _, _ = _publish_attestation_and_commitment(
        backend, effective_at_window=100
    )

    def _evil_loader(**kwargs):
        return LoadedDelta(
            run_id=kwargs["run_id"],
            window_id=kwargs["window_id"],
            merkle_root_hex="0" * 64,  # different from attestation
            raw_manifest_bytes=b"whatever",
            shard_digests=(),
        )

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_evil_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "rejected"
    assert "merkle_mismatch" in (outcome.reason or "")


def test_reject_on_smoke_mismatch(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(backend, effective_at_window=100)

    def _evil_smoke(delta):
        return "deadbeef" * 8  # wrong hash

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=_evil_smoke,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "rejected"
    assert "smoke_hash_mismatch" in (outcome.reason or "")


def test_reject_on_applier_raise(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _publish_attestation_and_commitment(backend, effective_at_window=100)

    def _angry_applier(delta):
        raise RuntimeError("model-load failure")

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_angry_applier,
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "rejected"
    assert "apply_failed" in (outcome.reason or "")
    # current_policy_window NOT advanced.
    assert consumer.current_policy_window == -1


def test_reject_on_attestation_missing(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    _, commitment, _ = _publish_attestation_and_commitment(
        backend, effective_at_window=100
    )
    # Delete the attestation blob.
    backend.delete(commitment.attestation_key)

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    assert outcome.state == "rejected"
    assert "attestation_missing" in (outcome.reason or "")


def test_reject_on_netuid_mismatch(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    # Publish an attestation for a different training netuid.
    bad_attestation = build_checkpoint_attestation(
        training_netuid=999,  # wrong
        inference_netuid=81,
        checkpoint_run_id="run-bad",
        checkpoint_window_id=5,
        merkle_root_hex="a" * 64,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val_x.json"],
        policy_authority_hotkey=AUTHORITY,
        smoke_hash_hex="c" * 64,
    )
    env = sign_envelope(CHECKPOINT_ATTESTATION_TYPE, bad_attestation, _signer())
    backend.put(bad_attestation.storage_key(), env.canonical_bytes())
    bad_commitment = build_policy_commitment(
        attestation=bad_attestation,
        effective_at_window=100,
        signer=_signer(),
    )
    # Fake the commitment storage key to the proper inference netuid so
    # the consumer sees it. (In practice the signer would set this key,
    # but we bypass to exercise the mismatch path.)
    fake_key = "commitments/81/policy/100.json"
    backend.put(
        fake_key,
        json.dumps(bad_commitment.to_dict(), sort_keys=True).encode("utf-8"),
    )

    consumer = PolicyConsumer(
        backend=backend,
        verifier=_verifier(),
        inference_netuid=81,
        training_netuid=3,
        delta_loader=_ok_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_Applier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    # training_netuid mismatch is filtered at commitment level → idle.
    assert outcome.state == "idle"


# ---------------------------------------------------------------------------
# Smoke runner determinism
# ---------------------------------------------------------------------------


def test_default_smoke_runner_deterministic():
    delta = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"manifest",
        shard_digests=("d1", "d2", "d0"),
    )
    h1 = default_smoke_runner(delta)
    h2 = default_smoke_runner(delta)
    assert h1 == h2
    assert len(h1) == 64


def test_default_smoke_runner_shard_order_invariant():
    """Same digests in different order → same smoke hash."""
    d1 = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"manifest",
        shard_digests=("d0", "d1", "d2"),
    )
    d2 = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"manifest",
        shard_digests=("d2", "d0", "d1"),
    )
    assert default_smoke_runner(d1) == default_smoke_runner(d2)


def test_default_smoke_runner_manifest_sensitive():
    d1 = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"manifest-a",
        shard_digests=("d0",),
    )
    d2 = LoadedDelta(
        run_id="r",
        window_id=1,
        merkle_root_hex="a" * 64,
        raw_manifest_bytes=b"manifest-b",
        shard_digests=("d0",),
    )
    assert default_smoke_runner(d1) != default_smoke_runner(d2)
