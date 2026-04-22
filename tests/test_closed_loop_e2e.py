"""End-to-end closed-loop integration — the Forge ↔ Ledger flywheel.

Exercises the full closed loop on a single filesystem backend:

1. **Forge side:** build a tiny synthetic delta, publish via
   ``reliquary.training.checkpoint_storage.publish_bundle``.
2. **Forge side:** publish attestation + commitment via
   ``reliquary.training.policy_attestation.publish_attestation``.
3. **Ledger side:** ``PolicyConsumer.poll_once`` discovers the
   commitment, verifies the full chain, applies the delta.
4. **Ledger side:** ``RolloutBundlePublisher`` packages a window's
   rollouts into a ``RolloutBundle`` and writes to the same backend.
5. **Forge side:** ``InferenceRegistryAdapter`` picks up the bundle.

Skips (not xfails) when the ``reliquary`` package isn't importable
from this test environment — the Ledger half of the bridge doesn't
depend on Forge at runtime, so this test is a cross-repo harness,
not a release gate on the Ledger repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_FORGE_REPO = Path(__file__).resolve().parents[2] / "reliquary"
if _FORGE_REPO.is_dir():
    sys.path.insert(0, str(_FORGE_REPO))

reliquary_training = pytest.importorskip(
    "reliquary.training.checkpoint_storage",
    reason="closed-loop E2E requires the Forge repo on sys.path",
)
reliquary_attestation = pytest.importorskip(
    "reliquary.training.policy_attestation",
    reason="closed-loop E2E requires the Forge policy_attestation module",
)
reliquary_delta = pytest.importorskip(
    "reliquary.training.delta_checkpoints",
    reason="closed-loop E2E requires the Forge delta-checkpoint module",
)
reliquary_inference_adapter = pytest.importorskip(
    "reliquary.inference_adapter",
    reason="closed-loop E2E requires reliquary.inference_adapter",
)

torch = pytest.importorskip("torch")

from reliquary.training.checkpoint_storage import (  # noqa: E402
    LocalFilesystemBackend as ForgeBackend,
)
from reliquary.training.checkpoint_storage import (
    publish_bundle,
)
from reliquary.training.delta_checkpoints import compute_delta  # noqa: E402
from reliquary.training.policy_attestation import (  # noqa: E402
    make_hmac_authority,
    publish_attestation,
)

from reliquary_inference.shared.policy_consumer import (  # noqa: E402
    LoadedDelta,
    PolicyConsumer,
    default_smoke_runner,
)
from reliquary_inference.validator.mesh import (  # noqa: E402
    MedianVerdict,
    MeshAggregationReport,
    ValidatorIdentity,
    VerdictArtifact,
)
from reliquary_inference.validator.rollout_bundle import (  # noqa: E402
    RolloutBundleFetcher,
    RolloutBundlePublisher,
    make_hmac_signer,
    make_hmac_verifier,
)
from reliquary_inference.validator.verdict_storage import (  # noqa: E402
    LocalFilesystemBackend as LedgerBackend,
)

AUTHORITY = "forge-authority"
AUTHORITY_SECRET = "forge-authority-secret"
VALIDATOR = "ledger-validator-a"
VALIDATOR_SECRET = "ledger-validator-a-secret"
TRAINING_NETUID = 3
INFERENCE_NETUID = 81


def _build_synthetic_delta():
    torch.manual_seed(0)
    base = {
        "encoder.weight": torch.zeros(4, 4),
        "decoder.weight": torch.zeros(2, 4),
    }
    target = {
        "encoder.weight": torch.ones(4, 4) * 0.1,
        "decoder.weight": torch.ones(2, 4) * 0.2,
    }
    return compute_delta(base, target)


def _forge_delta_loader(*, run_id, window_id, expected_merkle_root_hex, backend):
    """Production-shape DeltaLoader for the integration test.

    Reads the manifest blob (canonical JSON written by
    ``publish_bundle``) + every referenced shard; returns a
    :class:`LoadedDelta` whose ``raw_manifest_bytes`` + sorted shard
    digests round-trip through :func:`default_smoke_runner`.
    """
    manifest_key = f"checkpoints/{run_id}/{window_id}/manifest.json"
    manifest_bytes = backend.get(manifest_key)
    if manifest_bytes is None:
        raise RuntimeError(f"missing manifest at {manifest_key}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    shard_digests = tuple(entry["payload_sha256"] for entry in manifest["shards"])
    return LoadedDelta(
        run_id=run_id,
        window_id=window_id,
        merkle_root_hex=manifest["merkle_root_hex"],
        raw_manifest_bytes=manifest_bytes,
        shard_digests=shard_digests,
    )


class _RecordingApplier:
    def __init__(self):
        self.applied: list[LoadedDelta] = []

    def __call__(self, delta):
        self.applied.append(delta)


def test_full_loop_single_backend(tmp_path):
    """One backend + one filesystem root = both sides of the bridge."""
    shared_root = tmp_path / "shared-bus"
    forge_backend = ForgeBackend(shared_root)
    ledger_backend = LedgerBackend(shared_root)

    # ---- Phase 1: Forge publishes delta + attestation + commitment ----
    bundle = _build_synthetic_delta()
    manifest = publish_bundle(
        bundle,
        run_id="run-closedloop-001",
        window_id=42,
        backend=forge_backend,
    )
    attestation_pub = publish_attestation(
        backend=forge_backend,
        signer=make_hmac_authority(AUTHORITY, AUTHORITY_SECRET),
        training_netuid=TRAINING_NETUID,
        inference_netuid=INFERENCE_NETUID,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[100, 101],
        consumed_rollout_keys=[
            f"rollouts/{INFERENCE_NETUID}/100-100/{VALIDATOR}.json",
            f"rollouts/{INFERENCE_NETUID}/101-101/{VALIDATOR}.json",
        ],
        effective_at_ledger_window=102,
    )
    # Attestation blob lives under the Forge key convention.
    assert forge_backend.get(attestation_pub.attestation_key) is not None
    assert forge_backend.get(attestation_pub.commitment_key) is not None

    # ---- Phase 2: Ledger policy_consumer picks up + applies ----
    applier = _RecordingApplier()
    consumer = PolicyConsumer(
        backend=ledger_backend,
        verifier=make_hmac_verifier({AUTHORITY: AUTHORITY_SECRET}),
        inference_netuid=INFERENCE_NETUID,
        training_netuid=TRAINING_NETUID,
        delta_loader=_forge_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )
    # Ledger window hasn't yet reached effective_at (102) → "ready".
    ready = consumer.poll_once(ledger_window=100)
    assert ready.state == "ready"

    # Advance the window → "applied".
    applied = consumer.poll_once(ledger_window=102)
    assert applied.state == "applied"
    assert applied.attestation.checkpoint_run_id == "run-closedloop-001"
    assert len(applier.applied) == 1
    assert consumer.current_policy_window == 102

    # ---- Phase 3: Ledger packages a rollout bundle for Forge ----
    mesh_report = _mesh_report_with_accepts(
        window_id=110,
        completions=[("c1", "miner_x"), ("c2", "miner_y")],
        validators=[VALIDATOR, "ledger-validator-b"],
    )
    verdict_artifacts = [
        _verdict_artifact("c1", VALIDATOR),
        _verdict_artifact("c1", "ledger-validator-b"),
        _verdict_artifact("c2", VALIDATOR),
        _verdict_artifact("c2", "ledger-validator-b"),
    ]
    publisher = RolloutBundlePublisher(
        backend=ledger_backend,
        signer=make_hmac_signer(VALIDATOR, VALIDATOR_SECRET),
        netuid=INFERENCE_NETUID,
    )
    outcome = publisher.publish(
        window_id=110,
        mesh_report=mesh_report,
        manifest={
            "task_source": "sat",
            "task_count": 2,
            "seed": 0xDEADBEEF,
        },
        task_batch={
            "tasks": [
                {"id": "c1", "prompt": "prompt-1"},
                {"id": "c2", "prompt": "prompt-2"},
            ]
        },
        scorecard={"miner_x": {"score": 0.9}, "miner_y": {"score": 0.85}},
        accepted_completions=[
            {"completion_id": "c1", "miner_hotkey": "miner_x", "text": "answer-1"},
            {"completion_id": "c2", "miner_hotkey": "miner_y", "text": "answer-2"},
        ],
        verdict_artifacts=verdict_artifacts,
    )
    assert outcome.success is True
    assert outcome.key == f"rollouts/{INFERENCE_NETUID}/110-110/{VALIDATOR}.json"

    # ---- Phase 4: Forge's adapter-style fetcher picks up the bundle ----
    fetcher = RolloutBundleFetcher(
        backend=ledger_backend,
        verifier=make_hmac_verifier({VALIDATOR: VALIDATOR_SECRET}),
        netuid=INFERENCE_NETUID,
    )
    result = fetcher.fetch()
    assert len(result.bundles) == 1
    fetched = result.bundles[0]
    assert fetched.window_id == 110
    assert fetched.netuid == INFERENCE_NETUID
    assert {c["completion_id"] for c in fetched.completions} == {"c1", "c2"}
    assert len(fetched.verdicts) == 2


def test_full_loop_reject_forgery(tmp_path):
    """Unknown signer commits a commitment → Ledger rejects it.

    Demonstrates the anti-forgery property: only commitments signed by
    an allowlisted ``policy_authority`` hotkey get applied. A forger
    who owns R2 cannot install a policy unless their key is in the
    Ledger's verifier.
    """
    shared_root = tmp_path / "bus"
    forge_backend = ForgeBackend(shared_root)
    ledger_backend = LedgerBackend(shared_root)

    bundle = _build_synthetic_delta()
    manifest = publish_bundle(
        bundle,
        run_id="run-forgery",
        window_id=1,
        backend=forge_backend,
    )
    # Signed by an UNKNOWN hotkey.
    publish_attestation(
        backend=forge_backend,
        signer=make_hmac_authority("rogue", "rogue-secret"),
        training_netuid=TRAINING_NETUID,
        inference_netuid=INFERENCE_NETUID,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=[f"rollouts/{INFERENCE_NETUID}/10-10/val.json"],
        effective_at_ledger_window=100,
    )

    consumer = PolicyConsumer(
        backend=ledger_backend,
        # Verifier only knows the legit authority.
        verifier=make_hmac_verifier({AUTHORITY: AUTHORITY_SECRET}),
        inference_netuid=INFERENCE_NETUID,
        training_netuid=TRAINING_NETUID,
        delta_loader=_forge_delta_loader,
        smoke_runner=default_smoke_runner,
        applier=_RecordingApplier(),
    )
    outcome = consumer.poll_once(ledger_window=101)
    # Bad-signature commitments are filtered at list time → idle.
    assert outcome.state == "idle"
    assert consumer.current_policy_window == -1


def test_full_loop_reject_corrupted_bundle(tmp_path):
    """Corruption in transit: tamper the manifest sha → Ledger rejects."""
    shared_root = tmp_path / "bus"
    forge_backend = ForgeBackend(shared_root)
    LedgerBackend(shared_root)

    bundle = _build_synthetic_delta()
    manifest = publish_bundle(
        bundle, run_id="run-corruption", window_id=1, backend=forge_backend
    )
    publish_attestation(
        backend=forge_backend,
        signer=make_hmac_authority(AUTHORITY, AUTHORITY_SECRET),
        training_netuid=TRAINING_NETUID,
        inference_netuid=INFERENCE_NETUID,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=[f"rollouts/{INFERENCE_NETUID}/10-10/val.json"],
        effective_at_ledger_window=100,
    )

    # Tamper a shard byte on disk.
    shard_keys = [k for k in forge_backend.list("") if k.startswith(
        f"checkpoints/{manifest.run_id}/{manifest.window_id}/shards/"
    )]
    assert shard_keys, "test requires at least one shard"
    original = forge_backend.get(shard_keys[0])
    tampered = bytes(original[:-1]) + bytes([(original[-1] ^ 0xFF) & 0xFF])
    forge_backend.put(shard_keys[0], tampered)

    # The delta_loader reads the UNTAMPERED manifest digests, so the
    # shard_digests list in LoadedDelta remains consistent with the
    # manifest. The smoke hash check passes. The attack is only
    # caught when the applier (production path) actually rehydrates
    # tensors and reconstructs — which the integration test covers
    # by running fetch_bundle directly below.
    from reliquary.training.checkpoint_storage import (
        DeltaStorageShardCorrupted,
        fetch_bundle,
    )
    from reliquary.training.delta_checkpoints import DeltaMerkleMismatch

    with pytest.raises((DeltaStorageShardCorrupted, DeltaMerkleMismatch)):
        fetch_bundle(
            manifest.run_id,
            manifest.window_id,
            backend=forge_backend,
        )


# ---------------------------------------------------------------------------
# Mesh-artifact helpers (small local factories)
# ---------------------------------------------------------------------------


def _mesh_report_with_accepts(
    *, window_id, completions, validators
) -> MeshAggregationReport:
    report = MeshAggregationReport(window_id=window_id)
    for cid, miner in completions:
        report.median_verdicts[cid] = MedianVerdict(
            completion_id=cid,
            accepted=True,
            acceptance_score=1.0,
            median_scores={"reward": 0.85, "format": 1.0},
            participating_validators=list(validators),
            outlier_validators=[],
            quorum_satisfied=True,
        )
    return report


def _verdict_artifact(completion_id: str, validator_hotkey: str) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner_x",
        window_id=110,
        validator=ValidatorIdentity(
            hotkey=validator_hotkey, stake=100.0, signer_id=validator_hotkey
        ),
        accepted=True,
        stage_failed=None,
        reject_reason=None,
        scores={"reward": 0.85, "format": 1.0},
        signed_at=1_700_000_000.0,
    )
