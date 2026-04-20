"""Tests for :mod:`reliquary_inference.validator.rollout_bundle`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from reliquary_inference.validator.mesh import (
    MedianVerdict,
    MeshAggregationReport,
    ValidatorIdentity,
    VerdictArtifact,
)
from reliquary_inference.validator.rollout_bundle import (
    ROLLOUT_BUNDLE_METRIC_PUBLISHED,
    ROLLOUT_BUNDLE_METRIC_REGRESSED,
    ROLLOUT_BUNDLE_METRIC_SPOOLED,
    RolloutBundleFetcher,
    RolloutBundlePublisher,
    make_hmac_signer,
    make_hmac_verifier,
)
from reliquary_inference.validator.verdict_storage import (
    LocalFilesystemBackend,
    StorageBackend,
)
from reliquary_protocol import (
    BridgeEnvelope,
    RolloutBundle,
    envelope_from_dict,
    verify_envelope,
)


PRODUCER_HOTKEY = "val_a"
PRODUCER_SECRET = "hotkey-a-secret"


def _signer():
    return make_hmac_signer(PRODUCER_HOTKEY, PRODUCER_SECRET)


def _verifier():
    return make_hmac_verifier({PRODUCER_HOTKEY: PRODUCER_SECRET})


def _make_report(
    *,
    window_id: int,
    completion_ids: list[str],
    validators: list[str],
) -> MeshAggregationReport:
    report = MeshAggregationReport(window_id=window_id)
    for cid in completion_ids:
        report.median_verdicts[cid] = MedianVerdict(
            completion_id=cid,
            accepted=True,
            acceptance_score=1.0,
            median_scores={"reward": 0.75, "format": 1.0},
            participating_validators=list(validators),
            outlier_validators=[],
            quorum_satisfied=True,
        )
    return report


def _make_artifact(
    *, completion_id: str, validator_hotkey: str, accepted: bool = True
) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner_1",
        window_id=100,
        validator=ValidatorIdentity(
            hotkey=validator_hotkey, stake=100.0, signer_id=validator_hotkey
        ),
        accepted=accepted,
        stage_failed=None,
        reject_reason=None,
        scores={"reward": 0.75, "format": 1.0},
        signed_at=1_700_000_000.0,
    )


def _minimal_inputs(window_id: int = 100):
    accepted = [{"completion_id": "c1", "miner_hotkey": "m1", "text": "hi"}]
    rejected = [{"completion_id": "cRej", "miner_hotkey": "m2", "text": "bad"}]
    report = _make_report(
        window_id=window_id, completion_ids=["c1"], validators=["val_a", "val_b"]
    )
    artifacts = [
        _make_artifact(completion_id="c1", validator_hotkey="val_a"),
        _make_artifact(completion_id="c1", validator_hotkey="val_b"),
    ]
    manifest = {"task_source": "sat", "task_count": 1, "seed": 0xDEADBEEF}
    task_batch = {"tasks": [{"id": "c1", "prompt": "Solve: x+1=2"}]}
    scorecard = {"miner_1": {"score": 0.75}}
    return dict(
        window_id=window_id,
        mesh_report=report,
        manifest=manifest,
        task_batch=task_batch,
        scorecard=scorecard,
        accepted_completions=accepted + rejected,  # publisher filters
        verdict_artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Publish happy path
# ---------------------------------------------------------------------------


def test_publish_writes_bundle_at_canonical_key(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)

    outcome = pub.publish(**_minimal_inputs(window_id=100))

    assert outcome.success is True
    assert outcome.spooled_path is None
    assert outcome.key == "rollouts/81/100-100/val_a.json"
    # The blob is there and it parses as an envelope.
    raw = backend.get(outcome.key)
    assert raw is not None
    envelope = envelope_from_dict(json.loads(raw.decode("utf-8")))
    assert envelope.artifact_type == "rollout_bundle"
    assert envelope.signer_id == PRODUCER_HOTKEY
    assert verify_envelope(envelope, _verifier()) is True


def test_publish_filters_out_completions_not_in_median_verdicts(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)

    outcome = pub.publish(**_minimal_inputs(window_id=100))
    raw = backend.get(outcome.key)
    envelope = envelope_from_dict(json.loads(raw.decode("utf-8")))
    bundle = RolloutBundle.from_dict(json.loads(envelope.payload_json))

    completion_ids = {c["completion_id"] for c in bundle.completions}
    # cRej was not in median_verdicts → filtered.
    assert completion_ids == {"c1"}


def test_publish_includes_per_validator_provenance(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)

    outcome = pub.publish(**_minimal_inputs(window_id=100))
    raw = backend.get(outcome.key)
    envelope = envelope_from_dict(json.loads(raw.decode("utf-8")))
    bundle = RolloutBundle.from_dict(json.loads(envelope.payload_json))
    assert len(bundle.verdicts) == 1
    verdict = bundle.verdicts[0]
    assert verdict["completion_id"] == "c1"
    assert verdict["accepted"] is True
    assert len(verdict["per_validator"]) == 2
    assert sorted(v["validator_hotkey"] for v in verdict["per_validator"]) == [
        "val_a",
        "val_b",
    ]


def test_publish_counters_incremented(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)

    pub.publish(**_minimal_inputs(window_id=100))
    assert pub.metrics_counters[ROLLOUT_BUNDLE_METRIC_PUBLISHED] == 1
    assert pub.metrics_counters[ROLLOUT_BUNDLE_METRIC_SPOOLED] == 0


# ---------------------------------------------------------------------------
# Canonical bytes / determinism
# ---------------------------------------------------------------------------


def test_publish_canonical_bytes_deterministic_across_two_publishers(tmp_path):
    """Two publishers with identical inputs produce identical bundles.

    Pins 1) shuffled completion order, 2) shuffled artifact order —
    both must canonicalize to the same bytes.
    """
    b1 = LocalFilesystemBackend(tmp_path / "r1")
    b2 = LocalFilesystemBackend(tmp_path / "r2")
    pub1 = RolloutBundlePublisher(
        backend=b1, signer=_signer(), netuid=81
    )
    pub2 = RolloutBundlePublisher(
        backend=b2, signer=_signer(), netuid=81
    )
    published_at = "2026-04-20T00:00:00+00:00"
    shared = dict(
        window_id=100,
        manifest={"task_source": "sat", "task_count": 2, "seed": 0xDEADBEEF},
        task_batch={
            "tasks": [
                {"id": "c1", "prompt": "p1"},
                {"id": "c2", "prompt": "p2"},
            ]
        },
        scorecard=None,
        published_at=published_at,
    )
    completions_a = [
        {"completion_id": "c1", "miner_hotkey": "m1"},
        {"completion_id": "c2", "miner_hotkey": "m2"},
    ]
    completions_b = list(reversed(completions_a))
    report = _make_report(window_id=100, completion_ids=["c1", "c2"], validators=["val_a"])
    artifacts_a = [
        _make_artifact(completion_id="c1", validator_hotkey="val_a"),
        _make_artifact(completion_id="c2", validator_hotkey="val_a"),
    ]
    artifacts_b = list(reversed(artifacts_a))

    pub1.publish(
        **shared,
        mesh_report=report,
        accepted_completions=completions_a,
        verdict_artifacts=artifacts_a,
    )
    pub2.publish(
        **shared,
        mesh_report=report,
        accepted_completions=completions_b,
        verdict_artifacts=artifacts_b,
    )

    key = "rollouts/81/100-100/val_a.json"
    assert b1.get(key) == b2.get(key)


# ---------------------------------------------------------------------------
# Window monotonicity
# ---------------------------------------------------------------------------


def test_publish_rejects_window_regression(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    pub.publish(**_minimal_inputs(window_id=100))

    # Same window again → regression.
    outcome_same = pub.publish(**_minimal_inputs(window_id=100))
    assert outcome_same.success is False
    assert "last_published" in (outcome_same.last_error or "")

    # Older window → regression.
    outcome_old = pub.publish(**_minimal_inputs(window_id=50))
    assert outcome_old.success is False
    assert pub.metrics_counters[ROLLOUT_BUNDLE_METRIC_REGRESSED] == 2


def test_publish_accepts_monotonic_windows(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    r1 = pub.publish(**_minimal_inputs(window_id=100))
    r2 = pub.publish(**_minimal_inputs(window_id=101))
    r3 = pub.publish(**_minimal_inputs(window_id=102))
    assert all(r.success for r in (r1, r2, r3))


# ---------------------------------------------------------------------------
# Spool + flush
# ---------------------------------------------------------------------------


class _FlakyBackend:
    """Backend whose ``put`` fails until :meth:`unbreak` is called."""

    def __init__(self, wrapped: StorageBackend, fail_count: int) -> None:
        self._wrapped = wrapped
        self._fails_remaining = fail_count

    def put(self, key, data):
        if self._fails_remaining > 0:
            self._fails_remaining -= 1
            raise IOError(f"synthetic failure ({self._fails_remaining} left)")
        self._wrapped.put(key, data)

    def get(self, key):
        return self._wrapped.get(key)

    def list(self, prefix):
        return self._wrapped.list(prefix)

    def delete(self, key):
        self._wrapped.delete(key)

    def unbreak(self):
        self._fails_remaining = 0


def test_publish_spools_after_max_attempts(tmp_path):
    fs = LocalFilesystemBackend(tmp_path / "root")
    flaky = _FlakyBackend(fs, fail_count=99)
    spool = tmp_path / "spool"
    pub = RolloutBundlePublisher(
        backend=flaky,
        signer=_signer(),
        netuid=81,
        spool_dir=spool,
        max_attempts=3,
    )
    outcome = pub.publish(**_minimal_inputs(window_id=100))
    assert outcome.success is False
    assert outcome.spooled_path is not None
    assert Path(outcome.spooled_path).exists()
    assert pub.metrics_counters[ROLLOUT_BUNDLE_METRIC_SPOOLED] == 1


def test_flush_spool_drains_on_reconnect(tmp_path):
    fs = LocalFilesystemBackend(tmp_path / "root")
    flaky = _FlakyBackend(fs, fail_count=99)
    spool = tmp_path / "spool"
    pub = RolloutBundlePublisher(
        backend=flaky,
        signer=_signer(),
        netuid=81,
        spool_dir=spool,
        max_attempts=2,
    )
    # First publish fails → spooled.
    pub.publish(**_minimal_inputs(window_id=100))
    assert len(list(spool.iterdir())) == 1

    # Unbreak backend → flush drains the spool.
    flaky.unbreak()
    outcomes = pub.flush_spool()
    assert len(outcomes) == 1
    assert outcomes[0].success is True
    assert len(list(spool.iterdir())) == 0
    # The bundle landed at the canonical key.
    assert fs.get("rollouts/81/100-100/val_a.json") is not None


def test_flush_spool_preserves_failed_entries(tmp_path):
    fs = LocalFilesystemBackend(tmp_path / "root")
    flaky = _FlakyBackend(fs, fail_count=99)
    spool = tmp_path / "spool"
    pub = RolloutBundlePublisher(
        backend=flaky,
        signer=_signer(),
        netuid=81,
        spool_dir=spool,
        max_attempts=1,
    )
    pub.publish(**_minimal_inputs(window_id=100))
    # Flush while backend is still broken → entry stays on disk.
    outcomes = pub.flush_spool()
    assert len(outcomes) == 1
    assert outcomes[0].success is False
    assert len(list(spool.iterdir())) == 1


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def test_fetcher_returns_verified_bundles(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    pub.publish(**_minimal_inputs(window_id=100))
    pub.publish(**_minimal_inputs(window_id=101))

    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=81
    )
    result = fetcher.fetch()
    assert len(result.bundles) == 2
    assert sorted(b.window_id for b in result.bundles) == [100, 101]
    assert result.invalid == []


def test_fetcher_rejects_bad_signature(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    pub.publish(**_minimal_inputs(window_id=100))

    # Tamper: corrupt the envelope signature on disk.
    key = "rollouts/81/100-100/val_a.json"
    raw = backend.get(key)
    env = json.loads(raw.decode("utf-8"))
    env["signature"] = "0" * 64
    backend.put(key, json.dumps(env, sort_keys=True).encode("utf-8"))

    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=81
    )
    result = fetcher.fetch()
    assert result.bundles == []
    assert len(result.invalid) == 1
    assert result.invalid[0].reason == "bad_signature"


def test_fetcher_rejects_netuid_mismatch(tmp_path):
    # Publisher writes netuid=81; fetcher expects 99 → mismatch.
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    pub.publish(**_minimal_inputs(window_id=100))

    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=99
    )
    # Fetcher lists rollouts/99/ by default — no match, empty.
    result = fetcher.fetch()
    assert result.bundles == []
    # If we point it at the actual prefix we uncover the mismatch.
    result2 = fetcher.fetch(prefix="rollouts/81/")
    assert result2.bundles == []
    assert len(result2.invalid) == 1
    assert result2.invalid[0].reason == "netuid_mismatch"


def test_fetcher_rejects_wrong_artifact_type(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    # Inject a blob with artifact_type = "checkpoint_attestation" at a
    # rollouts/ key — must be rejected.
    backend.put(
        "rollouts/81/100-100/val_a.json",
        json.dumps(
            {
                "envelope_version": "1",
                "artifact_type": "checkpoint_attestation",
                "artifact_id": "0" * 64,
                "payload_json": "{}",
                "signer_id": "val_a",
                "signed_at": "2026-04-20T00:00:00+00:00",
                "signature": "deadbeef" * 8,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=81
    )
    result = fetcher.fetch()
    assert result.bundles == []
    assert len(result.invalid) == 1
    assert result.invalid[0].reason == "wrong_artifact_type"


def test_fetcher_handles_malformed_envelope(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    backend.put("rollouts/81/100-100/val_a.json", b"{not-json")
    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=81
    )
    result = fetcher.fetch()
    assert result.bundles == []
    assert result.invalid[0].reason == "malformed_envelope"


def test_fetcher_empty_backend_returns_empty(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    fetcher = RolloutBundleFetcher(
        backend=backend, verifier=_verifier(), netuid=81
    )
    result = fetcher.fetch()
    assert result.bundles == []
    assert result.invalid == []


# ---------------------------------------------------------------------------
# Shape matches InferenceRegistryAdapter expectations
# ---------------------------------------------------------------------------


def test_bundle_shape_matches_forge_adapter_expectations(tmp_path):
    """Forge's ``InferenceRegistryAdapter`` expects a window bundle
    with ``manifest``, ``task_batch``, ``scorecard``, ``completions``,
    ``verdicts``. All five must be present and the types must be
    JSON-compatible."""
    backend = LocalFilesystemBackend(tmp_path / "root")
    pub = RolloutBundlePublisher(backend=backend, signer=_signer(), netuid=81)
    pub.publish(**_minimal_inputs(window_id=100))
    raw = backend.get("rollouts/81/100-100/val_a.json")
    envelope = envelope_from_dict(json.loads(raw.decode("utf-8")))
    bundle_dict = json.loads(envelope.payload_json)

    for key in (
        "manifest",
        "task_batch",
        "scorecard",
        "completions",
        "verdicts",
        "netuid",
        "window_id",
        "window_range",
        "producer_hotkey",
        "published_at",
        "version",
    ):
        assert key in bundle_dict, f"missing {key} from bundle"
    # All values round-trip through JSON.
    json.dumps(bundle_dict)
