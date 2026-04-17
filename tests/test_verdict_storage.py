"""Tests for the verdict storage + signing pipeline.

Covers all acceptance-test bullets from
``private/reliquary-plan/notes/spec-verdict-storage.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from reliquary_inference.validator.mesh import (
    ValidatorIdentity,
    VerdictArtifact,
    aggregate_verdicts,
)
from reliquary_inference.validator.verdict_storage import (
    FetchResult,
    InvalidArtifactReport,
    LocalFilesystemBackend,
    PublishOutcome,
    StorageBackend,
    VerdictFetcher,
    VerdictPublisher,
    VerdictSigner,
    VerdictVerifier,
    _canonicalize,
    verdict_key,
)


HMAC_KEYS: dict[str, bytes] = {
    "mesh-A": b"key-A-super-secret",
    "mesh-B": b"key-B-super-secret",
    "mesh-C": b"key-C-super-secret",
}


def _sign_fn(hotkey: str):
    def _sign(data: bytes) -> str:
        return hmac.new(HMAC_KEYS[hotkey], data, hashlib.sha256).hexdigest()

    return _sign


def _verify(signer_id: str, data: bytes, signature: str, key: bytes) -> bool:
    expected = hmac.new(key, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _identity(hotkey: str, stake: float) -> ValidatorIdentity:
    return ValidatorIdentity(hotkey=hotkey, stake=stake, signer_id=hotkey)


def _verdict(
    *,
    completion_id: str,
    hotkey: str,
    stake: float = 40.0,
    window_id: int = 100,
    accepted: bool = True,
) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner-1",
        window_id=window_id,
        validator=_identity(hotkey, stake),
        accepted=accepted,
        stage_failed=None,
        reject_reason=None,
        scores={"correctness": 0.9, "format": 0.8},
        signed_at=1_700_000_000.0,
    )


def _publisher(tmp_path: Path, hotkey: str, *, netuid: int = 81) -> tuple[VerdictPublisher, LocalFilesystemBackend]:
    backend = LocalFilesystemBackend(tmp_path / "remote")
    signer = VerdictSigner(signer_id=hotkey, sign=_sign_fn(hotkey))
    publisher = VerdictPublisher(
        backend=backend,
        signer=signer,
        netuid=netuid,
        spool_dir=tmp_path / "spool" / hotkey,
    )
    return publisher, backend


def _fetcher(backend: StorageBackend, *, netuid: int = 81) -> VerdictFetcher:
    verifier = VerdictVerifier(
        expected_hotkeys={hotkey: HMAC_KEYS[hotkey] for hotkey in ("mesh-A", "mesh-B", "mesh-C")},
        verify=_verify,
    )
    return VerdictFetcher(backend=backend, verifier=verifier, netuid=netuid)


def test_canonical_bytes_stable_under_score_reordering() -> None:
    a = _verdict(completion_id="c1", hotkey="mesh-A")
    # Rebuild the same verdict with scores inserted in a different order.
    b = VerdictArtifact(
        completion_id=a.completion_id,
        miner_hotkey=a.miner_hotkey,
        window_id=a.window_id,
        validator=a.validator,
        accepted=a.accepted,
        stage_failed=a.stage_failed,
        reject_reason=a.reject_reason,
        scores={"format": a.scores["format"], "correctness": a.scores["correctness"]},
        signed_at=a.signed_at,
    )
    assert _canonicalize(a) == _canonicalize(b)


def test_publish_fetch_round_trip(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-roundtrip", hotkey="mesh-A")

    outcome = publisher.publish(verdict)
    assert outcome.success is True
    assert outcome.backend == "remote"
    assert outcome.key is not None

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(verdict.window_id)
    assert len(result.artifacts) == 1
    assert result.invalid == []

    fetched = result.artifacts[0]
    assert fetched.completion_id == verdict.completion_id
    assert fetched.validator.hotkey == "mesh-A"
    assert fetched.scores == verdict.scores
    assert fetched.accepted is True


def test_bad_signature_is_rejected(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-bad-sig", hotkey="mesh-A")
    outcome = publisher.publish(verdict)
    assert outcome.success is True

    # Tamper with the stored envelope's signature.
    raw = backend.get(outcome.key)
    envelope = json.loads(raw.decode("utf-8"))
    envelope["signature"] = "deadbeef" * 8
    backend.put(outcome.key, json.dumps(envelope, sort_keys=True).encode("utf-8"))

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(verdict.window_id)
    assert result.artifacts == []
    assert len(result.invalid) == 1
    assert result.invalid[0].reason == "bad_signature"


def test_tampered_payload_rejected(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-tamper-payload", hotkey="mesh-A")
    outcome = publisher.publish(verdict)
    assert outcome.success is True

    raw = backend.get(outcome.key)
    envelope = json.loads(raw.decode("utf-8"))
    # Flip accepted: True -> False; signature no longer binds the new payload.
    payload = json.loads(envelope["payload_json"])
    payload["accepted"] = False
    envelope["payload_json"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    backend.put(outcome.key, json.dumps(envelope, sort_keys=True).encode("utf-8"))

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(verdict.window_id)
    assert result.artifacts == []
    assert result.invalid[0].reason == "bad_signature"


def test_unknown_signer_rejected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "remote")
    hkeys = HMAC_KEYS.copy()
    hkeys["mesh-outsider"] = b"outsider-key"
    signer = VerdictSigner(signer_id="mesh-outsider", sign=lambda data: hmac.new(
        hkeys["mesh-outsider"], data, hashlib.sha256
    ).hexdigest())
    publisher = VerdictPublisher(
        backend=backend, signer=signer, netuid=81,
        spool_dir=tmp_path / "spool",
    )
    verdict = _verdict(completion_id="c-outsider", hotkey="mesh-outsider")
    publisher.publish(verdict)

    fetcher = _fetcher(backend)  # expected_hotkeys does NOT include mesh-outsider
    result = fetcher.fetch_window(verdict.window_id)
    assert result.artifacts == []
    assert result.invalid[0].reason == "unknown_signer"


def test_malformed_envelope_rejected(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-malformed", hotkey="mesh-A")
    outcome = publisher.publish(verdict)
    backend.put(outcome.key, b"{not valid json at all")

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(verdict.window_id)
    assert result.artifacts == []
    assert result.invalid[0].reason == "malformed_json"


def test_schema_mismatch_missing_field_rejected(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-schema-missing", hotkey="mesh-A")
    outcome = publisher.publish(verdict)

    raw = backend.get(outcome.key)
    envelope = json.loads(raw.decode("utf-8"))
    payload = json.loads(envelope["payload_json"])
    del payload["scores"]
    envelope["payload_json"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # Re-sign so we're testing schema, not signature.
    envelope["signature"] = _sign_fn("mesh-A")(envelope["payload_json"].encode("utf-8"))
    backend.put(outcome.key, json.dumps(envelope, sort_keys=True).encode("utf-8"))

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(verdict.window_id)
    assert result.artifacts == []
    assert result.invalid[0].reason == "schema_mismatch"


def test_window_mismatch_rejected(tmp_path: Path) -> None:
    publisher, backend = _publisher(tmp_path, "mesh-A")
    verdict = _verdict(completion_id="c-window-mismatch", hotkey="mesh-A", window_id=100)
    outcome = publisher.publish(verdict)

    # Re-publish the same envelope under a different window prefix.
    raw = backend.get(outcome.key)
    wrong_key = outcome.key.replace("/100/", "/101/")
    backend.put(wrong_key, raw)

    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(101)
    # The envelope under /101/ has payload.window_id=100 so it's rejected.
    assert result.artifacts == []
    assert result.invalid[0].reason == "schema_mismatch"


def test_publisher_spools_on_backend_failure(tmp_path: Path) -> None:
    class FailingBackend:
        def put(self, key, data):  # noqa: D401
            raise RuntimeError("r2 unavailable")

        def get(self, key):
            return None

        def list(self, prefix):
            return []

        def delete(self, key):
            pass

    signer = VerdictSigner(signer_id="mesh-A", sign=_sign_fn("mesh-A"))
    publisher = VerdictPublisher(
        backend=FailingBackend(),
        signer=signer,
        netuid=81,
        spool_dir=tmp_path / "spool",
        max_attempts=2,
    )
    verdict = _verdict(completion_id="c-spool", hotkey="mesh-A")
    outcome = publisher.publish(verdict)
    assert outcome.success is False
    assert outcome.backend == "spool"
    assert outcome.attempts == 2
    assert outcome.last_error is not None
    assert any((tmp_path / "spool").rglob("*.json".replace("*", ""))) or list(
        (tmp_path / "spool").rglob("*")
    )


def test_flush_spool_retries_and_succeeds(tmp_path: Path) -> None:
    # First publish with a failing backend, then swap it for a healthy one
    # via the publisher's backend attribute and flush_spool.
    class FailingBackend:
        def put(self, key, data):
            raise RuntimeError("r2 unavailable")

        def get(self, key):
            return None

        def list(self, prefix):
            return []

        def delete(self, key):
            pass

    signer = VerdictSigner(signer_id="mesh-A", sign=_sign_fn("mesh-A"))
    publisher = VerdictPublisher(
        backend=FailingBackend(),
        signer=signer,
        netuid=81,
        spool_dir=tmp_path / "spool",
        max_attempts=1,
    )
    verdict = _verdict(completion_id="c-flush", hotkey="mesh-A")
    publisher.publish(verdict)

    healthy = LocalFilesystemBackend(tmp_path / "remote")
    publisher.backend = healthy
    outcomes = publisher.flush_spool()

    assert len(outcomes) == 1
    assert outcomes[0].success is True
    assert outcomes[0].backend == "remote"
    # Spool directory is drained.
    assert not list((tmp_path / "spool").rglob("*.json")) and not list((tmp_path / "spool").rglob("*"))
    # Verdict now present in the healthy backend.
    assert healthy.list(f"verdicts/81/{verdict.window_id}/"), "flushed verdict missing from remote"


def test_atomic_publish_leaves_no_partial_file(tmp_path: Path) -> None:
    class BrokenSigner:
        signer_id = "mesh-A"

        def sign(self, data: bytes) -> str:
            raise RuntimeError("signer hardware unplugged")

    backend = LocalFilesystemBackend(tmp_path / "remote")
    publisher = VerdictPublisher(
        backend=backend,
        signer=BrokenSigner(),  # type: ignore[arg-type]
        netuid=81,
        spool_dir=tmp_path / "spool",
    )
    verdict = _verdict(completion_id="c-atomic", hotkey="mesh-A")
    outcome = publisher.publish(verdict)
    assert outcome.success is False
    assert outcome.backend == "none"
    # No file should have been written to the remote backend.
    assert backend.list("verdicts/") == []


def test_end_to_end_with_aggregator(tmp_path: Path) -> None:
    netuid = 81
    window = 500
    validators = [
        _identity("mesh-A", 40.0),
        _identity("mesh-B", 35.0),
        _identity("mesh-C", 15.0),
    ]

    # 3 validators each publish 4 verdicts to a shared backend.
    backend = LocalFilesystemBackend(tmp_path / "remote")
    for ident in validators:
        signer = VerdictSigner(signer_id=ident.hotkey, sign=_sign_fn(ident.hotkey))
        publisher = VerdictPublisher(backend=backend, signer=signer, netuid=netuid)
        for idx in range(4):
            publisher.publish(
                _verdict(completion_id=f"c{idx}", hotkey=ident.hotkey, stake=ident.stake, window_id=window)
            )

    fetcher = _fetcher(backend, netuid=netuid)
    fetch = fetcher.fetch_window(window)
    assert fetch.invalid == []
    assert len(fetch.artifacts) == 3 * 4

    report = aggregate_verdicts(
        fetch.artifacts, window_id=window, expected_validators=validators
    )
    assert report.missing_validators == []
    assert report.gated_validators == []
    for mv in report.median_verdicts.values():
        assert mv.accepted is True  # unanimous-accept honest verdicts


def test_local_backend_rejects_keys_escaping_root(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "root")
    with pytest.raises(ValueError):
        backend.put("../outside.json", b"x")


def test_fetch_window_returns_empty_on_unknown_window(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "remote")
    fetcher = _fetcher(backend)
    result = fetcher.fetch_window(999_999)
    assert result.artifacts == []
    assert result.invalid == []


def test_verdict_key_is_stable_and_human_readable() -> None:
    key = verdict_key(81, 500, "mesh-A", "c1")
    assert key == "verdicts/81/500/mesh-A/c1.json"
    assert "/" in key  # hierarchical prefix enables list-by-window
