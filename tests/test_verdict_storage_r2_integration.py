"""Integration test: VerdictPublisher + VerdictFetcher over a real R2 bucket.

Exercised only when `RELIQUARY_R2_CF_API_TOKEN` is set in the env — the
CI matrix can opt in by populating the secret, local devs can opt in by
sourcing `.env.local`. Skipped otherwise so the default test run stays
dep-free.

Proves gap #11 is closed: the Ledger verdict pipeline writes + reads
real bytes through the shared R2 backend in reliquary-protocol.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import pytest

_R2_TOKEN_SET = bool(os.environ.get("RELIQUARY_R2_CF_API_TOKEN"))
pytestmark = pytest.mark.skipif(
    not _R2_TOKEN_SET,
    reason="RELIQUARY_R2_CF_API_TOKEN not set; set it (or source .env.local) to run live R2 tests",
)

from reliquary_inference.validator.mesh import ValidatorIdentity, VerdictArtifact
from reliquary_inference.validator.verdict_storage import (
    VerdictFetcher,
    VerdictPublisher,
    VerdictSigner,
    VerdictVerifier,
    verdict_key,
)

try:  # pragma: no cover - skipped upstream when token is unset
    from reliquary_protocol.storage import R2ObjectBackend
except ImportError:  # pragma: no cover
    R2ObjectBackend = None  # type: ignore[assignment]


HMAC_KEY = b"ledger-integration-test-secret"
TEST_HOTKEY = "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZC"
TEST_NETUID = 8181  # isolated netuid — never collides with real SN81 traffic


def _signer() -> VerdictSigner:
    def sign(payload: bytes) -> str:
        return hmac.new(HMAC_KEY, payload, hashlib.sha256).hexdigest()

    return VerdictSigner(signer_id=TEST_HOTKEY, sign=sign)


def _verifier() -> VerdictVerifier:
    def verify(signer_id: str, payload: bytes, signature: str, _expected: object) -> bool:
        if signer_id != TEST_HOTKEY:
            return False
        expected = hmac.new(HMAC_KEY, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    return VerdictVerifier(expected_hotkeys={TEST_HOTKEY: {"stake": 40.0}}, verify=verify)


def _verdict(window_id: int, completion_id: str) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner-test",
        window_id=window_id,
        validator=ValidatorIdentity(hotkey=TEST_HOTKEY, stake=40.0),
        accepted=True,
        stage_failed=None,
        reject_reason=None,
        scores={"reward": 0.95},
        signed_at=float(int(time.time())),
    )


def _backend():
    assert R2ObjectBackend is not None, "reliquary-protocol not installed / importable"
    return R2ObjectBackend.from_env()


def _cleanup(backend, prefix: str) -> None:
    for key in backend.list(prefix):
        backend.delete(key)


def test_verdict_round_trip_over_r2() -> None:
    """Publish two verdicts → fetch the window → assert we got both back."""
    backend = _backend()
    window_id = 999_999_000 + int(time.time() % 1000)
    prefix = f"verdicts/{TEST_NETUID}/{window_id}/"
    _cleanup(backend, prefix)

    publisher = VerdictPublisher(backend=backend, signer=_signer(), netuid=TEST_NETUID)
    publisher.publish(_verdict(window_id, "c-round-trip-1"))
    publisher.publish(_verdict(window_id, "c-round-trip-2"))

    fetcher = VerdictFetcher(backend=backend, verifier=_verifier(), netuid=TEST_NETUID)
    fetched = fetcher.fetch_window(window_id)

    try:
        assert fetched.invalid == []
        ids = {a.completion_id for a in fetched.artifacts}
        assert ids == {"c-round-trip-1", "c-round-trip-2"}
        assert all(a.validator.hotkey == TEST_HOTKEY for a in fetched.artifacts)
    finally:
        _cleanup(backend, prefix)


def test_verdict_key_is_canonical() -> None:
    key = verdict_key(
        netuid=TEST_NETUID,
        window_id=42,
        validator_hotkey=TEST_HOTKEY,
        completion_id="c-abc",
    )
    assert key.startswith(f"verdicts/{TEST_NETUID}/42/")
    assert key.endswith("/c-abc.json")


def test_r2_rejects_tampered_envelope_on_fetch() -> None:
    """A byte-flip in the stored envelope must surface as `invalid`."""
    backend = _backend()
    window_id = 999_999_500 + int(time.time() % 500)
    prefix = f"verdicts/{TEST_NETUID}/{window_id}/"
    _cleanup(backend, prefix)

    publisher = VerdictPublisher(backend=backend, signer=_signer(), netuid=TEST_NETUID)
    publisher.publish(_verdict(window_id, "c-tamper"))

    key = verdict_key(
        netuid=TEST_NETUID,
        window_id=window_id,
        validator_hotkey=TEST_HOTKEY,
        completion_id="c-tamper",
    )
    original = backend.get(key)
    assert original is not None
    parsed = json.loads(original.decode("utf-8"))

    # Envelope = {payload_json, signature, signer_id}. Tamper payload_json
    # WITHOUT re-signing → signature must no longer match.
    inner = json.loads(parsed["payload_json"])
    inner["scores"]["reward"] = 0.0001
    parsed["payload_json"] = json.dumps(inner, sort_keys=True, separators=(",", ":"))
    backend.put(key, json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode())

    fetcher = VerdictFetcher(backend=backend, verifier=_verifier(), netuid=TEST_NETUID)
    fetched = fetcher.fetch_window(window_id)

    try:
        assert fetched.artifacts == [], "tampered verdict must NOT be accepted"
        assert len(fetched.invalid) == 1
        assert fetched.invalid[0].reason == "bad_signature"
    finally:
        _cleanup(backend, prefix)
