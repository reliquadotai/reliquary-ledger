"""End-to-end integration: VerdictPublisher + VerdictFetcher over R2Backend.

Uses the FakeS3 client from test_r2_backend.py to exercise the full
sign → publish → list → fetch → verify cycle without any network dep.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from reliquary_inference.validator.mesh import ValidatorIdentity, VerdictArtifact
from reliquary_inference.validator.r2_backend import R2Backend
from reliquary_inference.validator.verdict_storage import (
    VerdictFetcher,
    VerdictPublisher,
    VerdictSigner,
    VerdictVerifier,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_backend import FakeS3  # noqa: E402


NETUID = 81
HMAC_KEY = b"shared-hmac-key"
HOTKEY = "mesh-A"
STAKE = 40.0


def _sign(payload_bytes: bytes) -> str:
    return hmac.new(HMAC_KEY, payload_bytes, hashlib.sha256).hexdigest()


def _verify(
    _signer_id: str,
    payload_bytes: bytes,
    signature: str,
    _hotkey: object,
) -> bool:
    expected = hmac.new(HMAC_KEY, payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def _artifact(completion_id: str = "c-1", accepted: bool = True) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner-A",
        window_id=7,
        validator=ValidatorIdentity(hotkey=HOTKEY, stake=STAKE),
        accepted=accepted,
        stage_failed=None,
        reject_reason=None,
        scores={"reward": 1.0 if accepted else 0.0},
        signed_at=1_700_000_000.0,
    )


@pytest.fixture
def r2_backend() -> R2Backend:
    return R2Backend(bucket="reliquary-verdicts", client=FakeS3())


@pytest.fixture
def signer() -> VerdictSigner:
    return VerdictSigner(signer_id=HOTKEY, sign=_sign)


@pytest.fixture
def verifier() -> VerdictVerifier:
    return VerdictVerifier(
        expected_hotkeys={HOTKEY: STAKE},
        verify=_verify,
    )


# ---------------------------------------------------------------------------
# Round trip: publish → list → fetch
# ---------------------------------------------------------------------------


def test_publish_then_fetch_via_r2_backend(
    r2_backend: R2Backend, signer: VerdictSigner, verifier: VerdictVerifier
) -> None:
    publisher = VerdictPublisher(
        backend=r2_backend, signer=signer, netuid=NETUID
    )
    artifact = _artifact()
    publisher.publish(artifact)

    fetcher = VerdictFetcher(backend=r2_backend, verifier=verifier, netuid=NETUID)
    report = fetcher.fetch_window(window_id=7)
    assert len(report.artifacts) == 1
    fetched = report.artifacts[0]
    assert fetched.completion_id == "c-1"
    assert fetched.accepted is True
    assert fetched.validator.hotkey == HOTKEY


def test_fetch_rejects_tampered_payload_over_r2(
    r2_backend: R2Backend, signer: VerdictSigner, verifier: VerdictVerifier
) -> None:
    publisher = VerdictPublisher(
        backend=r2_backend, signer=signer, netuid=NETUID
    )
    artifact = _artifact()
    publisher.publish(artifact)

    # Directly mutate the bytes in the fake S3 store to simulate a MITM.
    fake = r2_backend.client  # type: ignore[assignment]
    target_key = next(k for (_, k) in fake.store if "c-1.json" in k)  # type: ignore[union-attr]
    raw = fake.store[("reliquary-verdicts", target_key)]  # type: ignore[index]
    envelope = json.loads(raw.decode("utf-8"))
    inner = json.loads(envelope["payload_json"])
    inner["accepted"] = not inner["accepted"]
    envelope["payload_json"] = json.dumps(inner, sort_keys=True, separators=(",", ":"))
    fake.store[("reliquary-verdicts", target_key)] = json.dumps(  # type: ignore[index]
        envelope, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    fetcher = VerdictFetcher(backend=r2_backend, verifier=verifier, netuid=NETUID)
    report = fetcher.fetch_window(window_id=7)
    # Tampered verdict does NOT land in artifacts; it's recorded as invalid
    # with a signature / tamper reason.
    assert len(report.artifacts) == 0
    assert len(report.invalid) >= 1


def test_multiple_verdicts_listed_and_fetched(
    r2_backend: R2Backend, signer: VerdictSigner, verifier: VerdictVerifier
) -> None:
    publisher = VerdictPublisher(
        backend=r2_backend, signer=signer, netuid=NETUID
    )
    for idx in range(5):
        publisher.publish(_artifact(completion_id=f"c-{idx}"))

    fetcher = VerdictFetcher(backend=r2_backend, verifier=verifier, netuid=NETUID)
    report = fetcher.fetch_window(window_id=7)
    assert len(report.artifacts) == 5
    completion_ids = {a.completion_id for a in report.artifacts}
    assert completion_ids == {f"c-{i}" for i in range(5)}


def test_empty_window_returns_empty_report(
    r2_backend: R2Backend, verifier: VerdictVerifier
) -> None:
    fetcher = VerdictFetcher(backend=r2_backend, verifier=verifier, netuid=NETUID)
    report = fetcher.fetch_window(window_id=999)
    assert report.artifacts == []
    assert report.invalid == []
    assert report.invalid == []


# ---------------------------------------------------------------------------
# R2-prefixed namespace works under the same publisher flow
# ---------------------------------------------------------------------------


def test_r2_prefix_scoping_does_not_leak_across_deployments() -> None:
    shared_client = FakeS3()
    backend_a = R2Backend(bucket="b", key_prefix="subnet-a", client=shared_client)
    backend_b = R2Backend(bucket="b", key_prefix="subnet-b", client=shared_client)

    signer_a = VerdictSigner(signer_id="v-a", sign=_sign)
    verifier_a = VerdictVerifier(expected_hotkeys={"v-a": 10.0}, verify=_verify)

    pub_a = VerdictPublisher(backend=backend_a, signer=signer_a, netuid=1)
    art_a = VerdictArtifact(
        completion_id="c-a",
        miner_hotkey="m",
        window_id=3,
        validator=ValidatorIdentity(hotkey="v-a", stake=10.0),
        accepted=True,
        stage_failed=None,
        reject_reason=None,
        scores={},
        signed_at=1.0,
    )
    pub_a.publish(art_a)

    # Under backend_a's key_prefix, the verdict is visible.
    fetcher_a = VerdictFetcher(backend=backend_a, verifier=verifier_a, netuid=1)
    report_a = fetcher_a.fetch_window(window_id=3)
    assert len(report_a.artifacts) == 1
    assert report_a.artifacts[0].validator.hotkey == "v-a"

    # Backend_b is scoped to a DIFFERENT prefix; no verdict visible.
    verifier_b = VerdictVerifier(expected_hotkeys={"v-b": 10.0}, verify=_verify)
    fetcher_b = VerdictFetcher(backend=backend_b, verifier=verifier_b, netuid=1)
    assert fetcher_b.fetch_window(window_id=3).artifacts == []
