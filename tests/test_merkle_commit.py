"""Tests for onchain merkle-commit helpers.

Uses an in-memory FakeCommitCallable to avoid any subtensor dep.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from reliquary_inference.chain.merkle_commit import (
    COMMITMENT_KEY_CHECKPOINT,
    COMMITMENT_KEY_VERDICT_MEDIAN,
    CommitContext,
    OnchainMerkleCommit,
    _compute_verdicts_merkle_root,
    commit_delta_checkpoint_merkle,
    commit_mesh_verdicts_merkle,
)


@dataclass
class FakeCommitResponse:
    success: bool = True
    attempts: int = 1


@dataclass
class FakeCommitCallable:
    calls: list[dict[str, Any]] = field(default_factory=list)
    response: FakeCommitResponse = field(default_factory=FakeCommitResponse)
    raise_on_call: Exception | None = None

    def __call__(self, *, policy_version: str, metadata_hash: str) -> FakeCommitResponse:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append(
            {"policy_version": policy_version, "metadata_hash": metadata_hash}
        )
        return self.response


@pytest.fixture
def ctx() -> CommitContext:
    return CommitContext(subnet="ledger", netuid=81)


# ---------------------------------------------------------------------------
# Merkle root computation
# ---------------------------------------------------------------------------


def test_empty_merkle_root_is_empty_sha256() -> None:
    assert _compute_verdicts_merkle_root([]) == hashlib.sha256(b"").hexdigest()


def test_single_payload_merkle_root_is_sha_of_leaf() -> None:
    payload = b"verdict-1"
    expected = hashlib.sha256(payload).hexdigest()
    assert _compute_verdicts_merkle_root([payload]) == expected


def test_merkle_root_commutative_under_input_order() -> None:
    payloads = [b"a", b"b", b"c"]
    root_a = _compute_verdicts_merkle_root(payloads)
    root_b = _compute_verdicts_merkle_root(list(reversed(payloads)))
    # Sorted internally → same root under any caller ordering.
    assert root_a == root_b


def test_merkle_root_different_payloads_different_root() -> None:
    a = _compute_verdicts_merkle_root([b"x", b"y"])
    b = _compute_verdicts_merkle_root([b"x", b"z"])
    assert a != b


# ---------------------------------------------------------------------------
# OnchainMerkleCommit
# ---------------------------------------------------------------------------


def test_commit_canonical_bytes_deterministic() -> None:
    record = OnchainMerkleCommit(
        commitment_key="reliquary_mesh_verdicts_ledger",
        window_id=7,
        merkle_root_hex="a" * 64,
        committed_at=1_700_000_000.0,
        kind="mesh_verdicts",
        metadata={"verdict_count": 3, "netuid": 81},
    )
    assert record.canonical_bytes() == record.canonical_bytes()


def test_commit_metadata_hash_is_hex_sha256() -> None:
    record = OnchainMerkleCommit(
        commitment_key="k",
        window_id=1,
        merkle_root_hex="0" * 64,
        committed_at=1.0,
        kind="delta_checkpoint",
        metadata={},
    )
    digest = record.metadata_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# commit_mesh_verdicts_merkle
# ---------------------------------------------------------------------------


def test_commit_mesh_verdicts_happy_path(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    payloads = [b"v1", b"v2", b"v3"]
    result = commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=payloads,
        ctx=ctx,
        committed_at=1_700_000_000.0,
        commit=fake,
    )
    assert result.success is True
    assert result.commit.commitment_key == f"{COMMITMENT_KEY_VERDICT_MEDIAN}_ledger"
    assert result.commit.window_id == 7
    assert result.commit.kind == "mesh_verdicts"
    assert result.commit.metadata["verdict_count"] == 3
    assert len(fake.calls) == 1
    # policy_version namespaces to the commitment_key.
    assert fake.calls[0]["policy_version"].endswith(
        f"{COMMITMENT_KEY_VERDICT_MEDIAN}_ledger"
    )


def test_commit_mesh_verdicts_chain_rejection(ctx: CommitContext) -> None:
    fake = FakeCommitCallable(response=FakeCommitResponse(success=False))
    result = commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=[b"v1"],
        ctx=ctx,
        committed_at=1.0,
        commit=fake,
    )
    assert result.success is False
    assert result.error is not None


def test_commit_mesh_verdicts_chain_exception_caught(ctx: CommitContext) -> None:
    fake = FakeCommitCallable(raise_on_call=RuntimeError("subtensor timeout"))
    result = commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=[b"v1"],
        ctx=ctx,
        committed_at=1.0,
        commit=fake,
    )
    assert result.success is False
    assert "subtensor timeout" in (result.error or "")


def test_commit_mesh_verdicts_empty_set_still_commits(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    result = commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=[],
        ctx=ctx,
        committed_at=1.0,
        commit=fake,
    )
    assert result.success is True
    assert result.commit.merkle_root_hex == hashlib.sha256(b"").hexdigest()
    assert result.commit.metadata["verdict_count"] == 0


# ---------------------------------------------------------------------------
# commit_delta_checkpoint_merkle
# ---------------------------------------------------------------------------


def test_commit_delta_checkpoint_happy_path(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    result = commit_delta_checkpoint_merkle(
        window_id=5,
        merkle_root_hex="a" * 64,
        from_checkpoint_hash="f" * 64,
        to_checkpoint_hash="b" * 64,
        shard_count=12,
        ctx=CommitContext(subnet="forge", netuid=3),
        committed_at=1_700_000_000.0,
        commit=fake,
    )
    assert result.success is True
    assert result.commit.commitment_key == f"{COMMITMENT_KEY_CHECKPOINT}_forge"
    assert result.commit.kind == "delta_checkpoint"
    assert result.commit.metadata["shard_count"] == 12
    assert result.commit.metadata["netuid"] == 3


def test_commit_delta_checkpoint_rejects_bad_merkle_root(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    with pytest.raises(ValueError):
        commit_delta_checkpoint_merkle(
            window_id=5,
            merkle_root_hex="not-hex",
            from_checkpoint_hash="",
            to_checkpoint_hash="",
            shard_count=0,
            ctx=ctx,
            committed_at=1.0,
            commit=fake,
        )


def test_commit_delta_checkpoint_rejects_short_merkle_root(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    with pytest.raises(ValueError):
        commit_delta_checkpoint_merkle(
            window_id=5,
            merkle_root_hex="a" * 63,  # one short
            from_checkpoint_hash="",
            to_checkpoint_hash="",
            shard_count=0,
            ctx=ctx,
            committed_at=1.0,
            commit=fake,
        )


def test_commit_delta_checkpoint_captures_hashes(ctx: CommitContext) -> None:
    fake = FakeCommitCallable()
    result = commit_delta_checkpoint_merkle(
        window_id=5,
        merkle_root_hex="a" * 64,
        from_checkpoint_hash="c" * 64,
        to_checkpoint_hash="d" * 64,
        shard_count=4,
        ctx=ctx,
        committed_at=1.0,
        commit=fake,
    )
    assert result.commit.metadata["from_checkpoint_hash"] == "c" * 64
    assert result.commit.metadata["to_checkpoint_hash"] == "d" * 64


# ---------------------------------------------------------------------------
# Subnet namespacing + determinism
# ---------------------------------------------------------------------------


def test_same_payload_different_subnets_different_commit_key() -> None:
    fake_a = FakeCommitCallable()
    fake_b = FakeCommitCallable()
    commit_mesh_verdicts_merkle(
        window_id=1,
        verdict_payloads=[b"x"],
        ctx=CommitContext(subnet="ledger", netuid=81),
        committed_at=1.0,
        commit=fake_a,
    )
    commit_mesh_verdicts_merkle(
        window_id=1,
        verdict_payloads=[b"x"],
        ctx=CommitContext(subnet="forge", netuid=3),
        committed_at=1.0,
        commit=fake_b,
    )
    # Chain sees DIFFERENT metadata_hash because the commitment_key is
    # namespaced (and netuid differs). This is what prevents SN81 and
    # SN3 overwriting each other's commit.
    assert fake_a.calls[0]["metadata_hash"] != fake_b.calls[0]["metadata_hash"]


def test_identical_inputs_produce_identical_commit_bytes(ctx: CommitContext) -> None:
    fake_a = FakeCommitCallable()
    fake_b = FakeCommitCallable()
    commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=[b"x", b"y"],
        ctx=ctx,
        committed_at=1.5,
        commit=fake_a,
    )
    commit_mesh_verdicts_merkle(
        window_id=7,
        verdict_payloads=[b"y", b"x"],  # order reversed
        ctx=ctx,
        committed_at=1.5,
        commit=fake_b,
    )
    assert fake_a.calls[0]["metadata_hash"] == fake_b.calls[0]["metadata_hash"]
