"""Onchain merkle-commit wiring for mesh verdicts + delta checkpoints.

Wraps the lower-level ``commit_policy_metadata`` chain primitive with a
domain-specific API so callers don't have to construct commitment keys
by hand. Two entry points:

- ``commit_mesh_verdicts_merkle``: commits the stake-weighted median
  verdict set's Merkle root for a given window so external observers
  can audit the consensus from chain state alone.
- ``commit_delta_checkpoint_merkle``: commits the checkpoint-bundle
  Merkle root so miners + sibling validators can verify they're
  downloading the blessed artifact before reconstruction.

The chain adapter itself stays in ``chain.adapter``. This module
composes the verdict-storage report or checkpoint manifest with a
``CommitContext`` and a commit-callable. Tests pass a fake callable
to avoid any subtensor / btcli dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol


COMMITMENT_KEY_VERDICT_MEDIAN = "reliquary_mesh_verdicts"
COMMITMENT_KEY_CHECKPOINT = "reliquary_checkpoint"


class CommitCallable(Protocol):
    """Minimal surface the onchain commit helper depends on.

    Maps 1:1 to ``ChainAdapter.commit_policy_metadata`` — we take a
    Protocol so tests can substitute a fake without importing bittensor.
    """

    def __call__(self, *, policy_version: str, metadata_hash: str) -> Any: ...


@dataclass(frozen=True)
class CommitContext:
    """Identifies the on-chain namespace a commit lands in.

    ``subnet`` matches the subnet name (``ledger`` / ``forge``) —
    keeps commits from one subnet from overwriting the other's even
    when both talk to the same subtensor.
    """

    subnet: str
    netuid: int


@dataclass(frozen=True)
class OnchainMerkleCommit:
    """Structured commit payload sent to the chain.

    The actual bytes stored onchain is the canonical JSON of this
    payload's ``as_dict()`` under the namespaced commitment key.
    """

    commitment_key: str
    window_id: int
    merkle_root_hex: str
    committed_at: float
    kind: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "commitment_key": self.commitment_key,
            "window_id": self.window_id,
            "merkle_root_hex": self.merkle_root_hex,
            "committed_at": self.committed_at,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def metadata_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass
class CommitResult:
    """Outcome returned by the commit helpers."""

    success: bool
    commit: OnchainMerkleCommit
    chain_response: Any = None
    error: str | None = None


def _compute_verdicts_merkle_root(
    verdict_payloads: list[bytes],
) -> str:
    """Binary Merkle root over canonical verdict bytes, sorted for determinism.

    Each payload's sha256 becomes a leaf; leaves are sorted ascending so
    validators in independent processes produce the same root over the
    same verdict set.
    """
    if not verdict_payloads:
        return hashlib.sha256(b"").hexdigest()
    leaves = sorted(
        hashlib.sha256(payload).digest() for payload in verdict_payloads
    )
    while len(leaves) > 1:
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[i] + leaves[i + 1]).digest()
            for i in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def commit_mesh_verdicts_merkle(
    *,
    window_id: int,
    verdict_payloads: list[bytes],
    ctx: CommitContext,
    committed_at: float,
    commit: CommitCallable,
    policy_version: str = "v1",
) -> CommitResult:
    """Commit the Merkle root of a window's verdict set to chain.

    Args:
        window_id: the training window this commit applies to.
        verdict_payloads: list of canonical-bytes-encoded verdict
            envelopes (typically each validator's ``VerdictArtifact``
            envelope bytes from ``verdict_storage.py``). The caller is
            responsible for having sorted / filtered them per whatever
            consensus policy applies — this function just hashes them.
        ctx: subnet + netuid namespace.
        committed_at: wall-clock timestamp of the commit (UTC seconds).
        commit: chain-adapter commit callable; tests pass a fake.
        policy_version: policy generation tag; bumps whenever the
            onchain schema changes.
    """
    merkle_root = _compute_verdicts_merkle_root(verdict_payloads)
    record = OnchainMerkleCommit(
        commitment_key=f"{COMMITMENT_KEY_VERDICT_MEDIAN}_{ctx.subnet}",
        window_id=window_id,
        merkle_root_hex=merkle_root,
        committed_at=float(committed_at),
        kind="mesh_verdicts",
        metadata={
            "verdict_count": len(verdict_payloads),
            "netuid": ctx.netuid,
            "policy_version": policy_version,
        },
    )
    return _dispatch_commit(record, commit, policy_version=policy_version)


def commit_delta_checkpoint_merkle(
    *,
    window_id: int,
    merkle_root_hex: str,
    from_checkpoint_hash: str,
    to_checkpoint_hash: str,
    shard_count: int,
    ctx: CommitContext,
    committed_at: float,
    commit: CommitCallable,
    policy_version: str = "v1",
) -> CommitResult:
    """Commit a delta-checkpoint bundle's Merkle root onchain.

    ``merkle_root_hex`` is the exact value returned by
    ``DeltaBundle.merkle_root_hex`` (or equivalently the recomputed
    root in ``fetch_bundle``). Callers must ensure ``window_id`` matches
    the bundle's target training window.
    """
    if len(merkle_root_hex) != 64 or not all(
        c in "0123456789abcdef" for c in merkle_root_hex
    ):
        raise ValueError(
            f"merkle_root_hex must be 64-char lowercase hex, got {merkle_root_hex!r}"
        )
    record = OnchainMerkleCommit(
        commitment_key=f"{COMMITMENT_KEY_CHECKPOINT}_{ctx.subnet}",
        window_id=window_id,
        merkle_root_hex=merkle_root_hex,
        committed_at=float(committed_at),
        kind="delta_checkpoint",
        metadata={
            "from_checkpoint_hash": from_checkpoint_hash,
            "to_checkpoint_hash": to_checkpoint_hash,
            "shard_count": shard_count,
            "netuid": ctx.netuid,
            "policy_version": policy_version,
        },
    )
    return _dispatch_commit(record, commit, policy_version=policy_version)


def _dispatch_commit(
    record: OnchainMerkleCommit,
    commit: CommitCallable,
    *,
    policy_version: str,
) -> CommitResult:
    try:
        response = commit(
            policy_version=f"{policy_version}.{record.commitment_key}",
            metadata_hash=record.metadata_hash(),
        )
    except Exception as exc:
        return CommitResult(
            success=False,
            commit=record,
            error=f"{type(exc).__name__}: {exc}",
        )

    ok = bool(getattr(response, "success", False))
    return CommitResult(
        success=ok,
        commit=record,
        chain_response=response,
        error=None if ok else f"chain rejected commit: {response!r}",
    )


__all__ = [
    "COMMITMENT_KEY_CHECKPOINT",
    "COMMITMENT_KEY_VERDICT_MEDIAN",
    "CommitCallable",
    "CommitContext",
    "CommitResult",
    "OnchainMerkleCommit",
    "commit_delta_checkpoint_merkle",
    "commit_mesh_verdicts_merkle",
]
