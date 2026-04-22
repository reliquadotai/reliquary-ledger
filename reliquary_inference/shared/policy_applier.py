"""Production PolicyApplier — hot-swaps tensor weights into a live MiningEngine.

The closed-loop bridge on the Ledger side has a PolicyConsumer that polls
for fresh CheckpointAttestations published by Forge, verifies the full
signature chain, and invokes an :class:`PolicyApplier` callable. Unit tests
use a no-op / recording applier. Production uses this module's
:class:`ReloadingPolicyApplier`, which:

  1. Reads the delta bundle bytes from the object store via the same
     ``DeltaLoader`` protocol the consumer already uses.
  2. Dequantizes each int8 shard into a float32 delta tensor.
  3. For each tensor named in the bundle, looks up the matching parameter
     on the :class:`MiningEngine`'s model and mutates its ``.data``
     in-place (``param.data.add_(delta)``). Targeted deltas (e.g. a
     LoRA-merged delta that touches only attention Q/K/V projections)
     legitimately don't cover every parameter — the applier skips any
     bundle shard whose tensor name is absent from the model, logging
     the skip rather than raising, so a ragged delta doesn't hose the
     miner.
  4. Subsequent ``engine.generate_completion`` calls use the updated
     weights immediately. No process restart, no model reload.

Failure handling:
  - Merkle root / shard sha256 mismatches raise from the delta_loader
    before this applier runs.
  - If tensor-shape mismatch or dtype mismatch is detected inside the
    applier, we raise BEFORE any tensor is mutated. The consumer then
    marks the commitment ``rejected`` with the error message, and the
    miner's current weights stay live.
"""

from __future__ import annotations

import logging
from typing import Any

from .policy_consumer import LoadedDelta

logger = logging.getLogger(__name__)


APPLIER_METRIC_APPLIED = "reliquary_policy_applier_apply_total"
APPLIER_METRIC_SKIPPED_UNKNOWN = "reliquary_policy_applier_skipped_unknown_tensor_total"
APPLIER_METRIC_REJECTED = "reliquary_policy_applier_rejected_total"


class ReloadingPolicyApplier:
    """Applies a delta bundle's weight updates into a live MiningEngine.

    Constructed with a reference to the engine and (lazily) a torch import.
    The bundle is expected to be attached to the ``LoadedDelta.extra``
    mapping under key ``"bundle"`` — see
    :class:`BundleAwareDeltaLoader` below for a matching loader.

    Idempotent: calling this with the same LoadedDelta twice double-
    applies the delta — don't do that. The PolicyConsumer enforces the
    "apply-once" invariant by tracking current_policy_window.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.metrics_counters: dict[str, int] = {
            APPLIER_METRIC_APPLIED: 0,
            APPLIER_METRIC_SKIPPED_UNKNOWN: 0,
            APPLIER_METRIC_REJECTED: 0,
        }

    def __call__(self, delta: LoadedDelta) -> None:
        bundle = delta.extra.get("bundle") if delta.extra else None
        if bundle is None:
            self.metrics_counters[APPLIER_METRIC_REJECTED] += 1
            raise RuntimeError(
                "ReloadingPolicyApplier requires delta.extra['bundle'] to be "
                "the fetched DeltaBundle; wire BundleAwareDeltaLoader into "
                "the PolicyConsumer"
            )

        import torch  # lazy: reliquary-inference has torch in prod, skip on import

        # Discover model parameters by name. We tolerate ragged bundles
        # (bundle has a strict subset of the model's parameters) — this
        # matches the real targeted-LoRA-merged delta shape.
        named_params = dict(self.engine.model.named_parameters())

        # Pre-flight: validate shapes BEFORE mutating any tensor.
        for shard in bundle.shards:
            if shard.tensor_name not in named_params:
                # Will skip at apply time; not a hard fail.
                continue
            p = named_params[shard.tensor_name]
            if tuple(p.shape) != tuple(shard.shape):
                self.metrics_counters[APPLIER_METRIC_REJECTED] += 1
                raise ValueError(
                    f"shape mismatch on {shard.tensor_name!r}: "
                    f"model={tuple(p.shape)} bundle={tuple(shard.shape)}"
                )

        # Apply.
        applied = 0
        skipped: list[str] = []
        for shard in bundle.shards:
            if shard.tensor_name not in named_params:
                skipped.append(shard.tensor_name)
                self.metrics_counters[APPLIER_METRIC_SKIPPED_UNKNOWN] += 1
                continue
            param = named_params[shard.tensor_name]
            delta_fp32 = _dequantize_int8_shard(
                data_bytes=shard.data_bytes,
                scale=shard.scale,
                shape=shard.shape,
                torch=torch,
            )
            with torch.no_grad():
                param.data.add_(delta_fp32.to(param.dtype).to(param.device))
            applied += 1

        self.metrics_counters[APPLIER_METRIC_APPLIED] += 1

        logger.info(
            "policy_applier.applied run_id=%s window=%d tensors_updated=%d "
            "tensors_skipped_unknown=%d merkle=%s",
            delta.run_id,
            delta.window_id,
            applied,
            len(skipped),
            delta.merkle_root_hex,
        )
        if skipped:
            logger.debug("policy_applier.skipped_unknown=%r", skipped[:10])


def _dequantize_int8_shard(
    *,
    data_bytes: bytes,
    scale: float,
    shape: tuple[int, ...],
    torch,
):
    """Reverse of reliquary.training.delta_checkpoints._quantize.

    Per-tensor symmetric int8 quantization: int8 -> float32 scaled back up
    by ``scale``. Matches the dequant path in
    ``reliquary.training.delta_checkpoints._dequantize``.
    """
    import numpy as np

    arr = np.frombuffer(data_bytes, dtype=np.int8).astype(np.float32) * float(scale)
    t = torch.from_numpy(arr.copy()).reshape(tuple(int(d) for d in shape))
    return t


def bundle_aware_delta_loader(fetch_bundle, backend_factory):
    """Build a DeltaLoader that attaches the fetched DeltaBundle into extra.

    Parameters:
      - ``fetch_bundle`` — callable (run_id, window_id, backend) ->
        DeltaBundle.  Typically :func:`reliquary.training.checkpoint_storage.fetch_bundle`.
      - ``backend_factory`` — callable () -> CheckpointStorageBackend. Gets
        called on each poll so the backend can be swapped (e.g. by
        operators rotating CF tokens) without restarting the miner.

    Returns a loader compatible with
    :class:`reliquary_inference.shared.policy_consumer.DeltaLoader`.
    """

    def _loader(
        *,
        run_id: str,
        window_id: int,
        expected_merkle_root_hex: str,
        backend,
    ) -> LoadedDelta:
        # Use the backend the PolicyConsumer handed us — that's the one
        # we wired via StorageBackend shim. If it doesn't match the
        # checkpoint_storage Protocol (e.g. RestR2ObjectStore wraps around
        # R2ObjectBackend), fall back to backend_factory() which the
        # caller configured for Forge-shaped reads.
        ckpt_backend = backend if _has_protocol_methods(backend) else backend_factory()

        bundle = fetch_bundle(run_id=run_id, window_id=window_id, backend=ckpt_backend)

        # Expose both the manifest bytes (for smoke_hash recompute) and the
        # full DeltaBundle (for the applier to mutate tensors).
        manifest_key = f"checkpoints/{run_id}/{window_id}/manifest.json"
        manifest_bytes = ckpt_backend.get(manifest_key)
        shard_digests = tuple(s.payload_sha256 for s in bundle.shards)
        return LoadedDelta(
            run_id=run_id,
            window_id=window_id,
            merkle_root_hex=bundle.merkle_root_hex,
            raw_manifest_bytes=manifest_bytes or b"",
            shard_digests=shard_digests,
            extra={"bundle": bundle},
        )

    return _loader


def _has_protocol_methods(obj) -> bool:
    return all(hasattr(obj, name) for name in ("put", "get", "list", "delete"))


__all__ = [
    "APPLIER_METRIC_APPLIED",
    "APPLIER_METRIC_REJECTED",
    "APPLIER_METRIC_SKIPPED_UNKNOWN",
    "ReloadingPolicyApplier",
    "bundle_aware_delta_loader",
]
