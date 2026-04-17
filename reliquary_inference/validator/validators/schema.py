"""Stage 1: schema check.

Verifies the completion artifact has all required top-level and payload
fields, carries our ledger protocol version, and does not replay a nonce
already seen in this window.
"""

from __future__ import annotations

from typing import Any

from ...protocol.constants import LEDGER_PROOF_VERSION
from .base import RejectReason, StageContext, StageResult, accept, reject


REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = ("producer_id", "payload")
REQUIRED_PAYLOAD_FIELDS: tuple[str, ...] = (
    "proof_version",
    "tokens",
    "commitments",
    "signature",
    "randomness",
    "nonce",
    "model_name",
    "layer_index",
    "task_source",
)


class SchemaStage:
    name: str = "schema"

    def check(self, context: StageContext) -> StageResult:
        missing_top = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in context.completion]
        if missing_top:
            return reject(
                self.name,
                RejectReason.SCHEMA_MISSING_FIELD,
                {"missing": missing_top, "scope": "top_level"},
            )

        payload = context.payload
        missing_payload = [f for f in REQUIRED_PAYLOAD_FIELDS if f not in payload]
        if missing_payload:
            return reject(
                self.name,
                RejectReason.SCHEMA_MISSING_FIELD,
                {"missing": missing_payload, "scope": "payload"},
            )

        version = payload.get("proof_version")
        if version != LEDGER_PROOF_VERSION:
            return reject(
                self.name,
                RejectReason.SCHEMA_VERSION_MISMATCH,
                {"expected": LEDGER_PROOF_VERSION, "observed": version},
            )

        nonce_key = _nonce_key(context)
        if nonce_key in context.seen_nonces:
            return reject(
                self.name,
                RejectReason.SCHEMA_DUPLICATE_NONCE,
                {"producer_id": nonce_key[0], "nonce": nonce_key[1]},
            )
        context.seen_nonces.add(nonce_key)

        return accept(self.name, {"proof_version": version, "nonce": nonce_key[1]})


def _nonce_key(context: StageContext) -> tuple[str, int]:
    payload = context.payload
    nonce_raw = payload.get("nonce")
    try:
        nonce_int = int(nonce_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        nonce_int = -1
    return (context.producer_id, nonce_int)
