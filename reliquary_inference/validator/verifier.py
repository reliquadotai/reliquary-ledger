"""Entry point for validator-side completion verification.

Delegates to the nine-stage pipeline in ``validator.pipeline`` while
preserving the legacy dict return shape used by callers (service layer,
weight aggregator, copycat pass). New code should consume ``VerdictResult``
directly via ``run_pipeline``.
"""

from __future__ import annotations

from typing import Any

from ..protocol.signatures import verify_commit_signature
from ..shared.modeling import load_model_bundle
from .pipeline import run_pipeline, default_stages, VerdictResult
from .validators.base import RejectReason, StageContext


# Legacy hard_fail_reason string mapping kept stable for downstream consumers
# (scorecard emitters, copycat pass, audit index). New reasons use the enum
# value verbatim; legacy reasons are mapped through this table.
_LEGACY_REASON_MAP: dict[RejectReason, str] = {
    RejectReason.SCHEMA_VERSION_MISMATCH: "invalid_proof_version",
    RejectReason.SCHEMA_DUPLICATE_NONCE: "duplicate_nonce",
    RejectReason.SCHEMA_MISSING_FIELD: "schema_missing_field",
    RejectReason.TOKENS_OUT_OF_VOCAB: "invalid_tokens",
    RejectReason.TOKENS_LENGTH_EXCEEDED: "invalid_tokens",
    RejectReason.PROMPT_BINDING_MISMATCH: "prompt_binding_mismatch",
    RejectReason.PROOF_SKETCH_MISMATCH: "proof_failed",
    RejectReason.PROOF_NO_POSITIONS_CHECKED: "proof_failed",
    RejectReason.ENVIRONMENT_FAILED_EVALUATION: "environment_failed_evaluation",
    RejectReason.SIGNATURE_INVALID: "invalid_signature",
}


def verify_completion(
    *,
    cfg: dict[str, Any],
    completion: dict[str, Any],
    task_batch: dict[str, Any],
    seen_nonces: set[tuple[str, int]],
) -> dict[str, Any]:
    """Verify a single completion artifact against the window's task batch.

    Returns a legacy-shape report dict. Internally runs the nine-stage
    pipeline; see ``validator.pipeline.run_pipeline`` for the new canonical
    return type.
    """
    bundle = load_model_bundle(
        str(task_batch["payload"]["model_ref"]),
        device=str(cfg["device"]),
        dtype_name=str(cfg.get("load_dtype", "auto")),
        require_flash_attention=bool(cfg.get("require_flash_attention", False)),
    )
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    payload = completion["payload"]

    report: dict[str, Any] = {
        "accepted": False,
        "hard_fail_reason": None,
        "soft_fail_reason": None,
        "checked_positions": 0,
        "passed_positions": 0,
        "task_source": payload["task_source"],
        "copycat_status": "pending",
        "signature_status": "pending",
        "task_binding_summary": {},
        "proof_summary": {},
        "semantic_result": {},
        "semantic_evaluation": {},
        "stage_failed": None,
        "reject_reason": None,
        "soft_flags": [],
    }

    signature_ok = verify_commit_signature(
        {
            "tokens": payload["tokens"],
            "commitments": payload["commitments"],
            "signature": payload["signature"],
            "signature_scheme": payload.get("signature_scheme", "local_hmac"),
            "signer_id": payload.get("signer_id"),
            "beacon": {"randomness": payload["randomness"]},
            "model": {"name": payload["model_name"], "layer_index": payload["layer_index"]},
            "proof_version": payload["proof_version"],
        },
        wallet_address=str(payload.get("signer_id") or completion["producer_id"]),
        secret=cfg.get("signing_secret") if payload.get("signature_scheme") == "local_hmac" else None,
    )
    report["signature_status"] = "ok" if signature_ok else "invalid_signature"
    if not signature_ok:
        report["hard_fail_reason"] = "invalid_signature"
        report["reject_reason"] = RejectReason.SIGNATURE_INVALID.value
        report["stage_failed"] = "signature"
        return report

    context = StageContext(
        completion=completion,
        task_batch=task_batch,
        seen_nonces=seen_nonces,
        model=model,
        tokenizer=tokenizer,
        randomness=str(payload["randomness"]),
        signing_secret=cfg.get("signing_secret"),
    )

    verdict = run_pipeline(default_stages(), context)
    _merge_verdict_into_report(verdict, context, report)
    return report


def _merge_verdict_into_report(
    verdict: VerdictResult,
    context: StageContext,
    report: dict[str, Any],
) -> None:
    report["accepted"] = verdict.accepted
    report["stage_failed"] = verdict.stage_failed
    if verdict.reason is not None:
        report["reject_reason"] = verdict.reason.value
        report["hard_fail_reason"] = _LEGACY_REASON_MAP.get(verdict.reason, verdict.reason.value)

    report["soft_flags"] = [
        {"stage": f.stage, "reason": f.reason.value if f.reason else None, "metadata": f.metadata}
        for f in verdict.soft_flags
    ]

    report["task_binding_summary"] = context.extras.get("task_binding_summary", {}) or {}
    report["proof_summary"] = context.extras.get("proof_summary", {}) or {}
    report["semantic_result"] = context.extras.get("semantic_result", {}) or {}
    report["semantic_evaluation"] = context.extras.get("semantic_evaluation", {}) or {}
    report["checked_positions"] = context.extras.get("checked_positions", 0) or 0
    report["passed_positions"] = context.extras.get("passed_positions", 0) or 0

    if not verdict.accepted and verdict.stage_failed == "environment":
        env_reason = context.extras.get("environment_reject_reason")
        if env_reason:
            report["soft_fail_reason"] = env_reason
