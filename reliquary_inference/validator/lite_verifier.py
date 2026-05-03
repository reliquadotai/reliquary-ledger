"""Lite-mode validator entry point.

Runs the 6 CPU stages independently against each completion and
borrows the 3 GPU-stage verdicts from a quorum of full validators'
published verdict-bundles. Combines the two signals into a final
verdict that:

* honours independent CPU-stage rejections as vetoes (lite validator
  never accepts a completion its own CPU stages rejected);
* requires ≥ ``lite_quorum`` full validators to have agreed on the
  GPU-stage verdict before accepting;
* abstains (treats as soft non-vote) when no GPU-stage quorum is
  reachable — lite validator does NOT artificially reject in that
  case, since the absence of full-validator signal is not evidence
  of cheating.

Compute footprint: tokenizer + ``AutoConfig`` (no GPU model). Runs
on a 1 vCPU / 1 GB RAM container.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from .mode import (
    DEFAULT_LITE_QUORUM,
    GPU_STAGES,
    gpu_stage_quorum_outcome,
)
from .pipeline import StagePolicy, default_stages, run_pipeline
from .validators.base import RejectReason, StageContext

logger = logging.getLogger(__name__)


# CPU-only stage subset for lite mode. Excludes ``proof``, ``logprob``,
# and ``distribution`` — those are the GPU stages, borrowed from the
# quorum of full validators.
LITE_ENABLED_STAGES = frozenset(
    {"schema", "tokens", "prompt", "termination", "environment", "reward"}
)


def _build_lite_model_stub(model_ref: str):
    """Construct a minimal stub object exposing ``.config`` for the
    CPU stages (tokens, termination) that read it. Avoids loading the
    full GPU model — uses ``transformers.AutoConfig`` (KB of metadata,
    no weights, no GPU)."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover — stage tests stub this
        raise RuntimeError(
            "lite-mode validator requires the `transformers` package "
            "(for tokenizer + AutoConfig only — no GPU is required)."
        ) from exc
    config = AutoConfig.from_pretrained(model_ref)
    return SimpleNamespace(config=config)


def build_lite_context(
    *,
    completion: dict[str, Any],
    task_batch: dict[str, Any],
    seen_nonces: set[tuple[str, int]],
    tokenizer,
    model_stub,
    randomness: str,
    signing_secret: bytes | None = None,
) -> StageContext:
    """Build a ``StageContext`` suitable for lite mode.

    The ``model`` field is the lightweight stub from
    :func:`_build_lite_model_stub` — the CPU stages access only
    ``model.config``, which the stub provides. The ``proof``,
    ``logprob``, and ``distribution`` stages would crash on the
    stub if invoked, but they are excluded by ``StagePolicy`` in
    :func:`verify_completion_lite`.
    """
    return StageContext(
        completion=completion,
        task_batch=task_batch,
        seen_nonces=seen_nonces,
        model=model_stub,
        tokenizer=tokenizer,
        randomness=randomness,
        signing_secret=signing_secret,
        extras={},
    )


def verify_completion_lite(
    *,
    cfg: dict[str, Any],
    completion: dict[str, Any],
    task_batch: dict[str, Any],
    seen_nonces: set[tuple[str, int]],
    tokenizer,
    model_stub,
    peer_full_verdicts_for_completion: list[dict[str, Any]],
    quorum: int = DEFAULT_LITE_QUORUM,
    enabled_stages: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Lite-mode analog of :func:`verifier.verify_completion`.

    Runs the CPU stages independently and combines the result with
    the GPU-stage quorum borrow from
    ``peer_full_verdicts_for_completion`` (the verdicts other full
    validators published for the same ``completion_id``).

    Returns a verdict report dict with the same shape as the full
    verifier's report PLUS a ``"lite_borrow"`` field documenting
    where the GPU-stage verdict came from.
    """
    payload = completion.get("payload") or {}
    report: dict[str, Any] = {
        "accepted": False,
        "hard_fail_reason": None,
        "soft_fail_reason": None,
        "checked_positions": 0,
        "passed_positions": 0,
        "task_source": payload.get("task_source"),
        "copycat_status": "pending",
        "signature_status": "pending",
        "task_binding_summary": {},
        "proof_summary": {},
        "semantic_result": {},
        "semantic_evaluation": {},
        "stage_failed": None,
        "reject_reason": None,
        "soft_flags": [],
        "lite_borrow": {
            "outcome": "pending",
            "metadata": {},
            "verdict_source": "lite",
        },
    }

    context = build_lite_context(
        completion=completion,
        task_batch=task_batch,
        seen_nonces=seen_nonces,
        tokenizer=tokenizer,
        model_stub=model_stub,
        randomness=str(payload.get("randomness", "")),
        signing_secret=cfg.get("signing_secret"),
    )

    # Run only the enabled CPU stages. Hard-fail short-circuits as in
    # full mode; the GPU stages are simply not in the enabled set.
    # Lite mode passes ``LITE_ENABLED_STAGES`` (the 6 CPU stages);
    # mirror mode passes an empty set, in which case no CPU
    # verification is done at all and the GPU-stage borrow decides
    # everything (mirror = pure aggregator).
    stages_to_run = enabled_stages if enabled_stages is not None else LITE_ENABLED_STAGES
    cpu_policy = StagePolicy(enabled_stages=set(stages_to_run))
    cpu_verdict = run_pipeline(default_stages(), context, policy=cpu_policy)

    if not cpu_verdict.accepted:
        # Independent CPU-stage veto. Lite validator rejects without
        # consulting the quorum — the CPU stages caught a defect a
        # quorum-borrow couldn't have absolved.
        report["accepted"] = False
        report["stage_failed"] = cpu_verdict.stage_failed
        if cpu_verdict.reason is not None:
            report["reject_reason"] = cpu_verdict.reason.value
            report["hard_fail_reason"] = cpu_verdict.reason.value
        report["lite_borrow"] = {
            "outcome": "skipped_cpu_veto",
            "metadata": {},
            "verdict_source": "lite_cpu_independent_reject",
        }
        return report

    # CPU stages all passed → consult the quorum on GPU stages.
    outcome, metadata = gpu_stage_quorum_outcome(
        peer_full_verdicts_for_completion, quorum=quorum
    )
    report["lite_borrow"] = {
        "outcome": outcome,
        "metadata": metadata,
        "verdict_source": f"lite_cpu_pass+borrow_{outcome}",
    }

    if outcome == "accept":
        report["accepted"] = True
        # Surface the borrowed proof-position counts so downstream
        # zone-filtering + audit reporting still works. We pick the
        # max checked/passed from the quorum so the lite validator's
        # scorecard reflects "at least one full validator checked N
        # positions" rather than zero.
        max_checked = 0
        max_passed = 0
        for v in peer_full_verdicts_for_completion:
            ps = v.get("proof_summary") or {}
            try:
                max_checked = max(max_checked, int(ps.get("checked_positions", 0)))
                max_passed = max(max_passed, int(ps.get("passed_positions", 0)))
            except (TypeError, ValueError):
                continue
        report["checked_positions"] = max_checked
        report["passed_positions"] = max_passed
        report["proof_summary"] = {
            "checked_positions": max_checked,
            "passed_positions": max_passed,
            "signature_status": "ok",
            "borrowed_from_quorum": True,
        }
        return report

    if outcome == "reject":
        report["accepted"] = False
        report["stage_failed"] = "borrowed_gpu"
        report["hard_fail_reason"] = "borrowed_gpu_reject_quorum"
        report["reject_reason"] = "borrowed_gpu_reject_quorum"
        return report

    # outcome == "abstain"
    report["accepted"] = False
    report["stage_failed"] = "lite_quorum_abstain"
    report["hard_fail_reason"] = "lite_quorum_abstain"
    report["reject_reason"] = "lite_quorum_abstain"
    return report


def index_peer_verdicts_by_completion(
    peer_verdicts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group a list of verdict artifacts by their ``completion_id`` so
    :func:`verify_completion_lite` can fetch all peer verdicts for the
    completion under judgement in a single dict lookup.

    The verdict artifact's payload is the dict shape produced by
    ``service.validate_window`` (one entry per completion judged).
    """
    by_id: dict[str, list[dict[str, Any]]] = {}
    for verdict in peer_verdicts:
        # Accept either an artifact envelope (with .payload) or the
        # raw payload dict — service.validate_window writes verdicts
        # as artifacts.
        payload = (
            verdict.get("payload")
            if isinstance(verdict.get("payload"), dict)
            else verdict
        )
        cid = payload.get("completion_id") if isinstance(payload, dict) else None
        if not cid:
            continue
        by_id.setdefault(str(cid), []).append(payload)
    return by_id


__all__ = [
    "LITE_ENABLED_STAGES",
    "build_lite_context",
    "index_peer_verdicts_by_completion",
    "verify_completion_lite",
]
