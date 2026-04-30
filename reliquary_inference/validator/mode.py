"""Validator mode dispatch + lite-mode quorum-borrow logic.

A Reliquary validator can run in one of three modes:

* ``full`` — canonical 9-stage independent verifier. Runs the proof
  (sketch), logprob, and distribution stages on a GPU. Mainnet weight-
  earners run this mode. Heavy compute, full security signal.
* ``lite`` — CPU-only operator-friendly mode. Runs the 6 CPU stages
  independently (schema, tokens, prompt, termination, environment,
  reward) and borrows the 3 GPU-stage verdicts from a quorum of full
  validators' published verdict-bundles. Independent veto on any CPU
  stage, abstain when no quorum, accept only when CPU passes AND
  ≥ ``lite_quorum`` full validators agreed. Preserves Yuma
  independence — lite validators sign their own verdicts and set
  their own weights.
* ``mirror`` — pure aggregator. Reads the mesh-aggregated weights and
  signs an identical weight extrinsic. No verification at all. The
  operator-friendliest tier; trades signal for ease.

The mode is chosen at startup via ``RELIQUARY_INFERENCE_VALIDATOR_MODE``
or ``--mode`` on the CLI.
"""

from __future__ import annotations

from typing import Iterable


VALIDATOR_MODE_FULL = "full"
VALIDATOR_MODE_LITE = "lite"
VALIDATOR_MODE_MIRROR = "mirror"

VALID_VALIDATOR_MODES = frozenset(
    {VALIDATOR_MODE_FULL, VALIDATOR_MODE_LITE, VALIDATOR_MODE_MIRROR}
)


# Stage-name partitioning. The names match ``validator.validators.base.STAGE_ORDER``.
CPU_STAGES = frozenset(
    {"schema", "tokens", "prompt", "termination", "environment", "reward"}
)
GPU_STAGES = frozenset({"proof", "logprob", "distribution"})


# Default quorum for lite-mode borrowing. With M full validators on the
# mesh, lite needs at least 2 of them to have published a verdict on
# the same completion before it accepts. Tunable via env var.
DEFAULT_LITE_QUORUM = 2

# Hard-fail reasons that indicate a GPU stage rejected the rollout.
# The lite validator borrows reject-on-GPU verdicts via the same path
# as accept verdicts: if ≥ quorum full validators rejected on a GPU
# stage, lite mirrors the reject.
GPU_STAGE_HARD_FAIL_REASONS = frozenset(
    {
        "proof_failed",
        "logprob_drift_exceeded",
        # distribution stage emits soft flags only; not a hard-reject signal.
    }
)


def normalise_mode(raw: str) -> str:
    """Normalise + validate a mode string. Defaults to ``full`` on
    empty/None input. Unknown values raise ``ValueError`` so a typo
    in the env file fails loudly at startup rather than silently
    falling through to ``full`` and giving the operator the wrong
    semantics."""
    if not raw:
        return VALIDATOR_MODE_FULL
    cleaned = str(raw).strip().lower()
    if cleaned not in VALID_VALIDATOR_MODES:
        raise ValueError(
            f"unknown validator mode {raw!r}; expected one of "
            f"{sorted(VALID_VALIDATOR_MODES)}"
        )
    return cleaned


def is_full_verdict(verdict_payload: dict) -> bool:
    """Heuristic: does this verdict come from a validator that ran the
    GPU stages? Identified by the ``proof_summary.checked_positions``
    field — full validators populate this with the actual count of
    sketch positions they checked (≥ 1 for any rollout that made it
    past the schema/tokens stages). Lite validators leave it at 0
    because they don't run the proof stage. This is automatic — no
    self-reporting in the verdict envelope, no trust in the producer's
    label. Anyone publishing a verdict claiming to be ``full`` would
    have to ALSO publish honest sketch-position counts, which they
    can't fake without actually doing the GPU work."""
    summary = verdict_payload.get("proof_summary") or {}
    try:
        return int(summary.get("checked_positions", 0)) > 0
    except (TypeError, ValueError):
        return False


def gpu_stage_quorum_outcome(
    full_verdicts: Iterable[dict],
    *,
    quorum: int,
) -> tuple[str, dict]:
    """Decide the borrowed GPU-stage verdict for a single completion
    given the full validators' verdicts.

    Returns ``(outcome, metadata)`` where outcome is one of:
      * ``"accept"``  — ≥ quorum full validators marked the completion
                       accepted (implies their proof, logprob, and
                       distribution stages all passed).
      * ``"reject"``  — ≥ quorum full validators rejected the
                       completion specifically on a GPU-stage hard-fail
                       reason (proof_failed, logprob_drift_exceeded).
      * ``"abstain"`` — insufficient signal. Lite validator declines
                       to either accept or reject this completion;
                       it's silently dropped from the mesh consensus
                       this validator contributes to.

    The CPU-stage rejections that come from full validators do NOT
    feed into this function — they're independent CPU-stage signals
    the lite validator is verifying anyway. We only look at full
    validators' GPU-stage signal here.
    """
    if quorum < 1:
        raise ValueError(f"quorum must be >= 1, got {quorum}")
    full = [v for v in full_verdicts if is_full_verdict(v)]
    n_full = len(full)
    accepts = sum(1 for v in full if bool(v.get("accepted")))
    rejects_on_gpu = sum(
        1
        for v in full
        if not bool(v.get("accepted"))
        and v.get("hard_fail_reason") in GPU_STAGE_HARD_FAIL_REASONS
    )
    metadata = {
        "n_full_verdicts_seen": n_full,
        "n_accepts": accepts,
        "n_rejects_on_gpu": rejects_on_gpu,
        "quorum_required": quorum,
    }
    if accepts >= quorum:
        return "accept", metadata
    if rejects_on_gpu >= quorum:
        return "reject", metadata
    return "abstain", metadata


__all__ = [
    "CPU_STAGES",
    "DEFAULT_LITE_QUORUM",
    "GPU_STAGES",
    "GPU_STAGE_HARD_FAIL_REASONS",
    "VALID_VALIDATOR_MODES",
    "VALIDATOR_MODE_FULL",
    "VALIDATOR_MODE_LITE",
    "VALIDATOR_MODE_MIRROR",
    "gpu_stage_quorum_outcome",
    "is_full_verdict",
    "normalise_mode",
]
