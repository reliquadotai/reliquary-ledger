"""Unit tests for the validator mode dispatch + lite-mode quorum borrow.

Covers the pure logic in ``validator/mode.py`` + the quorum decision
function. Per-stage lite verification (lite_verifier.verify_completion_lite)
tests live in test_lite_verifier.py — they need a tokenizer + AutoConfig
and run on a host where transformers is installed.
"""

from __future__ import annotations

import pytest

from reliquary_inference.validator.mode import (
    CPU_STAGES,
    DEFAULT_LITE_QUORUM,
    GPU_STAGES,
    GPU_STAGE_HARD_FAIL_REASONS,
    VALID_VALIDATOR_MODES,
    VALIDATOR_MODE_FULL,
    VALIDATOR_MODE_LITE,
    VALIDATOR_MODE_MIRROR,
    gpu_stage_quorum_outcome,
    is_full_verdict,
    normalise_mode,
)


# ────────  CPU + GPU stage partition  ────────


def test_cpu_and_gpu_stages_disjoint() -> None:
    assert CPU_STAGES.isdisjoint(GPU_STAGES)


def test_cpu_and_gpu_stages_cover_canonical_nine() -> None:
    canonical_nine = {
        "schema",
        "tokens",
        "prompt",
        "proof",
        "termination",
        "environment",
        "reward",
        "logprob",
        "distribution",
    }
    assert CPU_STAGES | GPU_STAGES == canonical_nine


def test_gpu_stage_hard_fail_reasons_subset_of_known() -> None:
    # Documents which reject reasons signal a GPU stage failure for
    # the borrow decision. Pinning the set so a future stage rename
    # can't silently drop a reason here.
    assert "proof_failed" in GPU_STAGE_HARD_FAIL_REASONS
    assert "logprob_drift_exceeded" in GPU_STAGE_HARD_FAIL_REASONS


# ────────  normalise_mode  ────────


def test_normalise_mode_defaults_to_full_on_empty() -> None:
    assert normalise_mode("") == VALIDATOR_MODE_FULL
    assert normalise_mode(None) == VALIDATOR_MODE_FULL  # type: ignore[arg-type]


def test_normalise_mode_accepts_canonical_modes() -> None:
    assert normalise_mode("full") == VALIDATOR_MODE_FULL
    assert normalise_mode("lite") == VALIDATOR_MODE_LITE
    assert normalise_mode("mirror") == VALIDATOR_MODE_MIRROR


def test_normalise_mode_normalises_case_and_whitespace() -> None:
    assert normalise_mode(" Full ") == VALIDATOR_MODE_FULL
    assert normalise_mode("LITE") == VALIDATOR_MODE_LITE


def test_normalise_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unknown validator mode"):
        normalise_mode("watchtower")


def test_valid_modes_set_matches_helpers() -> None:
    assert VALID_VALIDATOR_MODES == {
        VALIDATOR_MODE_FULL,
        VALIDATOR_MODE_LITE,
        VALIDATOR_MODE_MIRROR,
    }


# ────────  is_full_verdict  ────────


def test_is_full_verdict_true_when_proof_summary_has_positive_checked() -> None:
    v = {"proof_summary": {"checked_positions": 32}}
    assert is_full_verdict(v) is True


def test_is_full_verdict_false_when_zero_checked_positions() -> None:
    v = {"proof_summary": {"checked_positions": 0}}
    assert is_full_verdict(v) is False


def test_is_full_verdict_false_when_no_proof_summary() -> None:
    assert is_full_verdict({}) is False
    assert is_full_verdict({"proof_summary": None}) is False


def test_is_full_verdict_handles_garbage_safely() -> None:
    assert is_full_verdict({"proof_summary": {"checked_positions": "not-int"}}) is False


# ────────  gpu_stage_quorum_outcome  ────────


def _full_verdict(*, accepted: bool, hard_fail: str | None = None, checked: int = 32) -> dict:
    return {
        "accepted": accepted,
        "hard_fail_reason": hard_fail,
        "proof_summary": {"checked_positions": checked, "passed_positions": checked},
    }


def _lite_verdict(*, accepted: bool, hard_fail: str | None = None) -> dict:
    """Lite validators publish verdicts with checked_positions=0 (no
    proof stage). They must NOT count toward the GPU-stage quorum."""
    return {
        "accepted": accepted,
        "hard_fail_reason": hard_fail,
        "proof_summary": {"checked_positions": 0, "passed_positions": 0},
    }


def test_quorum_accept_on_two_full_accepts() -> None:
    verdicts = [_full_verdict(accepted=True), _full_verdict(accepted=True)]
    outcome, meta = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "accept"
    assert meta["n_accepts"] == 2


def test_quorum_reject_on_two_full_rejects_on_proof() -> None:
    verdicts = [
        _full_verdict(accepted=False, hard_fail="proof_failed"),
        _full_verdict(accepted=False, hard_fail="proof_failed"),
    ]
    outcome, meta = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "reject"
    assert meta["n_rejects_on_gpu"] == 2


def test_quorum_abstains_on_one_accept_one_reject() -> None:
    verdicts = [
        _full_verdict(accepted=True),
        _full_verdict(accepted=False, hard_fail="proof_failed"),
    ]
    outcome, _ = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "abstain"


def test_quorum_abstains_when_only_one_full_verdict_seen() -> None:
    verdicts = [_full_verdict(accepted=True)]
    outcome, _ = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "abstain"


def test_quorum_skips_lite_verdicts_in_count() -> None:
    """A lite verdict cannot be a quorum member — it doesn't have GPU
    signal. Two lites + one full = abstain (only 1 full counted)."""
    verdicts = [
        _lite_verdict(accepted=True),
        _lite_verdict(accepted=True),
        _full_verdict(accepted=True),
    ]
    outcome, meta = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "abstain"
    assert meta["n_full_verdicts_seen"] == 1
    assert meta["n_accepts"] == 1


def test_quorum_ignores_cpu_stage_rejections_for_gpu_decision() -> None:
    """A full validator that rejected on schema (CPU stage) does NOT
    contribute a GPU-stage signal. It's not counted as accept OR
    reject for the borrow."""
    verdicts = [
        _full_verdict(accepted=False, hard_fail="schema_missing_field"),
        _full_verdict(accepted=False, hard_fail="schema_missing_field"),
    ]
    outcome, meta = gpu_stage_quorum_outcome(verdicts, quorum=2)
    # Neither accept nor GPU-stage reject — abstain.
    assert outcome == "abstain"
    assert meta["n_accepts"] == 0
    assert meta["n_rejects_on_gpu"] == 0


def test_quorum_rejects_when_majority_split_but_gpu_rejects_meet_quorum() -> None:
    """3 full validators: 2 reject on GPU stage, 1 accepted. Quorum=2
    on GPU rejection wins; outcome is reject (not abstain) because the
    GPU-stage signal is conclusive."""
    verdicts = [
        _full_verdict(accepted=False, hard_fail="proof_failed"),
        _full_verdict(accepted=False, hard_fail="logprob_drift_exceeded"),
        _full_verdict(accepted=True),
    ]
    outcome, meta = gpu_stage_quorum_outcome(verdicts, quorum=2)
    assert outcome == "reject"
    assert meta["n_rejects_on_gpu"] == 2


def test_quorum_default_value_matches_constant() -> None:
    assert DEFAULT_LITE_QUORUM == 2


def test_quorum_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="quorum must be"):
        gpu_stage_quorum_outcome([_full_verdict(accepted=True)], quorum=0)


# ────────  index_peer_verdicts_by_completion (in lite_verifier)  ────────


def test_index_peer_verdicts_groups_by_completion_id() -> None:
    from reliquary_inference.validator.lite_verifier import (
        index_peer_verdicts_by_completion,
    )

    verdicts = [
        {"payload": {"completion_id": "c1", "accepted": True}},
        {"payload": {"completion_id": "c2", "accepted": False}},
        {"payload": {"completion_id": "c1", "accepted": True}},
    ]
    grouped = index_peer_verdicts_by_completion(verdicts)
    assert sorted(grouped.keys()) == ["c1", "c2"]
    assert len(grouped["c1"]) == 2
    assert len(grouped["c2"]) == 1


def test_index_peer_verdicts_handles_raw_payload_too() -> None:
    """Some callers pass the inner payload dict directly rather than
    the full artifact envelope. Both shapes should work."""
    from reliquary_inference.validator.lite_verifier import (
        index_peer_verdicts_by_completion,
    )

    raw = [{"completion_id": "c1", "accepted": True}]
    grouped = index_peer_verdicts_by_completion(raw)
    assert "c1" in grouped


def test_index_peer_verdicts_skips_entries_without_completion_id() -> None:
    from reliquary_inference.validator.lite_verifier import (
        index_peer_verdicts_by_completion,
    )

    verdicts = [{"payload": {"foo": "bar"}}]
    grouped = index_peer_verdicts_by_completion(verdicts)
    assert grouped == {}


# ────────  mirror mode (degenerate lite, zero CPU stages)  ────────


def test_mirror_mode_recognised_by_normalise() -> None:
    assert normalise_mode("mirror") == VALIDATOR_MODE_MIRROR
    assert normalise_mode(" MIRROR ") == VALIDATOR_MODE_MIRROR


def test_mirror_pipeline_accepts_empty_enabled_stages() -> None:
    """Verify the pipeline accepts an empty enabled_stages set (which
    is how mirror mode disables all CPU verification). Without an
    explicit test here, a future StagePolicy refactor could
    accidentally introduce a "must enable at least one stage" check
    and silently break mirror.
    """
    from reliquary_inference.validator.pipeline import (
        StagePolicy,
        default_stages,
        run_pipeline,
    )
    from reliquary_inference.validator.validators.base import StageContext

    # Empty completion / task_batch — pipeline should run zero stages
    # and return accepted=True without touching either field.
    ctx = StageContext(
        completion={"payload": {"task_id": "t", "tokens": [], "commitments": [],
                                "randomness": "00", "model_name": "x",
                                "layer_index": 0, "proof_version": "v",
                                "signature": "", "signature_scheme": "local_hmac",
                                "task_source": "math", "sample_index": 0,
                                "miner_id": "m", "producer_role": "miner"}},
        task_batch={"payload": {"tasks": [], "model_ref": "x", "model_name": "x"}},
        seen_nonces=set(),
        model=None,
        tokenizer=None,
    )
    verdict = run_pipeline(default_stages(), ctx, policy=StagePolicy(enabled_stages=set()))
    assert verdict.accepted is True
    assert verdict.stage_results == []
