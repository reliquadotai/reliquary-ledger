"""Tests for the nine-stage pipeline runner.

Uses mock stages so the pipeline plumbing can be verified independently of
the individual stage implementations.
"""

from __future__ import annotations

from typing import Callable

from reliquary_inference.validator.pipeline import (
    StagePolicy,
    default_stages,
    run_pipeline,
)
from reliquary_inference.validator.validators.base import (
    RejectReason,
    StageContext,
    StageResult,
    accept,
    reject,
    soft_flag,
)


class MockStage:
    def __init__(self, name: str, fn: Callable[[StageContext], StageResult]):
        self.name = name
        self._fn = fn
        self.called = 0

    def check(self, context: StageContext) -> StageResult:
        self.called += 1
        return self._fn(context)


def _ctx() -> StageContext:
    return StageContext(
        completion={"producer_id": "m1", "payload": {"task_source": "reasoning_tasks"}},
        task_batch={},
        seen_nonces=set(),
    )


def test_all_pass_pipeline_accepts() -> None:
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: accept("tokens")),
        MockStage("termination", lambda c: accept("termination")),
    ]
    verdict = run_pipeline(stages, _ctx())

    assert verdict.accepted is True
    assert verdict.stage_failed is None
    assert verdict.reason is None
    assert [r.stage for r in verdict.stage_results] == ["schema", "tokens", "termination"]
    assert all(s.called == 1 for s in stages)


def test_hard_fail_short_circuits_at_failing_stage() -> None:
    panic_stage = MockStage(
        "proof",
        lambda c: (_ for _ in ()).throw(RuntimeError("must not be called")),
    )
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: reject("tokens", RejectReason.TOKENS_OUT_OF_VOCAB)),
        panic_stage,
    ]
    verdict = run_pipeline(stages, _ctx())

    assert verdict.accepted is False
    assert verdict.stage_failed == "tokens"
    assert verdict.reason is RejectReason.TOKENS_OUT_OF_VOCAB
    assert panic_stage.called == 0


def test_soft_fail_does_not_halt_pipeline() -> None:
    later_stage = MockStage("environment", lambda c: accept("environment"))
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage(
            "distribution",
            lambda c: soft_flag(
                "distribution", RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND, {"median": 1.4}
            ),
        ),
        later_stage,
    ]
    verdict = run_pipeline(stages, _ctx())

    assert verdict.accepted is True
    assert later_stage.called == 1
    assert len(verdict.soft_flags) == 1
    assert verdict.soft_flags[0].reason is RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND


def test_exception_inside_stage_is_converted_to_hard_fail() -> None:
    stages = [
        MockStage("schema", lambda c: (_ for _ in ()).throw(ValueError("boom"))),
        MockStage("tokens", lambda c: accept("tokens")),
    ]
    verdict = run_pipeline(stages, _ctx())

    assert verdict.accepted is False
    assert verdict.stage_failed == "schema"
    assert verdict.reason is RejectReason.PIPELINE_STAGE_EXCEPTION
    assert "ValueError" in verdict.stage_results[0].metadata["exc"]


def test_policy_can_disable_stages() -> None:
    panic_stage = MockStage(
        "distribution",
        lambda c: (_ for _ in ()).throw(RuntimeError("must be skipped")),
    )
    policy = StagePolicy(enabled_stages={"schema", "tokens"})
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: accept("tokens")),
        panic_stage,
    ]
    verdict = run_pipeline(stages, _ctx(), policy=policy)

    assert verdict.accepted is True
    assert panic_stage.called == 0


def test_all_rejection_reasons_are_enum_members() -> None:
    stages = [MockStage("schema", lambda c: reject("schema", RejectReason.SCHEMA_MISSING_FIELD))]
    verdict = run_pipeline(stages, _ctx())
    assert isinstance(verdict.reason, RejectReason)


def test_default_stages_contain_all_nine_in_canonical_order() -> None:
    stages = default_stages()
    assert [s.name for s in stages] == [
        "schema",
        "tokens",
        "prompt",
        "proof",
        "termination",
        "environment",
        "reward",
        "logprob",
        "distribution",
    ]


def test_summary_metadata_includes_producer_and_task_source() -> None:
    stages = [MockStage("schema", lambda c: accept("schema"))]
    verdict = run_pipeline(stages, _ctx())
    assert verdict.metadata["producer_id"] == "m1"
    assert verdict.metadata["task_source"] == "reasoning_tasks"
