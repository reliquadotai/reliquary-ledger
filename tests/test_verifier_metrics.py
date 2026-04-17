"""Tests for the per-stage metrics collector + Prometheus exposition."""

from __future__ import annotations

from reliquary_inference.validator.metrics import StageMetrics, render_prometheus
from reliquary_inference.validator.pipeline import StagePolicy, run_pipeline
from reliquary_inference.validator.validators.base import (
    RejectReason,
    StageContext,
    StageResult,
    accept,
    reject,
    soft_flag,
)


class MockStage:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def check(self, context):
        return self._fn(context)


def _ctx() -> StageContext:
    return StageContext(
        completion={"producer_id": "m1", "payload": {"task_source": "reasoning_tasks"}},
        task_batch={},
        seen_nonces=set(),
    )


def test_metrics_records_accept() -> None:
    metrics = StageMetrics()
    metrics.record("schema", result="accept")
    snap = metrics.snapshot()
    assert snap["stage_results"][("schema", "accept")] == 1
    assert snap["rejections_by_reason"] == {}


def test_metrics_records_reject_with_reason() -> None:
    metrics = StageMetrics()
    metrics.record("tokens", result="reject", reason="tokens_out_of_vocab")
    snap = metrics.snapshot()
    assert snap["stage_results"][("tokens", "reject")] == 1
    assert snap["rejections_by_reason"][("tokens", "tokens_out_of_vocab")] == 1


def test_metrics_records_soft_flag() -> None:
    metrics = StageMetrics()
    metrics.record(
        "distribution",
        result="soft_flag",
        reason="distribution_median_out_of_band",
        soft_fail=True,
    )
    snap = metrics.snapshot()
    assert snap["stage_results"][("distribution", "soft_flag")] == 1
    assert snap["soft_flags_by_stage"][("distribution", "distribution_median_out_of_band")] == 1


def test_metrics_counters_increment() -> None:
    metrics = StageMetrics()
    for _ in range(5):
        metrics.record("schema", result="accept")
    metrics.record("schema", result="reject", reason="schema_missing_field")
    snap = metrics.snapshot()
    assert snap["stage_results"][("schema", "accept")] == 5
    assert snap["stage_results"][("schema", "reject")] == 1


def test_metrics_reset_clears_all_counters() -> None:
    metrics = StageMetrics()
    metrics.record("schema", result="accept")
    metrics.reset()
    assert metrics.snapshot()["stage_results"] == {}


def test_render_prometheus_emits_all_three_series() -> None:
    metrics = StageMetrics()
    metrics.record("schema", result="accept")
    metrics.record("tokens", result="reject", reason="tokens_out_of_vocab")
    metrics.record(
        "distribution",
        result="soft_flag",
        reason="distribution_median_out_of_band",
        soft_fail=True,
    )
    text = render_prometheus(metrics)
    assert "reliquary_verifier_stage_total" in text
    assert "reliquary_verifier_rejections_total" in text
    assert "reliquary_verifier_soft_flags_total" in text
    assert 'stage="schema",result="accept"} 1' in text
    assert 'stage="tokens",reason="tokens_out_of_vocab"} 1' in text
    assert 'stage="distribution",reason="distribution_median_out_of_band"} 1' in text


def test_render_prometheus_is_deterministic_in_ordering() -> None:
    metrics = StageMetrics()
    metrics.record("schema", result="accept")
    metrics.record("tokens", result="accept")
    metrics.record("proof", result="accept")
    text_a = render_prometheus(metrics)
    text_b = render_prometheus(metrics)
    assert text_a == text_b


def test_pipeline_records_stage_metrics_on_accept() -> None:
    metrics = StageMetrics()
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: accept("tokens")),
    ]
    run_pipeline(stages, _ctx(), metrics=metrics)
    snap = metrics.snapshot()
    assert snap["stage_results"][("schema", "accept")] == 1
    assert snap["stage_results"][("tokens", "accept")] == 1


def test_pipeline_records_reject_with_reason_code() -> None:
    metrics = StageMetrics()
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: reject("tokens", RejectReason.TOKENS_OUT_OF_VOCAB)),
    ]
    run_pipeline(stages, _ctx(), metrics=metrics)
    snap = metrics.snapshot()
    assert snap["stage_results"][("schema", "accept")] == 1
    assert snap["stage_results"][("tokens", "reject")] == 1
    assert snap["rejections_by_reason"][("tokens", "tokens_out_of_vocab")] == 1


def test_pipeline_records_soft_flag_and_continues() -> None:
    metrics = StageMetrics()
    stages = [
        MockStage(
            "distribution",
            lambda c: soft_flag(
                "distribution",
                RejectReason.DISTRIBUTION_MEDIAN_OUT_OF_BAND,
                {"median": 1.4},
            ),
        ),
        MockStage("environment", lambda c: accept("environment")),
    ]
    run_pipeline(stages, _ctx(), metrics=metrics)
    snap = metrics.snapshot()
    assert snap["stage_results"][("distribution", "soft_flag")] == 1
    assert snap["stage_results"][("environment", "accept")] == 1
    assert snap["soft_flags_by_stage"][("distribution", "distribution_median_out_of_band")] == 1


def test_pipeline_short_circuits_skip_metrics_for_downstream_stages() -> None:
    metrics = StageMetrics()
    panic = MockStage("proof", lambda c: (_ for _ in ()).throw(RuntimeError("must not be called")))
    stages = [
        MockStage("schema", lambda c: accept("schema")),
        MockStage("tokens", lambda c: reject("tokens", RejectReason.TOKENS_OUT_OF_VOCAB)),
        panic,
    ]
    run_pipeline(stages, _ctx(), metrics=metrics)
    snap = metrics.snapshot()
    assert ("proof", "accept") not in snap["stage_results"]
    assert ("proof", "reject") not in snap["stage_results"]


def test_pipeline_runs_without_metrics_argument() -> None:
    stages = [MockStage("schema", lambda c: accept("schema"))]
    result = run_pipeline(stages, _ctx())
    assert result.accepted is True
