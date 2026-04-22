"""Acceptance tests for the OpenTelemetry tracing facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from reliquary_inference.observability import tracing

pytestmark = pytest.mark.skipif(
    not tracing.is_otel_available(),
    reason="opentelemetry not installed in this environment",
)


@dataclass
class _StageResult:
    result: str
    reject_reason: str | None = None


@pytest.fixture
def recorder() -> tracing.InMemorySpanRecorder:
    tracing._reset_for_tests()
    rec = tracing.InMemorySpanRecorder()
    installed = tracing.configure_tracing("reliquary-test", exporters=[rec])
    assert installed, "expected fresh configuration"
    yield rec
    tracing._reset_for_tests()


def _flush() -> None:
    # Force any BatchSpanProcessor to flush so tests see spans deterministically.
    import opentelemetry.trace as trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


# ---------------------------------------------------------------------------
# get_tracer / configure_tracing
# ---------------------------------------------------------------------------


def test_get_tracer_returns_sdk_tracer_when_otel_present() -> None:
    tracer = tracing.get_tracer("reliquary-test")
    # Real SDK tracers implement start_as_current_span via __call__ with ctx mgr.
    assert hasattr(tracer, "start_as_current_span")


def test_configure_tracing_is_idempotent(recorder: tracing.InMemorySpanRecorder) -> None:
    second = tracing.configure_tracing("reliquary-test", exporters=[recorder])
    assert second is False


# ---------------------------------------------------------------------------
# InMemorySpanRecorder
# ---------------------------------------------------------------------------


def test_span_recorder_captures_span_name_and_attributes(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    tracer = tracing.get_tracer("reliquary-test")
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("reliquary.completion_id", "c-42")

    _flush()
    assert any(
        s["name"] == "test-span" and s["attributes"].get("reliquary.completion_id") == "c-42"
        for s in recorder.spans
    )


def test_span_recorder_clear_empties(recorder: tracing.InMemorySpanRecorder) -> None:
    tracer = tracing.get_tracer("reliquary-test")
    with tracer.start_as_current_span("one"):
        pass
    _flush()
    assert len(recorder.spans) >= 1
    recorder.clear()
    assert recorder.spans == []


# ---------------------------------------------------------------------------
# traced_stage
# ---------------------------------------------------------------------------


def test_traced_stage_emits_result_attribute(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_stage("tokens")
    def stage(ctx: dict) -> _StageResult:
        return _StageResult(result="accept")

    stage({"ctx": 1})
    _flush()
    stage_spans = [s for s in recorder.spans if s["name"] == "verifier.stage.tokens"]
    assert stage_spans
    assert stage_spans[-1]["attributes"].get("reliquary.stage") == "tokens"
    assert stage_spans[-1]["attributes"].get("reliquary.result") == "accept"


def test_traced_stage_emits_reject_reason_on_reject(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_stage("proof")
    def stage(ctx: dict) -> _StageResult:
        return _StageResult(result="reject", reject_reason="sketch_mismatch")

    stage({})
    _flush()
    reject_spans = [s for s in recorder.spans if s["name"] == "verifier.stage.proof"]
    assert reject_spans
    assert reject_spans[-1]["attributes"].get("reliquary.result") == "reject"
    assert reject_spans[-1]["attributes"].get("reliquary.reject_reason") == "sketch_mismatch"


def test_traced_stage_records_error_type_on_exception(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_stage("reward")
    def stage(ctx: dict) -> _StageResult:
        raise ValueError("bad thing")

    with pytest.raises(ValueError):
        stage({})
    _flush()
    error_spans = [s for s in recorder.spans if s["name"] == "verifier.stage.reward"]
    assert error_spans
    attrs = error_spans[-1]["attributes"]
    assert attrs.get("reliquary.result") == "error"
    assert attrs.get("reliquary.error_type") == "ValueError"


def test_nested_spans_share_trace_id(recorder: tracing.InMemorySpanRecorder) -> None:
    tracer = tracing.get_tracer("reliquary-test")

    @tracing.traced_stage("schema")
    def stage(ctx: dict) -> _StageResult:
        return _StageResult(result="accept")

    with tracer.start_as_current_span("window.process") as parent:
        if hasattr(parent, "get_span_context"):
            parent.get_span_context().trace_id
        stage({})

    _flush()
    stage_spans = [s for s in recorder.spans if s["name"] == "verifier.stage.schema"]
    window_spans = [s for s in recorder.spans if s["name"] == "window.process"]
    assert stage_spans
    assert window_spans
    # They share a trace_id by virtue of being nested in the same context.
    assert stage_spans[-1]["trace_id"] == window_spans[-1]["trace_id"]


def test_traced_stage_result_value_is_stringified(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    class _EnumLike:
        def __str__(self) -> str:
            return "soft_flag"

    @tracing.traced_stage("distribution")
    def stage(ctx: dict) -> Any:
        return _StageResult(result=_EnumLike())

    stage({})
    _flush()
    spans = [s for s in recorder.spans if s["name"] == "verifier.stage.distribution"]
    assert spans
    assert spans[-1]["attributes"].get("reliquary.result") == "soft_flag"


# ---------------------------------------------------------------------------
# No-op tracer invariants (exercised via _NoOpTracer directly)
# ---------------------------------------------------------------------------


def test_noop_tracer_supports_start_as_current_span_without_config() -> None:
    # Direct no-op even when OTEL present — tests the noop path's ergonomics.
    noop = tracing._NoOpTracer("test")
    with noop.start_as_current_span("x") as span:
        span.set_attribute("k", "v")
    # No assertions — we're verifying no exception.


def test_noop_span_supports_end_and_set_status() -> None:
    span = tracing._NoOpSpan()
    span.set_attribute("a", 1)
    span.set_status("OK")
    span.record_exception(RuntimeError("x"))
    span.end()
