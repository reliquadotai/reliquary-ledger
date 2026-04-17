"""Per-stage counters + Prometheus exposition for the nine-stage verifier.

Lightweight in-memory metrics: a ``StageMetrics`` collector tracks
``reliquary_verifier_stage_total{stage, result}`` and
``reliquary_verifier_rejections_total{stage, reason}``. The pipeline
increments these on every stage run; the existing metrics HTTP endpoint
can render them as Prometheus text via :func:`render_prometheus`.

No external dependencies — the ``prometheus_client`` package is nice to
have but not required here. Everything is a plain Python dict that
serializes to the standard exposition format.

Spec: private/reliquary-plan/notes/spec-nine-stage-verifier.md invariant 8.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


_STAGE_TOTAL = "reliquary_verifier_stage_total"
_REJECTIONS_TOTAL = "reliquary_verifier_rejections_total"
_SOFT_FLAGS_TOTAL = "reliquary_verifier_soft_flags_total"


@dataclass
class StageMetrics:
    """Thread-safe counter for verifier pipeline observability."""

    stage_results: dict[tuple[str, str], int] = field(default_factory=dict)
    rejections_by_reason: dict[tuple[str, str], int] = field(default_factory=dict)
    soft_flags_by_stage: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        stage: str,
        *,
        result: str,
        reason: str | None = None,
        soft_fail: bool = False,
    ) -> None:
        """Record one stage outcome.

        ``result`` is one of ``"accept"``, ``"reject"``, ``"soft_flag"``.
        ``reason`` is the RejectReason.value string on reject / soft_flag
        outcomes; None on accepts.
        """
        with self._lock:
            self.stage_results[(stage, result)] = (
                self.stage_results.get((stage, result), 0) + 1
            )
            if result == "reject" and reason:
                self.rejections_by_reason[(stage, reason)] = (
                    self.rejections_by_reason.get((stage, reason), 0) + 1
                )
            if soft_fail and reason:
                self.soft_flags_by_stage[(stage, reason)] = (
                    self.soft_flags_by_stage.get((stage, reason), 0) + 1
                )

    def snapshot(self) -> dict[str, dict[tuple[str, str], int]]:
        with self._lock:
            return {
                "stage_results": dict(self.stage_results),
                "rejections_by_reason": dict(self.rejections_by_reason),
                "soft_flags_by_stage": dict(self.soft_flags_by_stage),
            }

    def reset(self) -> None:
        with self._lock:
            self.stage_results.clear()
            self.rejections_by_reason.clear()
            self.soft_flags_by_stage.clear()


def render_prometheus(metrics: StageMetrics) -> str:
    """Render ``metrics`` as a Prometheus 0.0.4 text-exposition document."""
    lines: list[str] = []
    snap = metrics.snapshot()

    lines.append(f"# HELP {_STAGE_TOTAL} Verifier stage executions by result.")
    lines.append(f"# TYPE {_STAGE_TOTAL} counter")
    for (stage, result), value in sorted(snap["stage_results"].items()):
        lines.append(f'{_STAGE_TOTAL}{{stage="{stage}",result="{result}"}} {value}')

    lines.append(f"# HELP {_REJECTIONS_TOTAL} Verifier hard rejections by stage and reason code.")
    lines.append(f"# TYPE {_REJECTIONS_TOTAL} counter")
    for (stage, reason), value in sorted(snap["rejections_by_reason"].items()):
        lines.append(f'{_REJECTIONS_TOTAL}{{stage="{stage}",reason="{reason}"}} {value}')

    lines.append(f"# HELP {_SOFT_FLAGS_TOTAL} Verifier soft flags by stage and reason code.")
    lines.append(f"# TYPE {_SOFT_FLAGS_TOTAL} counter")
    for (stage, reason), value in sorted(snap["soft_flags_by_stage"].items()):
        lines.append(f'{_SOFT_FLAGS_TOTAL}{{stage="{stage}",reason="{reason}"}} {value}')

    return "\n".join(lines) + "\n"
