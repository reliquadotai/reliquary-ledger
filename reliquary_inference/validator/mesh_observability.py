"""Prometheus metrics for the validator mesh aggregator.

Translates ``MeshAggregationReport`` outcomes into Prometheus counters and
gauges for the validator-mesh Grafana dashboard. The aggregator itself
does not import this module; the operator wires observation by calling
``MeshMetrics.record_window(report)`` after each aggregation.

Spec: private/reliquary-plan/notes/spec-mesh-observability.md.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from reliquary_inference.validator.mesh import MeshAggregationReport


_COMPLETIONS_TOTAL = "reliquary_mesh_completions_total"
_DISAGREEMENT_RATE = "reliquary_mesh_validator_disagreement_rate"
_GATED_TOTAL = "reliquary_mesh_validators_gated_total"
_MISSING_TOTAL = "reliquary_mesh_validators_missing_total"
_LAST_WINDOW = "reliquary_mesh_last_window_observed"


def _escape_label_value(value: str) -> str:
    """Escape a Prometheus label value per the exposition format."""
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


@dataclass
class MeshMetrics:
    """Thread-safe collector over aggregator outcomes."""

    completions: dict[tuple[int, str], int] = field(default_factory=dict)
    disagreement_rate: dict[str, float] = field(default_factory=dict)
    gated: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    last_window: int | None = None
    _seen_windows: set[int] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_window(self, report: MeshAggregationReport) -> None:
        """Fold one aggregation report into the collector; idempotent per window_id."""
        with self._lock:
            if report.window_id in self._seen_windows:
                return
            self._seen_windows.add(report.window_id)

            for verdict in report.median_verdicts.values():
                outcome = "accepted" if verdict.accepted else "rejected"
                key = (report.window_id, outcome)
                self.completions[key] = self.completions.get(key, 0) + 1

            for hotkey, rate in report.validator_disagreement_rates.items():
                self.disagreement_rate[hotkey] = float(rate)

            for hotkey in report.gated_validators:
                self.gated[hotkey] = self.gated.get(hotkey, 0) + 1

            for hotkey in report.missing_validators:
                self.missing[hotkey] = self.missing.get(hotkey, 0) + 1

            self.last_window = report.window_id

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "completions": dict(self.completions),
                "disagreement_rate": dict(self.disagreement_rate),
                "gated": dict(self.gated),
                "missing": dict(self.missing),
                "last_window": self.last_window,
            }

    def reset(self) -> None:
        with self._lock:
            self.completions.clear()
            self.disagreement_rate.clear()
            self.gated.clear()
            self.missing.clear()
            self._seen_windows.clear()
            self.last_window = None


def render_mesh_prometheus(metrics: MeshMetrics) -> str:
    """Render ``metrics`` as Prometheus 0.0.4 text exposition."""
    snap = metrics.snapshot()
    lines: list[str] = []

    lines.append(
        f"# HELP {_COMPLETIONS_TOTAL} Mesh aggregator outcome per completion by window."
    )
    lines.append(f"# TYPE {_COMPLETIONS_TOTAL} counter")
    for (window_id, outcome), value in sorted(snap["completions"].items()):
        lines.append(
            f'{_COMPLETIONS_TOTAL}{{window_id="{window_id}",outcome="{outcome}"}} {value}'
        )

    lines.append(
        f"# HELP {_DISAGREEMENT_RATE} Latest disagreement rate per validator hotkey."
    )
    lines.append(f"# TYPE {_DISAGREEMENT_RATE} gauge")
    for hotkey, rate in sorted(snap["disagreement_rate"].items()):
        label = _escape_label_value(hotkey)
        lines.append(f'{_DISAGREEMENT_RATE}{{validator_hotkey="{label}"}} {rate}')

    lines.append(
        f"# HELP {_GATED_TOTAL} Windows in which a validator was gated."
    )
    lines.append(f"# TYPE {_GATED_TOTAL} counter")
    for hotkey, count in sorted(snap["gated"].items()):
        label = _escape_label_value(hotkey)
        lines.append(f'{_GATED_TOTAL}{{validator_hotkey="{label}"}} {count}')

    lines.append(
        f"# HELP {_MISSING_TOTAL} Windows in which a validator was missing."
    )
    lines.append(f"# TYPE {_MISSING_TOTAL} counter")
    for hotkey, count in sorted(snap["missing"].items()):
        label = _escape_label_value(hotkey)
        lines.append(f'{_MISSING_TOTAL}{{validator_hotkey="{label}"}} {count}')

    lines.append(f"# HELP {_LAST_WINDOW} Most recent window_id folded into the collector.")
    lines.append(f"# TYPE {_LAST_WINDOW} gauge")
    last_window = snap["last_window"]
    if last_window is not None:
        lines.append(f"{_LAST_WINDOW} {last_window}")

    return "\n".join(lines) + "\n"


__all__ = [
    "MeshMetrics",
    "render_mesh_prometheus",
]
