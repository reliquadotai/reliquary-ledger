"""Per-miner scoreboard metrics for the Epic 5 miner dashboard.

Aggregates ``VerdictArtifact`` outcomes by miner hotkey into Prometheus
counters + gauges so the Grafana "Miner scoreboard" dashboard can render
per-hotkey acceptance rate, per-score breakdowns, and rejection reason
histograms.

Subnet size is bounded (typically <= 256 miners), so per-miner label
cardinality is fine for Prometheus under the usual Reliquary deployment
envelope. Operators running beyond that envelope should aggregate at the
dashboard layer instead.

Spec companion: derives from `spec-validator-mesh.md` invariants.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from reliquary_inference.validator.mesh import VerdictArtifact


MINER_ACCEPTANCE_WINDOW = 32


_VERDICTS_TOTAL = "reliquary_miner_verdicts_total"
_ACCEPTANCE_RATE = "reliquary_miner_acceptance_rate"
_LAST_SCORE = "reliquary_miner_last_score"
_REJECTION_REASONS_TOTAL = "reliquary_miner_rejection_reasons_total"
_LAST_WINDOW_SEEN = "reliquary_miner_last_window_seen"


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


@dataclass
class MinerScoreboard:
    """Thread-safe per-miner-hotkey verdict aggregator."""

    verdicts: dict[tuple[str, str], int] = field(default_factory=dict)
    rejections: dict[tuple[str, str], int] = field(default_factory=dict)
    outcome_window: dict[str, Deque[bool]] = field(default_factory=dict)
    acceptance_rate: dict[str, float] = field(default_factory=dict)
    last_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    last_window_seen: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_verdict(self, verdict: VerdictArtifact) -> None:
        """Fold one verdict into the scoreboard."""
        miner = verdict.miner_hotkey
        if not miner:
            return

        outcome = "accepted" if verdict.accepted else "rejected"
        with self._lock:
            key = (miner, outcome)
            self.verdicts[key] = self.verdicts.get(key, 0) + 1

            if not verdict.accepted and verdict.reject_reason:
                reject_key = (miner, verdict.reject_reason)
                self.rejections[reject_key] = self.rejections.get(reject_key, 0) + 1

            window = self.outcome_window.setdefault(
                miner, deque(maxlen=MINER_ACCEPTANCE_WINDOW)
            )
            window.append(verdict.accepted)
            accepted_in_window = sum(1 for x in window if x)
            self.acceptance_rate[miner] = accepted_in_window / len(window)

            for score_name, score_value in verdict.scores.items():
                self.last_scores[(miner, score_name)] = float(score_value)

            self.last_window_seen[miner] = int(verdict.window_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "verdicts": dict(self.verdicts),
                "rejections": dict(self.rejections),
                "acceptance_rate": dict(self.acceptance_rate),
                "last_scores": dict(self.last_scores),
                "last_window_seen": dict(self.last_window_seen),
            }

    def reset(self) -> None:
        with self._lock:
            self.verdicts.clear()
            self.rejections.clear()
            self.outcome_window.clear()
            self.acceptance_rate.clear()
            self.last_scores.clear()
            self.last_window_seen.clear()


def render_miner_scoreboard_prometheus(scoreboard: MinerScoreboard) -> str:
    """Render ``scoreboard`` as Prometheus 0.0.4 text exposition."""
    snap = scoreboard.snapshot()
    lines: list[str] = []

    lines.append(f"# HELP {_VERDICTS_TOTAL} Verdict outcomes per miner hotkey.")
    lines.append(f"# TYPE {_VERDICTS_TOTAL} counter")
    for (miner, outcome), count in sorted(snap["verdicts"].items()):
        lines.append(
            f'{_VERDICTS_TOTAL}{{miner_hotkey="{_escape(miner)}",outcome="{outcome}"}} {count}'
        )

    lines.append(
        f"# HELP {_ACCEPTANCE_RATE} Rolling acceptance rate per miner over the last {MINER_ACCEPTANCE_WINDOW} verdicts."
    )
    lines.append(f"# TYPE {_ACCEPTANCE_RATE} gauge")
    for miner, rate in sorted(snap["acceptance_rate"].items()):
        lines.append(
            f'{_ACCEPTANCE_RATE}{{miner_hotkey="{_escape(miner)}"}} {rate}'
        )

    lines.append(
        f"# HELP {_LAST_SCORE} Most recent per-metric score per miner."
    )
    lines.append(f"# TYPE {_LAST_SCORE} gauge")
    for (miner, metric), value in sorted(snap["last_scores"].items()):
        lines.append(
            f'{_LAST_SCORE}{{miner_hotkey="{_escape(miner)}",metric="{_escape(metric)}"}} {value}'
        )

    lines.append(
        f"# HELP {_REJECTION_REASONS_TOTAL} Rejection reasons per miner hotkey."
    )
    lines.append(f"# TYPE {_REJECTION_REASONS_TOTAL} counter")
    for (miner, reason), count in sorted(snap["rejections"].items()):
        lines.append(
            f'{_REJECTION_REASONS_TOTAL}{{miner_hotkey="{_escape(miner)}",reason="{_escape(reason)}"}} {count}'
        )

    lines.append(
        f"# HELP {_LAST_WINDOW_SEEN} Most recent window_id a miner appeared in."
    )
    lines.append(f"# TYPE {_LAST_WINDOW_SEEN} gauge")
    for miner, window_id in sorted(snap["last_window_seen"].items()):
        lines.append(
            f'{_LAST_WINDOW_SEEN}{{miner_hotkey="{_escape(miner)}"}} {window_id}'
        )

    return "\n".join(lines) + "\n"


__all__ = [
    "MINER_ACCEPTANCE_WINDOW",
    "MinerScoreboard",
    "render_miner_scoreboard_prometheus",
]
