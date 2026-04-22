"""Acceptance tests for mesh observability metrics + Grafana dashboard.

Covers the 12 acceptance cases from
``private/reliquary-plan/notes/spec-mesh-observability.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from reliquary_inference.validator.mesh import (
    MedianVerdict,
    MeshAggregationReport,
)
from reliquary_inference.validator.mesh_observability import (
    MeshMetrics,
    render_mesh_prometheus,
)

_DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "deploy"
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "reliquary-validator-mesh.json"
)


def _verdict(completion_id: str, accepted: bool) -> MedianVerdict:
    return MedianVerdict(
        completion_id=completion_id,
        accepted=accepted,
        acceptance_score=1.0 if accepted else 0.0,
        median_scores={"reward": 1.0 if accepted else 0.0},
        participating_validators=["mesh-A"],
        outlier_validators=[],
        quorum_satisfied=True,
    )


def _report(
    *,
    window_id: int,
    accepted: int = 0,
    rejected: int = 0,
    disagreement: dict[str, float] | None = None,
    gated: list[str] | None = None,
    missing: list[str] | None = None,
) -> MeshAggregationReport:
    median_verdicts: dict[str, MedianVerdict] = {}
    for i in range(accepted):
        median_verdicts[f"c-acc-{i}"] = _verdict(f"c-acc-{i}", True)
    for i in range(rejected):
        median_verdicts[f"c-rej-{i}"] = _verdict(f"c-rej-{i}", False)
    return MeshAggregationReport(
        window_id=window_id,
        median_verdicts=median_verdicts,
        validator_disagreement_rates=disagreement or {},
        missing_validators=list(missing or []),
        gated_validators=list(gated or []),
    )


def test_record_empty_report() -> None:
    metrics = MeshMetrics()
    metrics.record_window(_report(window_id=42))
    output = render_mesh_prometheus(metrics)

    assert "reliquary_mesh_last_window_observed 42" in output
    assert "reliquary_mesh_completions_total{" not in output
    assert "reliquary_mesh_validator_disagreement_rate{" not in output


def test_accepted_vs_rejected_counts() -> None:
    metrics = MeshMetrics()
    metrics.record_window(_report(window_id=1, accepted=3, rejected=2))
    output = render_mesh_prometheus(metrics)

    assert 'reliquary_mesh_completions_total{window_id="1",outcome="accepted"} 3' in output
    assert 'reliquary_mesh_completions_total{window_id="1",outcome="rejected"} 2' in output


def test_per_validator_disagreement_gauge() -> None:
    metrics = MeshMetrics()
    metrics.record_window(
        _report(
            window_id=1,
            accepted=1,
            disagreement={"mesh-A": 0.0, "mesh-M": 0.7},
        )
    )
    output = render_mesh_prometheus(metrics)

    assert 'reliquary_mesh_validator_disagreement_rate{validator_hotkey="mesh-A"} 0.0' in output
    assert 'reliquary_mesh_validator_disagreement_rate{validator_hotkey="mesh-M"} 0.7' in output


def test_gated_counter_increments_once_per_window() -> None:
    metrics = MeshMetrics()
    metrics.record_window(_report(window_id=1, accepted=1, gated=["mesh-M"]))
    metrics.record_window(_report(window_id=2, accepted=1, gated=["mesh-M"]))
    output = render_mesh_prometheus(metrics)

    assert 'reliquary_mesh_validators_gated_total{validator_hotkey="mesh-M"} 2' in output


def test_missing_counter_increments() -> None:
    metrics = MeshMetrics()
    metrics.record_window(_report(window_id=1, missing=["mesh-C"]))
    output = render_mesh_prometheus(metrics)

    assert 'reliquary_mesh_validators_missing_total{validator_hotkey="mesh-C"} 1' in output


def test_idempotent_record() -> None:
    metrics = MeshMetrics()
    report = _report(window_id=5, accepted=2, gated=["mesh-M"])
    metrics.record_window(report)
    metrics.record_window(report)
    metrics.record_window(report)

    output = render_mesh_prometheus(metrics)
    assert 'reliquary_mesh_completions_total{window_id="5",outcome="accepted"} 2' in output
    assert 'reliquary_mesh_validators_gated_total{validator_hotkey="mesh-M"} 1' in output


def test_multi_window_accumulation() -> None:
    metrics = MeshMetrics()
    metrics.record_window(_report(window_id=1, accepted=2, rejected=1))
    metrics.record_window(_report(window_id=2, accepted=3, rejected=0))
    metrics.record_window(_report(window_id=3, accepted=1, rejected=4))
    output = render_mesh_prometheus(metrics)

    assert 'reliquary_mesh_completions_total{window_id="1",outcome="accepted"} 2' in output
    assert 'reliquary_mesh_completions_total{window_id="2",outcome="accepted"} 3' in output
    assert 'reliquary_mesh_completions_total{window_id="3",outcome="rejected"} 4' in output
    assert "reliquary_mesh_last_window_observed 3" in output


def test_render_format_stable() -> None:
    report = _report(
        window_id=7,
        accepted=4,
        rejected=1,
        disagreement={"mesh-A": 0.02, "mesh-B": 0.05},
        gated=["mesh-M"],
    )
    m1 = MeshMetrics()
    m2 = MeshMetrics()
    m1.record_window(report)
    m2.record_window(report)
    assert render_mesh_prometheus(m1) == render_mesh_prometheus(m2)


def test_label_escaping() -> None:
    metrics = MeshMetrics()
    metrics.record_window(
        _report(
            window_id=1,
            disagreement={"mesh\"weird\\": 0.5, "mesh\nnewline": 0.1},
        )
    )
    output = render_mesh_prometheus(metrics)
    assert 'validator_hotkey="mesh\\"weird\\\\"' in output
    assert 'validator_hotkey="mesh\\nnewline"' in output


def test_dashboard_json_parses() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    assert "title" in dashboard
    assert "panels" in dashboard
    assert "schemaVersion" in dashboard
    assert dashboard["uid"] == "reliquary-validator-mesh"


def test_dashboard_panels_present() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    panel_titles = {p.get("title") for p in dashboard["panels"]}
    assert "Acceptance rate" in panel_titles
    assert "Per-validator disagreement rate" in panel_titles
    assert "Gated validators (cumulative)" in panel_titles
    assert "Missing validators (cumulative)" in panel_titles


def test_dashboard_queries_reference_metrics() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    allowed_metrics = {
        "reliquary_mesh_completions_total",
        "reliquary_mesh_validator_disagreement_rate",
        "reliquary_mesh_validators_gated_total",
        "reliquary_mesh_validators_missing_total",
        "reliquary_mesh_last_window_observed",
    }
    metric_pattern = re.compile(r"reliquary_mesh_[a-z_]+")
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            found = metric_pattern.findall(expr)
            assert found, f"panel {panel['title']!r} target has no metric reference"
            for metric in found:
                assert metric in allowed_metrics, (
                    f"panel {panel['title']!r} references unknown metric {metric!r}"
                )
