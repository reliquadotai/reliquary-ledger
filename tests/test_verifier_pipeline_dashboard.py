"""Schema validation for the verifier-pipeline Grafana dashboard.

The metrics this dashboard consumes (``reliquary_verifier_stage_total``,
``reliquary_verifier_rejections_total``, ``reliquary_verifier_soft_flags_total``)
are already emitted by ``reliquary_inference/validator/metrics.py`` and
covered by `test_verifier_metrics.py`. The risk this test covers is
dashboard drift: we keep the JSON parseable, we keep the panel set
stable, and every PromQL reference only hits one of the three allowed
series.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "deploy"
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "reliquary-verifier-pipeline.json"
)

_ALLOWED_METRICS = {
    "reliquary_verifier_stage_total",
    "reliquary_verifier_rejections_total",
    "reliquary_verifier_soft_flags_total",
}
_METRIC_PATTERN = re.compile(r"reliquary_verifier_[a-z_]+")


def _load_dashboard() -> dict:
    return json.loads(_DASHBOARD_PATH.read_text())


def test_dashboard_parses() -> None:
    dashboard = _load_dashboard()
    assert dashboard["uid"] == "reliquary-verifier-pipeline"
    assert "panels" in dashboard
    assert "schemaVersion" in dashboard


def test_expected_panel_titles_present() -> None:
    dashboard = _load_dashboard()
    titles = {p["title"] for p in dashboard["panels"]}
    assert "Stage throughput by result" in titles
    assert "Rejection rate per stage" in titles
    assert "Top rejection reasons (cumulative)" in titles
    assert "Soft flags per stage" in titles


def test_all_queries_reference_allowed_metrics() -> None:
    dashboard = _load_dashboard()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            metrics = _METRIC_PATTERN.findall(expr)
            assert metrics, f"panel {panel['title']!r} target has no metric reference"
            for metric in metrics:
                assert metric in _ALLOWED_METRICS, (
                    f"panel {panel['title']!r} references unknown metric {metric!r}"
                )


def test_every_panel_has_at_least_one_target() -> None:
    dashboard = _load_dashboard()
    for panel in dashboard["panels"]:
        targets = panel.get("targets", [])
        assert targets, f"panel {panel['title']!r} has no targets"


def test_stat_panel_exposes_all_three_results() -> None:
    dashboard = _load_dashboard()
    stat_panels = [p for p in dashboard["panels"] if p.get("type") == "stat"]
    assert stat_panels, "expected at least one stat panel"
    legend_formats: set[str] = set()
    for panel in stat_panels:
        for target in panel.get("targets", []):
            legend_formats.add(target.get("legendFormat", ""))
    assert {"accept", "reject", "soft_flag"} <= legend_formats


def test_datasource_uid_is_prometheus() -> None:
    dashboard = _load_dashboard()
    for panel in dashboard["panels"]:
        ds = panel.get("datasource", {})
        assert ds.get("uid") == "prometheus"
        for target in panel.get("targets", []):
            t_ds = target.get("datasource", {})
            assert t_ds.get("uid") == "prometheus"
