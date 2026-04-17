"""Acceptance tests for the per-miner scoreboard + Grafana dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

from reliquary_inference.validator.mesh import ValidatorIdentity, VerdictArtifact
from reliquary_inference.validator.miner_scoreboard import (
    MINER_ACCEPTANCE_WINDOW,
    MinerScoreboard,
    render_miner_scoreboard_prometheus,
)


_DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "deploy"
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "reliquary-miner-scoreboard.json"
)

_ALLOWED_METRICS = {
    "reliquary_miner_verdicts_total",
    "reliquary_miner_acceptance_rate",
    "reliquary_miner_last_score",
    "reliquary_miner_rejection_reasons_total",
    "reliquary_miner_last_window_seen",
}
_METRIC_PATTERN = re.compile(r"reliquary_miner_[a-z_]+")


def _verdict(
    *,
    miner: str = "miner-A",
    validator: str = "mesh-A",
    window: int = 1,
    completion: str = "c-0",
    accepted: bool = True,
    reject_reason: str | None = None,
    scores: dict[str, float] | None = None,
    signed_at: float = 1_000_000.0,
) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion,
        miner_hotkey=miner,
        window_id=window,
        validator=ValidatorIdentity(hotkey=validator, stake=40.0),
        accepted=accepted,
        stage_failed=None,
        reject_reason=reject_reason,
        scores=scores or {"reward": 1.0 if accepted else 0.0},
        signed_at=signed_at,
    )


def test_record_single_accept_advances_counter() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(_verdict(accepted=True))
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'reliquary_miner_verdicts_total{miner_hotkey="miner-A",outcome="accepted"} 1' in output


def test_record_single_reject_advances_rejection_counter() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(
        _verdict(accepted=False, reject_reason="proof_sketch_mismatch")
    )
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'reliquary_miner_verdicts_total{miner_hotkey="miner-A",outcome="rejected"} 1' in output
    assert 'reliquary_miner_rejection_reasons_total{miner_hotkey="miner-A",reason="proof_sketch_mismatch"} 1' in output


def test_rolling_acceptance_rate() -> None:
    scoreboard = MinerScoreboard()
    for i in range(4):
        scoreboard.record_verdict(_verdict(completion=f"c-{i}", accepted=i != 0))
    snap = scoreboard.snapshot()
    assert snap["acceptance_rate"]["miner-A"] == 0.75


def test_acceptance_window_caps() -> None:
    scoreboard = MinerScoreboard()
    for i in range(MINER_ACCEPTANCE_WINDOW + 5):
        scoreboard.record_verdict(_verdict(completion=f"c-{i}", accepted=i < 5))
    snap = scoreboard.snapshot()
    assert snap["acceptance_rate"]["miner-A"] == 0.0


def test_last_scores_per_miner_metric() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(
        _verdict(
            accepted=True,
            scores={"reward": 0.9, "coherence": 0.8},
        )
    )
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'reliquary_miner_last_score{miner_hotkey="miner-A",metric="reward"} 0.9' in output
    assert 'reliquary_miner_last_score{miner_hotkey="miner-A",metric="coherence"} 0.8' in output


def test_multiple_miners_independent() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(_verdict(miner="miner-A", accepted=True))
    scoreboard.record_verdict(_verdict(miner="miner-B", accepted=False, reject_reason="bad_sig"))
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'miner_hotkey="miner-A",outcome="accepted"} 1' in output
    assert 'miner_hotkey="miner-B",outcome="rejected"} 1' in output


def test_last_window_seen_updates() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(_verdict(window=5))
    scoreboard.record_verdict(_verdict(window=7))
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'reliquary_miner_last_window_seen{miner_hotkey="miner-A"} 7' in output


def test_empty_miner_hotkey_skipped() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(_verdict(miner=""))
    assert scoreboard.snapshot()["verdicts"] == {}


def test_reset_clears_all() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(_verdict())
    scoreboard.reset()
    snap = scoreboard.snapshot()
    assert snap["verdicts"] == {}
    assert snap["acceptance_rate"] == {}


def test_render_format_stable() -> None:
    verdict = _verdict(accepted=True, scores={"reward": 0.7})
    s1 = MinerScoreboard()
    s2 = MinerScoreboard()
    s1.record_verdict(verdict)
    s2.record_verdict(verdict)
    assert render_miner_scoreboard_prometheus(s1) == render_miner_scoreboard_prometheus(s2)


def test_label_escaping_on_reject_reason() -> None:
    scoreboard = MinerScoreboard()
    scoreboard.record_verdict(
        _verdict(accepted=False, reject_reason='odd"reason\\')
    )
    output = render_miner_scoreboard_prometheus(scoreboard)
    assert 'reason="odd\\"reason\\\\"' in output


def test_dashboard_json_parses() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    assert dashboard["uid"] == "reliquary-miner-scoreboard"
    assert "panels" in dashboard
    assert "schemaVersion" in dashboard


def test_dashboard_panels_present() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    titles = {p.get("title") for p in dashboard["panels"]}
    assert "Top miners by acceptance rate" in titles
    assert "Top rejection reasons across miners" in titles
    assert "Per-miner last-score heatmap" in titles


def test_dashboard_queries_reference_miner_metrics() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            found = _METRIC_PATTERN.findall(expr)
            assert found, f"panel {panel['title']!r} has no metric reference"
            for metric in found:
                assert metric in _ALLOWED_METRICS, (
                    f"panel {panel['title']!r} references unknown metric {metric!r}"
                )
