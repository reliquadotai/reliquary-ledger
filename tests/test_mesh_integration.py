"""Tests for the mesh integration harness (CLI-level + serialization roundtrip)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reliquary_inference.validator.mesh_integration", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_produce_writes_verdict_payload(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.json"
    _run(
        "produce",
        "--validator-hotkey", "mesh-A",
        "--validator-stake", "40.0",
        "--scenario", "honest",
        "--window-id", "100",
        "--count", "8",
        "--output", str(out),
    )
    data = json.loads(out.read_text())
    assert data["window_id"] == 100
    assert data["validator_hotkey"] == "mesh-A"
    assert data["scenario"] == "honest"
    assert len(data["verdicts"]) == 8
    for v in data["verdicts"]:
        assert v["window_id"] == 100
        assert v["validator"]["hotkey"] == "mesh-A"
        assert v["validator"]["stake"] == 40.0
        assert v["accepted"] is True


def test_malicious_scenario_emits_rejections(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.json"
    _run(
        "produce",
        "--validator-hotkey", "mesh-M",
        "--validator-stake", "10.0",
        "--scenario", "malicious",
        "--window-id", "200",
        "--count", "4",
        "--output", str(out),
    )
    data = json.loads(out.read_text())
    for v in data["verdicts"]:
        assert v["accepted"] is False
        assert v["stage_failed"] == "proof"


def test_aggregate_two_honest_hosts_converges(tmp_path: Path) -> None:
    files = []
    for idx, (hotkey, stake) in enumerate([("mesh-A", 40.0), ("mesh-B", 40.0)]):
        f = tmp_path / f"verdicts_{hotkey}.json"
        _run(
            "produce",
            "--validator-hotkey", hotkey,
            "--validator-stake", str(stake),
            "--scenario", "honest",
            "--window-id", "500",
            "--count", "8",
            "--output", str(f),
        )
        files.append(str(f))

    report_path = tmp_path / "report.json"
    _run(
        "aggregate",
        "--input", *files,
        "--expected-hotkeys", "mesh-A=40.0", "mesh-B=40.0",
        "--output", str(report_path),
    )
    report = json.loads(report_path.read_text())
    assert report["window_id"] == 500
    assert report["total_completions"] == 8
    assert report["accepted"] == 8
    assert report["missing_validators"] == []
    assert report["gated_validators"] == []


def test_aggregate_detects_malicious_outlier(tmp_path: Path) -> None:
    files = []
    for hotkey, stake, scenario in [
        ("mesh-A", 40.0, "honest"),
        ("mesh-B", 40.0, "honest"),
        ("mesh-M", 10.0, "malicious"),
    ]:
        f = tmp_path / f"verdicts_{hotkey}.json"
        _run(
            "produce",
            "--validator-hotkey", hotkey,
            "--validator-stake", str(stake),
            "--scenario", scenario,
            "--window-id", "600",
            "--count", "8",
            "--output", str(f),
        )
        files.append(str(f))

    report_path = tmp_path / "report.json"
    _run(
        "aggregate",
        "--input", *files,
        "--expected-hotkeys", "mesh-A=40.0", "mesh-B=40.0", "mesh-M=10.0",
        "--output", str(report_path),
    )
    report = json.loads(report_path.read_text())
    assert report["window_id"] == 600
    assert report["accepted"] == 8, "honest majority should override malicious single outlier"
    assert report["disagreement_rates"]["mesh-M"] == 1.0
    assert "mesh-M" in report["gated_validators"]
    for verdict in report["median_verdicts"].values():
        assert "mesh-M" in verdict["outlier_validators"]


def test_aggregate_reports_missing_validators(tmp_path: Path) -> None:
    files = []
    f = tmp_path / "verdicts_A.json"
    _run(
        "produce",
        "--validator-hotkey", "mesh-A",
        "--validator-stake", "10.0",
        "--scenario", "honest",
        "--window-id", "700",
        "--count", "4",
        "--output", str(f),
    )
    files.append(str(f))

    report_path = tmp_path / "report.json"
    _run(
        "aggregate",
        "--input", *files,
        "--expected-hotkeys",
        "mesh-A=10.0", "mesh-B=10.0", "mesh-C=10.0",
        "--output", str(report_path),
    )
    report = json.loads(report_path.read_text())
    assert sorted(report["missing_validators"]) == ["mesh-B", "mesh-C"]
