"""Tests for the audit harness's checkpointing + progress features.

These land to unblock a real 100K-honest + 10K-adversarial campaign on
the Blackwell fleet — operators need resume semantics and periodic
progress to run unattended.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.audit_harness import (
    ADVERSARIAL_CLASSES,
    AuditReport,
    ClassReport,
    _checkpoint_load,
    _checkpoint_write,
    run_audit_campaign,
)


def _small_report(class_name: str = "honest") -> AuditReport:
    return AuditReport(
        timestamp="2026-04-18T00:00:00Z",
        host="test-host",
        torch_version="2.0.0",
        cuda_available=False,
        hidden_dim=256,
        challenge_k=32,
        trials_per_class=1,
        duration_seconds=0.0,
        classes={
            class_name: ClassReport(
                name=class_name,
                trials=1,
                accept_count=1,
                reject_count=0,
                false_negative_rate=0.0,
                false_positive_rate=0.0,
                median_min_sketch_diff=0.0,
            )
        },
    )


def test_checkpoint_write_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    report = _small_report()
    _checkpoint_write(path, report)
    loaded = _checkpoint_load(path)
    assert loaded is not None
    assert loaded.host == "test-host"
    assert "honest" in loaded.classes
    assert loaded.classes["honest"].trials == 1


def test_checkpoint_load_missing_returns_none(tmp_path: Path) -> None:
    assert _checkpoint_load(tmp_path / "absent.json") is None


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    _checkpoint_write(path, _small_report())
    leftover_tempfiles = [p for p in tmp_path.iterdir() if p.name.startswith(".audit-")]
    assert leftover_tempfiles == []


def test_checkpoint_write_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "ckpt.json"
    _checkpoint_write(path, _small_report())
    assert path.is_file()


def test_progress_callback_called_per_trial(tmp_path: Path) -> None:
    calls: list[tuple[str, int, int]] = []
    run_audit_campaign(
        honest_trials=3,
        adversarial_trials=2,
        progress_callback=lambda name, done, total: calls.append((name, done, total)),
    )
    # Expect 3 honest + 2 per adversarial class = 3 + 2*len(ADVERSARIAL_CLASSES).
    assert len(calls) == 3 + 2 * len(ADVERSARIAL_CLASSES)
    # Honest goes first and counts up.
    honest_calls = [c for c in calls if c[0] == "honest"]
    assert [c[1] for c in honest_calls] == [1, 2, 3]
    assert all(c[2] == 3 for c in honest_calls)


def test_progress_every_prints_to_stdout(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    run_audit_campaign(honest_trials=4, adversarial_trials=2, progress_every=2)
    captured = capsys.readouterr().out
    assert "[honest] 2/4" in captured
    assert "[honest] 4/4" in captured


def test_checkpoint_persists_after_each_class(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.json"
    report = run_audit_campaign(
        honest_trials=2, adversarial_trials=1, checkpoint_path=ckpt
    )
    assert ckpt.is_file()
    loaded = _checkpoint_load(ckpt)
    assert loaded is not None
    assert "honest" in loaded.classes
    for class_name in ADVERSARIAL_CLASSES:
        assert class_name in loaded.classes
    # Full report matches loaded checkpoint.
    assert set(report.classes) == set(loaded.classes)


def test_resume_skips_already_completed_classes(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.json"
    # First: only honest class persisted.
    seeded = _small_report()
    _checkpoint_write(ckpt, seeded)
    # Now resume — adversarial classes should be filled in, honest untouched.
    report = run_audit_campaign(
        honest_trials=3,  # different from the checkpoint; should NOT re-run.
        adversarial_trials=1,
        checkpoint_path=ckpt,
        resume=True,
    )
    assert report.classes["honest"].trials == 1  # preserved from checkpoint
    for class_name in ADVERSARIAL_CLASSES:
        assert class_name in report.classes


def test_resume_false_overwrites_existing_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.json"
    seeded = _small_report()
    _checkpoint_write(ckpt, seeded)
    report = run_audit_campaign(
        honest_trials=2, adversarial_trials=1, checkpoint_path=ckpt, resume=False
    )
    assert report.classes["honest"].trials == 2  # fresh run used honest_trials=2


def test_progress_callback_and_checkpoint_compose(tmp_path: Path) -> None:
    calls: list[tuple[str, int, int]] = []
    ckpt = tmp_path / "ckpt.json"
    run_audit_campaign(
        honest_trials=2,
        adversarial_trials=1,
        checkpoint_path=ckpt,
        progress_callback=lambda name, done, total: calls.append((name, done, total)),
    )
    assert calls  # callback invoked
    assert ckpt.is_file()  # checkpoint written


def test_checkpoint_file_contains_valid_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.json"
    run_audit_campaign(honest_trials=1, adversarial_trials=1, checkpoint_path=ckpt)
    parsed = json.loads(ckpt.read_text())
    assert parsed["host"]
    assert "classes" in parsed
    assert "honest" in parsed["classes"]
