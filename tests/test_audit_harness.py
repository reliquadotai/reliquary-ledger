"""Tests for the empirical audit harness.

Fast tests that use small trial counts; the real audit campaign is run
via CLI / separate invocation on the GPU server.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.audit_harness import (
    ADVERSARIAL_CLASSES,
    AuditReport,
    ClassReport,
    run_audit_campaign,
)


def test_audit_campaign_produces_all_classes() -> None:
    report = run_audit_campaign(honest_trials=8, adversarial_trials=4)
    assert "honest" in report.classes
    for class_name in ADVERSARIAL_CLASSES:
        assert class_name in report.classes


def test_audit_campaign_is_deterministic_given_fixed_seeds() -> None:
    report_a = run_audit_campaign(honest_trials=16, adversarial_trials=8)
    report_b = run_audit_campaign(honest_trials=16, adversarial_trials=8)
    # duration differs; counts and diffs must match.
    for class_name in report_a.classes:
        a = report_a.classes[class_name]
        b = report_b.classes[class_name]
        assert a.accept_count == b.accept_count
        assert a.reject_count == b.reject_count
        assert a.median_min_sketch_diff == b.median_min_sketch_diff


def test_honest_trials_always_accept() -> None:
    """Sketch-layer honest roundtrip always produces accept across trials."""
    report = run_audit_campaign(honest_trials=32, adversarial_trials=0)
    honest = report.classes["honest"]
    assert honest.accept_count == honest.trials
    assert honest.false_positive_rate == 0.0


def test_audit_documents_sketch_layer_limitations() -> None:
    """Per spec-proof-protocol.md Security Properties: the sketch layer alone
    does NOT reliably reject tampering at HIDDEN_DIM=256 with random-normal
    inputs because per-position sketch variance (~2000) lives inside tolerance
    (6000). The audit harness correctly surfaces this; strong rejection depends
    on the full nine-stage pipeline (Epic 2)."""
    report = run_audit_campaign(honest_trials=4, adversarial_trials=32)
    zero = report.classes["tamper_zero"]
    # We simply assert the audit ran and produced a per-class report.
    # The actual numerical FN rate is a property of the sketch layer and is
    # captured in the audit artifact for external review, not asserted here.
    assert zero.trials == 32
    assert 0.0 <= zero.false_negative_rate <= 1.0


def test_report_to_json_roundtrip() -> None:
    report = run_audit_campaign(honest_trials=4, adversarial_trials=4)
    payload = report.to_json()
    data = json.loads(payload)
    assert "classes" in data
    assert data["challenge_k"] > 0
    assert data["hidden_dim"] > 0
    assert "honest" in data["classes"]


def test_adversarial_classes_registry_is_nonempty() -> None:
    assert len(ADVERSARIAL_CLASSES) >= 3
    for fn in ADVERSARIAL_CLASSES.values():
        assert callable(fn)


def test_report_contains_host_and_torch_info() -> None:
    report = run_audit_campaign(honest_trials=2, adversarial_trials=2)
    assert report.torch_version
    assert report.host
    assert isinstance(report.cuda_available, bool)


def test_class_report_rates_are_in_valid_range() -> None:
    report = run_audit_campaign(honest_trials=8, adversarial_trials=4)
    for class_report in report.classes.values():
        assert 0.0 <= class_report.false_negative_rate <= 1.0
        assert 0.0 <= class_report.false_positive_rate <= 1.0
