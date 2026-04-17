"""Tests for the cross-GPU determinism harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary_inference.cross_gpu_audit import (
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_SCALES,
    DEFAULT_SEEDS,
    CrossGpuReport,
    SketchSample,
    collect_samples,
    compare_reports,
    run_cross_gpu_campaign,
)


def test_sample_count_is_product_of_dimensions() -> None:
    samples = collect_samples()
    expected = len(DEFAULT_SEEDS) * len(DEFAULT_SCALES) * len(DEFAULT_HIDDEN_DIMS)
    assert len(samples) == expected


def test_samples_are_deterministic_across_invocations() -> None:
    a = collect_samples()
    b = collect_samples()
    assert a == b


def test_report_samples_digest_is_stable() -> None:
    r1 = run_cross_gpu_campaign()
    r2 = run_cross_gpu_campaign()
    assert r1.samples_digest == r2.samples_digest


def test_report_contains_host_and_torch_metadata() -> None:
    report = run_cross_gpu_campaign(seeds=(0,), scales=(1.0,), hidden_dims=(128,))
    assert report.host
    assert report.torch_version
    assert isinstance(report.cuda_available, bool)
    assert len(report.samples) == 1
    assert report.samples[0].seed == 0


def test_compare_reports_detects_identical_runs() -> None:
    r1 = run_cross_gpu_campaign(seeds=(1, 2), scales=(1.0,), hidden_dims=(128,))
    r2 = run_cross_gpu_campaign(seeds=(1, 2), scales=(1.0,), hidden_dims=(128,))
    # Force a different "host" label so compare_reports groups them.
    r2.host = r1.host + "-twin"
    report = compare_reports([r1, r2])
    assert report["matching_digests"] is True
    assert report["mismatch_count"] == 0


def test_compare_reports_detects_sketch_drift() -> None:
    r1 = run_cross_gpu_campaign(seeds=(1,), scales=(1.0,), hidden_dims=(128,))
    # Fake a drifted report by mutating the sample.
    r2 = run_cross_gpu_campaign(seeds=(1,), scales=(1.0,), hidden_dims=(128,))
    r2.host = r1.host + "-drifted"
    original = r2.samples[0]
    r2.samples[0] = SketchSample(
        seed=original.seed,
        scale=original.scale,
        hidden_dim=original.hidden_dim,
        sketch=original.sketch + 1,
    )
    # Recompute digest after mutation.
    from reliquary_inference.cross_gpu_audit import _digest_samples

    r2.samples_digest = _digest_samples(r2.samples)

    report = compare_reports([r1, r2])
    assert report["matching_digests"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["seed"] == 1


def test_compare_reports_requires_at_least_one_report() -> None:
    with pytest.raises(ValueError):
        compare_reports([])


def test_compare_reports_surfaces_per_host_digest_map() -> None:
    r1 = run_cross_gpu_campaign(seeds=(1,), scales=(1.0,), hidden_dims=(128,))
    r2 = run_cross_gpu_campaign(seeds=(1,), scales=(1.0,), hidden_dims=(128,))
    r2.host = r1.host + "-twin"
    report = compare_reports([r1, r2])
    assert set(report["digest_by_host"].keys()) == {r1.host, r2.host}
    assert report["total_samples_per_host"] == {r1.host: 1, r2.host: 1}
