"""Tests for the health aggregator."""

from __future__ import annotations

from reliquary_inference.shared.health import (
    HealthSignals,
    HealthState,
    HealthThresholds,
    compute_health,
)


def _signals(**overrides) -> HealthSignals:
    defaults = dict(
        started_at=0.0,
        last_chain_ok_at=100.0,
        last_window_verified_at=100.0,
        last_proof_worker_heartbeat_at=100.0,
        model_loaded=True,
    )
    defaults.update(overrides)
    return HealthSignals(**defaults)


def test_all_fresh_is_ok() -> None:
    report = compute_health(_signals(), now_fn=lambda: 110.0)
    assert report.overall is HealthState.OK
    assert all(c.state is HealthState.OK for c in report.checks)
    assert report.uptime_seconds == 110.0


def test_missing_model_is_unhealthy() -> None:
    report = compute_health(_signals(model_loaded=False), now_fn=lambda: 110.0)
    assert report.overall is HealthState.UNHEALTHY
    model_check = next(c for c in report.checks if c.name == "model")
    assert model_check.state is HealthState.UNHEALTHY


def test_chain_disconnect_beyond_degraded_threshold() -> None:
    signals = _signals(last_chain_ok_at=0.0)
    report = compute_health(
        signals,
        now_fn=lambda: 90.0,
        thresholds=HealthThresholds(chain_disconnect_degraded_seconds=60.0, chain_disconnect_unhealthy_seconds=180.0),
    )
    assert report.overall is HealthState.DEGRADED


def test_chain_disconnect_beyond_unhealthy_threshold() -> None:
    signals = _signals(last_chain_ok_at=0.0)
    report = compute_health(
        signals,
        now_fn=lambda: 300.0,
        thresholds=HealthThresholds(chain_disconnect_degraded_seconds=60.0, chain_disconnect_unhealthy_seconds=180.0),
    )
    assert report.overall is HealthState.UNHEALTHY


def test_chain_never_connected_is_unhealthy() -> None:
    signals = _signals(last_chain_ok_at=None)
    report = compute_health(signals, now_fn=lambda: 100.0)
    assert report.overall is HealthState.UNHEALTHY


def test_last_window_stale_triggers_degraded() -> None:
    signals = _signals(last_window_verified_at=0.0)
    report = compute_health(
        signals,
        now_fn=lambda: 250.0,
        thresholds=HealthThresholds(
            last_window_degraded_seconds=180.0,
            last_window_unhealthy_seconds=600.0,
        ),
    )
    assert report.overall is HealthState.DEGRADED


def test_last_window_very_stale_triggers_unhealthy() -> None:
    signals = _signals(last_window_verified_at=0.0)
    report = compute_health(
        signals,
        now_fn=lambda: 700.0,
    )
    assert report.overall is HealthState.UNHEALTHY


def test_proof_worker_heartbeat_stale_unhealthy() -> None:
    signals = _signals(last_proof_worker_heartbeat_at=0.0)
    report = compute_health(
        signals,
        now_fn=lambda: 500.0,
    )
    assert report.overall is HealthState.UNHEALTHY
    pw_check = next(c for c in report.checks if c.name == "proof_worker")
    assert pw_check.state is HealthState.UNHEALTHY


def test_proof_worker_not_configured_does_not_degrade() -> None:
    signals = _signals(last_proof_worker_heartbeat_at=None)
    report = compute_health(signals, now_fn=lambda: 100.0)
    assert report.overall is HealthState.OK


def test_to_dict_shape_is_stable() -> None:
    report = compute_health(_signals(), now_fn=lambda: 110.0)
    data = report.to_dict()
    assert set(data) == {"overall", "checks", "generated_at", "uptime_seconds"}
    for check in data["checks"]:
        assert set(check) == {"name", "state", "detail", "metrics"}
