"""Health-check aggregator for mainnet deployments.

Assembles liveness + readiness signals from chain adapter state, the last
window verified, proof worker responsiveness, and model-loaded status.
Consumed by the ``/health`` HTTP endpoint and by systemd watchdog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class HealthState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """One component's health snapshot."""

    name: str
    state: HealthState
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HealthReport:
    overall: HealthState
    checks: list[HealthCheck]
    generated_at: float
    uptime_seconds: float

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.value,
            "checks": [
                {
                    "name": c.name,
                    "state": c.state.value,
                    "detail": c.detail,
                    "metrics": dict(c.metrics),
                }
                for c in self.checks
            ],
            "generated_at": self.generated_at,
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class HealthThresholds:
    """Operator-tunable thresholds for transition between ok/degraded/unhealthy."""

    chain_disconnect_degraded_seconds: float = 60.0
    chain_disconnect_unhealthy_seconds: float = 180.0
    last_window_degraded_seconds: float = 180.0
    last_window_unhealthy_seconds: float = 600.0
    proof_worker_unhealthy_seconds: float = 300.0


@dataclass
class HealthSignals:
    """Lightweight snapshot the producer writes on each loop iteration."""

    started_at: float
    last_chain_ok_at: float | None = None
    last_window_verified_at: float | None = None
    last_proof_worker_heartbeat_at: float | None = None
    model_loaded: bool = False


def compute_health(
    signals: HealthSignals,
    *,
    now_fn: Callable[[], float] = time.time,
    thresholds: HealthThresholds | None = None,
) -> HealthReport:
    thresholds = thresholds or HealthThresholds()
    now = now_fn()
    checks: list[HealthCheck] = []

    checks.append(_check_model_loaded(signals))
    checks.append(_check_chain(signals, now, thresholds))
    checks.append(_check_last_window(signals, now, thresholds))
    checks.append(_check_proof_worker(signals, now, thresholds))

    overall = _roll_up(checks)
    return HealthReport(
        overall=overall,
        checks=checks,
        generated_at=now,
        uptime_seconds=max(0.0, now - signals.started_at),
    )


def _check_model_loaded(signals: HealthSignals) -> HealthCheck:
    if signals.model_loaded:
        return HealthCheck("model", HealthState.OK, "loaded")
    return HealthCheck("model", HealthState.UNHEALTHY, "not loaded")


def _check_chain(signals: HealthSignals, now: float, thresholds: HealthThresholds) -> HealthCheck:
    if signals.last_chain_ok_at is None:
        return HealthCheck("chain", HealthState.UNHEALTHY, "never connected")
    age = now - signals.last_chain_ok_at
    if age > thresholds.chain_disconnect_unhealthy_seconds:
        state = HealthState.UNHEALTHY
    elif age > thresholds.chain_disconnect_degraded_seconds:
        state = HealthState.DEGRADED
    else:
        state = HealthState.OK
    return HealthCheck(
        "chain",
        state,
        f"last_ok_age_seconds={age:.1f}",
        {"last_ok_age_seconds": age},
    )


def _check_last_window(signals: HealthSignals, now: float, thresholds: HealthThresholds) -> HealthCheck:
    if signals.last_window_verified_at is None:
        return HealthCheck("last_window", HealthState.DEGRADED, "no window verified yet")
    age = now - signals.last_window_verified_at
    if age > thresholds.last_window_unhealthy_seconds:
        state = HealthState.UNHEALTHY
    elif age > thresholds.last_window_degraded_seconds:
        state = HealthState.DEGRADED
    else:
        state = HealthState.OK
    return HealthCheck(
        "last_window",
        state,
        f"age_seconds={age:.1f}",
        {"age_seconds": age},
    )


def _check_proof_worker(signals: HealthSignals, now: float, thresholds: HealthThresholds) -> HealthCheck:
    if signals.last_proof_worker_heartbeat_at is None:
        return HealthCheck("proof_worker", HealthState.OK, "not configured")
    age = now - signals.last_proof_worker_heartbeat_at
    if age > thresholds.proof_worker_unhealthy_seconds:
        return HealthCheck(
            "proof_worker",
            HealthState.UNHEALTHY,
            f"stale heartbeat age={age:.1f}s",
            {"heartbeat_age_seconds": age},
        )
    return HealthCheck(
        "proof_worker",
        HealthState.OK,
        f"heartbeat_age={age:.1f}s",
        {"heartbeat_age_seconds": age},
    )


def _roll_up(checks: list[HealthCheck]) -> HealthState:
    if any(c.state is HealthState.UNHEALTHY for c in checks):
        return HealthState.UNHEALTHY
    if any(c.state is HealthState.DEGRADED for c in checks):
        return HealthState.DEGRADED
    return HealthState.OK
