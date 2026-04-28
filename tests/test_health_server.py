"""Tests for the health HTTP wrapper.

Phase 3.1 of the mainnet-readiness push. Tests cover:

- ``HealthSignalsHolder`` round-trips updates under concurrency
- ``GET /health`` returns the structured HealthReport JSON
- ``GET /healthz`` returns the binary ``{ok, state}`` payload
- HTTP status code reflects state: OK→200, DEGRADED→200, UNHEALTHY→503
- ``GET /unknown`` returns 404
- ``OPTIONS /health`` returns 204 with CORS headers (preflight contract)
- Server can be made + closed cleanly via ``make_server``
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from http import HTTPStatus

import pytest

from reliquary_inference.shared.health import (
    HealthSignals,
    HealthState,
    HealthThresholds,
)
from reliquary_inference.shared.health_server import (
    HealthSignalsHolder,
    make_server,
)


def _free_port() -> int:
    """Bind to an OS-assigned port and immediately release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(url: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=2.0)
        return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read() if exc.fp else b""


# ────────  HealthSignalsHolder  ────────


def test_holder_round_trips_signals() -> None:
    initial = HealthSignals(started_at=100.0, model_loaded=False)
    holder = HealthSignalsHolder(initial)
    snap, _ = holder.snapshot()
    assert snap.started_at == 100.0
    assert snap.model_loaded is False

    updated = HealthSignals(started_at=100.0, model_loaded=True, last_chain_ok_at=150.0)
    holder.update(updated)
    snap, _ = holder.snapshot()
    assert snap.model_loaded is True
    assert snap.last_chain_ok_at == 150.0


def test_holder_concurrent_updates_do_not_corrupt() -> None:
    holder = HealthSignalsHolder(HealthSignals(started_at=0.0, model_loaded=False))

    def writer(value: float, count: int) -> None:
        for _ in range(count):
            holder.update(
                HealthSignals(started_at=0.0, model_loaded=True, last_chain_ok_at=value)
            )

    threads = [
        threading.Thread(target=writer, args=(1.0, 100)),
        threading.Thread(target=writer, args=(2.0, 100)),
        threading.Thread(target=writer, args=(3.0, 100)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap, _ = holder.snapshot()
    # The final value is one of {1.0, 2.0, 3.0}; the test guarantees no
    # torn read or partial state.
    assert snap.last_chain_ok_at in {1.0, 2.0, 3.0}


def test_holder_report_returns_compute_health_output() -> None:
    holder = HealthSignalsHolder(
        HealthSignals(
            started_at=time.time() - 10,
            model_loaded=True,
            last_chain_ok_at=time.time() - 5,
            last_window_verified_at=time.time() - 5,
        )
    )
    report = holder.report()
    assert report.overall is HealthState.OK
    component_names = [c.name for c in report.checks]
    assert "model" in component_names
    assert "chain" in component_names


# ────────  HTTP server  ────────


@pytest.fixture()
def serving_holder():
    """Spin up the health HTTP server on a free localhost port; tear down after."""
    holder = HealthSignalsHolder(
        HealthSignals(
            started_at=time.time() - 10,
            model_loaded=True,
            last_chain_ok_at=time.time() - 5,
            last_window_verified_at=time.time() - 5,
        )
    )
    port = _free_port()
    server = make_server(bind="127.0.0.1", port=port, holder=holder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield holder, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_get_health_returns_structured_report(serving_holder) -> None:
    holder, port = serving_holder
    status, headers, body = _request(f"http://127.0.0.1:{port}/health")
    assert status == HTTPStatus.OK
    assert headers.get("Content-Type", "").startswith("application/json")
    assert headers.get("Access-Control-Allow-Origin") == "*"
    payload = json.loads(body)
    assert payload["overall"] == "ok"
    assert "checks" in payload
    assert "uptime_seconds" in payload
    assert "generated_at" in payload
    assert any(c["name"] == "model" and c["state"] == "ok" for c in payload["checks"])


def test_get_healthz_returns_binary_payload(serving_holder) -> None:
    _, port = serving_holder
    status, headers, body = _request(f"http://127.0.0.1:{port}/healthz")
    assert status == HTTPStatus.OK
    payload = json.loads(body)
    assert payload == {"ok": True, "state": "ok"}


def test_get_health_returns_503_when_unhealthy(serving_holder) -> None:
    holder, port = serving_holder
    holder.update(HealthSignals(started_at=time.time(), model_loaded=False))
    status, _, body = _request(f"http://127.0.0.1:{port}/health")
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    payload = json.loads(body)
    assert payload["overall"] == "unhealthy"
    assert any(c["name"] == "model" and c["state"] == "unhealthy" for c in payload["checks"])


def test_get_health_returns_200_when_degraded(serving_holder) -> None:
    holder, port = serving_holder
    # Tighten thresholds so the existing chain age (5s) is degraded.
    holder._thresholds = HealthThresholds(  # noqa: SLF001 — test boundary
        chain_disconnect_degraded_seconds=2.0,
        chain_disconnect_unhealthy_seconds=3600.0,
    )
    status, _, body = _request(f"http://127.0.0.1:{port}/health")
    assert status == HTTPStatus.OK
    payload = json.loads(body)
    assert payload["overall"] == "degraded"


def test_unknown_path_returns_404(serving_holder) -> None:
    _, port = serving_holder
    status, _, _ = _request(f"http://127.0.0.1:{port}/unknown")
    assert status == HTTPStatus.NOT_FOUND


def test_query_string_is_ignored(serving_holder) -> None:
    _, port = serving_holder
    status, _, _ = _request(f"http://127.0.0.1:{port}/health?_=12345")
    assert status == HTTPStatus.OK


def test_options_request_returns_204_with_cors(serving_holder) -> None:
    _, port = serving_holder
    status, headers, _ = _request(f"http://127.0.0.1:{port}/health", method="OPTIONS")
    assert status == HTTPStatus.NO_CONTENT
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert "GET" in headers.get("Access-Control-Allow-Methods", "")
