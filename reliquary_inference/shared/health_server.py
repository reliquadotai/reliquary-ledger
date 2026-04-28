"""HTTP wrapper for the health-check aggregator.

Phase 3.1 of the mainnet-readiness push. The Tier 1 PRD shipped the
``compute_health(HealthSignals)`` logic in ``shared/health.py`` but
deferred the HTTP wrapper. This module closes that gap.

Design:

- ``HealthSignalsHolder`` is a thread-safe container the validator/miner
  main loop writes to on each iteration. The HTTP server reads from it
  on every request.
- ``serve_health(bind, port, holder)`` runs a minimal stdlib
  ``ThreadingHTTPServer`` exposing two endpoints:
    GET /health   -> 200 / 200 / 503 with structured HealthReport JSON
    GET /healthz  -> 200 / 200 / 503 with binary ``{ok: bool}`` payload
  HTTP status reflects state: OK -> 200, DEGRADED -> 200, UNHEALTHY -> 503.
- Returns 404 on unknown paths.
- CORS-enabled (Access-Control-Allow-Origin: *) so the public dashboard
  can probe a validator's health from a browser if the operator exposes
  the port.
- Logs are silenced (matches the existing ``serve_metrics`` quietness).

Wire-up in the validator main loop is operator-side scaffolding; this
module ships the server + holder + tests so it can be dropped into
``run-validator`` as a separate PR.
"""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .health import HealthReport, HealthSignals, HealthState, HealthThresholds, compute_health

logger = logging.getLogger(__name__)


class HealthSignalsHolder:
    """Thread-safe container for the most recent ``HealthSignals``.

    Producer (validator/miner main loop) calls ``update(...)`` after each
    iteration. Consumer (the HTTP handler) calls ``snapshot()`` to read
    the current state.
    """

    def __init__(self, initial: HealthSignals, *, thresholds: HealthThresholds | None = None) -> None:
        self._lock = threading.Lock()
        self._signals = initial
        self._thresholds = thresholds or HealthThresholds()

    def update(self, signals: HealthSignals) -> None:
        with self._lock:
            self._signals = signals

    def snapshot(self) -> tuple[HealthSignals, HealthThresholds]:
        with self._lock:
            return self._signals, self._thresholds

    def report(self) -> HealthReport:
        signals, thresholds = self.snapshot()
        return compute_health(signals, thresholds=thresholds)


_STATUS_FOR_STATE = {
    HealthState.OK: HTTPStatus.OK,
    HealthState.DEGRADED: HTTPStatus.OK,
    HealthState.UNHEALTHY: HTTPStatus.SERVICE_UNAVAILABLE,
}


def _build_handler(holder: HealthSignalsHolder) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _send(self, status: HTTPStatus, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/health", "/health/"}:
                report = holder.report()
                status = _STATUS_FOR_STATE[report.overall]
                payload = json.dumps(report.to_dict(), sort_keys=True).encode("utf-8")
                self._send(status, payload)
                return
            if path in {"/healthz", "/healthz/"}:
                report = holder.report()
                status = _STATUS_FOR_STATE[report.overall]
                payload = json.dumps(
                    {"ok": report.overall is HealthState.OK, "state": report.overall.value},
                    sort_keys=True,
                ).encode("utf-8")
                self._send(status, payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return HealthHandler


def make_server(
    *,
    bind: str,
    port: int,
    holder: HealthSignalsHolder,
) -> ThreadingHTTPServer:
    """Construct (but do not start) a health HTTP server.

    Separated from ``serve_health`` so callers can run the server in a
    daemon thread and shut it down cleanly.
    """
    handler_cls = _build_handler(holder)
    server = ThreadingHTTPServer((bind, port), handler_cls)
    return server


def serve_health(*, bind: str, port: int, holder: HealthSignalsHolder) -> None:
    """Blocking entry point: serve until interrupted."""
    server = make_server(bind=bind, port=port, holder=holder)
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = [
    "HealthSignalsHolder",
    "make_server",
    "serve_health",
]
