"""Static validation tests for the local monitoring docker-compose stack.

No docker required — we parse the compose manifest and validate its
shape so operators following the Epic 5 smoke runbook catch config
drift (missing healthcheck, changed port) at CI time rather than at
stack startup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_DEPLOY_ROOT = Path(__file__).resolve().parent.parent / "deploy" / "monitoring"
_COMPOSE_PATH = _DEPLOY_ROOT / "docker-compose.monitoring.yml"
_SMOKE_PATH = _DEPLOY_ROOT / "bin" / "monitoring-smoke.sh"


def _load_compose() -> dict:
    """Lightweight YAML loader — compose manifest is small and uses a fixed
    subset of YAML features, so we use PyYAML when available and fall back
    to a minimal hand-parser that covers the structure we ship."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - fallback path
        return _parse_compose_fallback(_COMPOSE_PATH.read_text())
    with _COMPOSE_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_compose_fallback(text: str) -> dict:
    """Extract a structural subset of the compose file using regex.

    Covers: top-level services + each service's ports, healthcheck, image.
    Enough for our shape assertions when PyYAML is absent.
    """
    services_block = re.search(
        r"^services:\n(.+?)(?=\n^volumes:|\Z)", text, re.DOTALL | re.MULTILINE
    )
    if not services_block:
        raise ValueError("services block not found")
    services: dict[str, dict] = {}
    current_service: str | None = None
    for line in services_block.group(1).splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 2 and stripped.rstrip(":") != stripped:
            current_service = stripped.strip().rstrip(":")
            services[current_service] = {
                "ports": [],
                "has_healthcheck": False,
                "raw": [],
            }
            continue
        if current_service is None:
            continue
        services[current_service]["raw"].append(line)
        if "healthcheck:" in stripped:
            services[current_service]["has_healthcheck"] = True
        if 'image:' in stripped:
            services[current_service]["image"] = stripped.split("image:", 1)[1].strip()
    for service, meta in services.items():
        meta["ports"] = [
            line.strip().lstrip("- ").strip('"')
            for line in meta["raw"]
            if re.match(r'\s*-\s*"?\d+:\d+', line)
        ]
    return {"services": services}


def test_compose_file_exists() -> None:
    assert _COMPOSE_PATH.is_file()


def test_compose_parses() -> None:
    data = _load_compose()
    assert "services" in data


def test_compose_has_three_services() -> None:
    data = _load_compose()
    services = set(data["services"])
    assert {"prometheus", "grafana", "jaeger"} <= services


def test_prometheus_service_ports_include_9090() -> None:
    data = _load_compose()
    prom = data["services"]["prometheus"]
    ports = prom.get("ports") or []
    port_strings = [str(p) for p in ports]
    assert any("9090" in p for p in port_strings)


def test_grafana_service_ports_include_3000() -> None:
    data = _load_compose()
    grafana = data["services"]["grafana"]
    port_strings = [str(p) for p in grafana.get("ports") or []]
    assert any("3000" in p for p in port_strings)


def test_jaeger_service_ports_cover_otlp_and_ui() -> None:
    data = _load_compose()
    jaeger = data["services"]["jaeger"]
    port_strings = [str(p) for p in jaeger.get("ports") or []]
    joined = " ".join(port_strings)
    assert "16686" in joined  # Jaeger UI
    assert "4317" in joined   # OTLP gRPC
    assert "4318" in joined   # OTLP HTTP


def test_every_service_declares_healthcheck() -> None:
    data = _load_compose()
    for service_name in ("prometheus", "grafana", "jaeger"):
        service = data["services"][service_name]
        # PyYAML returns a dict with "healthcheck"; fallback uses has_healthcheck.
        assert service.get("healthcheck") or service.get("has_healthcheck"), (
            f"service {service_name!r} has no healthcheck"
        )


def test_grafana_depends_on_prometheus() -> None:
    """Grafana's health depends on Prometheus; capture the explicit dep."""
    data = _load_compose()
    grafana = data["services"]["grafana"]
    # PyYAML: {"depends_on": [...]}, fallback: raw lines contain "depends_on".
    if "depends_on" in grafana:
        deps = grafana["depends_on"]
        if isinstance(deps, dict):
            assert "prometheus" in deps
        else:
            assert "prometheus" in deps
    else:
        raw = "".join(grafana.get("raw", []))
        assert "depends_on" in raw and "prometheus" in raw


def test_volumes_declared_for_prometheus_and_grafana() -> None:
    text = _COMPOSE_PATH.read_text()
    assert "prometheus_data" in text
    assert "grafana_data" in text


def test_prometheus_config_is_mounted_read_only() -> None:
    text = _COMPOSE_PATH.read_text()
    assert "./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in text


def test_grafana_dashboards_mounted_read_only() -> None:
    text = _COMPOSE_PATH.read_text()
    assert "./grafana/dashboards:/var/lib/grafana/dashboards:ro" in text


def test_grafana_provisioning_mounted_read_only() -> None:
    text = _COMPOSE_PATH.read_text()
    assert "./grafana/provisioning:/etc/grafana/provisioning:ro" in text


def test_smoke_script_exists_and_is_executable() -> None:
    assert _SMOKE_PATH.is_file()
    assert os.access(_SMOKE_PATH, os.X_OK), "monitoring-smoke.sh must be executable"


def test_smoke_script_uses_bash_strict_mode() -> None:
    text = _SMOKE_PATH.read_text()
    assert text.splitlines()[0].startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text


def test_smoke_script_probes_all_three_services() -> None:
    text = _SMOKE_PATH.read_text()
    assert "9090" in text or "PROM_URL" in text
    assert "3000" in text or "GRAFANA_URL" in text
    assert "16686" in text or "JAEGER_URL" in text


def test_smoke_script_accepts_override_flags() -> None:
    text = _SMOKE_PATH.read_text()
    for flag in ("--metrics-url", "--prom-url", "--grafana-url", "--jaeger-url"):
        assert flag in text


def test_smoke_script_exits_non_zero_on_failure() -> None:
    text = _SMOKE_PATH.read_text()
    assert "exit 1" in text
