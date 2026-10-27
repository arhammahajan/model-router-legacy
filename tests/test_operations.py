"""Prompt 14: cached probes, protected diagnostics, outcomes and CLI contracts."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import model_router as mr
import model_router.cli as router_cli
from test_discovery_refresh import FakeClock, LocalDiscovery, config as discovery_config
from test_http_capacity import ObservedServer, config as static_config




@pytest.fixture
def discovery():
    upstream = LocalDiscovery()
    try:
        yield upstream
    finally:
        upstream.close()


@contextmanager
def running(app):
    # Scheduling is deliberately stopped: these HTTP requests must use cache.
    server = ObservedServer(("127.0.0.1", 0), type("TestHandler", (mr.Handler,), {"app": app}))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            yield client, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        app.close()
        assert not thread.is_alive()
        assert server.errors.empty(), list(server.errors.queue)


def outcomes(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records
            if hasattr(record, "router_event") and record.router_event["event"] == "request_outcome"]


def test_public_probes_and_diagnostic_auth_are_network_free(discovery, monkeypatch, caplog):
    caplog.set_level("INFO", logger="model-router")
    cfg = discovery_config(discovery, catalog_interval=0, llama_interval=0)
    cfg.api_keys = ["dummy-router-secret"]
    app = mr.App(cfg)

    def forbidden(*args, **kwargs):
        pytest.fail("a cached operational endpoint attempted network work")

    monkeypatch.setattr(app.http, "send", forbidden)
    monkeypatch.setattr(app, "refresh_catalog", forbidden)
    monkeypatch.setattr(app, "refresh_llama_cpp", forbidden)
    attempts = discovery.attempts()
    with running(app) as (client, server):
        ids = set()
        for path in ("/livez", "/health", "/healthz", "/readyz"):
            response = client.get(path, headers={"X-Request-ID": "untrusted-id"})
            assert response.status_code == 200
            assert response.json() == {"status": "ready" if path == "/readyz" else "ok"}
            ids.add(response.headers["x-request-id"])
            assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])
            server.finished.get(timeout=3)
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            response = client.get("/diagnostics", headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"]
            assert "provider_details" not in response.text
            server.finished.get(timeout=3)
        for headers in ({"Authorization": "Bearer dummy-router-secret"},
                        {"x-api-key": "dummy-router-secret"}):
            response = client.get("/diagnostics", headers=headers)
            assert response.status_code == 200
            assert response.json()["readiness"]["ready"] is True
            server.finished.get(timeout=3)
        assert len(ids) == 4
        assert discovery.attempts() == attempts
    records = outcomes(caplog)
    assert len(records) == 8
    assert {record["http_status"] for record in records} == {200, 401}
    assert all(record["elapsed_ms"] >= 0 for record in records)
    assert "dummy-router-secret" not in caplog.text


def test_diagnostics_disabled_without_router_auth():
    app = mr.App(static_config(auth_mode="none"))
    with running(app) as (client, server):
        response = client.get("/diagnostics", headers={"Authorization": "Bearer any-key"})
        server.finished.get(timeout=3)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "diagnostics_disabled"




def test_partial_outage_and_retained_catalog_routes(discovery):
    cfg = discovery_config(discovery, catalog_interval=0, llama_interval=0)
    # Only discovered routes count; dynamic provider/model dispatch alone does not.
    cfg.routes.clear()
    cfg.public_routes.clear()
    clock = FakeClock()
    app = mr.App(cfg, clock=clock)
    with running(app) as (client, server):
        discovery.fail("one")
        app.refresh_llama_cpp(force=True)
        assert app.readiness_payload()["usable_routes"] == 2  # catalog + two
        assert client.get("/readyz").status_code == 200
        server.finished.get(timeout=3)
        discovery.fail("two", "catalog")
        clock.advance_to(100)
        app.refresh_llama_cpp(force=True)
        app.refresh_catalog(force=True)
        # Catalog failure does not prove that its generation endpoints failed.
        assert app.readiness_payload()["usable_routes"] == 1
        assert app.discovery_summary()["catalog"] == {
            "stale": True, "age_seconds": 100, "stale_age_seconds": 0,
            "consecutive_failures": 1,
        }
        # If local routes are all that remain, known failed servers are unready.
        app._publish_registry(catalog=mr.CatalogSnapshot())
        response = client.get("/readyz")
        server.finished.get(timeout=3)
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}
        assert app.resolve("local-one") is not None  # stale routes remain dispatchable
        discovery.recover()
        app.refresh_llama_cpp(force=True)
        assert app.readiness_payload()["ready"] is True


def test_never_discovered_and_successful_empty_inventory_are_unready(discovery):
    discovery.fail("one", "two", "catalog")
    cfg = discovery_config(discovery, catalog_interval=0, llama_interval=0)
    cfg.routes.clear()
    cfg.public_routes.clear()
    app = mr.App(cfg)
    with running(app) as (client, server):
        assert app.catalog_status()["loaded"] is False
        assert app.llama_cpp_status()["loaded"] is False
        assert client.get("/readyz").status_code == 503
        server.finished.get(timeout=3)
        assert client.get("/livez").json() == {"status": "ok"}
        server.finished.get(timeout=3)
        discovery.recover()
        discovery.models = {"catalog": [], "one": [], "two": []}
        app.refresh_catalog(force=True)
        app.refresh_llama_cpp(force=True)
        assert app.catalog_status()["loaded"] is True
        assert app.readiness_payload()["ready"] is False


def test_explicit_routes_survive_first_discovery_failure_and_disabled_routes_do_not_count(discovery):
    discovery.fail("one", "two", "catalog")
    cfg = discovery_config(discovery, catalog_interval=0, llama_interval=0)
    app = mr.App(cfg)
    try:
        assert app.readiness_payload()["usable_routes"] == 1
        assert app.catalog_status()["loaded"] is False
        cfg.routes["explicit"] = replace(cfg.routes["explicit"], enabled=False)
        app._publish_registry()
        assert app.readiness_payload()["ready"] is False
        assert app.readiness_payload()["enabled_routes"] == 0
    finally:
        app.close()




def cli_config(discovery=None):
    raw = {"server": {"auth_mode": "none"}, "providers": {"static": {
        "dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1", "api_key": "dummy-key"}},
        "models": [{"name": "explicit", "provider": "static"}]}
    if discovery:
        raw["models"] = []
        raw["catalog"] = {"url": discovery.url + "/catalog", "refresh_seconds": 0,
                          "rules": [{"match": ".", "provider": "static"}]}
    return raw


@pytest.mark.parametrize("flags,expected", [(["--validate-config"], 0),
    (["--validate-config", "--offline"], 0), (["--check", "--offline"], 4),
    (["--check-readiness", "--offline"], 4)])
def test_offline_cli_never_constructs_runtime(tmp_path, monkeypatch, capsys, flags, expected):
    path = tmp_path / "router.json"
    raw = cli_config()
    raw["catalog"] = {"url": "http://127.0.0.1:1/catalog", "refresh_seconds": 0,
                      "rules": [{"match": ".", "provider": "static"}]}
    path.write_text(json.dumps(raw))
    def forbidden(*args, **kwargs):
        pytest.fail("offline CLI constructed a network-capable runtime")
    monkeypatch.setattr(router_cli, "App", forbidden)
    assert mr.main(["--no-default-env", "-c", str(path), *flags]) == expected
    assert json.loads(capsys.readouterr().out)["status"] == ("valid" if expected == 0 else "indeterminate")


@pytest.mark.parametrize("flag", ["--check", "--check-readiness"])
@pytest.mark.parametrize("failed", [False, True])
def test_cli_readiness_exit_and_startup_only_attempt(discovery, tmp_path, capsys, flag, failed):
    if failed:
        discovery.fail("catalog")
    path = tmp_path / "router.json"
    path.write_text(json.dumps(cli_config(discovery)))
    assert mr.main(["--no-default-env", "-c", str(path), flag]) == (3 if failed else 0)
    assert discovery.attempts() == (1, 0, 0)
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is not failed
    assert payload["discovery"]["catalog"]["consecutive_failures"] == int(failed)
    assert not any(t.name == "model-router-discovery" for t in threading.enumerate())


@pytest.mark.parametrize("flags", [["--validate-config"], ["--check"], ["--check", "--offline"]])
def test_cli_config_failures_exit_two_in_actual_process(tmp_path, flags):
    path = tmp_path / "router.json"
    raw = cli_config()
    raw["server"]["port"] = "invalid"
    path.write_text(json.dumps(raw))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "model_router.py"),
                             "--no-default-env", "-c", str(path), *flags],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "server.port" in result.stderr
    assert "Traceback" not in result.stderr






