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


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    monkeypatch.delenv("MODEL_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_ROUTER_API_KEYS", raising=False)


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


def test_diagnostics_sanitizes_all_details_and_omits_admin_settings(discovery, monkeypatch):
    cfg = discovery_config(discovery, catalog_interval=0, llama_interval=0)
    cfg.api_keys = ["dummy-router-secret"]
    provider = cfg.providers["static"]
    cfg.providers["static"] = replace(provider, headers={"X-Secret": "dummy-header-secret"},
                                      litellm_kwargs={"metadata": {"hidden": "admin-only-value"}})
    app = mr.App(cfg)

    def failed(*args, **kwargs):
        raise RuntimeError('dummy-router-secret dummy-static-key dummy-catalog-key '
                           'dummy-header-secret Bearer unknown-token x-api-key: arbitrary-key '
                           + 'x' * 6000)

    monkeypatch.setattr(app.http, "get", failed)
    app.refresh_catalog(force=True)
    app.refresh_llama_cpp(force=True)
    with running(app) as (client, server):
        response = client.get("/diagnostics", headers={"x-api-key": "dummy-router-secret"})
        server.finished.get(timeout=3)
        assert response.status_code == 200
        details = response.json()
        assert details["readiness"]["ready"] is True
        assert details["provider_details"][0]["reachability"] == "unavailable"
        assert details["catalog"]["refresh"]["consecutive_failures"] == 1
        assert details["catalog"]["refresh"]["stale"] is True
        assert details["llama_cpp"]["servers"][0]["consecutive_failures"] == 1
        assert len(details["catalog"]["error"]) <= mr.MAX_ERROR_MESSAGE
        assert "<redacted>" in response.text
        for secret in ("dummy-router-secret", "dummy-static-key", "dummy-catalog-key",
                       "dummy-header-secret", "unknown-token", "arbitrary-key", "admin-only-value"):
            assert secret not in response.text
        assert "litellm_kwargs" not in response.text
        assert "headers" not in response.text


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


@pytest.mark.parametrize("translated", [False, True])
def test_request_outcomes_correlate_selection_without_payloads(translated, caplog):
    caplog.set_level("DEBUG", logger="model-router")
    cfg = static_config(auth_mode="required", api_keys=["dummy-router-secret"])
    app = mr.App(cfg)
    app.verbose = True
    if translated:
        cfg.providers["local"] = replace(cfg.providers["local"], dialect="anthropic")
        cfg.routes["test"] = replace(cfg.routes["test"], dialect="anthropic")
        app._publish_registry()
    app.dispatch = lambda *args: mr.Plan(body=b'{"completion":"private completion"}',
                                       headers={"X-Request-ID": "upstream-id"})
    with running(app) as (client, server):
        response = client.post("/v1/chat/completions?private=private-query", json={
            "model": "test", "messages": [{"role": "user", "content": "private prompt"}],
        }, headers={"Authorization": "Bearer dummy-router-secret"})
        server.finished.get(timeout=3)
        denied = client.post("/v1/chat/completions", json={"model": "test"})
        server.finished.get(timeout=3)
        missing = client.post("/v1/chat/completions", json={
            "model": "local/private-client-model", "messages": [],
        }, headers={"x-api-key": "dummy-router-secret"})
        server.finished.get(timeout=3)
    record, = [r for r in outcomes(caplog) if r["request_id"] == response.headers["x-request-id"]]
    assert record["http_status"] == 200
    assert record["model"] == record["upstream_model"] == "test"
    assert record["provider"] == "local"
    assert record["translation_mode"] == ("translated" if translated else "passthrough")
    assert response.headers.get_list("x-request-id") == [record["request_id"]]
    assert record["request_id"] != "upstream-id"
    assert record["elapsed_ms"] >= 0
    assert denied.status_code == 401
    assert missing.status_code == 200  # dynamic routing still works, names omitted from logs
    dynamic, = [r for r in outcomes(caplog) if r["request_id"] == missing.headers["x-request-id"]]
    assert dynamic["model"] is dynamic["upstream_model"] is None
    for value in ("private prompt", "private completion", "private-query", "dummy-router-secret",
                  "private-client-model"):
        assert value not in caplog.text


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


def test_cli_unexpected_failure_exit_one_and_verbose_stays_body_free(tmp_path, monkeypatch, caplog):
    path = tmp_path / "router.json"
    path.write_text(json.dumps(cli_config()))
    def failed(*args, **kwargs):
        raise RuntimeError("dummy-key failure")
    monkeypatch.setattr(router_cli, "App", failed)
    caplog.set_level("INFO", logger="model-router")
    assert mr.main(["--no-default-env", "-c", str(path), "--check", "--verbose"]) == 1
    assert "dummy-key" not in caplog.text
    import logging
    assert logging.getLogger("LiteLLM").getEffectiveLevel() >= logging.WARNING


def test_cli_help_explains_migration_and_invalid_offline_usage(capsys):
    with pytest.raises(SystemExit) as result:
        mr.main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for term in ("--validate-config", "--check-readiness", "--check", "startup-only", "indeterminate"):
        assert term in help_text
    with pytest.raises(SystemExit) as result:
        mr.main(["--offline"])
    assert result.value.code == 2


def test_structured_logs_remain_sanitized_valid_json_during_shutdown(caplog):
    cfg = static_config()
    cfg.api_keys = ["dummy-router-secret"]
    app = mr.App(cfg)
    app.close()  # filters are detached before a final handler might finish
    caplog.set_level("INFO", logger="model-router")
    app.log_event(event="discovery_refresh", first="dummy-router-secret " + "x" * 6000,
                  second="Bearer unknown-token " + "y" * 6000)
    record, = [r for r in caplog.records if hasattr(r, "router_event")]
    payload = json.loads(record.getMessage())
    assert len(payload["first"]) == len(payload["second"]) == mr.MAX_ERROR_MESSAGE
    assert "dummy-router-secret" not in record.getMessage()
    assert "unknown-token" not in record.getMessage()
