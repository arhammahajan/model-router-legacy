"""Review the actual bundled Go routing against frozen public evidence, offline."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

import model_router as mr


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
ENDPOINTS = json.loads((FIXTURES / "opencode_go_endpoints.json").read_text())["models"]
UNMAPPED = {
    "glm-5.1", "grok-4.6", "iceman-contributor-free", "kimi-k2.6",
    "minimax-m2.5", "minimax-m2.7", "omen-alpha", "qwen3.6-plus", "qwen3.7-max",
}


@pytest.fixture
def bundled_raw(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Load the shipped file, without invoking CLI credential-file loading.
    monkeypatch.setenv("OPENCODE_API_KEY", "dummy-go-provider-key")
    monkeypatch.setenv("MODEL_ROUTER_API_KEY", "")
    monkeypatch.setenv("MODEL_ROUTER_API_KEYS", "")
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    return mr._load_config_file(ROOT / "router.toml")


@pytest.fixture
def offline_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "payload": json.loads((FIXTURES / "opencode_go_catalog.json").read_text()),
        "requests": [],
    }

    def send(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        # Replace send itself: unexpected requests fail instead of using any
        # real transport. Inbound router tests use local http.client sockets.
        state["requests"].append(request)
        if request.method == "GET" and str(request.url) == "https://opencode.ai/inference/go/v1/models":
            assert "authorization" not in request.headers
            assert "x-api-key" not in request.headers
            assert request.headers["user-agent"] == f"litellm-model-router/{mr.__version__}"
            return httpx.Response(200, json=state["payload"], request=request)
        if request.method == "GET" and request.url.host == "127.0.0.1" and request.url.path == "/health":
            return httpx.Response(503, request=request)
        if request.method == "POST" and str(request.url) in {
            info["endpoint"] for info in ENDPOINTS.values()
        }:
            request.read()
            return httpx.Response(200, json={"ok": True}, request=request)
        pytest.fail(f"Unexpected outbound request: {request.method} {request.url}")

    monkeypatch.setattr(httpx.Client, "send", send)
    return state


@pytest.fixture
def bundled_app(bundled_raw: dict[str, Any], offline_catalog: dict[str, Any]) -> Iterator[mr.App]:
    app = mr.App(mr.build_config(bundled_raw))
    try:
        yield app
    finally:
        app.close()


def test_bundled_mapping_matches_entire_reviewed_endpoint_table(bundled_app: mr.App) -> None:
    app = bundled_app
    assert app.config.catalog is not None
    assert not app.config.catalog.rules  # no family prefix or catch-all guesses
    assert set(app.config.catalog.model_providers) == set(ENDPOINTS)
    assert {row["id"] for row in app.models_payload()["data"]} == set(ENDPOINTS)
    for model, expected in ENDPOINTS.items():
        route = app.resolve(model)
        assert route is not None
        provider = app.provider(route.provider_name)
        assert route.upstream_model == model
        assert route.dialect == provider.dialect == expected["dialect"]
        endpoint = mr._upstream_url(provider, route.dialect)
        if route.dialect == "gemini":
            endpoint += "/models/" + model
        assert endpoint == expected["endpoint"]
        assert provider.api_key == "dummy-go-provider-key"
        assert provider.headers["User-Agent"] == f"litellm-model-router/{mr.__version__}"
    assert app.catalog_status()["loaded"] is True
    assert app.catalog_status()["error"] is None
    assert set(app.catalog_status()["unmapped_models"]) == UNMAPPED
    assert "dummy-go-provider-key" not in json.dumps(app.models_payload())


@pytest.mark.parametrize(("model", "dialect"), [
    ("glm-5.3-flash", "openai-chat"),
    ("kimi-k3.1-pro", "openai-chat"),
    ("deepseek-v4.1-flash", "openai-chat"),
    ("mimo-v2.6-pro", "openai-chat"),
    ("minimax-m3.1-flash", "openai-chat"),
    ("minimax-m3", "anthropic"),
    ("qwen3.8-max", "anthropic"),
    ("qwen3.8-flash", "anthropic"),
    ("qwen3.7-plus", "anthropic"),
    ("claude-haiku-5-5", "anthropic"),
    ("grok-4.7", "openai-responses"),
    ("gpt-6-luna", "openai-responses"),
    ("muse-spark-1.3-contributor", "openai-responses"),
])
def test_bundled_routes_send_to_documented_endpoint_with_trusted_key(
    bundled_app: mr.App, offline_catalog: dict[str, Any], model: str, dialect: str,
) -> None:
    route = bundled_app.resolve(model)
    assert route is not None and route.dialect == dialect
    body: dict[str, Any] = {"model": model, "stream": False}
    body.update({"input": "Hi"} if dialect == "openai-responses" else {
        "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32,
    })
    result = bundled_app.dispatch(dialect, route, body, {
        "Authorization": "Bearer dummy-client-key", "x-api-key": "dummy-client-key",
    })
    assert result.status == 200
    request = offline_catalog["requests"][-1]
    assert str(request.url) == ENDPOINTS[model]["endpoint"]
    assert json.loads(request.content)["model"] == model
    if dialect == "anthropic":
        assert request.headers["x-api-key"] == "dummy-go-provider-key"
        assert "authorization" not in request.headers
    else:
        assert request.headers["authorization"] == "Bearer dummy-go-provider-key"
        assert "x-api-key" not in request.headers


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses", "anthropic"])
def test_unknown_catalog_models_are_diagnosed_without_dispatch(
    bundled_app: mr.App, offline_catalog: dict[str, Any], ingress: str,
) -> None:
    new_models = {"claude-new", "gemini-new", "gpt-new", "qwen-new", "minimax-new", "brand-new"}
    offline_catalog["payload"]["data"].extend({"id": model} for model in new_models)
    bundled_app.refresh_catalog(force=True)
    assert set(bundled_app.catalog_status()["unmapped_models"]) == UNMAPPED | new_models
    listed = {row["id"] for row in bundled_app.models_payload()["data"]}
    assert not listed & (UNMAPPED | new_models)

    server = mr.serve(bundled_app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        for model in sorted(UNMAPPED | new_models | {"absent-model"}):
            path = {"openai-chat": "/v1/chat/completions", "openai-responses": "/v1/responses",
                    "anthropic": "/v1/messages"}[ingress]
            body = {"model": model, "input": "Hi", "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Hi"}]}
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                response = connection.getresponse()
                data = json.loads(response.read())
                assert response.status == 404
                if model == "absent-model":
                    assert "not configured" in data["error"]["message"]
                else:
                    assert "endpoint compatibility is unknown" in data["error"]["message"]
                    assert "catalog.model_providers" in data["error"]["message"]
                    if ingress != "anthropic":
                        assert data["error"]["code"] == "model_endpoint_unknown"
            finally:
                connection.close()
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert set(json.loads(response.read())["catalog"]["unmapped_models"]) == UNMAPPED | new_models
        finally:
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert not any(request.method == "POST" for request in offline_catalog["requests"])


def test_exact_overrides_beat_rules_and_explicit_routes_beat_catalog(
    bundled_raw: dict[str, Any], offline_catalog: dict[str, Any],
) -> None:
    bundled_raw["catalog"]["rules"] = [{"match": ".", "provider": "opencode-openai"}]
    bundled_raw["catalog"]["model_providers"]["minimax-m3"] = "opencode-chat"
    bundled_raw["catalog"]["model_providers"]["qwen3.6-plus"] = "opencode-anthropic"
    bundled_raw["catalog"]["model_providers"]["not-in-catalog"] = "opencode-chat"
    bundled_raw["models"] = [
        {"name": "deepseek-v4.1-flash", "provider": "opencode-anthropic", "aliases": ["explicit"]},
    ]
    app = mr.App(mr.build_config(bundled_raw))
    try:
        assert app.resolve("minimax-m3").dialect == "openai-chat"
        assert app.resolve("qwen3.6-plus").dialect == "anthropic"
        assert app.resolve("grok-4.6").dialect == "openai-responses"  # explicit rule opt-in
        assert app.resolve("not-in-catalog") is None
        assert app.resolve("deepseek-v4.1-flash").dialect == "anthropic"
        assert app.resolve("explicit") is app.resolve("deepseek-v4.1-flash")
        assert app.catalog_status()["unmapped_models"] == []
    finally:
        app.close()


def test_explicit_route_can_enable_unmapped_model_without_catalog_guess(
    bundled_raw: dict[str, Any], offline_catalog: dict[str, Any],
) -> None:
    bundled_raw["models"] = [{"name": "qwen3.6-plus", "provider": "opencode-chat"}]
    app = mr.App(mr.build_config(bundled_raw))
    try:
        assert app.resolve("qwen3.6-plus").dialect == "openai-chat"
        assert set(app.catalog_status()["unmapped_models"]) == UNMAPPED - {"qwen3.6-plus"}
    finally:
        app.close()


def test_all_unmapped_and_empty_refreshes_replace_old_routes(
    bundled_app: mr.App, offline_catalog: dict[str, Any],
) -> None:
    offline_catalog["payload"] = {"data": [{"id": "unreviewed"}]}
    bundled_app.refresh_catalog(force=True)
    assert bundled_app.models_payload()["data"] == []
    assert bundled_app.resolve("gpt-6-luna") is None
    assert bundled_app.catalog_status()["models"] == 0
    assert bundled_app.catalog_status()["loaded"] is True
    assert bundled_app.catalog_status()["error"] is None
    assert bundled_app.catalog_status()["unmapped_models"] == ["unreviewed"]
    offline_catalog["payload"] = {"data": []}
    bundled_app.refresh_catalog(force=True)
    assert bundled_app.catalog_status()["unmapped_models"] == []
    assert bundled_app.resolve("unreviewed") is None


def test_catalog_cannot_choose_credentials_or_endpoints(
    bundled_raw: dict[str, Any], offline_catalog: dict[str, Any],
) -> None:
    offline_catalog["payload"] = {"data": [{
        "id": "minimax-m3", "provider": "opencode-openai", "dialect": "openai-responses",
        "api_key": "catalog-key", "base_url": "https://untrusted.invalid",
        "metadata": {"provider": "opencode-openai", "dialect": "openai-responses"},
    }]}
    app = mr.App(mr.build_config(bundled_raw))
    try:
        route = app.resolve("minimax-m3")
        assert route.dialect == "anthropic"
        app.dispatch("anthropic", route, {"model": "minimax-m3", "max_tokens": 32,
                     "messages": [{"role": "user", "content": "Hi"}]}, {})
        request = offline_catalog["requests"][-1]
        assert str(request.url) == ENDPOINTS["minimax-m3"]["endpoint"]
        assert request.headers["x-api-key"] == "dummy-go-provider-key"
        row = app.models_payload()["data"][0]
        assert row["metadata"]["dialect"] == "anthropic"
        assert row["metadata"]["provider"] == "opencode-anthropic"
    finally:
        app.close()


@pytest.mark.parametrize("mapping", [
    None, [], "opencode-chat", {"": "opencode-chat"}, {" spaced ": "opencode-chat"},
    {1: "opencode-chat"}, {"model": "missing"}, {"model": {"api_key": "untrusted"}},
])
def test_exact_mapping_validation(bundled_raw: dict[str, Any], mapping: Any) -> None:
    bundled_raw["catalog"]["model_providers"] = mapping
    with pytest.raises(SystemExit, match="catalog.model_providers"):
        mr.build_config(bundled_raw)
