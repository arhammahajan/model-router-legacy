"""Built-in Muse discovery uses the router inventory and generation transport."""
from __future__ import annotations

import httpx
import pytest

import model_router as mr
from test_model_router import _body, _config, router_config, upstream  # noqa: F401


@pytest.mark.parametrize("path", ["/muse-code/models", "/v1/muse-code/models"])
def test_muse_catalog_limits_and_standard_inventory_are_independent(upstream, path):
    raw = _config(upstream)
    raw["models"][2]["metadata"] = {"display_name": "Responses model"}
    raw["models"] += [
        {"name": "fallback", "provider": "resp"},
        {"name": "disabled", "provider": "resp", "enabled": False},
    ]
    with router_config(raw) as base:
        response = httpx.get(base + path)
        standard = httpx.get(base + "/v1/models").json()
    assert response.status_code == 200
    payload = response.json()
    rows = {row["id"]: row for row in payload["data"]}
    assert set(rows) == {"m-anth", "m-chat", "m-resp", "fallback"}
    assert rows["m-resp"]["metadata"]["muse-code"] == {
        "display_label": "Responses model", "is_hidden": False,
        "limit": {"context": 128000, "output": 16384},
    }
    assert rows["fallback"]["metadata"]["muse-code"]["limit"] == {
        "context": 128000, "output": 16384,
    }
    assert payload["first_id"] == payload["data"][0]["id"]
    assert payload["last_id"] == payload["data"][-1]["id"]
    assert payload["has_more"] is False
    assert {row["id"] for row in standard["data"]} == {"m-anth", "m-chat", "m-resp", "fallback"}
    assert all("muse-code" not in row["metadata"] for row in standard["data"])
    assert upstream.requests == []  # Discovery must not call an upstream or run inference.


def test_muse_catalog_uses_reported_limits(monkeypatch, upstream):
    app = mr.App(mr.build_config(_config(upstream)))
    try:
        payload = app.models_payload()
        entry = next(row for row in payload["data"] if row["id"] == "m-resp")
        limits = entry["metadata"]["router"]["limits"]
        limits["context_window"]["value"] = 8192
        limits["max_output_tokens"]["value"] = 1024
        monkeypatch.setattr(app, "models_payload", lambda: payload)
        result = app.muse_models_payload()
        row = next(row for row in result["data"] if row["id"] == "m-resp")
        assert row["metadata"]["muse-code"]["limit"] == {
            "context": 8192, "output": 1024,
        }
    finally:
        app.close()


def test_muse_catalog_without_native_responses_keeps_other_protocols(upstream):
    raw = _config(upstream)
    raw["models"] = raw["models"][:2]
    app = mr.App(mr.build_config(raw))
    try:
        assert {row["id"] for row in app.muse_models_payload()["data"]} == {"m-anth", "m-chat"}
    finally:
        app.close()


@pytest.mark.parametrize("path", ["/muse-code/models", "/v1/muse-code/models"])
def test_muse_catalog_requires_the_same_auth_as_models(upstream, path):
    with router_config(_config(upstream, api_keys=["router-secret"])) as base:
        assert httpx.get(base + path).status_code == 401
        assert httpx.get(base + path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        for headers in ({"Authorization": "Bearer router-secret"}, {"x-api-key": "router-secret"}):
            response = httpx.get(base + path, headers=headers)
            assert response.status_code == 200
            assert {row["id"] for row in response.json()["data"]} == {"m-anth", "m-chat", "m-resp"}
    assert upstream.requests == []


@pytest.mark.parametrize("stream", [False, True])
def test_muse_uses_existing_responses_transport_after_discovery(upstream, stream):
    with router_config(_config(upstream)) as base:
        catalog = httpx.get(base + "/v1/muse-code/models").json()
        model = next(row["id"] for row in catalog["data"] if row["id"] == "m-resp")
        body = {**_body("openai-responses", model, stream), "store": False,
                "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"}}
        response = httpx.post(base + "/v1/responses", json=body,
                              headers={"x-session-id": "muse-conversation"}, timeout=10)
    assert response.status_code == 200, response.text
    if stream:
        assert "response.completed" in response.text
    else:
        assert response.json()["object"] == "response"
    request = upstream.last_request("/v1/responses")
    assert request["body"] == {**body, "model": "upstream-responses"}
    assert request["headers"]["x-opencode-session"] == "muse-conversation"
