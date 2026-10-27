"""Prompt 11: conversation affinity with real adapters and local fake upstreams."""
from __future__ import annotations

from email.message import Message

import httpx
import pytest

import model_router as mr
import model_router.config as router_configuration
import model_router.constants as router_constants
import model_router.headers as router_headers
from test_model_router import PATHS, _body, _config, _with_gemini, router_config, upstream


INGRESSES = tuple(PATHS)
DESTINATIONS = (
    ("m-chat", "/v1/chat/completions"),
    ("m-resp", "/v1/responses"),
    ("m-anth", "/v1/messages"),
    ("m-gem", "/v1beta/models/gemini-2.5-flash:generateContent"),
)


def session_config(upstream):
    raw = _with_gemini(_config(upstream), upstream)
    raw["providers"]["gem"]["session_header"] = "x-opencode-session"
    return raw


def provider(**policy):
    return router_configuration._build_provider("test", {
        "dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1",
        "api_key": "dummy-provider-key", "session_header": "x-opencode-session",
        **policy,
    })




@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,path", DESTINATIONS)
def test_explicit_header_wins_over_native_headers_and_cache_fallback(
    upstream, ingress, model, path,
):
    raw = session_config(upstream)
    for settings in raw["providers"].values():
        settings["session_prompt_cache_key"] = True
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, model, False), "prompt_cache_key": "cache-conversation",
        }, headers={
            "X-OpenCode-Session": "explicit-conversation",
            "x-claude-code-session-id": "claude-conversation",
            "session-id": "codex-conversation", "x-session-id": "other-conversation",
        }, timeout=30)
    assert response.status_code == 200, response.text
    request = upstream.last_request(path)
    assert request["headers"]["x-opencode-session"] == "explicit-conversation"
    assert not (set(router_constants._NATIVE_SESSION_HEADERS) & request["headers"].keys())






@pytest.mark.parametrize("ingress", INGRESSES)
def test_required_identity_fails_before_network_and_recovers(upstream, ingress):
    raw = session_config(upstream)
    for settings in raw["providers"].values():
        settings["session_missing"] = "reject"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-chat", False))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["message"] == (
            "Conversation identity required: send x-opencode-session with the same value on every turn"
        )
        assert error.get("code", error["type"]) == (
            "invalid_request_error" if ingress == "anthropic" else "missing_session_id"
        )
        assert upstream.requests == []
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-chat", False),
                              headers={"x-opencode-session": "recovered-conversation"})
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("header", router_constants._NATIVE_SESSION_HEADERS)
def test_each_reviewed_native_header_is_recognized_and_validated(header):
    p = provider()
    assert router_headers._session_id(p, {}, {header.upper(): "conversation-1"}) == "conversation-1"
    with pytest.raises(mr.RouterError) as error:
        router_headers._session_id(p, {}, {header: ""})
    assert error.value.status == 400


def test_configured_native_priority_and_disabling_sources():
    p = provider(session_source_headers=["SESSION-ID", "x-claude-code-session-id"],
                 session_prompt_cache_key=True)
    sources = {"x-claude-code-session-id": "claude", "session-id": "codex"}
    assert router_headers._session_id(p, {"prompt_cache_key": "cache"}, sources) == "codex"
    assert router_headers._session_id(p, {}, dict(reversed(list(sources.items())))) == "codex"
    assert router_headers._session_id(provider(session_source_headers=[]), {}, sources) is None
    assert router_headers._session_id(provider(), {"prompt_cache_key": "cache"}, {}) is None
    assert router_headers._session_id(provider(session_source_headers=[], session_prompt_cache_key=True),
                          {"prompt_cache_key": "cache"}, sources) == "cache"








@pytest.mark.parametrize("ingress", INGRESSES)
def test_duplicate_identity_headers_are_rejected_at_http_boundary(upstream, ingress):
    with router_config(session_config(upstream)) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-chat", False),
                              headers=[("x-opencode-session", "first"),
                                       ("X-OpenCode-Session", "second")])
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Duplicate conversation identity header"
    assert upstream.requests == []








def test_session_policy_disabled_retains_native_passthrough_behavior():
    p = provider(session_header="")
    assert router_headers._session_id(p, {}, {"session-id": "client-conversation"}) is None
    headers = router_headers._upstream_request_headers(p, "openai-chat", {"session-id": "client-conversation"})
    assert headers["session-id"] == "client-conversation"
