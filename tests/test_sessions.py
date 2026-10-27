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
@pytest.mark.parametrize("stream", [False, True])
def test_repeated_turns_and_independent_conversations_on_the_wire(
    upstream, ingress, model, path, stream,
):
    raw = session_config(upstream)
    # The installed adapter simulates streaming for unknown Responses models.
    raw["models"][2]["model"] = "gpt-4.1"
    native_header = {
        "openai-chat": "X-Session-Id", "openai-responses": "SeSsIoN-Id",
        "anthropic": "X-Claude-Code-Session-Id",
    }[ingress]
    expected = ["conversation-A", "conversation-B", "conversation-A", "conversation-B"]
    with router_config(raw) as base, httpx.Client(timeout=30) as client:
        for turn, identity in enumerate(expected):
            body = _body(ingress, model, stream)
            # Prompt contents grow while identity remains the same; conversations
            # A and B also start with identical prompts and use the same API key.
            text = "hello" if turn < 2 else "hello, next turn"
            if ingress == "openai-responses":
                body["input"] = text
            else:
                body["messages"][0]["content"] = text
            response = client.post(base + PATHS[ingress], json=body, headers={
                native_header: identity, "Authorization": "Bearer dummy-client-key",
            })
            assert response.status_code == 200, response.text
    sent = [request for request in upstream.requests if request["body"] is not None]
    assert len(sent) == 4
    assert [request["headers"]["x-opencode-session"] for request in sent] == expected
    for request in sent:
        expected_path = path.replace("generateContent", "streamGenerateContent") if stream else path
        assert request["path"] == expected_path
        assert native_header.lower() not in request["headers"]
        assert "dummy-client-key" not in str(request["headers"])
        assert any(value == "upstream-key" or value == "Bearer upstream-key"
                   for value in request["headers"].values())


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
@pytest.mark.parametrize("model,path", DESTINATIONS)
def test_cache_fallback_is_consumed_on_bridges_and_preserved_on_native_calls(
    upstream, ingress, model, path,
):
    raw = session_config(upstream)
    for settings in raw["providers"].values():
        settings["session_prompt_cache_key"] = True
    with router_config(raw) as base, httpx.Client(timeout=30) as client:
        for identity in ("cache-A", "cache-B", "cache-A"):
            response = client.post(base + PATHS[ingress], json={
                **_body(ingress, model, False), "prompt_cache_key": identity,
            })
            assert response.status_code == 200, response.text
    sent = [request for request in upstream.requests if request["path"] == path]
    assert [request["headers"]["x-opencode-session"] for request in sent] == [
        "cache-A", "cache-B", "cache-A",
    ]
    for request in sent:
        assert ("prompt_cache_key" in request["body"]) == (PATHS[ingress] == path)


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,path", DESTINATIONS)
def test_absent_identity_is_omitted_without_credential_or_prompt_derivation(
    upstream, ingress, model, path,
):
    with router_config(session_config(upstream)) as base, httpx.Client(timeout=30) as client:
        for _ in range(2):
            response = client.post(base + PATHS[ingress], json=_body(ingress, model, False))
            assert response.status_code == 200, response.text
    assert all("x-opencode-session" not in request["headers"] for request in upstream.requests)


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


@pytest.mark.parametrize("invalid", [None, 1, {}, [], "", " ", "abc def", "abc\tdef",
                                     "abc\r\ninjected: yes", "abc\x00def", "abc\x7f", "café",
                                     "a,b", "a" * 257])
def test_invalid_values_fail_even_when_a_higher_priority_source_is_valid(invalid):
    p = provider(session_prompt_cache_key=True)
    for payload, headers in (
        ({"prompt_cache_key": invalid}, {"x-opencode-session": "valid"}),
        ({}, {"x-opencode-session": "valid", "session-id": invalid}),
    ):
        with pytest.raises(mr.RouterError) as error:
            router_headers._session_id(p, payload, headers)
        assert error.value.status == 400
        assert error.value.code == "invalid_session_id"
    valid = "A" * 256
    assert router_headers._session_id(p, {}, {"x-opencode-session": valid}) == valid


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("header_value", ["bad value", "bad,value", "a" * 257, ""])
def test_invalid_header_is_rejected_at_http_boundary(upstream, ingress, header_value):
    with router_config(session_config(upstream)) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-chat", False),
                              headers={"x-opencode-session": header_value})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"].startswith("Conversation identity must be 1-256 ASCII characters:")
    assert error.get("code", error["type"]) == (
        "invalid_request_error" if ingress == "anthropic" else "invalid_session_id"
    )
    assert upstream.requests == []


@pytest.mark.parametrize("same_value", [False, True])
def test_duplicate_headers_rejected_for_real_http_messages_and_case_variant_maps(same_value):
    values = {"x-opencode-session": "first", "X-OpenCode-Session": "first" if same_value else "second"}
    message = Message()
    for key, value in values.items():
        message[key] = value
    for headers in (message, values):
        with pytest.raises(mr.RouterError) as error:
            router_headers._session_id(provider(), {}, headers)
        assert error.value.code == "invalid_session_id"


@pytest.mark.parametrize("ingress", INGRESSES)
def test_duplicate_identity_headers_are_rejected_at_http_boundary(upstream, ingress):
    with router_config(session_config(upstream)) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-chat", False),
                              headers=[("x-opencode-session", "first"),
                                       ("X-OpenCode-Session", "second")])
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Duplicate conversation identity header"
    assert upstream.requests == []


def test_custom_target_and_disabled_native_sources_are_consistent(upstream):
    raw = session_config(upstream)
    for settings in raw["providers"].values():
        settings.update(session_header="X-Conversation-Affinity", session_source_headers=[])
    with router_config(raw) as base:
        for model in ("m-chat", "m-anth"):
            response = httpx.post(base + PATHS["openai-chat"], json=_body("openai-chat", model, False),
                                  headers={"x-session-id": "ignored", "session-id": "ignored"})
            assert response.status_code == 200, response.text
            request = upstream.requests[-1]
            assert "x-conversation-affinity" not in request["headers"]
            assert "x-session-id" not in request["headers"]
            assert "session-id" not in request["headers"]
            response = httpx.post(base + PATHS["openai-chat"], json=_body("openai-chat", model, False),
                                  headers={"X-Conversation-Affinity": "custom-1"})
            assert response.status_code == 200, response.text
            assert upstream.requests[-1]["headers"]["x-conversation-affinity"] == "custom-1"


@pytest.mark.parametrize("policy", [
    {"session_header": "Authorization"}, {"session_header": "X-API-Key"},
    {"session_header": "x-goog-api-key"}, {"session_header": "Cookie"},
    {"session_header": "Host"}, {"session_header": "Content-Length"},
    {"session_header": "anthropic-beta"}, {"session_header": "bad header"},
    {"session_header": "x-session\r\n"}, {"session_header": False},
    {"session_source_headers": "session-id"}, {"session_source_headers": [1]},
    {"session_source_headers": ["authorization"]},
    {"session_source_headers": ["x-request-id"]},
    {"session_source_headers": ["session-id", "SESSION-ID"]},
    {"session_prompt_cache_key": "true"}, {"session_prompt_cache_key": 1},
    {"session_missing": "random"},
    {"session_header": "", "session_missing": "reject"},
])
def test_invalid_session_policy_is_a_config_error(policy):
    with pytest.raises(SystemExit, match="session"):
        provider(**policy)


@pytest.mark.parametrize("level", ["headers", "kwargs_headers", "kwargs_extra_headers"])
@pytest.mark.parametrize("name", ["X-OpenCode-Session", "SESSION-ID", "X-Conversation-Affinity"])
def test_static_global_sessions_are_rejected(level, name):
    static = {name: "global-session"}
    policy = ({"headers": static} if level == "headers" else
              {"litellm_kwargs": {level.removeprefix("kwargs_"): static}})
    with pytest.raises(SystemExit, match="static session headers"):
        provider(session_header="x-conversation-affinity", **policy)


def test_session_policy_disabled_retains_native_passthrough_behavior():
    p = provider(session_header="")
    assert router_headers._session_id(p, {}, {"session-id": "client-conversation"}) is None
    headers = router_headers._upstream_request_headers(p, "openai-chat", {"session-id": "client-conversation"})
    assert headers["session-id"] == "client-conversation"
