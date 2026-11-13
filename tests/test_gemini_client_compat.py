"""Direct Antigravity compatibility against the real router and fake upstreams."""
import hashlib
import json
from uuid import uuid4

import pytest

from model_router.client_compat import prepare_gemini
from test_gemini_ingress import (
    HELLO, NATIVE_PATH, STREAM_PATH, configured, generate, stream,
)
from test_model_router import router_config, upstream


MODELS = [
    ("m-gem", NATIVE_PATH),
    ("m-chat", "/v1/chat/completions"),
    ("m-resp", "/v1/responses"),
    ("m-anth", "/v1/messages"),
]


def config(upstream, *, enabled=True):
    raw = configured(upstream, bearer_session_compat=enabled)
    raw["server"]["auth_mode"] = "none"
    for provider in raw["providers"].values():
        provider.update(session_header="x-opencode-session", session_missing="reject")
    return raw


def token():
    return "router-session-" + str(uuid4())


def acp_identity(session):
    return "agy-acp-" + hashlib.sha256(json.dumps(session, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("model,path", MODELS)
@pytest.mark.parametrize("carrier", ["x-goog-api-key", "x-api-key", "bearer", "query"])
def test_public_gemini_tokens_preserve_affinity_across_streams(upstream, model, path, carrier):
    first, second = token(), token()
    with router_config(config(upstream)) as base:
        for session, streaming in [(first, False), (first, True), (second, False)]:
            headers, params = {}, {"alt": "sse"} if streaming else {}
            if carrier == "query":
                params["key"] = session
            elif carrier == "bearer":
                headers["Authorization"] = "Bearer " + session
            else:
                headers[carrier] = session
            response = (stream if streaming else generate)(
                base, model, headers=headers, params=params,
            )
            assert response.status_code == 200, response.text
            if streaming:
                assert "data:" in response.text
            request = upstream.last_request(STREAM_PATH if streaming and model == "m-gem" else path)
            assert request["headers"]["x-opencode-session"] == session
            assert request["headers"].get("authorization") != "Bearer " + session
            assert request["headers"].get("x-goog-api-key") != session
            assert request["headers"].get("x-api-key") != session
            assert "key=" not in request["query"]
            assert "sessionId" not in request["body"]


@pytest.mark.parametrize("model,path", MODELS)
def test_acp_conversation_wins_over_per_launch_token_and_is_consumed(upstream, model, path):
    with router_config(config(upstream)) as base:
        for session, streaming in [("conversation one / ☃", False), ("conversation one / ☃", True),
                                   ("conversation two", False)]:
            response = (stream if streaming else generate)(
                base, model, {**HELLO, "sessionId": session},
                headers={"x-goog-api-key": token()},
            )
            assert response.status_code == 200, response.text
            request = upstream.last_request(STREAM_PATH if streaming and model == "m-gem" else path)
            assert request["headers"]["x-opencode-session"] == acp_identity(session)
            assert "sessionId" not in request["body"]


def test_native_identity_has_priority_and_duplicate_headers_still_fail(upstream):
    with router_config(config(upstream)) as base:
        response = generate(base, "m-chat", {**HELLO, "sessionId": "body-session"}, headers={
            "x-opencode-session": "native", "x-goog-api-key": token(),
        })
        assert response.status_code == 200, response.text
        assert upstream.last_request("/v1/chat/completions")["headers"]["x-opencode-session"] == "native"
        response = generate(base, "m-gem", {**HELLO, "sessionId": "body-session"}, headers=[
            ("x-opencode-session", "one"), ("x-opencode-session", "two"),
            ("x-goog-api-key", token()),
        ])
        assert response.status_code == 400
        assert "Duplicate conversation identity header" in response.text


@pytest.mark.parametrize("session", [False, 0, [], {}])
def test_invalid_acp_identity_is_a_client_error(upstream, session):
    with router_config(config(upstream)) as base:
        response = generate(base, "m-chat", {**HELLO, "sessionId": session},
                            headers={"x-goog-api-key": token()})
    assert response.status_code == 400
    assert "sessionId" in response.text


def test_null_or_missing_acp_identity_uses_the_launch_token():
    session = token()
    for value in (None, ""):
        payload, fallback = prepare_gemini({**HELLO, "sessionId": value}, {"x-goog-api-key": session})
        assert payload == HELLO and fallback == session
    assert prepare_gemini(HELLO, {"x-goog-api-key": "ordinary-client-key"}) == (HELLO, None)
    assert prepare_gemini({**HELLO, "sessionId": "conversation"}, {}) == (HELLO, acp_identity("conversation"))


def test_conflicting_public_tokens_are_rejected(upstream):
    with router_config(config(upstream)) as base:
        response = generate(base, "m-chat", headers={"x-goog-api-key": token()},
                            params={"key": token()})
    assert response.status_code == 400
    assert "Conflicting public session tokens" in response.text


def test_compatibility_off_keeps_native_gemini_payloads_and_ignores_tokens(upstream):
    body = {**HELLO, "sessionId": "native-vendor-field"}
    raw = config(upstream, enabled=False)
    for provider in raw["providers"].values():
        provider["session_missing"] = "omit"
    with router_config(raw) as base:
        response = generate(base, "m-gem", body, headers={"x-goog-api-key": token()})
    assert response.status_code == 200, response.text
    request = upstream.last_request(NATIVE_PATH)
    assert request["body"] == body
    assert "x-opencode-session" not in request["headers"]


def test_session_metadata_does_not_bypass_gemini_validation(upstream):
    with router_config(config(upstream)) as base:
        response = generate(base, "m-gem", {"sessionId": "conversation", "contents": []})
    assert response.status_code == 400
    assert "contents" in response.text


def test_query_session_tokens_and_body_identity_are_not_logged(upstream, caplog):
    caplog.set_level("INFO", logger="model-router")
    session, body_session = token(), "private-conversation-identity"
    with router_config(config(upstream)) as base:
        response = generate(base, "m-chat", {**HELLO, "sessionId": body_session},
                            params={"key": session})
    assert response.status_code == 200, response.text
    assert session not in caplog.text and body_session not in caplog.text


def test_json_array_streaming_keeps_session_affinity(upstream):
    session = token()
    with router_config(config(upstream)) as base:
        response = stream(base, "m-chat", params={"key": session})
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
    assert upstream.last_request("/v1/chat/completions")["headers"]["x-opencode-session"] == session
