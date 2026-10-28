"""Gemini API ingress: SDK paths, passthrough, translation, streaming, auth and failures.

Everything runs against local fake upstreams with the real router and the pinned
LiteLLM adapters. No credentials, no network and no paid calls are involved.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any
from urllib.parse import quote

import httpx
import pytest

import model_router as mr
from test_model_router import (
    ANTHROPIC_TEXT, CHAT_TEXT, FakeUpstream, RESPONSES_TEXT, _chat_completion,
    _config, _gemini_response, _parse_sse, _with_gemini, router_config, upstream,
)
from test_protocol_contracts import ARGS, RESULT, response_for
from test_retries import application
from test_operations import running

PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/"
       "iZk9HQAAAABJRU5ErkJggg==")
PDF = base64.b64encode(b"%PDF-1.4 minimal").decode()
WAV = base64.b64encode(b"RIFF....WAVEfmt ").decode()

HELLO = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
CROSS = [
    ("m-chat", CHAT_TEXT, "/v1/chat/completions"),
    ("m-anth", ANTHROPIC_TEXT, "/v1/messages"),
    ("m-resp", RESPONSES_TEXT, "/v1/responses"),
]
CROSS_IDS = [model for model, _, _ in CROSS]
NATIVE_PATH = "/v1beta/models/gemini-2.5-flash:generateContent"
STREAM_PATH = "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
LOOKUP = {"functionDeclarations": [{
    "name": "lookup", "description": "Find a thing",
    "parameters": {"type": "OBJECT", "required": ["q"], "properties": {
        "q": {"type": "STRING", "description": "query"},
        "n": {"type": "INTEGER", "nullable": True},
    }}}]}
LOOKUP_SCHEMA = {"type": "object", "required": ["q"], "properties": {
    "q": {"type": "string", "description": "query"}, "n": {"type": ["integer", "null"]}}}


def configured(upstream: FakeUpstream, *, lossy: bool = False, api_keys: list[str] | None = None,
               anthropic_model: str = "claude-sonnet-4-20250514", **server: Any) -> dict[str, Any]:
    raw = _with_gemini(_config(upstream, api_keys=api_keys), upstream)
    # A model name LiteLLM knows to think; unknown names have no adapter mapping.
    raw["models"][0]["model"] = anthropic_model
    raw["server"].update(server)
    for provider in raw["providers"].values():
        provider["compatibility"] = "lossy" if lossy else "strict"
    return raw


def endpoint(base: str, model: str, method: str = "generateContent", version: str = "v1beta") -> str:
    return f"{base}/{version}/models/{model}:{method}"


def generate(base: str, model: str, body: dict[str, Any] | None = None, **kwargs: Any) -> httpx.Response:
    return httpx.post(endpoint(base, model), json=HELLO if body is None else body,
                      timeout=30, **kwargs)


def stream(base: str, model: str, body: dict[str, Any] | None = None, **kwargs: Any) -> httpx.Response:
    params = kwargs.pop("params", {"alt": "sse"})
    return httpx.post(endpoint(base, model, "streamGenerateContent"),
                      json=HELLO if body is None else body, params=params, timeout=30, **kwargs)


def frames(response: httpx.Response) -> list[dict[str, Any]]:
    return [json.loads(event["data"]) for event in _parse_sse(response.text) if "data" in event]


def parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [part for candidate in payload["candidates"] for part in candidate["content"]["parts"]]


def text(payload: dict[str, Any]) -> str:
    return "".join(part.get("text", "") for part in parts(payload) if not part.get("thought"))


def losses(response: httpx.Response) -> list[dict[str, str]]:
    return json.loads(response.headers["x-model-router-loss"])


def sent(upstream: FakeUpstream, path: str) -> dict[str, Any]:
    return upstream.last_request(path)["body"]


def assert_gemini_error(response: httpx.Response, status: int, name: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == status and error["status"] == name and error["message"]
    return error


# ---------------------------------------------------------------------------
# Same-protocol passthrough
# ---------------------------------------------------------------------------

def test_native_generate_content_is_forwarded_without_translation(upstream):
    body = {
        **HELLO,
        "system_instruction": {"parts": [{"text": "be brief"}]},
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        "generationConfig": {"temperature": 0.2, "futureField": {"x": 1}},
        "cachedContent": "cachedContents/abc",
        "tools": [{"googleSearch": {}}],
    }
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-gem", body, headers={
            "x-goog-api-key": "client-secret", "x-goog-user-project": "someone-else",
            "x-goog-api-client": "google-genai-sdk/1.0 gl-python/3.12"})
    assert response.status_code == 200 and response.json() == _gemini_response()
    request = upstream.last_request(NATIVE_PATH)
    # The router adds only the path-derived model; the body is otherwise untouched.
    assert request["body"] == body
    assert request["query"] == ""
    assert request["headers"]["x-goog-api-key"] == "upstream-key"
    assert "client-secret" not in json.dumps(request["headers"])
    assert "x-goog-user-project" not in request["headers"]
    assert "x-model-router-loss" not in response.headers


def test_native_stream_is_relayed_byte_for_byte(upstream):
    first, last = _gemini_response(), _gemini_response()
    first["candidates"][0].pop("finishReason")
    wire = [f"data: {json.dumps(first)}\r\n\r\n".encode(), f"data: {json.dumps(last)}\r\n\r\n".encode()]
    upstream.stream_frames = wire
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-gem")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"".join(wire)
    request = upstream.last_request(STREAM_PATH)
    assert request["query"] == "alt=sse"
    assert "stream" not in request["body"] and "model" not in request["body"]


def test_native_stream_without_alt_sse_does_not_request_sse(upstream):
    upstream.stream_frames = [b'[{"candidates":[]}]']
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-gem", params={})
    assert response.status_code == 200 and response.content == b'[{"candidates":[]}]'
    assert upstream.last_request(STREAM_PATH)["query"] == ""


@pytest.mark.parametrize("path", [
    "/v1beta/models/m-gem:generateContent",
    "/v1/models/m-gem:generateContent",
    "/models/m-gem:generateContent",
    "/v1beta/models/m%2Dgem:generateContent",
])
def test_sdk_path_spellings_resolve_the_same_route(upstream, path):
    with router_config(configured(upstream)) as base:
        response = httpx.post(base + path, json=HELLO, timeout=30)
    assert response.status_code == 200, response.text
    assert upstream.last_request(NATIVE_PATH)


def test_provider_qualified_names_keep_their_slash_unescaped_or_escaped(upstream):
    with router_config(configured(upstream)) as base:
        raw = httpx.post(base + "/v1beta/models/gem/gemini-2.5-flash:generateContent",
                         json=HELLO, timeout=30)
        escaped = httpx.post(base + f"/v1beta/models/{quote('gem/gemini-2.5-flash', safe='')}:generateContent",
                             json=HELLO, timeout=30)
    assert raw.status_code == escaped.status_code == 200
    assert [r["path"] for r in upstream.requests] == [NATIVE_PATH, NATIVE_PATH]


@pytest.mark.parametrize("name", ["gem/%2E%2E/other", "gem/x%3Fkey=1", "gem/x%26y", "gem/x%23", "gem/a%25b"])
def test_dynamic_upstream_names_cannot_rewrite_the_upstream_url(upstream, name):
    with router_config(configured(upstream)) as base:
        response = httpx.post(f"{base}/v1beta/models/{name}:generateContent", json=HELLO, timeout=30)
    assert response.status_code == 400, response.text
    assert upstream.requests == []


def test_native_upstream_errors_keep_their_status_and_redact_secrets(upstream):
    upstream.error_status = 429
    upstream.error_body = json.dumps({"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "quota exceeded for key upstream-key"}}).encode()
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-gem")
    error = assert_gemini_error(response, 429, "RESOURCE_EXHAUSTED")
    assert "upstream-key" not in response.text and "quota exceeded" in error["message"]
    assert response.headers["retry-after"] == "0"
    assert "do-not-forward" not in json.dumps(dict(response.headers))


def test_native_non_json_upstream_errors_are_wrapped_as_gemini_errors(upstream):
    upstream.error_status = 502
    upstream.error_body = b"<html>bad gateway ?key=AIzaSyDUMMYDUMMYDUMMYDUMMYDUMMYDUMMY123</html>"
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-gem")
    assert_gemini_error(response, 502, "UNAVAILABLE")
    assert "AIza" not in response.text


# ---------------------------------------------------------------------------
# Authentication and credential isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["header", "query", "bearer", "x-api-key"])
def test_router_keys_are_accepted_from_every_gemini_client_spelling(upstream, how):
    kwargs: dict[str, Any] = {}
    if how == "header":
        kwargs["headers"] = {"x-goog-api-key": "router-secret"}
    elif how == "query":
        kwargs["params"] = {"key": "router-secret"}
    elif how == "bearer":
        kwargs["headers"] = {"authorization": "Bearer router-secret"}
    else:
        kwargs["headers"] = {"x-api-key": "router-secret"}
    with router_config(configured(upstream, api_keys=["router-secret"])) as base:
        response = generate(base, "m-chat", **kwargs)
    assert response.status_code == 200, response.text
    request = upstream.last_request("/v1/chat/completions")
    # Router credentials are never an upstream credential, in headers or the URL.
    assert request["headers"]["authorization"] == "Bearer upstream-key"
    assert "router-secret" not in json.dumps(request["headers"]) + request["query"]
    assert "router-secret" not in json.dumps(request["body"])


@pytest.mark.parametrize("kwargs", [
    {}, {"headers": {"x-goog-api-key": "wrong"}}, {"params": {"key": "wrong"}},
    {"headers": {"x-goog-api-key": ""}},
])
def test_missing_or_wrong_router_keys_are_rejected_in_gemini_form(upstream, kwargs):
    with router_config(configured(upstream, api_keys=["router-secret"])) as base:
        response = generate(base, "m-gem", **kwargs)
        listed = httpx.get(base + "/v1beta/models", timeout=30, **kwargs)
    assert_gemini_error(response, 401, "UNAUTHENTICATED")
    assert_gemini_error(listed, 401, "UNAUTHENTICATED")
    assert upstream.requests == []


def test_a_client_key_is_never_forwarded_even_when_router_auth_is_off(upstream):
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-gem", headers={"x-goog-api-key": "AIzaSyCLIENTKEYCLIENTKEYCLIENTKEY12345"},
                            params={"key": "AIzaSyCLIENTKEYCLIENTKEYCLIENTKEY12345"})
    assert response.status_code == 200
    request = upstream.last_request(NATIVE_PATH)
    assert request["query"] == ""
    assert "CLIENTKEY" not in json.dumps(request["headers"])


def test_credentials_in_urls_headers_and_errors_never_reach_logs(upstream, caplog):
    caplog.set_level(logging.DEBUG)
    upstream.error_status = 400
    upstream.error_body = b'{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"bad key=router-secret"}}'
    with router_config(configured(upstream, api_keys=["router-secret"])) as base:
        response = generate(base, "m-gem", params={"key": "router-secret"})
        ok = generate(base, "no-such-model", params={"key": "router-secret"})
    assert_gemini_error(ok, 404, "NOT_FOUND")
    assert response.status_code == 400 and "router-secret" not in response.text
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "router-secret" not in logged and "upstream-key" not in logged
    assert "key=" not in logged


def test_only_a_fixed_label_is_logged_for_gemini_paths(upstream, caplog):
    caplog.set_level(logging.INFO)
    with router_config(configured(upstream)) as base:
        generate(base, "m-gem")
        httpx.get(base + "/v1beta/models/m-gem", timeout=30)
        generate(base, "an-arbitrary-client-model-name")
    events = [json.loads(r.getMessage()) for r in caplog.records
              if hasattr(r, "router_event") and r.router_event["event"] == "request_outcome"]
    assert {event["path"] for event in events} == {
        "/v1beta/models/*:generateContent", "/v1beta/models/*"}
    assert "an-arbitrary-client-model-name" not in json.dumps(events)
    assert events[0]["ingress"] == "gemini" and events[0]["translation_mode"] == "passthrough"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_model_list_and_get_use_the_gemini_resource_shape(upstream):
    raw = configured(upstream)
    raw["models"][1]["metadata"] = {"display_name": "Chat Model"}
    with router_config(raw) as base:
        listed = httpx.get(base + "/v1beta/models", timeout=30)
        one = httpx.get(base + "/v1beta/models/m-chat", timeout=30)
        missing = httpx.get(base + "/v1beta/models/nope", timeout=30)
    assert listed.status_code == 200
    names = [model["name"] for model in listed.json()["models"]]
    assert names == ["models/m-anth", "models/m-chat", "models/m-resp", "models/m-gem"]
    assert "nextPageToken" not in listed.json()
    model = one.json()
    assert model["name"] == "models/m-chat" and model["displayName"] == "Chat Model"
    assert model["supportedGenerationMethods"] == ["generateContent", "streamGenerateContent"]
    # Limits that were never reported are omitted, not invented.
    assert "inputTokenLimit" not in model and "outputTokenLimit" not in model
    assert_gemini_error(missing, 404, "NOT_FOUND")


def test_model_list_paginates_with_opaque_tokens(upstream):
    with router_config(configured(upstream)) as base:
        first = httpx.get(base + "/v1beta/models", params={"pageSize": 3}, timeout=30).json()
        second = httpx.get(base + "/v1beta/models", params={
            "pageSize": 3, "pageToken": first["nextPageToken"]}, timeout=30).json()
        bad = httpx.get(base + "/v1beta/models", params={"pageToken": "garbage!"}, timeout=30)
        huge = httpx.get(base + "/v1beta/models", params={"pageSize": 5000}, timeout=30)
    assert [m["name"] for m in first["models"]] == ["models/m-anth", "models/m-chat", "models/m-resp"]
    assert [m["name"] for m in second["models"]] == ["models/m-gem"]
    assert "nextPageToken" not in second
    assert_gemini_error(bad, 400, "INVALID_ARGUMENT")
    assert_gemini_error(huge, 400, "INVALID_ARGUMENT")


def test_the_openai_model_list_is_unchanged(upstream):
    with router_config(configured(upstream)) as base:
        payload = httpx.get(base + "/v1/models", timeout=30).json()
    assert payload["object"] == "list" and "models" not in payload
    found = {entry["id"]: entry for entry in payload["data"]}
    # A Gemini ingress exists now, so every non-Gemini route is reachable through it.
    assert found["m-chat"]["metadata"]["router"]["protocols"]["translated"]["gemini"]["value"] is True
    assert found["m-gem"]["metadata"]["router"]["protocols"]["translated"]["gemini"]["value"] is False
    assert found["m-gem"]["metadata"]["router"]["protocols"]["native"]["gemini"]["value"] is True


# ---------------------------------------------------------------------------
# Cross-protocol generateContent
# ---------------------------------------------------------------------------

BASIC = {
    "systemInstruction": {"parts": [{"text": "be brief"}]},
    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
    "generationConfig": {"maxOutputTokens": 64, "temperature": 0.5, "topP": 0.9},
}


@pytest.mark.parametrize("model,expected,path", CROSS, ids=CROSS_IDS)
def test_generate_content_is_translated_to_every_other_protocol(upstream, model, expected, path):
    with router_config(configured(upstream)) as base:
        response = generate(base, model, BASIC)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["candidates"] == [{
        "content": {"role": "model", "parts": [{"text": expected}]},
        "finishReason": "STOP", "index": 0}]
    assert payload["usageMetadata"] == {
        "promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}
    assert payload["modelVersion"] == {"m-chat": "upstream-chat", "m-anth": "claude-sonnet-4-20250514",
                                       "m-resp": "upstream-responses"}[model]
    assert "x-model-router-loss" not in response.headers
    body = sent(upstream, path)
    assert "be brief" in json.dumps(body) and "hello" in json.dumps(body)
    if model == "m-chat":
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1] == {"role": "user", "content": "hello"}
        assert (body["max_tokens"], body["temperature"], body["top_p"]) == (64, 0.5, 0.9)
    elif model == "m-anth":
        assert (body["max_tokens"], body["temperature"], body["top_p"]) == (64, 0.5, 0.9)
    else:
        assert (body["max_output_tokens"], body["temperature"], body["top_p"]) == (64, 0.5, 0.9)
        assert body["instructions"] == "be brief"
    # Router and client credentials stay out of the translated request.
    assert "x-goog-api-key" not in upstream.last_request(path)["headers"]


def test_snake_case_requests_are_accepted_as_on_the_public_api(upstream):
    body = {"system_instruction": {"parts": [{"text": "be brief"}]},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generation_config": {"max_output_tokens": 32, "top_p": 0.5, "stop_sequences": ["END"]}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    request = sent(upstream, "/v1/chat/completions")
    assert (request["max_tokens"], request["top_p"], request["stop"]) == (32, 0.5, ["END"])


def test_multi_turn_history_and_role_defaults(upstream):
    body = {"contents": [
        {"parts": [{"text": "one"}]},
        {"role": "model", "parts": [{"text": "two"}, {"text": " three"}]},
        {"role": "user", "parts": [{"text": "four"}]},
    ]}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    assert sent(upstream, "/v1/chat/completions")["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two three"},
        {"role": "user", "content": "four"}]


@pytest.mark.parametrize("finish,expected", [("length", "MAX_TOKENS"), ("content_filter", "SAFETY"),
                                             ("stop", "STOP")])
def test_chat_finish_reasons_map_to_gemini_finish_reasons(upstream, finish, expected):
    payload = _chat_completion("cut")
    payload["choices"][0]["finish_reason"] = finish
    upstream.response_payload = payload
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat")
    assert response.json()["candidates"][0]["finishReason"] == expected


def test_reasoning_tokens_are_reported_as_gemini_thoughts(upstream):
    payload = _chat_completion("answer")
    payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
                        "completion_tokens_details": {"reasoning_tokens": 20},
                        "prompt_tokens_details": {"cached_tokens": 4}}
    upstream.response_payload = payload
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat")
    assert response.json()["usageMetadata"] == {
        "promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 40,
        "thoughtsTokenCount": 20, "cachedContentTokenCount": 4}


@pytest.mark.parametrize("include", [True, False, None])
def test_thought_summaries_are_returned_only_when_requested(upstream, include):
    payload = _chat_completion("answer")
    payload["choices"][0]["message"]["reasoning_content"] = "private reasoning"
    upstream.response_payload = payload
    body = dict(HELLO)
    if include is not None:
        body["generationConfig"] = {"thinkingConfig": {"includeThoughts": include}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    expected = [{"text": "private reasoning", "thought": True}, {"text": "answer"}] if include else [
        {"text": "answer"}]
    assert parts(response.json()) == expected


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected,path", CROSS, ids=CROSS_IDS)
def test_stream_generate_content_emits_gemini_sse_for_every_protocol(upstream, model, expected, path):
    with router_config(configured(upstream)) as base:
        response = stream(base, model, BASIC)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in response.text and "event:" not in response.text
    chunks = frames(response)
    assert chunks and all("candidates" in chunk for chunk in chunks)
    assert "".join(text(chunk) for chunk in chunks) == expected
    final = chunks[-1]
    assert final["candidates"][0]["finishReason"] == "STOP"
    assert final["usageMetadata"]["promptTokenCount"] == 5
    assert all("finishReason" not in c["candidates"][0] for c in chunks[:-1])
    if model != "m-resp":  # LiteLLM simulates streaming for Responses models it does not know
        assert sent(upstream, path)["stream"] is True


def test_stream_without_alt_sse_is_a_json_array(upstream):
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat", params={})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    chunks = response.json()
    assert isinstance(chunks, list) and "".join(text(c) for c in chunks) == CHAT_TEXT
    assert chunks[-1]["candidates"][0]["finishReason"] == "STOP"


def chat_frame(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None) -> bytes:
    chunk: dict[str, Any] = {"id": "chatcmpl_x", "object": "chat.completion.chunk", "created": 1,
                             "model": "upstream-chat",
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        chunk["usage"] = usage
        chunk["choices"] = [] if finish is None else chunk["choices"]
    return f"data: {json.dumps(chunk)}\n\n".encode()


def test_streamed_text_arrives_incrementally_and_finishes_once(upstream):
    upstream.stream_frames = [
        chat_frame({"role": "assistant", "content": ""}),
        chat_frame({"content": "Hel"}), chat_frame({"content": "lo"}),
        chat_frame({}, "length"),
        chat_frame({}, None, {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}),
        b"data: [DONE]\n\n"]
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat")
    chunks = frames(response)
    assert [text(c) for c in chunks] == ["Hel", "lo", ""]
    assert chunks[-1]["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert chunks[-1]["usageMetadata"] == {
        "promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": 9}
    assert {c["responseId"] for c in chunks} == {"chatcmpl_x"}


def test_a_function_call_split_across_deltas_becomes_one_whole_call(upstream):
    pieces = [ARGS[:4], ARGS[4:19], ARGS[19:22], ARGS[22:]]
    upstream.stream_frames = [
        chat_frame({"role": "assistant", "content": "Looking. "}),
        chat_frame({"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
                                    "function": {"name": "lookup", "arguments": pieces[0]}}]}),
        *[chat_frame({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
          for piece in pieces[1:]],
        chat_frame({}, "tool_calls"),
        chat_frame({}, None, {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}),
        b"data: [DONE]\n\n"]
    body = {**HELLO, "tools": [LOOKUP]}
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat", body)
    chunks = frames(response)
    calls = [p["functionCall"] for c in chunks for p in c["candidates"][0]["content"]["parts"]
             if "functionCall" in p]
    assert calls == [{"id": "call_9", "name": "lookup", "args": json.loads(ARGS)}]
    assert chunks[0]["candidates"][0]["content"]["parts"] == [{"text": "Looking. "}]
    assert chunks[-1]["candidates"][0]["finishReason"] == "STOP"
    assert sum("functionCall" in p for c in chunks for p in c["candidates"][0]["content"]["parts"]) == 1


def test_streamed_thoughts_follow_include_thoughts(upstream):
    upstream.stream_frames = [
        chat_frame({"role": "assistant", "reasoning_content": "hmm "}),
        chat_frame({"content": "answer"}), chat_frame({}, "stop"), b"data: [DONE]\n\n"]
    with router_config(configured(upstream)) as base:
        hidden = stream(base, "m-chat")
        shown = stream(base, "m-chat", {**HELLO, "generationConfig": {
            "thinkingConfig": {"includeThoughts": True}}})
    assert [p for c in frames(hidden) for p in parts(c)] == [{"text": "answer"}]
    assert [p for c in frames(shown) for p in parts(c)] == [
        {"text": "hmm ", "thought": True}, {"text": "answer"}]


def test_an_upstream_failure_before_the_first_byte_is_a_real_http_error(upstream):
    upstream.error_status = 429
    upstream.error_body = b'{"error":{"message":"slow down"}}'
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat")
    assert_gemini_error(response, 429, "RESOURCE_EXHAUSTED")
    assert response.headers["retry-after"] == "0"


def test_a_mid_stream_failure_ends_with_an_error_frame_and_no_done_marker(upstream):
    upstream.stream_error = "backend exploded"
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat")
    assert response.status_code == 200
    assert "[DONE]" not in response.text
    last = frames(response)[-1]
    assert last["error"]["code"] >= 500 and "backend exploded" in last["error"]["message"]
    assert last["error"]["status"] in {"UNAVAILABLE", "INTERNAL"}


def test_a_mid_stream_failure_without_alt_sse_closes_the_json_array(upstream):
    upstream.stream_error = "backend exploded"
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-chat", params={})
    assert response.status_code == 200
    chunks = response.json()
    assert chunks[-1]["error"]["status"] in {"UNAVAILABLE", "INTERNAL"}
    assert "backend exploded" in chunks[-1]["error"]["message"]


@pytest.mark.parametrize("params", [{"alt": "xml"}, [("alt", "sse"), ("alt", "json")]])
def test_an_unknown_alt_selector_is_rejected(upstream, params):
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-gem", params=params)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


def test_alt_json_is_accepted_and_not_forwarded_as_sse(upstream):
    upstream.stream_frames = [b"[]"]
    with router_config(configured(upstream)) as base:
        response = stream(base, "m-gem", params={"alt": "json"})
    assert response.status_code == 200
    assert upstream.last_request(STREAM_PATH)["query"] == ""


def test_an_oversized_query_string_is_a_clean_gemini_error(upstream):
    with router_config(configured(upstream)) as base:
        response = httpx.post(endpoint(base, "m-gem") + "?" + "&".join(f"p{i}=1" for i in range(80)),
                              json=HELLO, timeout=30)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


def test_empty_text_parts_are_not_forwarded_as_empty_blocks(upstream):
    body = {"contents": [{"role": "user", "parts": [{"text": ""}, {"text": "hi"}, {"text": ""}]}]}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-anth", body)
    assert response.status_code == 200, response.text
    assert "".join(json.dumps(m["content"]) for m in sent(upstream, "/v1/messages")["messages"]).count('"text": ""') == 0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def upstream_tools(upstream: FakeUpstream, model: str) -> list[dict[str, Any]]:
    path = {"m-chat": "/v1/chat/completions", "m-anth": "/v1/messages",
            "m-resp": "/v1/responses"}[model]
    return sent(upstream, path)["tools"]


@pytest.mark.parametrize("model", CROSS_IDS)
def test_function_declarations_become_json_schema_tools(upstream, model):
    with router_config(configured(upstream)) as base:
        response = generate(base, model, {**HELLO, "tools": [LOOKUP]})
    assert response.status_code == 200, response.text
    tool, = upstream_tools(upstream, model)
    schema = (tool.get("function") or tool).get("parameters") or tool.get("input_schema")
    assert (tool.get("function") or tool)["name"] == "lookup"
    assert schema["required"] == ["q"]
    assert schema["properties"]["q"]["type"] == "string"
    assert schema["properties"]["n"]["type"] in (["integer", "null"], "integer")


def test_the_chat_tool_definition_is_exactly_the_converted_schema(upstream):
    with router_config(configured(upstream)) as base:
        generate(base, "m-chat", {**HELLO, "tools": [LOOKUP]})
    assert upstream_tools(upstream, "m-chat") == [{"type": "function", "function": {
        "name": "lookup", "description": "Find a thing", "parameters": LOOKUP_SCHEMA}}]


def test_json_schema_declarations_are_not_rewritten(upstream):
    schema = {"type": "object", "properties": {"type": {"type": "string"}}, "additionalProperties": False}
    tool = {"functionDeclarations": [{"name": "lookup", "parametersJsonSchema": schema}]}
    with router_config(configured(upstream)) as base:
        generate(base, "m-chat", {**HELLO, "tools": [tool]})
    assert upstream_tools(upstream, "m-chat")[0]["function"]["parameters"] == schema


@pytest.mark.parametrize("model", CROSS_IDS)
def test_function_calls_in_the_answer_become_function_call_parts(upstream, model):
    upstream.response_payload = response_for(model, tool=True)
    with router_config(configured(upstream)) as base:
        response = generate(base, model, {**HELLO, "tools": [LOOKUP]})
    assert response.status_code == 200, response.text
    call, = [p["functionCall"] for p in parts(response.json()) if "functionCall" in p]
    assert call["name"] == "lookup" and call["args"] == json.loads(ARGS) and call["id"]
    assert response.json()["candidates"][0]["finishReason"] == "STOP"


def history(call_id: str | None) -> dict[str, Any]:
    call: dict[str, Any] = {"name": "lookup", "args": json.loads(ARGS)}
    result: dict[str, Any] = {"name": "lookup", "response": {"output": RESULT}}
    if call_id:
        call["id"] = result["id"] = call_id
    return {"contents": [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"functionCall": call, "thoughtSignature": "opaque"}]},
        {"role": "user", "parts": [{"functionResponse": result}]},
    ], "tools": [LOOKUP]}


@pytest.mark.parametrize("call_id", [None, "call-abc"])
@pytest.mark.parametrize("model", CROSS_IDS)
def test_tool_history_pairs_each_result_with_its_call(upstream, model, call_id):
    with router_config(configured(upstream)) as base:
        response = generate(base, model, history(call_id))
    assert response.status_code == 200, response.text
    path = {"m-chat": "/v1/chat/completions", "m-anth": "/v1/messages", "m-resp": "/v1/responses"}[model]
    request = sent(upstream, path)
    dump = json.dumps(request)
    assert RESULT in dump and "lookup" in dump and "opaque" not in dump
    if model == "m-chat":
        assistant, = [m for m in request["messages"] if m.get("tool_calls")]
        call, = assistant["tool_calls"]
        result, = [m for m in request["messages"] if m["role"] == "tool"]
        assert call["id"] == result["tool_call_id"] == (call_id or "call_1")
        assert json.loads(call["function"]["arguments"]) == json.loads(ARGS)
        assert json.loads(result["content"]) == {"output": RESULT}
    elif model == "m-anth":
        blocks = [b for m in request["messages"] if isinstance(m["content"], list) for b in m["content"]]
        use, = [b for b in blocks if b["type"] == "tool_use"]
        result, = [b for b in blocks if b["type"] == "tool_result"]
        assert use["id"] == result["tool_use_id"] and use["input"] == json.loads(ARGS)
    else:
        call, = [i for i in request["input"] if i.get("type") == "function_call"]
        result, = [i for i in request["input"] if i.get("type") == "function_call_output"]
        assert call["call_id"] == result["call_id"]


def test_ids_the_target_cannot_carry_are_replaced_consistently(upstream):
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", history("weird id/with.chars"))
    assert response.status_code == 200
    messages = sent(upstream, "/v1/chat/completions")["messages"]
    call_id = next(m for m in messages if m.get("tool_calls"))["tool_calls"][0]["id"]
    assert call_id == "call_1" == next(m for m in messages if m["role"] == "tool")["tool_call_id"]


def test_a_function_response_without_its_call_is_rejected(upstream):
    body = {"contents": [{"role": "user", "parts": [{"functionResponse": {
        "name": "lookup", "response": {"output": "x"}}}]}], "tools": [LOOKUP]}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


@pytest.mark.parametrize("config,expected", [
    ({"mode": "AUTO"}, "auto"),
    ({"mode": "ANY"}, "required"),
    ({"mode": "NONE"}, "none"),
    ({"mode": "ANY", "allowedFunctionNames": ["lookup"]},
     {"type": "function", "function": {"name": "lookup"}}),
])
def test_function_calling_modes_map_to_tool_choice(upstream, config, expected):
    body = {**HELLO, "tools": [LOOKUP], "toolConfig": {"functionCallingConfig": config}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    assert sent(upstream, "/v1/chat/completions")["tool_choice"] == expected


def test_a_narrowed_allowed_list_is_reported_as_an_approximation(upstream):
    two = {"functionDeclarations": [
        {"name": "lookup", "parametersJsonSchema": {"type": "object"}},
        {"name": "other", "parametersJsonSchema": {"type": "object"}},
        {"name": "third", "parametersJsonSchema": {"type": "object"}}]}
    body = {**HELLO, "tools": [two], "toolConfig": {"functionCallingConfig": {
        "mode": "ANY", "allowedFunctionNames": ["lookup", "other"]}}}
    with router_config(configured(upstream)) as strict:
        refused = generate(strict, "m-chat", body)
    assert_gemini_error(refused, 400, "INVALID_ARGUMENT")
    assert "toolConfig.functionCallingConfig.allowedFunctionNames" in refused.json()["error"]["message"]
    with router_config(configured(upstream, lossy=True)) as lossy:
        response = generate(lossy, "m-chat", body)
    assert response.status_code == 200
    assert [t["function"]["name"] for t in upstream_tools(upstream, "m-chat")] == ["lookup", "other"]
    assert losses(response)[0]["action"] == "approximated"


def test_native_gemini_tools_are_refused_in_strict_mode_and_reported_in_lossy_mode(upstream):
    body = {**HELLO, "tools": [{"googleSearch": {}}, LOOKUP]}
    with router_config(configured(upstream)) as base:
        refused = generate(base, "m-chat", body)
    error = assert_gemini_error(refused, 400, "INVALID_ARGUMENT")
    assert "tools[0].googleSearch" in error["message"]
    assert error["details"][0]["reason"] == "TRANSLATION_LOSS"
    assert upstream.requests == []
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200
    assert [loss["field"] for loss in losses(response)] == ["tools[0].googleSearch"]
    # The translatable function tool survives next to the dropped native one.
    assert [t["function"]["name"] for t in upstream_tools(upstream, "m-chat")] == ["lookup"]


def test_tool_config_without_tools_is_reported_not_silently_applied(upstream):
    body = {**HELLO, "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200
    assert [loss["field"] for loss in losses(response)] == ["toolConfig.functionCallingConfig"]
    assert "tool_choice" not in sent(upstream, "/v1/chat/completions")


# ---------------------------------------------------------------------------
# Multimodal input
# ---------------------------------------------------------------------------

def media(mime: str, data: str, **extra: Any) -> dict[str, Any]:
    return {"contents": [{"role": "user", "parts": [
        {"text": "what is this"}, {"inlineData": {"mimeType": mime, "data": data, **extra}}]}]}


def test_inline_images_reach_each_protocol_in_its_own_form(upstream):
    with router_config(configured(upstream)) as base:
        for model in CROSS_IDS:
            assert generate(base, model, media("image/png", PNG)).status_code == 200
    chat = sent(upstream, "/v1/chat/completions")["messages"][0]["content"]
    assert chat == [{"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]
    anthropic = sent(upstream, "/v1/messages")["messages"][0]["content"]
    image, = [b for b in anthropic if b["type"] == "image"]
    assert image["source"] == {"type": "base64", "media_type": "image/png", "data": PNG}
    responses = json.dumps(sent(upstream, "/v1/responses")["input"])
    assert "input_image" in responses and PNG in responses


def test_inline_pdfs_become_documents(upstream):
    with router_config(configured(upstream)) as base:
        for model in CROSS_IDS:
            assert generate(base, model, media("application/pdf", PDF, displayName="a b.pdf")).status_code == 200
    chat = sent(upstream, "/v1/chat/completions")["messages"][0]["content"][1]
    assert chat["type"] == "file" and chat["file"]["file_data"].endswith(PDF)
    assert chat["file"]["filename"] == "a b.pdf"
    assert any(b["type"] == "document" for b in sent(upstream, "/v1/messages")["messages"][0]["content"])
    assert "input_file" in json.dumps(sent(upstream, "/v1/responses")["input"])


def test_audio_is_carried_only_where_chat_defines_it(upstream):
    with router_config(configured(upstream)) as base:
        chat = generate(base, "m-chat", media("audio/wav", WAV))
        refused = generate(base, "m-anth", media("audio/wav", WAV))
    assert chat.status_code == 200
    content = sent(upstream, "/v1/chat/completions")["messages"][0]["content"]
    assert content[1] == {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}}
    error = assert_gemini_error(refused, 400, "INVALID_ARGUMENT")
    assert "contents[0].parts[1].inlineData" in error["message"]


@pytest.mark.parametrize("body,field", [
    (media("video/mp4", PNG), "contents[0].parts[1].inlineData"),
    (media("image/heic", PNG), "contents[0].parts[1].inlineData"),
    ({"contents": [{"role": "user", "parts": [
        {"text": "hi"}, {"fileData": {"mimeType": "image/png",
                                      "fileUri": "https://example.test/a.png"}}]}]},
     "contents[0].parts[1]"),
])
def test_media_without_an_established_equivalent_is_never_silently_dropped(upstream, body, field):
    with router_config(configured(upstream)) as base:
        refused = generate(base, "m-chat", body)
    assert field in assert_gemini_error(refused, 400, "INVALID_ARGUMENT")["message"]
    assert upstream.requests == []
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200
    assert [loss["field"] for loss in losses(response)] == [field]
    assert "image_url" not in json.dumps(sent(upstream, "/v1/chat/completions"))


def test_a_request_with_only_dropped_content_is_rejected_even_in_lossy_mode(upstream):
    body = {"contents": [{"role": "user", "parts": [{"fileData": {
        "mimeType": "image/png", "fileUri": "https://example.test/a.png"}}]}]}
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-chat", body)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


@pytest.mark.parametrize("data", ["not base64!!", ""])
def test_malformed_inline_data_is_rejected(upstream, data):
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", media("image/png", data))
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")


# ---------------------------------------------------------------------------
# Strict / lossy policy for generation controls
# ---------------------------------------------------------------------------

def test_safety_settings_are_refused_in_strict_mode_and_reported_in_lossy_mode(upstream):
    body = {**HELLO, "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT",
                                         "threshold": "BLOCK_NONE"}]}
    with router_config(configured(upstream)) as base:
        refused = generate(base, "m-chat", body)
    assert "safetySettings" in assert_gemini_error(refused, 400, "INVALID_ARGUMENT")["message"]
    assert upstream.requests == []
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200
    assert [loss["field"] for loss in losses(response)] == ["safetySettings"]
    assert "safety" not in json.dumps(sent(upstream, "/v1/chat/completions")).lower()


@pytest.mark.parametrize("key", ["cachedContent", "continuationToken"])
@pytest.mark.parametrize("lossy", [False, True])
def test_state_held_by_gemini_is_never_dropped(upstream, key, lossy):
    with router_config(configured(upstream, lossy=lossy)) as base:
        response = generate(base, "m-chat", {**HELLO, key: "cachedContents/abc"})
    error = assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert error["details"][0]["reason"] == "UNSUPPORTED_GEMINI_STATE"
    assert upstream.requests == []


@pytest.mark.parametrize("model", ["m-chat", "m-anth"])
def test_stop_sequences_and_top_k_reach_protocols_that_define_them(upstream, model):
    body = {**HELLO, "generationConfig": {"stopSequences": ["END"], "topK": 40, "maxOutputTokens": 9}}
    with router_config(configured(upstream)) as base:
        response = generate(base, model, body)
    assert response.status_code == 200, response.text
    request = sent(upstream, {"m-chat": "/v1/chat/completions", "m-anth": "/v1/messages"}[model])
    assert request.get("stop") == ["END"] or request.get("stop_sequences") == ["END"]
    assert request["top_k"] == 40


def test_controls_a_destination_cannot_apply_are_named_by_their_gemini_path(upstream):
    body = {**HELLO, "generationConfig": {
        "stopSequences": ["END"], "topK": 40, "seed": 7, "candidateCount": 2, "maxOutputTokens": 32}}
    with router_config(configured(upstream)) as base:
        refused = generate(base, "m-resp", body)
    message = assert_gemini_error(refused, 400, "INVALID_ARGUMENT")["message"]
    for name in ("generationConfig.stopSequences", "generationConfig.topK",
                 "generationConfig.seed", "generationConfig.candidateCount"):
        assert name in message
    assert "generationConfig.maxOutputTokens" not in message
    with router_config(configured(upstream, lossy=True)) as base:
        response = generate(base, "m-resp", body)
    assert response.status_code == 200
    assert {loss["field"] for loss in losses(response)} == {
        "generationConfig.stopSequences", "generationConfig.topK", "generationConfig.seed",
        "generationConfig.candidateCount"}
    assert sent(upstream, "/v1/responses")["max_output_tokens"] == 32


def test_thinking_budget_is_exact_on_anthropic_and_reported_elsewhere(upstream):
    body = {**HELLO, "generationConfig": {"maxOutputTokens": 4096,
                                           "thinkingConfig": {"thinkingBudget": 2048}}}
    with router_config(configured(upstream)) as base:
        anthropic = generate(base, "m-anth", body)
        chat = generate(base, "m-chat", body)
    assert anthropic.status_code == 200, anthropic.text
    assert sent(upstream, "/v1/messages")["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "generationConfig.thinkingConfig.thinkingBudget" in assert_gemini_error(
        chat, 400, "INVALID_ARGUMENT")["message"]


def test_dynamic_thinking_budget_is_the_destination_default(upstream):
    body = {**HELLO, "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    request = sent(upstream, "/v1/chat/completions")
    assert not {"reasoning_effort", "thinking"} & request.keys()


def test_thinking_level_is_an_approximation_and_never_renamed(upstream):
    body = {**HELLO, "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    assert sent(upstream, "/v1/chat/completions")["reasoning_effort"] == "high"


def test_a_zero_thinking_budget_disables_thinking(upstream):
    body = {**HELLO, "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert response.status_code == 200, response.text
    assert sent(upstream, "/v1/chat/completions")["reasoning_effort"] == "none"


def test_budget_and_level_together_are_invalid(upstream):
    body = {**HELLO, "generationConfig": {"thinkingConfig": {"thinkingBudget": 1, "thinkingLevel": "LOW"}}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")


def test_json_mode_and_schemas_become_response_format(upstream):
    schema = {"type": "OBJECT", "properties": {"answer": {"type": "STRING"}}, "required": ["answer"]}
    body = {**HELLO, "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}
    plain = {**HELLO, "generationConfig": {"responseMimeType": "application/json"}}
    with router_config(configured(upstream)) as base:
        generate(base, "m-chat", body)
        structured = sent(upstream, "/v1/chat/completions")["response_format"]
        generate(base, "m-chat", plain)
        json_mode = sent(upstream, "/v1/chat/completions")["response_format"]
        refused = generate(base, "m-anth", body)
    assert structured == {"type": "json_schema", "json_schema": {"name": "response", "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}
    assert json_mode == {"type": "json_object"}
    assert "generationConfig.responseMimeType" in assert_gemini_error(
        refused, 400, "INVALID_ARGUMENT")["message"]


def test_a_schema_needs_the_json_mime_type(upstream):
    body = {**HELLO, "generationConfig": {"responseSchema": {"type": "OBJECT"}}}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")


# ---------------------------------------------------------------------------
# Validation and unsupported surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["m-chat", "m-gem"])
@pytest.mark.parametrize("body", [
    {}, {"contents": []}, {"contents": "hi"}, {"contents": [{"role": "user"}]},
    {"contents": [{"role": "user", "parts": ["hi"]}]},
    {"contents": [{"role": "user", "parts": [{"text": "x"}]}], "tools": {}},
    {"contents": [{"role": "user", "parts": [{"text": "x"}]}], "generationConfig": []},
    {"contents": [{"role": "user", "parts": [{"text": "x"}]}], "model": "m-anth"},
    {"contents": [{"role": "user", "parts": [{"text": "x"}]}], "stream": True},
])
def test_malformed_requests_are_rejected_before_any_upstream_call(upstream, model, body):
    with router_config(configured(upstream)) as base:
        response = generate(base, model, body)
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


@pytest.mark.parametrize("config,field", [
    ({"bogus": 1}, "bogus"), ({"temperature": 5}, "temperature"), ({"temperature": "hot"}, "temperature"),
    ({"topP": 2}, "topP"), ({"maxOutputTokens": 0}, "maxOutputTokens"), ({"topK": -1}, "topK"),
    ({"stopSequences": [""]}, "stopSequences"), ({"maxOutputTokens": True}, "maxOutputTokens"),
])
def test_invalid_generation_config_is_rejected(upstream, config, field):
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", {**HELLO, "generationConfig": config})
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")
    assert upstream.requests == []


def test_unknown_names_do_not_echo_untrusted_text(upstream):
    body = {**HELLO, "totally<script>alert(1)</script>": 1}
    with router_config(configured(upstream)) as base:
        response = generate(base, "m-chat", body)
    assert "<script>" not in response.text
    assert_gemini_error(response, 400, "INVALID_ARGUMENT")


def test_invalid_json_and_empty_bodies_use_the_gemini_error_shape(upstream):
    with router_config(configured(upstream)) as base:
        garbage = httpx.post(endpoint(base, "m-chat"), content=b"{nope", timeout=30)
        empty = httpx.post(endpoint(base, "m-chat"), content=b"", timeout=30)
        array = httpx.post(endpoint(base, "m-chat"), json=[1], timeout=30)
    for response in (garbage, empty, array):
        assert_gemini_error(response, 400, "INVALID_ARGUMENT")


def test_unknown_models_unsupported_methods_and_tuned_models_have_gemini_errors(upstream):
    with router_config(configured(upstream)) as base:
        unknown = generate(base, "no-such-model")
        tuned = httpx.post(base + "/v1beta/tunedModels/x:generateContent", json=HELLO, timeout=30)
        counted = httpx.post(endpoint(base, "m-chat", "countTokens"), json=HELLO, timeout=30)
        embed = httpx.post(endpoint(base, "m-chat", "embedContent"), json={}, timeout=30)
        other = httpx.post(base + "/v1beta/models/m-chat:somethingElse", json=HELLO, timeout=30)
    assert "models/no-such-model" in assert_gemini_error(unknown, 404, "NOT_FOUND")["message"]
    assert_gemini_error(tuned, 404, "NOT_FOUND")
    assert_gemini_error(counted, 501, "UNIMPLEMENTED")
    assert_gemini_error(embed, 501, "UNIMPLEMENTED")
    assert_gemini_error(other, 404, "NOT_FOUND")
    assert upstream.requests == []


def test_unsupported_methods_still_require_authentication(upstream):
    with router_config(configured(upstream, api_keys=["router-secret"])) as base:
        response = httpx.post(endpoint(base, "m-chat", "countTokens"), json=HELLO, timeout=30)
    assert_gemini_error(response, 401, "UNAUTHENTICATED")


def test_the_openai_and_anthropic_endpoints_are_unaffected(upstream):
    with router_config(configured(upstream)) as base:
        chat = httpx.post(base + "/v1/chat/completions", json={
            "model": "m-gem", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}, timeout=30)
        missing = httpx.post(base + "/v1/nope", json={}, timeout=30)
    assert chat.status_code == 200 and chat.json()["choices"][0]["message"]["content"]
    assert missing.status_code == 404 and "message" in missing.json()["error"]
    assert "status" not in missing.json()["error"]


# ---------------------------------------------------------------------------
# Limits, upstream failures, retries and deadlines
# ---------------------------------------------------------------------------

def test_the_body_limit_applies_with_a_gemini_error(upstream):
    with router_config(configured(upstream, max_body_bytes=2048)) as base:
        big = generate(base, "m-gem", {**HELLO, "pad": "x" * 4096})
        fine = generate(base, "m-gem")
    assert_gemini_error(big, 413, "INVALID_ARGUMENT")
    assert fine.status_code == 200


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp"])
def test_upstream_errors_on_translated_routes_are_gemini_errors(upstream, model):
    upstream.error_status = 500
    upstream.error_body = b'{"error":{"message":"boom for upstream-key"}}'
    with router_config(configured(upstream)) as base:
        response = generate(base, model)
    assert response.status_code == 500, response.text
    assert response.json()["error"]["status"] == "INTERNAL"
    assert "upstream-key" not in response.text


def test_generation_capacity_is_shared_with_the_other_protocols(upstream, monkeypatch):
    with router_config(configured(upstream, max_generations=1)) as base:
        server_acquire = mr.RouterHTTPServer.acquire_generation
        monkeypatch.setattr(mr.RouterHTTPServer, "acquire_generation", lambda self, c: False)
        response = generate(base, "m-gem")
        monkeypatch.setattr(mr.RouterHTTPServer, "acquire_generation", server_acquire)
        recovered = generate(base, "m-gem")
    assert_gemini_error(response, 503, "UNAVAILABLE")
    assert response.headers["retry-after"] == "1"
    assert recovered.status_code == 200
    assert len(upstream.requests) == 1


@pytest.mark.parametrize("model", ["m-gem", "m-chat"])
def test_retry_backoff_and_the_overall_deadline_apply_to_gemini_requests(peer, model):
    peer.script = [429]
    with application(peer, deadline=1.4) as (app, clock), running(app) as (client, server):
        reply = client.post(f"/v1beta/models/{model}:generateContent", json=HELLO)
        assert reply.status_code == 504, reply.text
        error = reply.json()["error"]
        assert error["status"] == "DEADLINE_EXCEEDED"
        assert "will not replay" in error["message"]
        assert peer.attempts >= 2


@pytest.fixture
def peer():
    from test_retries import SequenceUpstream
    fake = SequenceUpstream().start()
    try:
        yield fake
    finally:
        fake.stop()
        fake._thread.join(3)
