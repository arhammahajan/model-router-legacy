"""Prompt 15: provider wire contracts through installed adapters, never adapter stubs.

Synthetic fixtures follow the source links in fixtures/README.md. Function JSON
is fragmented only on protocols that document argument deltas; Gemini sends a
complete functionCall object, not invented partial JSON strings.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from test_model_router import (
    INGRESSES, PATHS, TARGETS, UPSTREAM_PATHS, _anthropic_message, _anthropic_sse,
    _body, _chat_completion, _chat_sse, _config, _gemini_response, _parse_sse,
    _responses_response, _responses_sse, _text_from_response, _with_gemini,
    router_config, upstream,
)
from test_translation_policy import control

ARGS = '{"q":"snow ☃ and \\"quotes\\""}'
# Split a JSON key/value and a string containing Unicode and escaped quotes.
FRAGMENTS = [ARGS[:4], ARGS[4:19], ARGS[19:22], ARGS[22:]]
RESULT = "lookup-result-42"
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=='


def config(upstream):
    raw = _with_gemini(_config(upstream), upstream)
    # Known Responses model selects native SSE in LiteLLM instead of simulation.
    raw["models"][2]["model"] = "gpt-4.1"
    for provider in raw["providers"].values():
        provider["litellm_kwargs"] = {"num_retries": 0}
    return raw


def response_for(model, *, tool=False):
    if model == "m-anth":
        payload = _anthropic_message("answer", tool)
        if tool:
            payload["content"] = payload["content"][:1]
            payload["content"][0]["input"] = json.loads(ARGS)
    elif model == "m-chat":
        payload = _chat_completion("answer", tool)
        if tool:
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = ARGS
    elif model == "m-resp":
        payload = _responses_response("answer", tool)
        if tool:
            payload["output"] = payload["output"][:1]
            payload["output"][0]["arguments"] = ARGS
    else:
        payload = _gemini_response()
        payload["candidates"][0]["content"]["parts"] = ([{
            "functionCall": {"name": "lookup", "args": json.loads(ARGS)},
        }] if tool else [{"text": "answer"}])
    return payload


def call_from(ingress, payload):
    if ingress == "openai-chat":
        call, = payload["choices"][0]["message"]["tool_calls"]
        return call["id"], call["function"]["name"], json.loads(call["function"]["arguments"])
    if ingress == "openai-responses":
        call, = [item for item in payload["output"] if item["type"] == "function_call"]
        return call["call_id"], call["name"], json.loads(call["arguments"])
    call, = [item for item in payload["content"] if item["type"] == "tool_use"]
    return call["id"], call["name"], call["input"]


def continuation(ingress, body, payload, call_id):
    body = copy.deepcopy(body)
    if ingress == "openai-chat":
        body["messages"] += [payload["choices"][0]["message"],
                             {"role": "tool", "tool_call_id": call_id, "content": RESULT}]
    elif ingress == "openai-responses":
        body["input"] = [{"role": "user", "content": "hello"}, *payload["output"],
                         {"type": "function_call_output", "call_id": call_id, "output": RESULT}]
    else:
        body["messages"] += [
            {"role": "assistant", "content": payload["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                           "content": RESULT}]},
        ]
    return body


def assert_history(model, sent):
    """Both the assistant call and matching result must reach the upstream."""
    if model == "m-gem":
        parts = [part for item in sent["contents"] for part in item["parts"]]
        call, = [p["function_call"] for p in parts if "function_call" in p]
        result, = [p["function_response"] for p in parts if "function_response" in p]
        assert call["name"] == result["name"] == "lookup"
        assert call["args"] == json.loads(ARGS)
        assert RESULT in json.dumps(result["response"])
    elif model == "m-resp":
        call, = [p for p in sent["input"] if p.get("type") == "function_call"]
        result, = [p for p in sent["input"] if p.get("type") == "function_call_output"]
        assert call["call_id"] == result["call_id"]
        assert call["name"] == "lookup" and json.loads(call["arguments"]) == json.loads(ARGS)
        assert result["output"] == RESULT or result["output"] == [{"type": "input_text", "text": RESULT}]
    elif model == "m-chat":
        assistant, = [m for m in sent["messages"] if m.get("tool_calls")]
        call, = assistant["tool_calls"]
        result, = [m for m in sent["messages"] if m["role"] == "tool"]
        assert call["id"] == result["tool_call_id"]
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == json.loads(ARGS)
        assert result["content"] == RESULT
    else:
        parts = [p for m in sent["messages"] for p in m["content"] if isinstance(p, dict)]
        call, = [p for p in parts if p["type"] == "tool_use"]
        result, = [p for p in parts if p["type"] == "tool_result"]
        assert call["id"] == result["tool_use_id"]
        assert call["name"] == "lookup" and call["input"] == json.loads(ARGS)
        assert RESULT in json.dumps(result["content"])


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_tool_call_and_result_round_trip(upstream, ingress, model, text):
    body = {**_body(ingress, model, False), **control(ingress, "tools")}
    with router_config(config(upstream)) as base:
        upstream.response_payload = response_for(model, tool=True)
        first = httpx.post(base + PATHS[ingress], json=body, timeout=15)
        assert first.status_code == 200, first.text
        payload = first.json()
        call_id, name, args = call_from(ingress, payload)
        assert call_id and name == "lookup" and args == json.loads(ARGS)
        if ingress == "openai-chat":
            assert payload["choices"][0]["finish_reason"] == "tool_calls"
        elif ingress == "anthropic":
            assert payload["stop_reason"] == "tool_use"
        else:
            assert payload["status"] == "completed"
        upstream.response_payload = response_for(model)
        second = httpx.post(base + PATHS[ingress],
                            json=continuation(ingress, body, payload, call_id), timeout=15)
        assert second.status_code == 200, second.text
        assert _text_from_response(ingress, second.json()) == "answer"
    assert len(upstream.requests) == 2
    assert_history(model, upstream.last_request(UPSTREAM_PATHS[text])["body"])


def frame(payload, name=None):
    return ((f"event: {name}\n" if name else "") +
            "data: " + json.dumps(payload) + "\n\n").encode()


def tool_frames(model):
    final = response_for(model, tool=True)
    if model == "m-chat":
        chunks = []
        for i, fragment in enumerate(FRAGMENTS):
            chunk = json.loads(_chat_sse("")[0])
            call = {"index": 0, "function": {"arguments": fragment}}
            if i == 0:
                call.update(id="call_1", type="function")
                call["function"]["name"] = "lookup"
            chunk["choices"][0]["delta"] = {"tool_calls": [call]}
            chunks.append(frame(chunk))
        terminal = json.loads(_chat_sse("")[1])
        terminal["choices"][0]["finish_reason"] = "tool_calls"
        return chunks + [frame(terminal), b"data: [DONE]\n\n"]
    if model == "m-anth":
        start = _anthropic_sse("")[0]
        events = [start, ("content_block_start", {"type": "content_block_start", "index": 0,
                  "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}})]
        events += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": fragment}})
                   for fragment in FRAGMENTS]
        events += [("content_block_stop", {"type": "content_block_stop", "index": 0}),
                   ("message_delta", {"type": "message_delta", "delta": {
                       "stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 3}}),
                   ("message_stop", {"type": "message_stop"})]
        return [frame(p, name) for name, p in events]
    if model == "m-resp":
        item = final["output"][0]
        added = {**item, "status": "in_progress", "arguments": ""}
        events = _responses_sse("")[:2] + [
            ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": added})]
        events += [("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": item["id"], "delta": fragment}) for fragment in FRAGMENTS]
        events += [("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": item["id"], "arguments": ARGS}),
            ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "response": final})]
        return [frame({**p, "sequence_number": i}, name) for i, (name, p) in enumerate(events)]
    return [frame(final)]


def assert_terminal(ingress, reply):
    assert reply.status_code == 200, reply.text
    assert reply.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(reply.text)
    payloads = [json.loads(f["data"]) for f in frames if f["data"] != "[DONE]"]
    assert not any(p.get("error") or p.get("type") == "error" for p in payloads), reply.text
    if ingress == "openai-chat":
        assert frames[-1] == {"data": "[DONE]"}
        assert sum(f["data"] == "[DONE]" for f in frames) == 1
    elif ingress == "openai-responses":
        kinds = [p["type"] for p in payloads]
        assert kinds[0] == "response.created"
        assert kinds[-1] == "response.completed" and kinds.count("response.completed") == 1
        assert payloads[-1]["response"]["status"] == "completed"
        assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
    else:
        kinds = [p["type"] for p in payloads]
        assert kinds[0] == "message_start"
        assert kinds[-2:] == ["message_delta", "message_stop"]
        assert kinds.count("message_stop") == 1
        opened = set()
        for p in payloads:
            if p["type"] == "content_block_start":
                assert p["index"] not in opened
                opened.add(p["index"])
            elif p["type"] == "content_block_delta":
                assert p["index"] in opened
            elif p["type"] == "content_block_stop":
                assert p["index"] in opened
                opened.remove(p["index"])
        assert not opened
    return payloads


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_streamed_tool_arguments_reassemble_and_terminate(upstream, ingress, model, text):
    upstream.stream_frames = tool_frames(model)
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, model, True), **control(ingress, "tools")}, timeout=15)
    payloads = assert_terminal(ingress, reply)
    fragments = []
    if ingress == "openai-chat":
        calls = [call for p in payloads for c in p.get("choices", [])
                 for call in c["delta"].get("tool_calls") or []]
        assert {call["index"] for call in calls} == {0}
        assert calls[0]["id"] and calls[0]["function"]["name"] == "lookup"
        fragments = [c["function"].get("arguments", "") for c in calls]
        assert any(c.get("finish_reason") == "tool_calls" for p in payloads for c in p.get("choices", []))
    elif ingress == "openai-responses":
        added, = [p for p in payloads if p["type"] == "response.output_item.added" and p["item"]["type"] == "function_call"]
        deltas = [p for p in payloads if p["type"] == "response.function_call_arguments.delta"]
        assert all(p["item_id"] == added["item"]["id"] and p["output_index"] == added["output_index"] for p in deltas)
        assert payloads.index(added) < payloads.index(deltas[0])
        done, = [p for p in payloads if p["type"] == "response.function_call_arguments.done"]
        item_done, = [p for p in payloads if p["type"] == "response.output_item.done" and p["item"]["type"] == "function_call"]
        assert done["item_id"] == item_done["item"]["id"] == added["item"]["id"]
        assert json.loads(done["arguments"]) == json.loads(ARGS)
        assert payloads.index(deltas[-1]) < payloads.index(done) < payloads.index(item_done) < len(payloads) - 1
        fragments = [p["delta"] for p in deltas]
        assert call_from(ingress, payloads[-1]["response"])[1:] == ("lookup", json.loads(ARGS))
    else:
        added, = [p for p in payloads if p["type"] == "content_block_start" and p["content_block"]["type"] == "tool_use"]
        assert added["content_block"]["name"] == "lookup" and added["content_block"]["id"]
        fragments = [p["delta"]["partial_json"] for p in payloads if p["type"] == "content_block_delta" and p["delta"]["type"] == "input_json_delta"]
        assert payloads[-2]["delta"]["stop_reason"] == "tool_use"
    assert json.loads("".join(fragments)) == json.loads(ARGS)
    path = UPSTREAM_PATHS[text].replace(":generateContent", ":streamGenerateContent")
    assert upstream.last_request(path)["body"].get("tools")


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_inline_image_reaches_provider_without_fetch(upstream, ingress, model, text):
    body = _body(ingress, model, False)
    if ingress == "openai-chat":
        body["messages"][0]["content"] = [{"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}]
    elif ingress == "openai-responses":
        body["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "data:image/png;base64," + PNG}]}]
    else:
        body["messages"][0]["content"] = [{"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=body, timeout=15)
    assert reply.status_code == 200, reply.text
    sent = upstream.last_request(UPSTREAM_PATHS[text])["body"]
    if model == "m-gem":
        parts = sent["contents"][0]["parts"]
        image, = [p["inline_data"] for p in parts if "inline_data" in p]
        assert image == {"mime_type": "image/png", "data": PNG}
    elif model == "m-anth":
        image, = [p for p in sent["messages"][0]["content"] if p["type"] == "image"]
        assert image["source"] == {"type": "base64", "media_type": "image/png", "data": PNG}
    elif model == "m-chat":
        image, = [p for p in sent["messages"][0]["content"] if p["type"] == "image_url"]
        assert image["image_url"]["url"] == "data:image/png;base64," + PNG
    else:
        parts = sent["input"][0]["content"]
        image, = [p for p in parts if p["type"] == "input_image"]
        assert image["image_url"] == "data:image/png;base64," + PNG
    assert len(upstream.requests) == 1


def metered_response(model):
    payload = response_for(model)
    if model == "m-chat":
        payload["usage"] = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2}}
    elif model == "m-resp":
        payload["usage"] = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 2}}
    elif model == "m-anth":
        payload["usage"] = {"input_tokens": 6, "output_tokens": 5,
            "cache_creation_input_tokens": 2, "cache_read_input_tokens": 4,
            "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 0},
            "server_tool_use": {"web_search_requests": 1}}
    else:
        payload["usageMetadata"] = {"promptTokenCount": 12, "candidatesTokenCount": 3,
            "totalTokenCount": 17, "cachedContentTokenCount": 4, "thoughtsTokenCount": 2}
    return payload


def metered_frames(model, payload):
    if model == "m-chat":
        # The final Chat usage chunk has choices:[], not a finish delta.
        chunks = [frame(json.loads(p)) for p in _chat_sse("answer")]
        chunks[-1] = frame({k: v for k, v in json.loads(_chat_sse("answer")[1]).items() if k != "usage"})
        chunks += [frame({"id": "chatcmpl_upstream", "object": "chat.completion.chunk",
                          "created": 1, "model": "upstream-chat", "choices": [], "usage": payload["usage"]}),
                   b"data: [DONE]\n\n"]
        return chunks
    if model == "m-resp":
        events = _responses_sse("answer")
        events[-1][1]["response"] = payload
        return [frame({**p, "sequence_number": i}, n) for i, (n, p) in enumerate(events)]
    if model == "m-anth":
        events = _anthropic_sse("answer")
        events[0][1]["message"]["usage"] = {**payload["usage"], "output_tokens": 0}
        events[-2][1]["usage"] = {"output_tokens": 5}
        return [frame(p, n) for n, p in events]
    return [frame(payload)]


def usage_from(ingress, reply, stream):
    if not stream:
        return reply.json()["usage"]
    payloads = assert_terminal(ingress, reply)
    if ingress == "openai-chat":
        return [p["usage"] for p in payloads if p.get("usage")][-1]
    if ingress == "openai-responses":
        return payloads[-1]["response"]["usage"]
    usage = {}
    for p in payloads:
        if p["type"] == "message_start":
            usage.update(p["message"]["usage"])
        elif p["type"] == "message_delta":
            usage.update(p.get("usage", {}))
    return usage


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_usage_cache_and_reasoning_accounting(upstream, ingress, model, text, stream):
    payload = metered_response(model)
    upstream.response_payload = payload
    upstream.stream_frames = metered_frames(model, payload)
    body = _body(ingress, model, stream)
    if ingress == "openai-chat" and stream:
        body["stream_options"] = {"include_usage": True}
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=body, timeout=15)
    assert reply.status_code == 200, reply.text
    usage = usage_from(ingress, reply, stream)
    if ingress == "openai-chat":
        assert usage["prompt_tokens"] == 12, usage
        assert usage["completion_tokens"] == 5 and usage["total_tokens"] == 17, usage
        assert usage["prompt_tokens_details"]["cached_tokens"] == 4, usage
        if model != "m-anth":  # Messages reports thinking as output, without an exact subcount.
            assert usage["completion_tokens_details"]["reasoning_tokens"] == 2, usage
        else:
            assert usage["cache_creation_input_tokens"] == 2, usage
    elif ingress == "openai-responses":
        assert usage["input_tokens"] == 12 and usage["output_tokens"] == 5, usage
        assert usage["total_tokens"] == 17, usage
        assert usage["input_tokens_details"]["cached_tokens"] == 4, usage
        if model != "m-anth":
            assert usage["output_tokens_details"]["reasoning_tokens"] == 2, usage
    else:
        assert usage["output_tokens"] == 5, usage
        assert usage["cache_read_input_tokens"] == 4, usage
        assert usage["input_tokens"] == (6 if model == "m-anth" else 8), usage
        if model == "m-anth":
            assert usage["cache_creation_input_tokens"] == 2, usage
            assert usage["server_tool_use"]["web_search_requests"] == 1, usage


@pytest.mark.parametrize("status", [400, 429, 503])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_actual_upstream_http_errors_and_rate_limit_headers(upstream, ingress, model, text, stream, status):
    # All routes exist. The error must originate in the actual upstream HTTP request.
    upstream.error_status = status
    if model == "m-gem":
        error = {"error": {"code": status, "status": {400: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE"}[status], "message": "fixture upstream refused"}}
    elif model == "m-anth":
        error = {"type": "error", "error": {"type": {400: "invalid_request_error", 429: "rate_limit_error", 503: "api_error"}[status], "message": "fixture upstream refused"}}
    else:
        error = {"error": {"type": "invalid_request_error" if status == 400 else "rate_limit_error" if status == 429 else "server_error", "code": str(status), "message": "fixture upstream refused"}}
    upstream.error_body = json.dumps(error).encode()
    upstream.error_headers = {"X-RateLimit-Remaining-Requests": "0", "X-RateLimit-Reset-Requests": "2s",
                              "anthropic-ratelimit-tokens-remaining": "0"}
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, stream), timeout=15)
    assert len(upstream.requests) == 1  # no hidden retries or local lookup failure
    assert upstream.requests[0]["path"] == UPSTREAM_PATHS[text].replace(
        ":generateContent", ":streamGenerateContent" if stream else ":generateContent")
    assert reply.status_code == status, reply.text
    assert reply.headers["content-type"].startswith("application/json"), reply.text
    payload = reply.json()
    assert payload["error"]["message"] and "fixture upstream refused" in payload["error"]["message"]
    if ingress == "anthropic":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == {400: "invalid_request_error", 429: "rate_limit_error", 503: "api_error"}[status]
    assert reply.headers["retry-after"] == "0"
    assert reply.headers["x-ratelimit-remaining-requests"] == "0"
    assert reply.headers["x-ratelimit-reset-requests"] == "2s"
    assert reply.headers["anthropic-ratelimit-tokens-remaining"] == "0"
    assert "x-upstream-diagnostic" not in reply.headers


@pytest.mark.parametrize("ingress,model", [
    ("openai-chat", "m-anth"), ("openai-responses", "m-anth"), ("anthropic", "m-anth"),
    ("openai-chat", "m-gem"), ("openai-responses", "m-gem"),
    ("openai-chat", "m-resp"), ("openai-responses", "m-chat"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_supported_reasoning_controls_reach_provider(upstream, ingress, model, stream):
    raw = config(upstream)
    if model == "m-anth":
        raw["models"][0]["model"] = "claude-sonnet-4-20250514"
    if model == "m-resp":
        raw["models"][2]["model"] = "o3"
    body = {**_body(ingress, model, stream), **control(ingress, "reasoning")}
    body["max_tokens" if ingress != "openai-responses" else "max_output_tokens"] = 32768
    with router_config(raw) as base:
        reply = httpx.post(base + PATHS[ingress], json=body, timeout=15)
    assert reply.status_code == 200, reply.text
    assert "x-model-router-loss" not in reply.headers
    sent = upstream.requests[-1]["body"]
    if model == "m-anth":
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    elif model == "m-gem":
        assert sent["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 4096, "includeThoughts": True}
    elif model == "m-resp":
        assert sent["reasoning"]["effort"] == "high"
    else:
        assert sent["reasoning_effort"] == "high"
    if stream:
        assert_terminal(ingress, reply)


@pytest.mark.parametrize("model", ["m-anth", "m-gem"])
@pytest.mark.parametrize("ingress", ["openai-chat", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_thinking_content_and_signature_where_supported(upstream, model, ingress, stream):
    payload = response_for(model)
    if model == "m-anth":
        payload["content"].insert(0, {"type": "thinking", "thinking": "checking fixture", "signature": "c2lnbmF0dXJl"})
        events = _anthropic_sse("answer")
        events[1][1]["index"] = events[2][1]["index"] = events[3][1]["index"] = 1
        thought_events = [
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "checking fixture"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "c2lnbmF0dXJl"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]
        upstream.stream_frames = [frame(p, n) for n, p in [events[0], *thought_events, *events[1:]]]
    else:
        payload["candidates"][0]["content"]["parts"].insert(0, {"text": "checking fixture", "thought": True})
        upstream.stream_frames = [frame(payload)]
    upstream.response_payload = payload
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, stream), timeout=15)
    assert reply.status_code == 200, reply.text
    if stream:
        events = assert_terminal(ingress, reply)
        if ingress == "openai-chat":
            reasoning = "".join(c["delta"].get("reasoning_content") or "" for p in events for c in p.get("choices", []))
        else:
            reasoning = "".join(p["delta"].get("thinking", "") for p in events if p["type"] == "content_block_delta")
        assert reasoning == "checking fixture"
        assert "answer" in reply.text
    elif ingress == "openai-chat":
        message = reply.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "checking fixture"
        assert message["content"] == "answer"
    else:
        thought, = [p for p in reply.json()["content"] if p["type"] == "thinking"]
        assert thought["thinking"] == "checking fixture"
        if model == "m-anth":
            assert thought["signature"] == "c2lnbmF0dXJl"


PROVIDER_USAGE = {"cost": 0, "opencode": {"billing_source": "go", "list_cost": 0.00412,
    "allowance": 0.003, "tokens": {"input": 6, "output": 5, "reasoning": 2,
    "cache_read": 4, "cache_write_5m": 2, "cache_write_1h": 0}}}


@pytest.mark.parametrize("ingress,model", [("openai-chat", "m-chat"),
    ("openai-responses", "m-resp"), ("anthropic", "m-anth")])
@pytest.mark.parametrize("stream", [False, True])
def test_native_provider_usage_extensions_are_preserved(upstream, ingress, model, stream):
    payload = metered_response(model)
    payload["usage"].update(copy.deepcopy(PROVIDER_USAGE))
    upstream.response_payload = payload
    upstream.stream_frames = metered_frames(model, payload)
    if model == "m-anth":
        # Go sends billing on the final message_delta, not message_start.
        first = json.loads(upstream.stream_frames[0].decode().split("data: ")[1])
        for key in PROVIDER_USAGE:
            first["message"]["usage"].pop(key)
        upstream.stream_frames[0] = frame(first, "message_start")
        final = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                 "usage": {"output_tokens": 5, **PROVIDER_USAGE}}
        upstream.stream_frames[-2] = frame(final, "message_delta")
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, stream), timeout=15)
    assert reply.status_code == 200, reply.text
    usage = usage_from(ingress, reply, stream)
    for key, value in PROVIDER_USAGE.items():
        assert usage[key] == value
    if not stream:
        assert reply.content == json.dumps(payload).encode()
    else:
        assert reply.content == b"".join(upstream.stream_frames)


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
@pytest.mark.parametrize("when", ["first", "midstream"])
def test_provider_sse_error_is_terminal(upstream, ingress, model, text, when):
    message = "fixture stream failure"
    if model == "m-anth":
        error = frame({"type": "error", "error": {"type": "api_error", "message": message}}, "error")
    elif model == "m-resp":
        error = frame({"type": "error", "code": "server_error", "message": message,
                       "param": None, "sequence_number": 5 if when == "midstream" else 0}, "error")
    elif model == "m-gem":
        error = frame({"error": {"code": 500, "status": "INTERNAL", "message": message}})
    else:
        error = frame({"error": {"type": "server_error", "code": "server_error", "message": message}})
    complete = metered_frames(model, metered_response(model))
    prefix = []
    if when == "midstream":
        if model == "m-gem":
            initial = response_for(model)
            initial["candidates"][0].pop("finishReason")
            prefix = [frame(initial)]
        else:
            prefix = complete[:3 if model == "m-anth" else 5 if model == "m-resp" else 1]
    upstream.stream_frames = [*prefix, error, *complete]
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, True), timeout=15)
    native = model == {"openai-chat": "m-chat", "openai-responses": "m-resp", "anthropic": "m-anth"}[ingress]
    assert len(upstream.requests) == 1
    if native:
        assert reply.status_code == 200
        assert reply.content == b"".join(upstream.stream_frames)
        return  # native SSE is deliberately opaque, including embedded errors
    expected_status = 200 if when == "midstream" or (ingress, model) == ("anthropic", "m-resp") else 500
    assert reply.status_code == expected_status, reply.text
    expected_message = "Response API in-stream error" if model == "m-resp" else message
    assert expected_message in reply.text, reply.text
    if reply.status_code != 200:
        assert when == "first"
        assert "error" in reply.json()
        assert reply.headers["content-type"].startswith("application/json")
        return
    events = _parse_sse(reply.text)
    payloads = [json.loads(f["data"]) for f in events if f["data"] != "[DONE]"]
    assert payloads[-1].get("type") == "error" or payloads[-1].get("error"), reply.text
    assert "response.completed" not in reply.text and "message_stop" not in reply.text
    if ingress == "openai-chat":
        assert events[-1] == {"data": "[DONE]"}
    elif ingress == "openai-responses":
        assert events[-1]["event"] == "error"
        assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
    else:
        assert events[-1]["event"] == "error"


@pytest.mark.parametrize("ingress,model", [("openai-chat", "m-anth"),
    ("openai-responses", "m-chat"), ("anthropic", "m-gem")])
def test_forwarded_rate_limit_diagnostics_are_sanitized(upstream, ingress, model):
    upstream.error_body = b'{"error":{"message":"fixture refused"}}'
    upstream.error_headers = {"X-RateLimit-Reset": "upstream-key Bearer private-fixture-token " + "x" * 5000}
    with router_config(config(upstream)) as base:
        reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, False), timeout=15)
    assert reply.status_code == 429
    value = reply.headers["x-ratelimit-reset"]
    assert "upstream-key" not in value and "private-fixture-token" not in value
    assert len(value) <= 4000 and "REDACTED" in value.upper()
    assert "x-upstream-diagnostic" not in reply.headers
