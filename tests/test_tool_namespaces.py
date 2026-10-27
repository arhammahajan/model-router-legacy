"""Namespaced coding tools survive non-native Responses routes and later turns."""
from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from model_router.tool_namespaces import flatten_namespaces, restore_namespaces
from test_model_router import (
    _anthropic_message, _chat_completion, _config, _with_gemini, router_config, upstream,
)


def request():
    return {"model": "m-chat", "input": "Look up hello", "max_output_tokens": 64,
            "tools": [{"type": "namespace", "name": "muse", "tools": [{
                "type": "function", "name": "lookup", "description": "Look up a term",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                               "required": ["q"], "additionalProperties": False},
            }]}], "tool_choice": {"type": "function", "namespace": "muse", "name": "lookup"}}


def test_namespace_history_and_choice_round_trip_without_mutating_the_request():
    body = request()
    body["input"] = [{"type": "function_call", "namespace": "muse", "name": "lookup",
                      "arguments": '{"q":"hello"}', "call_id": "call_1"},
                     {"type": "function_call_output", "call_id": "call_1", "output": "found"}]
    original = deepcopy(body)
    flat, names = flatten_namespaces(body)
    tool = flat["tools"][0]
    assert body == original
    assert tool["parameters"] == body["tools"][0]["tools"][0]["parameters"]
    assert len(tool["name"]) <= 64
    assert flat["input"][0]["name"] == flat["tool_choice"]["name"] == tool["name"]
    assert "namespace" not in flat["input"][0]
    assert restore_namespaces(flat["input"], names) == body["input"]
    other = request()
    other["tools"][0]["name"] = "other"
    assert flatten_namespaces(other)[0]["tools"][0]["name"] != tool["name"]


@pytest.mark.parametrize("model,path", [
    ("m-chat", "/v1/chat/completions"), ("m-anth", "/v1/messages"),
    ("m-gem", "/v1beta/models/gemini-2.5-flash:generateContent"),
])
def test_namespaced_functions_reach_each_upstream_and_return_the_original_identity(upstream, model, path):
    body = {**request(), "model": model, "store": False,
            "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"}}
    alias = flatten_namespaces(body)[0]["tools"][0]["name"]
    if model == "m-chat":
        response = _chat_completion("", tool=True)
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = alias
    elif model == "m-anth":
        response = _anthropic_message("", tool=True)
        response["content"][0]["name"] = alias
    else:
        response = {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": alias, "args": {"q": "hello"}}}
        ]}, "finishReason": "STOP", "index": 0}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3,
                                      "totalTokenCount": 8}}
    upstream.response_payload = response
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["compatibility"] = "lossy"
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/responses", json=body, timeout=30)
    assert response.status_code == 200, response.text
    calls = [item for item in response.json()["output"] if item["type"] == "function_call"]
    assert len(calls) == 1
    assert calls[0]["namespace"] == "muse"
    assert calls[0]["name"] == "lookup"
    assert json.loads(calls[0]["arguments"]) == {"q": "hello"}
    sent = upstream.last_request(path)["body"]
    assert alias in json.dumps(sent["tools"])
    assert "tools" not in {loss["field"] for loss in json.loads(response.headers["x-model-router-loss"])}


def test_streaming_restores_namespace_on_added_done_and_completed_events(upstream):
    body = {**request(), "stream": True}
    alias = flatten_namespaces(body)[0]["tools"][0]["name"]
    base = {"id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1,
            "model": "upstream-chat"}
    frames = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": alias, "arguments": '{"q":"hello"}'},
        }]}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    upstream.stream_frames = [b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames]
    upstream.stream_frames.append(b"data: [DONE]\n\n")
    with router_config(_config(upstream)) as base_url:
        response = httpx.post(base_url + "/v1/responses", json=body, timeout=30)
    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    calls = [event["item"] for event in events if event["type"] in (
        "response.output_item.added", "response.output_item.done")]
    calls += next(event["response"]["output"] for event in events if event["type"] == "response.completed")
    assert calls
    assert all(call["name"] == "lookup" and call["namespace"] == "muse" for call in calls)
