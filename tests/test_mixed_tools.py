"""Mixed function/native tool lists: per-tool translation policy, tool_choice and round trips.

All offline: a local fake upstream, the real router and the real LiteLLM adapters.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from model_router.tools import plan_tools
from test_model_router import (
    PATHS, _anthropic_message, _chat_completion, _config, router_config, upstream,
)

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
          "additionalProperties": False}
LOOKUP = {"type": "function", "name": "lookup", "description": "Look up", "parameters": SCHEMA}
SEARCH = {"type": "function", "name": "search", "description": "Search", "parameters": SCHEMA}
WEB = {"type": "web_search"}
MCP = {"type": "mcp", "server_label": "docs", "server_url": "http://127.0.0.1:9/mcp"}
CUSTOM = {"type": "custom", "name": "patch", "description": "Freeform"}

CHAT = ("m-chat", "/v1/chat/completions")
ANTH = ("m-anth", "/v1/messages")
TARGETS = [CHAT, ANTH]


def raw_config(upstream: Any, compatibility: str) -> dict[str, Any]:
    raw = _config(upstream)
    for provider in raw["providers"].values():
        provider["compatibility"] = compatibility
    return raw


def responses_request(model: str, tools: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"model": model, "input": "Look up hello", "max_output_tokens": 64,
            "tools": tools, **extra}


def post(raw: dict[str, Any], path: str, body: dict[str, Any]) -> httpx.Response:
    with router_config(raw) as base:
        return httpx.post(base + path, json=body, timeout=30)


def losses(response: httpx.Response) -> list[dict[str, str]]:
    return json.loads(response.headers.get("x-model-router-loss", "[]"))


def sent_names(body: dict[str, Any]) -> list[str]:
    """Function names the upstream actually received, for chat or Anthropic wire shapes."""
    return [tool.get("name") or tool["function"]["name"] for tool in body.get("tools", [])]


# --- mixed lists -----------------------------------------------------------------------------

@pytest.mark.parametrize("model,path", TARGETS)
def test_lossy_mixed_list_keeps_functions_emulates_custom_and_reports_native_tools(upstream, model, path):
    body = responses_request(model, [LOOKUP, WEB, SEARCH, MCP, CUSTOM],
                             tool_choice="auto", parallel_tool_calls=False)
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(path)["body"]
    assert sent_names(sent) == ["lookup", "search", "patch"]
    # Function definitions are unchanged by the translation.
    schema = (sent["tools"][0].get("input_schema") or sent["tools"][0]["function"]["parameters"])
    assert schema["properties"] == SCHEMA["properties"]
    assert {(item["field"], item["action"]) for item in losses(response)} == {
        ("tools[1]", "dropped"), ("tools[3]", "dropped"), ("tools[4]", "approximated")}
    reasons = {item["field"]: item["reason"] for item in losses(response)}
    assert "'web_search'" in reasons["tools[1]"]
    assert "'mcp'" in reasons["tools[3]"]
    assert "custom tool input is emulated" in reasons["tools[4]"]
    # tool_choice stays valid; the parallel constraint survives because tools remain.
    if model == "m-chat":
        assert sent["tool_choice"] == "auto"
        assert sent["parallel_tool_calls"] is False
    else:
        assert sent["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


@pytest.mark.parametrize("model,path", TARGETS)
def test_strict_mixed_list_is_rejected_precisely_before_any_network_call(upstream, model, path):
    response = post(raw_config(upstream, "strict"), PATHS["openai-responses"],
                    responses_request(model, [LOOKUP, WEB]))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "translation_loss"
    assert "tools[1]" in error["message"] and "'web_search'" in error["message"]
    assert "tools[0]" not in error["message"]
    assert upstream.requests == []


def test_untrusted_tool_type_is_not_echoed_into_diagnostics(upstream):
    hostile = {"type": "x\r\nX-Injected: 1 " + "A" * 200}
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"],
                    responses_request(CHAT[0], [LOOKUP, hostile]))
    assert response.status_code == 200, response.text
    assert "X-Injected" not in response.headers["x-model-router-loss"]
    assert "'unknown'" in losses(response)[0]["reason"]
    assert "x-injected" not in response.headers


def test_namespace_with_function_and_native_child_keeps_the_function(upstream):
    namespace = {"type": "namespace", "name": "muse", "tools": [
        {"type": "function", "name": "lookup", "parameters": SCHEMA}, CUSTOM]}
    lossy = post(raw_config(upstream, "lossy"), PATHS["openai-responses"],
                 responses_request(CHAT[0], [namespace, SEARCH]))
    assert lossy.status_code == 200, lossy.text
    names = sent_names(upstream.last_request(CHAT[1])["body"])
    assert len(names) == 3 and "search" in names and all(name.startswith("ns_") for name in names[:2])
    assert [(item["field"], item["action"]) for item in losses(lossy)] == [("tools[0].tools[1]", "approximated")]
    upstream.requests.clear()
    strict = post(raw_config(upstream, "strict"), PATHS["openai-responses"],
                  responses_request(CHAT[0], [namespace, SEARCH]))
    assert strict.status_code == 400 and "custom tool input is emulated" in strict.json()["error"]["message"]
    assert upstream.requests == []


# --- function-only and native-only lists ------------------------------------------------------

@pytest.mark.parametrize("compatibility", ["strict", "lossy"])
@pytest.mark.parametrize("model,path", TARGETS)
def test_function_only_list_translates_without_any_loss(upstream, compatibility, model, path):
    body = responses_request(model, [LOOKUP, SEARCH], tool_choice="required",
                             parallel_tool_calls=True)
    response = post(raw_config(upstream, compatibility), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    sent = upstream.last_request(path)["body"]
    assert sent_names(sent) == ["lookup", "search"]
    assert sent["tool_choice"] in ("required", {"type": "any"}, {"type": "any", "disable_parallel_tool_use": False})


@pytest.mark.parametrize("model,path", TARGETS)
def test_native_only_list_drops_tools_and_every_dependent_control(upstream, model, path):
    body = responses_request(model, [WEB, MCP], tool_choice="required", parallel_tool_calls=True)
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(path)["body"]
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        assert key not in sent, sent
    assert {item["field"] for item in losses(response)} == {
        "tools[0]", "tools[1]", "tool_choice", "parallel_tool_calls"}
    upstream.requests.clear()
    strict = post(raw_config(upstream, "strict"), PATHS["openai-responses"], body)
    assert strict.status_code == 400
    assert upstream.requests == []


def test_native_tools_still_reach_a_responses_upstream_untouched(upstream):
    body = responses_request("m-resp", [LOOKUP, WEB, MCP], tool_choice={"type": "web_search"})
    response = post(raw_config(upstream, "strict"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request("/v1/responses")["body"]
    assert sent["tools"] == [LOOKUP, WEB, MCP]
    assert sent["tool_choice"] == {"type": "web_search"}


# --- tool_choice ------------------------------------------------------------------------------

def test_forced_function_choice_survives_native_tool_removal(upstream):
    body = responses_request(CHAT[0], [WEB, LOOKUP],
                             tool_choice={"type": "function", "name": "lookup"})
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(CHAT[1])["body"]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert [item["field"] for item in losses(response)] == ["tools[0]"]


@pytest.mark.parametrize("choice", [{"type": "web_search"}, {"type": "mcp", "server_label": "docs"}])
def test_choice_forcing_a_removed_native_tool_is_dropped_and_reported(upstream, choice):
    body = responses_request(CHAT[0], [LOOKUP, WEB, CUSTOM], tool_choice=choice)
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(CHAT[1])["body"]
    assert sent_names(sent) == ["lookup", "patch"]
    assert "tool_choice" not in sent
    assert {"field": "tool_choice", "reason": "forced tool was removed",
            "action": "dropped"} in losses(response)


def test_anthropic_forced_native_tool_keeps_the_parallel_constraint(upstream):
    bash = {"type": "bash_20250124", "name": "bash"}
    plain = {"name": "lookup", "description": "Look up", "input_schema": SCHEMA}
    body = {"model": CHAT[0], "max_tokens": 64,
            "messages": [{"role": "user", "content": "run it"}], "tools": [bash, plain],
            "tool_choice": {"type": "tool", "name": "bash", "disable_parallel_tool_use": True}}
    response = post(raw_config(upstream, "lossy"), PATHS["anthropic"], body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(CHAT[1])["body"]
    assert sent_names(sent) == ["lookup"]
    assert sent["tool_choice"] == "auto"
    assert sent["parallel_tool_calls"] is False
    assert {(item["field"], item["action"]) for item in losses(response)} == {
        ("tools[0]", "dropped"), ("tool_choice", "approximated")}
    upstream.requests.clear()
    assert post(raw_config(upstream, "strict"), PATHS["anthropic"], body).status_code == 400
    assert upstream.requests == []


def test_reconcile_keeps_the_parallel_constraint_when_the_forced_tool_is_removed():
    # The decision preserves this constraint; the router applies its native Chat field.
    params = {"tools": [{"type": "bash_20250124", "name": "bash"}, {"name": "lookup"}],
              "tool_choice": {"type": "tool", "name": "bash", "disable_parallel_tool_use": True}}
    params = plan_tools(params, "anthropic", "openai-chat").request
    assert params["tools"] == [{"name": "lookup"}]
    assert params["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_anthropic_custom_tools_are_functions_and_typed_tools_are_native(upstream):
    custom = {"type": "custom", "name": "lookup", "description": "Look up", "input_schema": SCHEMA}
    response = post(raw_config(upstream, "strict"), PATHS["anthropic"], {
        "model": CHAT[0], "max_tokens": 64, "messages": [{"role": "user", "content": "go"}],
        "tools": [custom], "tool_choice": {"type": "tool", "name": "lookup"}})
    assert response.status_code == 200, response.text
    assert sent_names(upstream.last_request(CHAT[1])["body"]) == ["lookup"]


def test_chat_custom_tool_is_emulated_for_an_anthropic_upstream(upstream):
    fn = {"type": "function", "function": {"name": "lookup", "parameters": SCHEMA}}
    custom = {"type": "custom", "custom": {"name": "patch"}}
    response = post(raw_config(upstream, "lossy"), PATHS["openai-chat"], {
        "model": ANTH[0], "max_tokens": 64, "messages": [{"role": "user", "content": "go"}],
        "tools": [custom, fn], "tool_choice": {"type": "custom", "custom": {"name": "patch"}}})
    assert response.status_code == 200, response.text
    sent = upstream.last_request(ANTH[1])["body"]
    assert sent_names(sent) == ["patch", "lookup"]
    assert sent["tool_choice"] == {"type": "tool", "name": "patch"}
    assert {(item["field"], item["action"]) for item in losses(response)} == {("tools[0]", "approximated")}


# --- multi-turn round trips -------------------------------------------------------------------

HISTORY = [
    {"role": "user", "content": [{"type": "input_text", "text": "Look up hello"}]},
    {"type": "function_call", "call_id": "call_prev", "name": "lookup",
     "arguments": '{"q":"hello"}'},
    {"type": "function_call_output", "call_id": "call_prev", "output": "found: world"},
    {"role": "user", "content": [{"type": "input_text", "text": "Now search it"}]},
]


def chat_tool_frames() -> list[bytes]:
    base = {"id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1,
            "model": "upstream-chat"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_new", "type": "function",
            "function": {"name": "search", "arguments": ""}}]}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": '{"q":"wor'}}]}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": 'ld"}'}}]}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    return [b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks] + [b"data: [DONE]\n\n"]


def anthropic_tool_frames() -> list[bytes]:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_t", "type": "message", "role": "assistant", "model": "upstream-anthropic",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "call_new", "name": "search", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": '{"q":"wor'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": 'ld"}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events]


def tool_message_blocks(body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(assistant tool calls, tool results) from a chat or Anthropic request body."""
    calls, results = [], []
    for message in body["messages"]:
        if message["role"] == "tool":
            results.append({"id": message["tool_call_id"], "content": message["content"]})
        for call in message.get("tool_calls") or []:
            calls.append({"id": call["id"], "name": call["function"]["name"],
                          "arguments": json.loads(call["function"]["arguments"])})
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if block.get("type") == "tool_use":
                    calls.append({"id": block["id"], "name": block["name"],
                                  "arguments": block["input"]})
                if block.get("type") == "tool_result":
                    content = block["content"]
                    text = content if isinstance(content, str) else content[0]["text"]
                    results.append({"id": block["tool_use_id"], "content": text})
    return calls, results


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model,path", TARGETS)
def test_tool_call_round_trip_with_history_through_a_mixed_list(upstream, model, path, stream):
    if stream:
        upstream.stream_frames = chat_tool_frames() if model == "m-chat" else anthropic_tool_frames()
    elif model == "m-chat":
        upstream.response_payload = _chat_completion("", tool=True)
        call = upstream.response_payload["choices"][0]["message"]["tool_calls"][0]
        call.update(id="call_new", function={"name": "search", "arguments": '{"q":"world"}'})
    else:
        upstream.response_payload = _anthropic_message("", tool=True)
        upstream.response_payload["content"][0].update(
            id="call_new", name="search", input={"q": "world"})
    body = responses_request(model, [LOOKUP, WEB, SEARCH], stream=stream)
    body["input"] = HISTORY
    response = post(raw_config(upstream, "lossy"), PATHS["openai-responses"], body)
    assert response.status_code == 200, response.text
    assert [item["field"] for item in losses(response)] == ["tools[1]"]

    # The previous turn reaches the upstream with its ID, arguments and result intact.
    calls, results = tool_message_blocks(upstream.last_request(path)["body"])
    assert calls == [{"id": "call_prev", "name": "lookup", "arguments": {"q": "hello"}}]
    assert results == [{"id": "call_prev", "content": "found: world"}]

    # The new call comes back as a Responses function_call with the same ID and arguments.
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        completed = next(event for event in events if event["type"] == "response.completed")
        output = completed["response"]["output"]
        deltas = "".join(event["delta"] for event in events
                         if event["type"] == "response.function_call_arguments.delta")
        assert deltas == '{"q":"world"}'
    else:
        output = response.json()["output"]
    function_calls = [item for item in output if item["type"] == "function_call"]
    assert len(function_calls) == 1
    assert function_calls[0]["call_id"] == "call_new"
    assert function_calls[0]["name"] == "search"
    assert json.loads(function_calls[0]["arguments"]) == {"q": "world"}
