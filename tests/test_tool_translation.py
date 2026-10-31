"""Tool decisions and custom calls through real pinned adapters and local HTTP."""
from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from model_router.common import RouterError
from model_router.tools import decide_tool, plan_tools
from test_model_router import (
    PATHS, _anthropic_message, _chat_completion, _config, _gemini_response,
    _responses_response, _with_gemini, router_config, upstream,
)

TEXT = '*** Begin Patch\n+snow ☃ and "quotes" \\ path\n*** End Patch'
CUSTOM = {"type": "custom", "name": "patch", "description": "Apply a patch"}
INGRESSES = ["openai-chat", "openai-responses"]
EMULATED = [(source, target) for source in INGRESSES
            for target in ("m-chat", "m-anth", "m-gem")
            if not (source == "openai-chat" and target == "m-chat")]
UPSTREAM_PATH = {"m-chat": "/v1/chat/completions", "m-anth": "/v1/messages",
                 "m-gem": "/v1beta/models/gemini-2.5-flash:generateContent", "m-resp": "/v1/responses"}
DIALECT = {"m-chat": "openai-chat", "m-anth": "anthropic", "m-gem": "gemini", "m-resp": "openai-responses"}


def configured(upstream, policy="lossy"):
    raw = _with_gemini(_config(upstream), upstream)
    raw["models"][2]["model"] = "gpt-4.1"
    for provider in raw["providers"].values():
        provider["compatibility"] = policy
    return raw


def request(source, target, *, stream=False, namespace=None):
    custom = deepcopy(CUSTOM)
    body = {"model": target, "stream": stream}
    if source == "openai-chat":
        body.update(messages=[{"role": "user", "content": "patch this"}],
                    tools=[{"type": "custom", "custom": {k: v for k, v in custom.items() if k != "type"}}],
                    tool_choice={"type": "custom", "custom": {"name": "patch"}})
    else:
        body.update(input="patch this", tools=[custom], tool_choice={"type": "custom", "name": "patch"})
        if namespace:
            body["tools"] = [{"type": "namespace", "name": namespace, "tools": [custom]}]
            body["tool_choice"]["namespace"] = namespace
    return body


def answer(model, name="patch", text=TEXT):
    args = {"content": text}
    if model == "m-chat":
        data = _chat_completion("", tool=True)
        data["choices"][0]["message"]["tool_calls"][0].update(
            id="call_new", function={"name": name, "arguments": json.dumps(args)})
    elif model == "m-anth":
        data = _anthropic_message("", tool=True)
        data["content"] = [{"type": "tool_use", "id": "call_new", "name": name, "input": args}]
    elif model == "m-gem":
        data = _gemini_response()
        data["candidates"][0]["content"]["parts"] = [{"functionCall": {"id": "call_new", "name": name, "args": args}}]
    else:
        data = _responses_response("")
        data["output"] = [{"type": "custom_tool_call", "id": "ctc_new", "call_id": "call_new",
                           "name": name, "input": text, "status": "completed"}]
    return data


def frame(data, event=None):
    return (("event: " + event + "\n" if event else "") + "data: " + json.dumps(data) + "\n\n").encode()


def streamed_answer(model, name="patch", text=TEXT):
    args = json.dumps({"content": text})
    fragments = [args[:4], args[4:18], args[18:]]
    if model == "m-chat":
        base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "upstream"}
        frames = []
        for index, fragment in enumerate(fragments):
            call = {"index": 0, "function": {"arguments": fragment}}
            if index == 0:
                call.update(id="call_new", type="function")
                call["function"]["name"] = name
            frames.append(frame({**base, "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}))
        return frames + [frame({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}), b"data: [DONE]\n\n"]
    if model == "m-anth":
        message = _anthropic_message("")
        message.update(content=[], stop_reason=None)
        events = [
            {"type": "message_start", "message": message},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "call_new", "name": name, "input": {}}},
            *[{"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": part}} for part in fragments],
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
             "usage": {"output_tokens": 3}}, {"type": "message_stop"},
        ]
        return [frame(event, event["type"]) for event in events]
    if model == "m-gem":
        return [frame(answer(model, name, text))]
    final = answer(model, name, text)
    item = final["output"][0]
    events = [
        {"type": "response.created", "response": {**final, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "input": "", "status": "in_progress"}},
        {"type": "response.custom_tool_call_input.delta", "item_id": item["id"], "output_index": 0, "delta": text},
        {"type": "response.custom_tool_call_input.done", "item_id": item["id"], "output_index": 0, "input": text},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": final},
    ]
    return [frame({**event, "sequence_number": index}, event["type"]) for index, event in enumerate(events)]


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def returned_call(source, payload):
    if source == "openai-chat":
        call, = payload["choices"][0]["message"]["tool_calls"]
        assert call["type"] == "custom"
        return call["id"], call["custom"]["name"], call["custom"]["input"]
    call, = [item for item in payload["output"] if item["type"] == "custom_tool_call"]
    return call["call_id"], call["name"], call["input"]


@pytest.mark.parametrize("source,target", EMULATED)
@pytest.mark.parametrize("stream", [False, True])
def test_custom_tool_round_trip_through_real_adapters(upstream, source, target, stream):
    body = request(source, target, stream=stream)
    upstream.response_payload = answer(target)
    if stream:
        upstream.stream_frames = streamed_answer(target)
    with router_config(configured(upstream)) as base:
        response = httpx.post(base + PATHS[source], json=body, timeout=30)
        assert response.status_code == 200, response.text
        assert json.loads(response.headers["x-model-router-loss"]) == [{
            "field": "tools[0]", "action": "approximated",
            "reason": "custom tool input is emulated by a string-valued function; "
                      "native freeform/grammar generation is not preserved"}]
        if stream:
            output = events(response)
            assert not any(event.get("type") == "error" for event in output), output
            if source == "openai-responses":
                assert not any("function_call_arguments" in event["type"] for event in output)
                assert "".join(event["delta"] for event in output
                               if event["type"] == "response.custom_tool_call_input.delta") == TEXT
                payload = next(event["response"] for event in output if event["type"] == "response.completed")
                assert [event["sequence_number"] for event in output] == list(range(len(output)))
            else:
                calls = [call for event in output for choice in event.get("choices", [])
                         for call in choice["delta"].get("tool_calls", [])]
                payload = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}
        else:
            payload = response.json()
        call_id, name, text = returned_call(source, payload)
        assert call_id and name == "patch" and text == TEXT
        # Send the call/result back through the same plan on a second request.
        continuation = deepcopy(body)
        continuation["stream"] = False
        if source == "openai-responses":
            continuation["input"] = [{"role": "user", "content": "patch this"}, *payload["output"],
                                     {"type": "custom_tool_call_output", "call_id": call_id, "output": "applied"}]
        else:
            continuation["messages"] += [payload["choices"][0]["message"],
                                         {"role": "tool", "tool_call_id": call_id, "content": "applied"}]
        upstream.stream_frames = None
        second = httpx.post(base + PATHS[source], json=continuation, timeout=30)
        assert second.status_code == 200, second.text
    sent = upstream.last_request(UPSTREAM_PATH[target])["body"]
    assert "applied" in json.dumps(sent)
    if target == "m-chat":
        call = next(message for message in sent["messages"] if message.get("tool_calls"))["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"content": TEXT}
        assert sent["tool_choice"] == {"type": "function", "function": {"name": "patch"}}
    elif target == "m-anth":
        call = next(block for message in sent["messages"] if isinstance(message.get("content"), list)
                    for block in message["content"] if block.get("type") == "tool_use")
        assert call["input"] == {"content": TEXT}
        assert sent["tool_choice"]["name"] == "patch"
    else:
        parts = [part for content in sent["contents"] for part in content["parts"]]
        call = next(part.get("function_call", part.get("functionCall")) for part in parts
                    if "function_call" in part or "functionCall" in part)
        assert call["args"] == {"content": TEXT}


@pytest.mark.parametrize("source,target", EMULATED)
def test_strict_policy_refuses_custom_emulation_before_upstream(upstream, source, target):
    with router_config(configured(upstream, "strict")) as base:
        response = httpx.post(base + PATHS[source], json=request(source, target), timeout=30)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "translation_loss"
    assert upstream.requests == []


@pytest.mark.parametrize("stream", [False, True])
def test_chat_custom_to_responses_uses_native_custom_contract(upstream, stream):
    upstream.response_payload = answer("m-resp")
    upstream.stream_frames = streamed_answer("m-resp") if stream else None
    with router_config(configured(upstream, "strict")) as base:
        response = httpx.post(base + PATHS["openai-chat"], json=request("openai-chat", "m-resp", stream=stream), timeout=30)
        assert response.status_code == 200, response.text
        assert "x-model-router-loss" not in response.headers
        assert upstream.last_request("/v1/responses")["body"]["tools"][0]["type"] == "custom"
        if not stream:
            payload = response.json()
        else:
            calls = [call for event in events(response) for choice in event.get("choices", [])
                     for call in choice["delta"].get("tool_calls", [])]
            payload = {"choices": [{"message": {"role": "assistant", "tool_calls": calls}}]}
        call_id, name, text = returned_call("openai-chat", payload)
        assert name == "patch" and text == TEXT
        continuation = request("openai-chat", "m-resp")
        continuation["messages"] += [payload["choices"][0]["message"],
                                     {"role": "tool", "tool_call_id": call_id, "content": "applied"}]
        upstream.stream_frames = None
        second = httpx.post(base + PATHS["openai-chat"], json=continuation, timeout=30)
        assert second.status_code == 200, second.text
    history = upstream.last_request("/v1/responses")["body"]["input"]
    call, = [item for item in history if item.get("type") == "custom_tool_call"]
    result, = [item for item in history if item.get("type") == "custom_tool_call_output"]
    assert call["input"] == TEXT and call["call_id"] == result["call_id"] == call_id


def test_namespace_custom_tools_keep_distinct_identities_and_original_loss_paths():
    body = request("openai-responses", "m-chat", namespace="one")
    body["tools"].append({"type": "namespace", "name": "two", "tools": [CUSTOM]})
    body["input"] = [{"type": "custom_tool_call", "call_id": "old", "namespace": "two",
                      "name": "patch", "input": TEXT},
                     {"type": "custom_tool_call_output", "call_id": "old", "output": "ok"}]
    original = deepcopy(body)
    plan = plan_tools(body, "openai-responses", "openai-chat")
    assert body == original
    names = [tool["name"] for tool in plan.request["tools"]]
    assert len(set(names)) == 2
    assert plan.request["tool_choice"]["name"] == names[0]
    assert plan.request["input"][0]["name"] == names[1]
    assert [loss["field"] for loss in plan.losses] == ["tools[0].tools[0]", "tools[1].tools[0]"]
    for name, namespace in zip(names, ("one", "two")):
        restored = plan.response({"output": [{"type": "function_call", "name": name, "id": "fc_1",
                                             "call_id": "call", "arguments": json.dumps({"content": TEXT})}]})
        assert restored["output"][0]["namespace"] == namespace
        assert restored["output"][0]["name"] == "patch"
        assert restored["output"][0]["input"] == TEXT


@pytest.mark.parametrize("value", ['{"content": 1}', 'not JSON', '{"content":"x","extra":1}', '[]'])
def test_invalid_custom_arguments_fail_with_safe_error(value):
    plan = plan_tools(request("openai-responses", "m-chat"), "openai-responses", "openai-chat")
    with pytest.raises(RouterError) as error:
        plan.response({"output": [{"type": "function_call", "name": "patch", "arguments": value}]})
    assert error.value.status == 502
    assert error.value.code == "invalid_tool_call"
    assert value not in str(error.value)


def test_tool_decisions_distinguish_client_emulation_and_hosted_tools():
    assert decide_tool("openai-responses", "anthropic", "custom", "tools[0]").outcome == "approximated"
    assert decide_tool("anthropic", "openai-chat", "bash_20250124", "tools[0]").execution == "client"
    assert decide_tool("gemini", "anthropic", "googleSearch", "tools[0]").outcome == "unsupported"
    assert decide_tool("openai-chat", "openai-responses", "custom", "tools[0]").outcome == "exact"


def test_removed_tool_history_is_refused_even_under_lossy_policy():
    body = {"tools": [{"name": "bash", "type": "bash_20250124"}], "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "old", "name": "bash", "input": {}}]}]}
    with pytest.raises(RouterError) as error:
        plan_tools(body, "anthropic", "openai-chat")
    assert error.value.code == "unsupported_tool_history"


@pytest.mark.parametrize("target", ["m-chat", "m-anth", "m-gem"])
@pytest.mark.parametrize("stream", [False, True])
def test_namespaced_custom_calls_survive_real_adapter_responses(upstream, target, stream):
    body = request("openai-responses", target, stream=stream, namespace="editor")
    plan = plan_tools(body, "openai-responses", DIALECT[target])
    alias = plan.request["tools"][0]["name"]
    upstream.response_payload = answer(target, alias)
    upstream.stream_frames = streamed_answer(target, alias) if stream else None
    with router_config(configured(upstream)) as base:
        response = httpx.post(base + PATHS["openai-responses"], json=body, timeout=30)
    assert response.status_code == 200, response.text
    if stream:
        output = events(response)
        payload = next(event["response"] for event in output if event["type"] == "response.completed")
        for event in output:
            if event["type"] in ("response.output_item.added", "response.output_item.done"):
                assert event["item"]["type"] == "custom_tool_call"
                assert event["item"]["namespace"] == "editor"
                assert event["item"]["name"] == "patch"
    else:
        payload = response.json()
    call, = payload["output"]
    assert call["type"] == "custom_tool_call" and call["namespace"] == "editor"
    assert call["input"] == TEXT and call["name"] == "patch"
    assert json.loads(response.headers["x-model-router-loss"])[0]["field"] == "tools[0].tools[0]"


@pytest.mark.parametrize("source", INGRESSES)
def test_custom_grammar_is_carried_as_instructions_with_a_loss(source):
    body = request(source, "m-anth")
    grammar = {"syntax": "regex", "definition": "[a-z]+"}
    if source == "openai-chat":
        body["tools"][0]["custom"]["format"] = {"type": "grammar", "grammar": grammar}
    else:
        body["tools"][0]["format"] = {"type": "grammar", **grammar}
    plan = plan_tools(body, source, "anthropic")
    function = plan.request["tools"][0].get("function", plan.request["tools"][0])
    assert "[a-z]+" in function["description"]
    assert plan.losses[0]["action"] == "approximated"
    assert "grammar" in plan.losses[0]["reason"]


def test_chat_grammar_reaches_native_responses_without_emulation(upstream):
    body = request("openai-chat", "m-resp")
    body["tools"][0]["custom"]["format"] = {
        "type": "grammar", "grammar": {"syntax": "regex", "definition": "[a-z]+"}}
    upstream.response_payload = answer("m-resp", text="abc")
    with router_config(configured(upstream, "strict")) as base:
        response = httpx.post(base + PATHS["openai-chat"], json=body, timeout=30)
    assert response.status_code == 200, response.text
    tool = upstream.last_request("/v1/responses")["body"]["tools"][0]
    assert tool["format"] == {"type": "grammar", "syntax": "regex", "definition": "[a-z]+"}
    assert returned_call("openai-chat", response.json())[2] == "abc"


def response_tool_events(name="patch", text=TEXT, *, item_id="fc_new", call_id="call_new", index=0):
    args = json.dumps({"content": text})
    item = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": args,
            "status": "completed"}
    return [
        {"type": "response.output_item.added", "output_index": index,
         "item": {**item, "arguments": "", "status": "in_progress"}},
        {"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": index, "delta": args},
        {"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": index, "arguments": args},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def test_interleaved_custom_and_function_events_preserve_identity_and_payloads():
    body = request("openai-responses", "m-chat")
    body["tools"].append({"type": "function", "name": "lookup", "parameters": {"type": "object"}})
    plan = plan_tools(body, "openai-responses", "openai-chat")
    a = response_tool_events()
    b = response_tool_events("lookup", item_id="fc_lookup", call_id="call_lookup", index=1)
    final = {"type": "response.completed", "response": {"output": [a[-1]["item"], b[-1]["item"]]}}
    converted = list(plan.responses_stream([a[0], b[0], a[1], b[1], b[2], a[2], b[3], a[3], final]))
    custom_events = [event for event in converted if event["type"].startswith("response.custom_tool_call_input.")]
    assert [event["type"] for event in custom_events] == [
        "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done"]
    assert all(event["item_id"] == "ctc_new" and event["output_index"] == 0 for event in custom_events)
    assert b[1] in converted and b[2] in converted
    assert converted[-1]["response"]["output"][0]["input"] == TEXT
    assert converted[-1]["response"]["output"][1] == b[-1]["item"]


@pytest.mark.parametrize("failure", ["bad_json", "changed_final", "missing_completion", "late_delta"])
def test_invalid_custom_streams_never_complete(failure):
    plan = plan_tools(request("openai-responses", "m-chat"), "openai-responses", "openai-chat")
    stream = response_tool_events()
    if failure == "bad_json":
        stream[1]["delta"] = stream[2]["arguments"] = "bad JSON"
    elif failure == "changed_final":
        stream[3]["item"]["arguments"] = '{"content":"changed"}'
    elif failure == "late_delta":
        stream.insert(3, dict(stream[1]))
    seen = []
    with pytest.raises(RouterError) as error:
        for event in plan.responses_stream(stream):
            seen.append(event)
    assert error.value.code == "invalid_tool_call"
    assert not any(event["type"] == "response.completed" for event in seen)


def test_stream_error_does_not_decode_or_complete_partial_custom_calls():
    plan = plan_tools(request("openai-responses", "m-chat"), "openai-responses", "openai-chat")
    stream = response_tool_events()[:2]
    stream[1]["delta"] = '{"content":'
    error = {"type": "error", "code": "upstream_error", "message": "interrupted"}
    converted = list(plan.responses_stream([*stream, error, {"type": "response.completed"}]))
    assert converted[-1] == error
    assert not any(event["type"] == "response.custom_tool_call_input.done" for event in converted)


def test_custom_stream_buffer_limit_is_enforced(monkeypatch):
    import model_router.tools as tool_module
    monkeypatch.setattr(tool_module, "MAX_TOOL_ARGUMENTS", 5)
    plan = plan_tools(request("openai-responses", "m-chat"), "openai-responses", "openai-chat")
    with pytest.raises(RouterError) as error:
        list(plan.responses_stream(response_tool_events()))
    assert error.value.code == "invalid_tool_call"


@pytest.mark.parametrize("native_kind", ["web_search_call", "mcp_approval_request", "computer_call_output"])
def test_native_tool_history_cannot_be_silently_discarded(native_kind):
    body = request("openai-responses", "m-chat")
    body["input"] = [{"type": native_kind, "id": "vendor_state"}]
    with pytest.raises(RouterError) as error:
        plan_tools(body, "openai-responses", "openai-chat")
    assert error.value.code == "unsupported_tool_history"


def test_custom_identity_is_restored_in_echoed_tool_configuration():
    body = request("openai-responses", "m-chat", namespace="editor")
    plan = plan_tools(body, "openai-responses", "openai-chat")
    response = plan.response({"tools": plan.request["tools"], "tool_choice": plan.request["tool_choice"]})
    assert response["tools"] == body["tools"]
    assert response["tool_choice"] == body["tool_choice"]


@pytest.mark.parametrize("source,target", [("openai-chat", "m-chat"), ("openai-responses", "m-resp")])
def test_native_custom_tools_bypass_planning_and_preserve_responses(upstream, source, target):
    body = request(source, target)
    data = answer("m-resp")
    if source == "openai-chat":
        data = _chat_completion("", tool=True)
        data["choices"][0]["message"]["tool_calls"] = [
            {"id": "call_new", "type": "custom", "custom": {"name": "patch", "input": TEXT}}]
    upstream.response_payload = data
    with router_config(configured(upstream, "strict")) as base:
        response = httpx.post(base + PATHS[source], json=body, timeout=30)
    assert response.status_code == 200, response.text
    assert response.json() == data
    assert "x-model-router-loss" not in response.headers
    assert upstream.last_request(UPSTREAM_PATH[target])["body"]["tools"] == body["tools"]


def test_chat_interleaved_function_and_custom_calls_keep_text_and_ids():
    body = request("openai-chat", "m-anth")
    body["tools"].append({"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}})
    plan = plan_tools(body, "openai-chat", "anthropic")
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "starting", "tool_calls": [
            {"index": 0, "id": "custom_id", "function": {"name": "pa", "arguments": '{"content":'}},
            {"index": 1, "id": "function_id", "function": {"name": "lookup", "arguments": '{"q":'}},
        ]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": '"hello"}'}},
            {"index": 0, "function": {"name": "tch", "arguments": json.dumps(TEXT) + '}'}},
        ]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"total_tokens": 8}},
    ]
    output = list(plan.chat_stream(chunks))
    assert output[0]["choices"][0]["delta"] == {"content": "starting"}
    calls = output[1]["choices"][0]["delta"]["tool_calls"]
    assert calls[0] == {"index": 0, "id": "custom_id", "type": "custom", "custom": {"name": "patch", "input": TEXT}}
    assert calls[1]["id"] == "function_id" and calls[1]["index"] == 1
    assert json.loads(calls[1]["function"]["arguments"]) == {"q": "hello"}
    assert output[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert [chunk["usage"] for chunk in output if "usage" in chunk] == [{"total_tokens": 8}]


def test_model_tool_capability_rejection_supersedes_emulation(upstream):
    raw = configured(upstream)
    raw["capabilities"] = {"overrides": {"m-chat": {"tools": {"supported": False}}}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["openai-responses"], json=request("openai-responses", "m-chat"), timeout=30)
    assert response.status_code == 200, response.text
    assert "tools" not in upstream.last_request("/v1/chat/completions")["body"]
    assert all(loss["action"] == "dropped" for loss in json.loads(response.headers["x-model-router-loss"]))


def test_forced_choice_cannot_change_the_declared_tool_type():
    body = request("openai-responses", "m-chat")
    body["tool_choice"] = {"type": "function", "name": "patch"}
    with pytest.raises(RouterError) as error:
        plan_tools(body, "openai-responses", "openai-chat")
    assert error.value.code == "invalid_tool"


def test_orphaned_custom_result_cannot_be_translated_without_its_call():
    body = request("openai-responses", "m-chat")
    body["input"] = [{"type": "custom_tool_call_output", "call_id": "missing", "output": "ok"}]
    with pytest.raises(RouterError) as error:
        plan_tools(body, "openai-responses", "openai-chat")
    assert error.value.code == "invalid_tool"
