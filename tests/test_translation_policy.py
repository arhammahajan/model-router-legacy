"""Prompt 10 contracts: real adapters, fake keys and local wire assertions."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import model_router as mr
from test_model_router import (
    PATHS, TRANSLATED_ROUTES, _body, _config, _llama_only_config, _with_gemini,
    llama_server, router_config, upstream,
)


def control(ingress: str, feature: str) -> dict[str, Any]:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    if feature == "reasoning":
        return ({"reasoning_effort": "high"} if ingress == "openai-chat" else
                {"reasoning": {"effort": "high"}} if ingress == "openai-responses" else
                {"thinking": {"type": "enabled", "budget_tokens": 4096}})
    if feature == "tools":
        return {"tools": ([{"type": "function", "function": {"name": "lookup", "parameters": schema}}]
                          if ingress == "openai-chat" else
                          [{"type": "function", "name": "lookup", "parameters": schema}]
                          if ingress == "openai-responses" else
                          [{"name": "lookup", "input_schema": schema}])}
    return ({"response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}}
            if ingress == "openai-chat" else
            {"text": {"format": {"type": "json_schema", "name": "result", "schema": schema}}}
            if ingress == "openai-responses" else
            {"output_format": {"type": "json_schema", "schema": schema}})


@pytest.mark.parametrize("ingress", ["anthropic", "openai-chat"])
@pytest.mark.parametrize("lossy", [False, True])
def test_gemini_cache_hints_never_create_an_extra_cache_service_request(upstream, ingress, lossy):
    raw = _with_gemini(_config(upstream), upstream)
    raw["providers"]["gem"]["compatibility"] = "lossy" if lossy else "strict"
    body = _body(ingress, "m-gem", False)
    cached = {"type": "text", "text": "Keep this prompt", "cache_control": {"type": "ephemeral"}}
    if ingress == "anthropic":
        body["system"] = [cached]
    else:
        body["messages"].insert(0, {"role": "system", "content": [cached]})
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=body, timeout=30)
    if lossy:
        assert response.status_code == 200, response.text
        assert "cache_control" in {loss["field"] for loss in json.loads(response.headers["x-model-router-loss"])}
        assert len(upstream.requests) == 1
        assert upstream.requests[0]["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
        assert "Keep this prompt" in json.dumps(upstream.requests[0]["body"])
    else:
        assert response.status_code == 400
        assert upstream.requests == []


@pytest.mark.parametrize("lossy", [False, True])
def test_discovered_local_models_use_the_configured_translation_policy(llama_server, lossy):
    raw = _llama_only_config(llama_server, compatibility="lossy" if lossy else "strict")
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/responses", json={
            "model": "llama-test", "input": "hello", "max_output_tokens": 32,
            "store": False, "include": ["reasoning.encrypted_content"],
        }, timeout=30)
    if lossy:
        assert response.status_code == 200, response.text
        assert {loss["field"] for loss in json.loads(response.headers["x-model-router-loss"])} == {"store", "include"}
        sent = llama_server.last_request("/v1/chat/completions")["body"]
        assert sent["model"] == "llama-test"
        assert sent["messages"][0]["content"] == "hello"
    else:
        assert response.status_code == 400
        assert not any(request["path"] == "/v1/chat/completions" for request in llama_server.requests)


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES)
@pytest.mark.parametrize("feature", ["reasoning", "tools", "schema"])
def test_known_unsupported_controls_fail_before_network(upstream, ingress, model, path, feature):
    raw = _config(upstream)
    raw["capabilities"] = {"overrides": {model: {
        "reasoning": {"supported": False}, "tools": {"supported": False},
        "structured_output": {"supported": False},
    }}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, model, False), **control(ingress, feature),
        })
    assert response.status_code == 400, response.text
    assert "same-protocol" in response.text
    assert "lossy" in response.text
    assert upstream.requests == []


@pytest.mark.parametrize("ingress,model,path", [
    ("openai-chat", "m-resp", "/v1/responses"),
    ("openai-responses", "m-chat", "/v1/chat/completions"),
    ("anthropic", "m-chat", "/v1/chat/completions"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_lossy_mode_reports_removed_controls_on_json_and_sse(
    upstream, ingress, model, path, stream,
):
    raw = _config(upstream)
    provider = {"m-chat": "chat", "m-resp": "resp"}[model]
    raw["providers"][provider]["compatibility"] = "lossy"
    raw["capabilities"] = {"overrides": {model: {
        "reasoning": {"supported": False}, "tools": {"supported": False},
        "structured_output": {"supported": False},
    }}}
    body = _body(ingress, model, stream)
    for feature in ("reasoning", "tools", "schema"):
        body.update(control(ingress, feature))
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=body, timeout=30)
    assert response.status_code == 200, response.text
    losses = json.loads(response.headers["x-model-router-loss"])
    assert len(losses) == 3
    assert all(item["action"] == "dropped" for item in losses)
    sent = upstream.last_request(path)["body"]
    for key in ("tools", "reasoning", "reasoning_effort", "thinking", "response_format", "output_format"):
        assert not sent.get(key), sent
    assert not sent.get("text", {}).get("format"), sent






@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("lossy", [False, True])
def test_unmapped_adapter_reasoning_is_never_silent(upstream, ingress, lossy):
    raw = _config(upstream)
    if lossy:
        raw["providers"]["anth"]["compatibility"] = "lossy"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, "m-anth", False), **control(ingress, "reasoning"),
        })
    if lossy:
        assert response.status_code == 200, response.text
        assert json.loads(response.headers["x-model-router-loss"])[0]["action"] == "dropped"
        assert "thinking" not in upstream.last_request("/v1/messages")["body"]
    else:
        assert response.status_code == 400, response.text
        assert upstream.requests == []






@pytest.mark.parametrize("bad", ["unknown", True, 1])
def test_invalid_policy_is_rejected(upstream, bad):
    raw = _config(upstream)
    raw["providers"]["chat"]["compatibility"] = bad
    with pytest.raises(SystemExit, match="compatibility"):
        mr.build_config(raw)


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("lossy", [False, True])
def test_effort_to_anthropic_budget_is_an_approximation(upstream, ingress, lossy):
    raw = _config(upstream)
    raw["models"][0]["model"] = "claude-3-7-sonnet-20250219"
    raw["providers"]["anth"]["compatibility"] = "lossy" if lossy else "strict"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, "m-anth", False), **control(ingress, "reasoning"),
            ("max_tokens" if ingress == "openai-chat" else "max_output_tokens"): 40000,
        })
    if not lossy:
        assert response.status_code == 400, response.text
        assert "approximate" in response.text
        assert upstream.requests == []
        return
    assert response.status_code == 200, response.text
    losses = json.loads(response.headers["x-model-router-loss"])
    assert [(item["field"].split(".")[0], item["action"]) for item in losses] == [
        ("reasoning_effort" if ingress == "openai-chat" else "reasoning", "approximated")]
    sent = upstream.last_request("/v1/messages")["body"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_legacy_drop_params_selects_diagnosed_lossy_policy(upstream):
    raw = _config(upstream)
    raw["providers"]["anth"]["drop_params"] = True
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["openai-chat"], json={
            **_body("openai-chat", "m-anth", False), "reasoning_effort": "high",
        })
    assert response.status_code == 200, response.text
    assert json.loads(response.headers["x-model-router-loss"])[0]["field"] == "reasoning_effort"
    assert "thinking" not in upstream.last_request("/v1/messages")["body"]






@pytest.mark.parametrize("ingress,model,path", [
    ("openai-chat", "m-chat", "/v1/chat/completions"),
    ("openai-responses", "m-resp", "/v1/responses"),
    ("anthropic", "m-anth", "/v1/messages"),
])
def test_native_vendor_fields_bypass_translation_policy(upstream, ingress, model, path):
    raw = _config(upstream)
    raw["capabilities"] = {"overrides": {model: {"reasoning": {"supported": False}}}}
    body = {**_body(ingress, model, False), **control(ingress, "reasoning"),
            "vendor_future_control": {"option": "keep"}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=body)
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    sent = upstream.last_request(path)["body"]
    for key, item in body.items():
        if key != "model":
            assert sent[key] == item






@pytest.mark.parametrize("model", ["m-gem", "m-anth"])
@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_chat_store_false_is_the_default_and_loses_nothing(upstream, model, policy):
    """Chat Completions stores nothing unless asked, so `store: false` needs no mapping."""
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["compatibility"] = policy
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64, "store": False, "stream_options": {"include_usage": True}}
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/chat/completions", json=body, timeout=30)
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    assert all("store" not in request["body"] for request in upstream.requests)


@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_chat_store_true_cannot_be_honored_by_a_destination_without_stored_completions(upstream, policy):
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["compatibility"] = policy
    body = {"model": "m-gem", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64, "store": True}
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/chat/completions", json=body, timeout=30)
    if policy == "strict":
        assert response.status_code == 400 and "store" in response.text
    else:
        (loss,) = json.loads(response.headers["x-model-router-loss"])
        assert loss["field"] == "store" and loss["action"] == "dropped"


@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_codex_client_metadata_stays_out_of_translated_adapter_context(upstream, lossy, stream):
    raw = _config(upstream)
    raw["providers"]["chat"]["compatibility"] = "lossy" if lossy else "strict"
    body = {**_body("openai-responses", "m-chat", stream),
            "client_metadata": {"session_id": "codex-session", "api_base": "untrusted"}}
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/responses", json=body, timeout=30)
    if lossy:
        assert response.status_code == 200, response.text
        assert "client_metadata" in {
            loss["field"] for loss in json.loads(response.headers["x-model-router-loss"])
        }
        assert upstream.requests
        assert "client_metadata" not in upstream.requests[0]["body"]
        assert "untrusted" not in json.dumps(upstream.requests[0]["body"])
    else:
        assert response.status_code == 400
        assert upstream.requests == []
