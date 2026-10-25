"""Prompt 10 contracts: real adapters, fake keys and local wire assertions."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import model_router as mr
from test_model_router import (
    PATHS, TRANSLATED_ROUTES, _body, _config, _with_gemini, router_config, upstream,
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


@pytest.mark.parametrize("ingress,model,path", [
    ("openai-chat", "m-resp", "/v1/responses"),
    ("openai-responses", "m-chat", "/v1/chat/completions"),
    ("anthropic", "m-chat", "/v1/chat/completions"),
    ("anthropic", "m-resp", "/v1/responses"),
])
def test_unknown_metadata_allows_mapped_schema(upstream, ingress, model, path):
    with router_config(_config(upstream)) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, model, False), **control(ingress, "schema"),
        })
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    sent = upstream.last_request(path)["body"]
    fmt = sent.get("response_format") or sent["text"]["format"]
    schema = fmt.get("json_schema", fmt)["schema"]
    assert schema["properties"]["answer"] == {"type": "string"}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("ingress,model,path", [
    ("openai-responses", "m-chat", "/v1/chat/completions"),
    ("openai-chat", "m-resp", "/v1/responses"),
])
def test_unknown_effort_metadata_does_not_disable_mapping(upstream, ingress, model, path):
    raw = _config(upstream)
    raw["capabilities"] = {"overrides": {model: {"reasoning": {
        "supported": None, "effort": {"supported": None, "values": None},
    }}}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, model, False), **control(ingress, "reasoning"),
        })
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    sent = upstream.last_request(path)["body"]
    assert (sent["reasoning_effort"] if "reasoning_effort" in sent else sent["reasoning"]["effort"]) == "high"


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


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES + [
    (ingress, "m-gem", "/v1beta/models/gemini-2.5-flash:generateContent")
    for ingress in PATHS
] + [(ingress, model, path) for ingress, model, path in [
    ("anthropic", "m-anth", "/v1/messages"),
    ("openai-chat", "m-chat", "/v1/chat/completions"),
    ("openai-responses", "m-resp", "/v1/responses"),
]])
def test_dialect_headers_and_credentials_on_the_wire(upstream, ingress, model, path):
    raw = _with_gemini(_config(upstream), upstream)
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, model, False), headers={
            "AnThRoPiC-BeTa": "test-beta", "AnThRoPiC-VeRsIoN": "2023-06-01",
            "OpEnAI-BeTa": "test-openai-beta", "Authorization": "Bearer client-key",
            "X-Api-Key": "client-key", "api-key": "client-key", "x-goog-api-key": "client-key",
            "Cookie": "client-key", "Proxy-Authorization": "Bearer client-key",
            "X-Arbitrary-Transport": "must-not-bridge",
        }, timeout=30)
    assert response.status_code == 200, response.text
    headers = upstream.last_request(path)["headers"]
    assert not any("client-key" in item for item in headers.values())
    if ingress != raw["providers"][{"m-anth": "anth", "m-chat": "chat", "m-resp": "resp", "m-gem": "gem"}[model]]["dialect"]:
        assert "x-arbitrary-transport" not in headers
        if model == "m-anth":
            assert headers["anthropic-beta"] == "test-beta"
            assert headers["anthropic-version"] == "2023-06-01"
            assert "openai-beta" not in headers
        elif model in ("m-chat", "m-resp"):
            assert headers["openai-beta"] == "test-openai-beta"
            assert "anthropic-beta" not in headers
            assert "anthropic-version" not in headers
        else:
            assert "openai-beta" not in headers and "anthropic-beta" not in headers
    assert (headers.get("authorization") == "Bearer upstream-key"
            or headers.get("x-api-key") == "upstream-key"
            or headers.get("x-goog-api-key") == "upstream-key")


@pytest.mark.parametrize("ingress,model,path", [
    ("openai-chat", "m-anth", "/v1/messages"),
    ("openai-responses", "m-anth", "/v1/messages"),
    ("anthropic", "m-anth", "/v1/messages"),
])
def test_header_precedence_is_case_insensitive(upstream, ingress, model, path):
    raw = _config(upstream)
    raw["providers"]["anth"].update({
        "headers": {"ANTHROPIC-BETA": "server-beta", "ANTHROPIC-VERSION": "server-version"},
        "litellm_kwargs": {"headers": {"anthropic-beta": "kwargs-beta"},
                           "extra_headers": {"Anthropic-Beta": "extra-beta"}},
    })
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, model, False), headers={
            "anthropic-beta": "client-beta", "anthropic-version": "client-version",
        })
    assert response.status_code == 200, response.text
    headers = upstream.last_request(path)["headers"]
    assert headers["anthropic-beta"] == "server-beta"
    assert headers["anthropic-version"] == "server-version"


@pytest.mark.parametrize("bad", ["unknown", True, 1])
def test_invalid_policy_is_rejected(upstream, bad):
    raw = _config(upstream)
    raw["providers"]["chat"]["compatibility"] = bad
    with pytest.raises(SystemExit, match="compatibility"):
        mr.build_config(raw)


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
def test_supported_effort_to_anthropic_budget_reaches_upstream(upstream, ingress):
    raw = _config(upstream)
    raw["models"][0]["model"] = "claude-3-7-sonnet-20250219"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, "m-anth", False), **control(ingress, "reasoning"),
            ("max_tokens" if ingress == "openai-chat" else "max_output_tokens"): 40000,
        })
    assert response.status_code == 200, response.text
    assert "x-model-router-loss" not in response.headers
    sent = upstream.last_request("/v1/messages")["body"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}


@pytest.mark.parametrize("model,path", [("m-chat", "/v1/chat/completions"), ("m-resp", "/v1/responses")])
@pytest.mark.parametrize("lossy", [False, True])
def test_token_budget_approximation_requires_explicit_lossy_mode(upstream, model, path, lossy):
    raw = _config(upstream)
    raw["providers"]["chat" if model == "m-chat" else "resp"]["compatibility"] = "lossy" if lossy else "strict"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["anthropic"], json={
            **_body("anthropic", model, False), **control("anthropic", "reasoning"),
        })
    if not lossy:
        assert response.status_code == 400
        assert upstream.requests == []
        return
    assert response.status_code == 200, response.text
    assert json.loads(response.headers["x-model-router-loss"]) == [{
        "field": "thinking", "reason": "token budget becomes an approximate effort", "action": "approximated",
    }]
    sent = upstream.last_request(path)["body"]
    assert (sent.get("reasoning_effort") or sent["reasoning"]["effort"]) == "high"


@pytest.mark.parametrize("lossy", [False, True])
def test_known_effort_values_are_checked_without_echoing_values(upstream, lossy):
    raw = _config(upstream)
    raw["providers"]["chat"]["compatibility"] = "lossy" if lossy else "strict"
    raw["capabilities"] = {"overrides": {"m-chat": {"reasoning": {
        "supported": True, "effort": {"supported": True, "values": ["low", "high"]},
    }}}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["openai-responses"], json={
            **_body("openai-responses", "m-chat", False), "reasoning": {"effort": "client-secret"},
        })
    assert "client-secret" not in response.text
    assert "client-secret" not in str(response.headers)
    if lossy:
        assert response.status_code == 200, response.text
        assert json.loads(response.headers["x-model-router-loss"])[0]["field"] == "reasoning"
        assert "reasoning_effort" not in upstream.last_request("/v1/chat/completions")["body"]
    else:
        assert response.status_code == 400
        assert upstream.requests == []


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES)
def test_lossy_policy_does_not_admit_unknown_or_administrative_settings(upstream, ingress, model, path):
    raw = _config(upstream)
    for provider in raw["providers"].values():
        provider["compatibility"] = "lossy"
    with router_config(raw) as base:
        for setting in ({"unknown_generation_control": 42}, {"drop_params": True},
                        {"extra_body": {"extra_headers": {"authorization": "client-key"}}}):
            response = httpx.post(base + PATHS[ingress], json={
                **_body(ingress, model, False), **setting,
            })
            assert response.status_code == 400, response.text
    assert upstream.requests == []


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


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
def test_unknown_metadata_gemini_reasoning_mapping(upstream, ingress):
    raw = _with_gemini(_config(upstream), upstream)
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, "m-gem", False), **control(ingress, "reasoning"),
        }, timeout=30)
    assert response.status_code == 200, response.text
    sent = upstream.last_request("/v1beta/models/gemini-2.5-flash:generateContent")["body"]
    assert sent["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096


@pytest.mark.parametrize("lossy", [False, True])
def test_known_unsupported_sampling_extension(upstream, lossy):
    raw = _config(upstream)
    raw["providers"]["chat"]["compatibility"] = "lossy" if lossy else "strict"
    raw["capabilities"] = {"overrides": {"m-chat": {"sampling": {"min_p": {"supported": False}}}}}
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["openai-responses"], json={
            **_body("openai-responses", "m-chat", False), "extra_body": {"min_p": 0.2, "top_k": 12},
        })
    if not lossy:
        assert response.status_code == 400
        assert upstream.requests == []
    else:
        assert response.status_code == 200, response.text
        assert json.loads(response.headers["x-model-router-loss"])[0]["field"] == "extra_body.min_p"
        sent = upstream.last_request("/v1/chat/completions")["body"]
        assert "min_p" not in sent and sent["top_k"] == 12


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


def test_unmapped_native_stop_control_is_diagnosed(upstream):
    raw = _config(upstream)
    raw["providers"]["resp"]["compatibility"] = "lossy"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS["anthropic"], json={
            **_body("anthropic", "m-resp", False), "stop_sequences": ["END"],
        })
    assert response.status_code == 200, response.text
    assert json.loads(response.headers["x-model-router-loss"])[0]["field"] == "stop_sequences"
    sent = upstream.last_request("/v1/responses")["body"]
    assert "stop" not in sent and "stop_sequences" not in sent


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("level", ["client", "kwargs_headers", "kwargs_extra"])
def test_translated_header_precedence_below_provider_headers(upstream, ingress, level):
    raw = _config(upstream)
    kwargs = {}
    if level != "client":
        kwargs["headers"] = {"AnThRoPiC-VeRsIoN": "kwargs-version", "AnThRoPiC-BeTa": "kwargs-beta"}
    if level == "kwargs_extra":
        kwargs["extra_headers"] = {"ANTHROPIC-VERSION": "extra-version", "ANTHROPIC-BETA": "extra-beta"}
    raw["providers"]["anth"]["litellm_kwargs"] = kwargs
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, "m-anth", False), headers={
            "anthropic-version": "client-version", "anthropic-beta": "client-beta",
        })
    assert response.status_code == 200, response.text
    prefix = {"client": "client", "kwargs_headers": "kwargs", "kwargs_extra": "extra"}[level]
    headers = upstream.last_request("/v1/messages")["headers"]
    assert headers["anthropic-version"] == prefix + "-version"
    assert headers["anthropic-beta"] == prefix + "-beta"
