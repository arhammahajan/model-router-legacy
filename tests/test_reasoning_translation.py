"""Reasoning-control translation: exact mappings, approximations and unsupported values.

Offline: real LiteLLM adapters against local fake upstreams, asserting what reaches
the wire. Strict policy must reject whatever is not exact; lossy policy must apply
it and report exactly what was approximated or dropped.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from model_router.reasoning import plan_reasoning
from test_model_router import (
    PATHS, _anthropic_sse, _body, _chat_sse, _config, _llama_only_config,
    _with_gemini, llama_server, router_config, upstream,
)

WIRE = {"m-anth": "/v1/messages", "m-chat": "/v1/chat/completions",
        "m-resp": "/v1/responses",
        "m-gem": "/v1beta/models/gemini-2.5-flash:generateContent",
        "llama-test": "/v1/chat/completions"}
NATIVE = {"openai-chat": "m-chat", "openai-responses": "m-resp", "anthropic": "m-anth"}
LOSS = "x-model-router-loss"
SECRET_PROMPT = "PRIVATE-PROMPT-7f3a"


def raw_config(upstream, *, policy="strict", overrides=None, llama=None, anthropic_model=None):
    raw = _with_gemini(_config(upstream), upstream)
    # A model name LiteLLM knows to think; unknown names have no adapter mapping.
    raw["models"][0]["model"] = anthropic_model or "claude-sonnet-4-20250514"
    for provider in raw["providers"].values():
        provider["compatibility"] = policy
    if overrides:
        raw["capabilities"] = {"overrides": overrides}
    if llama is not None:
        raw["llama_cpp"] = {"servers": [llama.url], "timeout": 1.0, "refresh_seconds": 0,
                            "compatibility": policy}
    return raw


def send(raw, ingress, model, controls, *, stream=False):
    body = {**_body(ingress, model, stream), **controls}
    key = "max_output_tokens" if ingress == "openai-responses" else "max_tokens"
    body[key] = 32768
    with router_config(raw) as base:
        return httpx.post(base + PATHS[ingress], json=body, timeout=30)


def wire(server, model):
    return server.last_request(WIRE[model])["body"]


def losses(response):
    return json.loads(response.headers[LOSS])


def effort_for(ingress, value):
    return ({"reasoning_effort": value} if ingress == "openai-chat"
            else {"reasoning": {"effort": value}})


def total_requests(upstream, llama=None):
    return len(upstream.requests) + (len(llama.requests) if llama else 0)


# --------------------------------------------------------------------------
# Exact mappings: accepted by strict policy, nothing reported.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
def test_effort_none_disables_anthropic_thinking_explicitly(upstream, ingress):
    response = send(raw_config(upstream), ingress, "m-anth", effort_for(ingress, "none"))
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(upstream, "m-anth")
    assert sent["thinking"] == {"type": "disabled"}
    assert "output_config" not in sent


@pytest.mark.parametrize("target", ["m-chat", "m-resp"])
def test_anthropic_thinking_disabled_becomes_effort_none(upstream, target):
    response = send(raw_config(upstream), "anthropic", target, {"thinking": {"type": "disabled"}})
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(upstream, target)
    assert (sent["reasoning_effort"] if target == "m-chat" else sent["reasoning"]["effort"]) == "none"


@pytest.mark.parametrize("ingress,controls", [
    ("openai-chat", {"reasoning_effort": "none"}),
    ("openai-responses", {"reasoning": {"effort": "none"}}),
    ("anthropic", {"thinking": {"type": "disabled"}}),
])
def test_llama_thinking_disabled_is_explicit_for_every_ingress(llama_server, upstream, ingress, controls):
    response = send(raw_config(upstream, llama=llama_server), ingress, "llama-test", controls)
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(llama_server, "llama-test")
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert not {"reasoning_effort", "reasoning", "thinking", "thinking_budget_tokens"} & sent.keys()


def test_anthropic_budget_is_the_llama_budget_not_an_effort_bucket(llama_server, upstream):
    response = send(raw_config(upstream, llama=llama_server), "anthropic", "llama-test",
                    {"thinking": {"type": "enabled", "budget_tokens": 3000}})
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(llama_server, "llama-test")
    assert sent["thinking_budget_tokens"] == 3000
    assert "reasoning_effort" not in sent and "chat_template_kwargs" not in sent


def test_anthropic_budget_is_the_gemini_budget(upstream):
    response = send(raw_config(upstream), "anthropic", "m-gem",
                    {"thinking": {"type": "enabled", "budget_tokens": 3000}})
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    assert wire(upstream, "m-gem")["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 3000


def test_thinking_extension_budget_reaches_gemini_exactly(upstream):
    response = send(raw_config(upstream), "openai-chat", "m-gem",
                    {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2048}}})
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    assert wire(upstream, "m-gem")["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 2048


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("ingress,target", [("openai-chat", "m-resp"), ("openai-responses", "m-chat")])
def test_chat_and_responses_efforts_are_one_enum_and_never_renamed(upstream, ingress, target, effort):
    response = send(raw_config(upstream), ingress, target, effort_for(ingress, effort))
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(upstream, target)
    assert (sent["reasoning"]["effort"] if target == "m-resp" else sent["reasoning_effort"]) == effort


@pytest.mark.parametrize("effort,expected", [
    ("minimal", 1024), ("low", 2048), ("medium", 8192), ("high", 16384), ("xhigh", 32768),
])
@pytest.mark.parametrize("ingress", ["openai-responses"])
def test_documented_llama_effort_budgets_remain_accepted(llama_server, upstream, ingress, effort, expected):
    response = send(raw_config(upstream, llama=llama_server), ingress, "llama-test",
                    effort_for(ingress, effort))
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    assert wire(llama_server, "llama-test")["thinking_budget_tokens"] == expected


# --------------------------------------------------------------------------
# Capability validation: known facts decide, unknown facts stay neutral.
# --------------------------------------------------------------------------

def values_override(model, values, **extra):
    return {model: {"reasoning": {"supported": True, "effort": {"supported": True, "values": values},
                                  **extra}}}


@pytest.mark.parametrize("policy", ["strict", "lossy"])
@pytest.mark.parametrize("ingress,model", [("openai-responses", "m-chat"), ("openai-chat", "m-resp")])
def test_effort_outside_known_values_is_never_clamped(upstream, policy, ingress, model):
    raw = raw_config(upstream, policy=policy, overrides=values_override(model, ["low", "high"]))
    response = send(raw, ingress, model, effort_for(ingress, "medium"))
    if policy == "strict":
        assert response.status_code == 400, response.text
        assert "known values" in response.text
        assert upstream.requests == []
    else:
        assert response.status_code == 200, response.text
        (loss,) = losses(response)
        assert loss["action"] == "dropped" and "known values" in loss["reason"]
        sent = wire(upstream, model)
        assert "reasoning_effort" not in sent and "reasoning" not in sent


@pytest.mark.parametrize("ingress,model", [("openai-responses", "m-chat"), ("openai-chat", "m-resp")])
def test_effort_inside_known_values_passes_exactly(upstream, ingress, model):
    raw = raw_config(upstream, overrides=values_override(model, ["low", "high"]))
    response = send(raw, ingress, model, effort_for(ingress, "high"))
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers


@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_disable_is_rejected_when_the_destination_cannot_express_none(upstream, policy):
    raw = raw_config(upstream, policy=policy, overrides=values_override("m-chat", ["low", "high"]))
    response = send(raw, "anthropic", "m-chat", {"thinking": {"type": "disabled"}})
    if policy == "strict":
        assert response.status_code == 400 and upstream.requests == []
    else:
        (loss,) = losses(response)
        assert loss["field"] == "thinking" and loss["action"] == "dropped"
        assert "cannot express 'none'" in loss["reason"]
        assert "reasoning_effort" not in wire(upstream, "m-chat")


@pytest.mark.parametrize("ingress,controls", [
    ("openai-chat", {"reasoning_effort": "none"}),
    ("anthropic", {"thinking": {"type": "disabled"}}),
])
def test_model_that_cannot_turn_thinking_off_rejects_disable(upstream, ingress, controls):
    target = "m-anth" if ingress == "openai-chat" else "m-chat"
    raw = raw_config(upstream, overrides={target: {"reasoning": {"supported": True,
                                                                  "toggle": {"supported": False}}}})
    response = send(raw, ingress, target, controls)
    assert response.status_code == 400, response.text
    assert "disable" in response.text
    assert upstream.requests == []


@pytest.mark.parametrize("ingress,target,controls", [
    ("openai-chat", "m-anth", {"reasoning_effort": "none"}),
    ("openai-responses", "m-chat", {"reasoning": {"effort": "none"}}),
    ("anthropic", "m-resp", {"thinking": {"type": "disabled"}}),
])
def test_disable_on_a_model_that_does_not_reason_is_already_satisfied(upstream, ingress, target, controls):
    raw = raw_config(upstream, overrides={target: {"reasoning": {"supported": False}}})
    response = send(raw, ingress, target, controls)
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(upstream, target)
    assert not {"thinking", "reasoning", "reasoning_effort"} & sent.keys()


@pytest.mark.parametrize("controls", [
    {"reasoning_effort": "high"}, {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2048}}},
])
def test_enabling_reasoning_on_a_model_that_does_not_reason_is_a_loss(upstream, controls):
    raw = raw_config(upstream, overrides={"m-anth": {"reasoning": {"supported": False}}})
    response = send(raw, "openai-chat", "m-anth", controls)
    assert response.status_code == 400, response.text
    assert upstream.requests == []


@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_adapter_without_a_mapping_for_the_model_is_reported(upstream, policy):
    """LiteLLM refuses `thinking` for a model name it does not know; that is a loss, not a crash."""
    raw = raw_config(upstream, policy=policy, anthropic_model="upstream-anthropic")
    response = send(raw, "openai-chat", "m-anth", {"reasoning_effort": "none"})
    if policy == "strict":
        assert response.status_code == 400 and "no mapping" in response.text
        assert upstream.requests == []
    else:
        assert response.status_code == 200, response.text
        (loss,) = losses(response)
        assert loss["field"] == "reasoning_effort" and loss["action"] == "dropped"
        assert "thinking" not in wire(upstream, "m-anth")


@pytest.mark.parametrize("effort", ["xhigh", "max"])
@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_gemini_efforts_without_a_counterpart_are_reported_not_clamped(upstream, policy, ingress, effort):
    response = send(raw_config(upstream, policy=policy), ingress, "m-gem", effort_for(ingress, effort))
    if policy == "strict":
        assert response.status_code == 400 and "no control" in response.text
        assert upstream.requests == []
        return
    assert response.status_code == 200, response.text
    (loss,) = losses(response)
    assert loss["action"] == "dropped" and "Gemini" in loss["reason"]
    assert "thinkingConfig" not in wire(upstream, "m-gem").get("generationConfig", {})


@pytest.mark.parametrize("effort", ["max", "extreme"])
@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_llama_has_no_control_for_max_or_unknown_levels(llama_server, upstream, policy, effort):
    raw = raw_config(upstream, policy=policy, llama=llama_server)
    response = send(raw, "openai-responses", "llama-test", {"reasoning": {"effort": effort}})
    if policy == "strict":
        assert response.status_code == 400, response.text
        assert llama_server.requests and not any(
            r["path"] == "/v1/chat/completions" for r in llama_server.requests)
        return
    assert response.status_code == 200, response.text
    (loss,) = losses(response)
    assert loss["field"] == "reasoning.effort" and loss["action"] == "dropped"
    sent = wire(llama_server, "llama-test")
    assert not {"reasoning_effort", "reasoning", "thinking_budget_tokens"} & sent.keys()


def test_gemini_disable_is_exact_only_when_the_model_is_known_to_allow_it(upstream):
    controls = {"thinking": {"type": "disabled"}}
    unknown = send(raw_config(upstream, policy="lossy"), "anthropic", "m-gem", controls)
    (loss,) = losses(unknown)
    assert loss["action"] == "approximated" and "Gemini" in loss["reason"]
    assert send(raw_config(upstream), "anthropic", "m-gem", controls).status_code == 400
    known = raw_config(upstream, overrides={"m-gem": {"reasoning": {"supported": True,
                                                                    "toggle": {"supported": True}}}})
    exact = send(known, "anthropic", "m-gem", controls)
    assert exact.status_code == 200 and LOSS not in exact.headers
    assert wire(upstream, "m-gem")["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


def test_budget_outside_known_bounds_is_dropped_not_clamped(upstream):
    raw = raw_config(upstream, policy="lossy",
                     overrides={"m-chat": {"reasoning": {"supported": True, "budget": {
                         "supported": True, "min_tokens": 1024, "max_tokens": 8192}}}})
    response = send(raw, "anthropic", "m-chat", {"thinking": {"type": "enabled", "budget_tokens": 20000}})
    (loss,) = losses(response)
    assert loss["action"] == "dropped" and "bounds" in loss["reason"]
    assert "reasoning_effort" not in wire(upstream, "m-chat")


# --------------------------------------------------------------------------
# Approximations: strict rejects, lossy applies and reports "approximated".
# --------------------------------------------------------------------------

APPROXIMATIONS = [
    ("openai-chat", "m-anth", {"reasoning_effort": "medium"}, "reasoning_effort"),
    ("openai-responses", "m-anth", {"reasoning": {"effort": "medium"}}, "reasoning.effort"),
    ("openai-chat", "m-gem", {"reasoning_effort": "low"}, "reasoning_effort"),
    ("openai-responses", "m-gem", {"reasoning": {"effort": "high"}}, "reasoning.effort"),
    ("anthropic", "m-chat", {"thinking": {"type": "enabled", "budget_tokens": 4096}}, "thinking"),
    ("anthropic", "m-resp", {"thinking": {"type": "enabled", "budget_tokens": 4096}}, "thinking"),
    ("anthropic", "m-chat", {"output_config": {"effort": "high"}}, "output_config.effort"),
    ("anthropic", "m-resp", {"output_config": {"effort": "low"}}, "output_config.effort"),
]


@pytest.mark.parametrize("ingress,target,controls,field", APPROXIMATIONS)
def test_strict_rejects_every_approximation_before_the_network(upstream, ingress, target, controls, field):
    response = send(raw_config(upstream), ingress, target, controls)
    assert response.status_code == 400, response.text
    assert "approximate" in response.text and field in response.text
    assert upstream.requests == []


@pytest.mark.parametrize("ingress,target,controls,field", APPROXIMATIONS)
def test_lossy_reports_every_approximation(upstream, ingress, target, controls, field):
    response = send(raw_config(upstream, policy="lossy"), ingress, target, controls)
    assert response.status_code == 200, response.text
    assert [(item["field"], item["action"]) for item in losses(response)] == [(field, "approximated")]


def test_effort_levels_are_not_equated_across_scopes(upstream):
    """OpenAI reasoning effort and Anthropic all-output effort share names, not meaning."""
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat",
                    {"output_config": {"effort": "high"}})
    (loss,) = losses(response)
    assert "governs all output" in loss["reason"]
    assert wire(upstream, "m-chat")["reasoning_effort"] == "high"


@pytest.mark.parametrize("effort", ["max", "xhigh", "minimal"])
def test_anthropic_effort_is_carried_only_under_the_same_name(upstream, effort):
    """No renaming: with no published values only low/medium/high exist on both sides."""
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat",
                    {"output_config": {"effort": effort}})
    (loss,) = losses(response)
    assert loss["action"] == "dropped"
    assert "reasoning_effort" not in wire(upstream, "m-chat")


def test_anthropic_effort_uses_the_routes_published_values(upstream):
    raw = raw_config(upstream, policy="lossy",
                     overrides=values_override("m-resp", ["low", "medium", "high", "xhigh"]))
    response = send(raw, "anthropic", "m-resp", {"output_config": {"effort": "xhigh"}})
    assert [item["action"] for item in losses(response)] == ["approximated"]
    assert wire(upstream, "m-resp")["reasoning"]["effort"] == "xhigh"


def test_budget_and_effort_together_keep_the_level_and_report_the_budget(upstream):
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat",
                    {"thinking": {"type": "enabled", "budget_tokens": 4096},
                     "output_config": {"effort": "low"}})
    found = {item["field"]: item for item in losses(response)}
    assert found["thinking"]["action"] == "dropped" and "superseded" in found["thinking"]["reason"]
    assert found["output_config.effort"]["action"] == "approximated"
    assert wire(upstream, "m-chat")["reasoning_effort"] == "low"


def test_budget_survives_when_the_effort_has_no_destination(llama_server, upstream):
    response = send(raw_config(upstream, policy="lossy", llama=llama_server), "anthropic", "llama-test",
                    {"thinking": {"type": "enabled", "budget_tokens": 4096},
                     "output_config": {"effort": "low"}})
    assert [(i["field"], i["action"]) for i in losses(response)] == [("output_config.effort", "dropped")]
    assert wire(llama_server, "llama-test")["thinking_budget_tokens"] == 4096


def test_disabled_thinking_wins_over_an_effort_level(upstream):
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat",
                    {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}})
    assert [(i["field"], i["action"]) for i in losses(response)] == [("output_config.effort", "dropped")]
    assert wire(upstream, "m-chat")["reasoning_effort"] == "none"


# --------------------------------------------------------------------------
# Unsupported semantics.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["m-chat", "m-resp", "m-gem", "llama-test"])
def test_adaptive_thinking_has_no_equivalent_and_is_not_invented(llama_server, upstream, target):
    raw = raw_config(upstream, policy="lossy", llama=llama_server)
    response = send(raw, "anthropic", target, {"thinking": {"type": "adaptive"}})
    (loss,) = losses(response)
    assert loss["field"] == "thinking" and loss["action"] == "dropped"
    sent = wire(llama_server if target == "llama-test" else upstream, target)
    assert not {"reasoning_effort", "thinking", "thinking_budget_tokens"} & sent.keys()
    assert "reasoning" not in sent


@pytest.mark.parametrize("target", ["m-chat", "m-resp", "m-gem", "llama-test"])
def test_thinking_display_is_reported_when_the_destination_has_no_such_control(llama_server, upstream, target):
    raw = raw_config(upstream, policy="lossy", llama=llama_server)
    response = send(raw, "anthropic", target,
                    {"thinking": {"type": "disabled", "display": "omitted"}})
    assert "thinking.display" in {item["field"] for item in losses(response)}


@pytest.mark.parametrize("target", ["m-anth", "m-chat", "m-gem", "llama-test"])
@pytest.mark.parametrize("controls,field", [
    ({"reasoning": {"summary": "auto"}}, "reasoning.summary"),
    ({"reasoning": {"generate_summary": "concise"}}, "reasoning.generate_summary"),
    ({"include": ["reasoning.encrypted_content"]}, "include"),
    ({"store": False}, "store"),
])
def test_responses_only_controls_are_never_claimed_to_be_honored(llama_server, upstream, target, controls, field):
    def generated():
        return [r for r in upstream.requests + llama_server.requests
                if r["path"] in WIRE.values() and r["body"] is not None]

    response = send(raw_config(upstream, llama=llama_server), "openai-responses", target, controls)
    assert response.status_code == 400 and field in response.text
    assert generated() == []

    response = send(raw_config(upstream, policy="lossy", llama=llama_server),
                    "openai-responses", target, controls)
    assert response.status_code == 200, response.text
    (loss,) = losses(response)
    assert loss["field"] == field and loss["action"] == "dropped"
    sent = wire(llama_server if target == "llama-test" else upstream, target)
    reasoning = sent.get("reasoning") if isinstance(sent.get("reasoning"), dict) else {}
    assert not {"store", "include", "summary", "generate_summary"} & (set(sent) | set(reasoning))


# --------------------------------------------------------------------------
# Invalid or ambiguous explicit choices fail loudly.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("thinking", [
    {"budget_tokens": 2048},
    {"type": "turbo"},
    {"type": "enabled"},
    {"type": "enabled", "budget_tokens": 0},
    {"type": "disabled", "budget_tokens": 2048},
    {"type": "adaptive", "budget_tokens": 2048},
    {"type": "enabled", "budget_tokens": 2048, "display": "loud"},
])
def test_malformed_thinking_is_rejected_instead_of_defaulted(upstream, thinking):
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat", {"thinking": thinking})
    assert response.status_code == 400, response.text
    assert upstream.requests == []


def test_effort_and_thinking_object_cannot_both_be_honored(upstream):
    response = send(raw_config(upstream, policy="lossy"), "openai-chat", "m-anth", {
        "reasoning_effort": "high", "extra_body": {"thinking": {"type": "disabled"}}})
    assert response.status_code == 400 and "cannot be combined" in response.text
    assert upstream.requests == []


# --------------------------------------------------------------------------
# Native protocol passthrough is untouched, whatever the policy and capabilities.
# --------------------------------------------------------------------------

PASSTHROUGH = [
    ("anthropic", {"thinking": {"type": "adaptive", "display": "omitted"},
                   "output_config": {"effort": "max"}}),
    ("anthropic", {"thinking": {"type": "enabled", "budget_tokens": 2048}}),
    ("openai-responses", {"reasoning": {"effort": "max", "summary": "detailed"}, "store": False,
                          "include": ["reasoning.encrypted_content"]}),
    ("openai-chat", {"reasoning_effort": "max", "vendor_future_control": {"keep": True}}),
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ingress,controls", PASSTHROUGH)
def test_native_routes_forward_reasoning_controls_unchanged(upstream, ingress, controls, stream):
    model = NATIVE[ingress]
    raw = raw_config(upstream, overrides={model: {"reasoning": {"supported": False}}})
    response = send(raw, ingress, model, controls, stream=stream)
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    sent = wire(upstream, model)
    for key, value in controls.items():
        assert sent[key] == value


def test_llama_chat_passthrough_keeps_explicit_native_fields_over_effort(llama_server):
    with router_config(_llama_only_config(llama_server)) as base:
        response = httpx.post(base + PATHS["openai-chat"], json={
            **_body("openai-chat", "llama-test", False), "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": True}})
    assert response.status_code == 200 and LOSS not in response.headers
    assert wire(llama_server, "llama-test")["chat_template_kwargs"] == {"enable_thinking": True}


# --------------------------------------------------------------------------
# Streaming.
# --------------------------------------------------------------------------

def sse_payloads(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.parametrize("ingress,target,controls,check", [
    ("openai-chat", "m-anth", {"reasoning_effort": "none"},
     lambda sent: sent["thinking"] == {"type": "disabled"}),
    ("openai-responses", "m-chat", {"reasoning": {"effort": "none"}},
     lambda sent: sent["reasoning_effort"] == "none"),
    ("anthropic", "m-chat", {"thinking": {"type": "disabled"}},
     lambda sent: sent["reasoning_effort"] == "none"),
    ("anthropic", "m-resp", {"output_config": {"effort": "low"}},
     lambda sent: sent["reasoning"]["effort"] == "low"),
])
def test_streaming_requests_get_the_same_mapping_and_loss_header(upstream, ingress, target, controls, check):
    response = send(raw_config(upstream, policy="lossy"), ingress, target, controls, stream=True)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert check(wire(upstream, target))
    assert not any(p.get("type") == "error" or p.get("error") for p in sse_payloads(response.text))
    if "output_config" in controls:
        assert [i["action"] for i in losses(response)] == ["approximated"]
    else:
        assert LOSS not in response.headers


def test_streaming_rejection_happens_before_any_bytes(upstream):
    response = send(raw_config(upstream), "openai-responses", "m-gem", effort_for("openai-responses", "max"),
                    stream=True)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert upstream.requests == []


def test_anthropic_thinking_stream_surfaces_as_chat_reasoning_content(upstream):
    def frame(event, payload):
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
    start = _anthropic_sse("x")[0]
    upstream.stream_frames = [
        frame(*start),
        frame("content_block_start", {"type": "content_block_start", "index": 0,
                                      "content_block": {"type": "thinking", "thinking": ""}}),
        frame("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "thinking_delta", "thinking": "considering"}}),
        frame("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "signature_delta", "signature": "sig"}}),
        frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        frame("content_block_start", {"type": "content_block_start", "index": 1,
                                      "content_block": {"type": "text", "text": ""}}),
        frame("content_block_delta", {"type": "content_block_delta", "index": 1,
                                      "delta": {"type": "text_delta", "text": "answer"}}),
        frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        frame("message_delta", {"type": "message_delta",
                                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                "usage": {"output_tokens": 3}}),
        frame("message_stop", {"type": "message_stop"}),
    ]
    raw = raw_config(upstream, policy="lossy")
    response = send(raw, "openai-chat", "m-anth", {"extra_body": {"thinking": {
        "type": "enabled", "budget_tokens": 2048}}}, stream=True)
    assert response.status_code == 200, response.text
    assert wire(upstream, "m-anth")["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    deltas = [c["choices"][0]["delta"] for c in sse_payloads(response.text) if c.get("choices")]
    assert "considering" in "".join(d.get("reasoning_content") or "" for d in deltas)
    assert "answer" in "".join(d.get("content") or "" for d in deltas)


def test_chat_reasoning_stream_surfaces_as_anthropic_thinking_delta(upstream):
    chunk = json.loads(_chat_sse("x")[0])
    chunk["choices"][0]["delta"] = {"role": "assistant", "reasoning_content": "weighing options"}
    final = json.loads(_chat_sse("x")[1])
    answer = json.loads(_chat_sse("answer")[0])
    answer["choices"][0]["delta"] = {"content": "answer"}
    upstream.stream_frames = [f"data: {json.dumps(c)}\n\n".encode() for c in (chunk, answer, final)] + [
        b"data: [DONE]\n\n"]
    response = send(raw_config(upstream, policy="lossy"), "anthropic", "m-chat",
                    {"thinking": {"type": "disabled"}}, stream=True)
    assert response.status_code == 200, response.text
    assert wire(upstream, "m-chat")["reasoning_effort"] == "none"
    deltas = [p["delta"] for p in sse_payloads(response.text) if p.get("type") == "content_block_delta"]
    assert any(d.get("type") == "thinking_delta" and "weighing" in d["thinking"] for d in deltas)
    assert any(d.get("type") == "text_delta" and d["text"] == "answer" for d in deltas)


# --------------------------------------------------------------------------
# Diagnostics never carry prompts or credentials.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["strict", "lossy"])
def test_loss_diagnostics_contain_no_prompt_or_credentials(upstream, policy):
    raw = raw_config(upstream, policy=policy)
    body = {"model": "m-anth", "input": SECRET_PROMPT, "max_output_tokens": 64,
            "reasoning": {"effort": "high", "summary": "auto"}, "store": False,
            "include": ["reasoning.encrypted_content"], "instructions": SECRET_PROMPT}
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/responses", json=body, timeout=30)
    # A successful body is model output and may echo the request; diagnostics may not.
    shown = json.dumps(dict(response.headers)) + (response.text if response.status_code != 200 else "")
    assert SECRET_PROMPT not in shown and "upstream-key" not in shown
    if policy == "lossy":
        assert {i["field"] for i in losses(response)} >= {
            "reasoning.summary", "reasoning.effort", "store", "include"}


# --------------------------------------------------------------------------
# The planner itself: pure, so the whole mapping table is cheap to pin down.
# --------------------------------------------------------------------------

def facts(**known):
    return lambda path: known.get(path.replace(".", "_"))


def plan(ingress, target, params, llama=False, **known):
    return plan_reasoning(ingress, target, llama, params, facts(**known))


@pytest.mark.parametrize("values", [None, ("low", "high"), ("none", "minimal", "low", "medium", "high", "xhigh", "max")])
@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "bogus"])
@pytest.mark.parametrize("target", ["openai-chat", "openai-responses", "anthropic", "gemini"])
def test_the_planner_never_changes_the_name_of_an_effort(target, effort, values):
    result = plan("openai-chat", target, {"reasoning_effort": effort}, reasoning_effort_values=values)
    sent = result.completion.get("reasoning_effort")
    assert sent in (None, effort)
    if sent is None and effort != "none" and target in ("openai-chat", "openai-responses") \
            and (values is None or effort in values):
        assert "reasoning_effort" not in result.strip  # left in place for the adapter, unchanged


def test_the_planner_does_not_mutate_its_input():
    params = {"reasoning": {"effort": "none", "summary": "auto"}}
    before = json.dumps(params)
    plan("openai-responses", "anthropic", params)
    assert json.dumps(params) == before


def test_toggle_native_value_is_used_when_the_route_publishes_one():
    result = plan("openai-chat", "anthropic", {"reasoning_effort": "none"},
                  reasoning_toggle_native_parameter={"protocol": "anthropic", "path": "thinking.type",
                                                     "values": {"enabled": "enabled", "disabled": "off"}})
    assert result.completion["thinking"] == {"type": "off"}


# --------------------------------------------------------------------------
# Gemini effort: exact only when the adapter sends it as a native thinking level
# and the route publishes the level.
# --------------------------------------------------------------------------

LEVEL_MODEL = "gemini-4-argon"  # newer than Gemini 2: the adapter emits thinkingLevel
LEVEL_FACTS = {"reasoning": {"supported": True,
                             "effort": {"supported": True, "values": ["low", "medium", "high"]}}}


def gemini_levels_config(upstream, *, model=LEVEL_MODEL, facts=LEVEL_FACTS, policy="strict"):
    raw = raw_config(upstream, policy=policy, overrides={"m-gem": facts} if facts else None)
    next(m for m in raw["models"] if m["name"] == "m-gem")["model"] = model
    return raw


def gemini_wire(upstream, model=LEVEL_MODEL):
    return upstream.last_request(f"/v1beta/models/{model}:generateContent")["body"]


@pytest.mark.parametrize("ingress", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_published_gemini_level_is_exact_under_strict_policy(upstream, ingress, effort):
    response = send(gemini_levels_config(upstream), ingress, "m-gem", effort_for(ingress, effort))
    assert response.status_code == 200, response.text
    assert LOSS not in response.headers
    thinking = gemini_wire(upstream)["generationConfig"]["thinkingConfig"]
    assert thinking["thinkingLevel"] == effort and "thinkingBudget" not in thinking


@pytest.mark.parametrize("raw_kwargs,effort,reason", [
    # Budget-based model: the same word is only an estimate of a token budget.
    ({"model": "gemini-2.5-flash"}, "high", "thinking budget or level"),
    # No published values: the route has not said it accepts this level.
    ({"facts": None}, "high", "thinking budget or level"),
    # The adapter has no native level of this name on every model.
    ({"facts": {"reasoning": {"supported": True, "effort": {
        "supported": True, "values": ["minimal", "low", "medium", "high"]}}}},
     "minimal", "thinking budget or level"),
])
def test_gemini_effort_stays_approximated_without_a_native_level(upstream, raw_kwargs, effort, reason):
    raw = gemini_levels_config(upstream, **raw_kwargs, policy="lossy")
    response = send(raw, "openai-chat", "m-gem", {"reasoning_effort": effort})
    assert response.status_code == 200, response.text
    (loss,) = losses(response)
    assert loss["action"] == "approximated" and reason in loss["reason"]
    assert send(gemini_levels_config(upstream, **raw_kwargs), "openai-chat", "m-gem",
                {"reasoning_effort": effort}).status_code == 400


def test_gemini_level_outside_the_published_values_is_still_refused(upstream):
    response = send(gemini_levels_config(upstream), "openai-chat", "m-gem", {"reasoning_effort": "xhigh"})
    assert response.status_code == 400 and "known values" in response.text
    assert upstream.requests == []
