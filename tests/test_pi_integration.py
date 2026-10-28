"""The extension and the installed pi client against the router, with local fixtures only.

pi's own request builders run (tests/fixtures/pi_harness.mjs) and send what they
build over HTTP to an in-process router backed by fake upstreams. This proves which
extension workarounds are still needed: nothing here reimplements pi.

Skipped when node or the installed pi client is not available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from test_model_router import (
    FakeLlamaServer, _config, _with_gemini, llama_server, router_config, upstream,  # noqa: F401  (fixtures)
)

ROOT = Path(__file__).resolve().parent.parent
EXTENSION = ROOT / "integrations" / "pi" / "model-router.ts"
HARNESS = Path(__file__).parent / "fixtures" / "pi_harness.mjs"
LOSS = "x-model-router-loss"


def _pi_ai() -> Path | None:
    explicit = os.environ.get("PI_AI_DIR")
    candidates = [Path(explicit)] if explicit else []
    install = Path.home() / ".pi" / "agent" / "install"
    try:
        version = (install / "current-version").read_text().strip()
        candidates.append(install / "releases" / version / "node_modules"
                          / "@earendil-works" / "pi-ai")
    except OSError:
        pass
    return next((path for path in candidates if (path / "dist" / "models.js").exists()), None)


PI_AI = _pi_ai()
pytestmark = pytest.mark.skipif(
    PI_AI is None or shutil.which("node") is None,
    reason="needs node and an installed pi client (or PI_AI_DIR)")

LEVEL_FACTS = {"reasoning": {"supported": True, "effort": {
    "supported": True, "values": ["low", "medium", "high"]}}}
CHAT_FACTS = {"reasoning": {"supported": True, "effort": {
    "supported": True, "values": ["none", "low", "high", "max"]}}}
# A route whose facts say it thinks but publish no effort levels.
UNPUBLISHED_FACTS = {"reasoning": {"supported": True}}


def _run(upstream, *, policy="strict", llama=None, strata=None):
    raw = _with_gemini(_config(upstream), upstream)
    # gemini-4-argon is newer than Gemini 2: the adapter sends a native thinking level.
    raw["models"].append({"name": "m-gem3", "provider": "gem", "model": "gemini-4-argon"})
    for provider in raw["providers"].values():
        provider["compatibility"] = policy
        provider["session_header"] = "x-opencode-session"
    raw["capabilities"] = {"overrides": {
        "m-gem3": LEVEL_FACTS, "m-gem": UNPUBLISHED_FACTS, "m-chat": CHAT_FACTS,
        "m-anth": {"reasoning": {"supported": True, "toggle": {"supported": True}}},
    }}
    if llama is not None or strata is not None:
        raw["llama_cpp"] = {"servers": [s.url for s in (llama, strata) if s is not None],
                            "timeout": 1.0, "refresh_seconds": 0,
                            "compatibility": policy}
    with router_config(raw) as base:
        env = {**os.environ, "MODEL_ROUTER_URL": base, "MODEL_ROUTER_API_KEY": "unused"}
        done = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", str(HARNESS),
             str(PI_AI), str(EXTENSION)],
            capture_output=True, text=True, timeout=120, env=env, check=False)
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    return {m["id"]: m for m in output["models"]}, output["records"]


def test_pi_requests_to_gemini_chat_are_accepted_under_strict_policy(upstream):
    """Pi's own Chat request needs no supportsStore / supportsUsageInStreaming override."""
    models, records = _run(upstream)
    gem = models["m-gem3"]
    assert gem["reasoning"] is True and "compat" not in gem
    sent = [r for r in records if r["model"] == "m-gem3"]
    assert {r["level"] for r in sent} == {"off", "low", "medium", "high"}
    for record in sent:
        assert record["request"]["store"] is False
        assert record["request"]["stream_options"] == {"include_usage": True}
        assert record["status"] == 200 and record["loss"] is None, record
    assert {r["level"]: r["request"].get("reasoning_effort") for r in sent} == {
        "off": None, "low": "low", "medium": "medium", "high": "high"}
    # Pi's efforts arrive at Gemini as native thinking levels, not budgets.
    wire = [req["body"]["generationConfig"]["thinkingConfig"] for req in upstream.requests
            if req["path"].startswith("/v1beta/models/gemini-4-argon:")
            and "thinkingConfig" in req["body"].get("generationConfig", {})]
    assert [w["thinkingLevel"] for w in wire] == ["low", "medium", "high"]


def test_gemini_without_published_effort_levels_keeps_thinking_off_in_pi(upstream):
    """Unpublished levels mean a token-budget estimate, which strict policy refuses."""
    models, records = _run(upstream)
    assert models["m-gem"]["reasoning"] is False
    sent = [r for r in records if r["model"] == "m-gem"]
    assert [r["level"] for r in sent] == ["off"]
    assert sent[0]["status"] == 200 and "reasoning_effort" not in sent[0]["request"]


def test_pi_offers_only_the_levels_a_route_publishes(upstream):
    models, records = _run(upstream)
    assert models["m-gem3"]["thinkingLevelMap"] == {"minimal": None}
    assert models["m-chat"]["thinkingLevelMap"] == {
        "minimal": None, "medium": None, "max": "max", "off": "none"}
    chat = {r["level"]: r for r in records if r["model"] == "m-chat"}
    assert set(chat) == {"off", "low", "high", "max"}
    assert chat["off"]["request"]["reasoning_effort"] == "none"
    assert all(r["status"] == 200 and r["loss"] is None for r in chat.values()), chat


def test_the_router_cannot_hide_a_level_from_pi(upstream):
    """Level availability is client metadata; the router can only refuse what is sent."""
    raw = _with_gemini(_config(upstream), upstream)
    raw["models"].append({"name": "m-gem3", "provider": "gem", "model": "gemini-4-argon"})
    raw["capabilities"] = {"overrides": {"m-gem3": LEVEL_FACTS}}
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/chat/completions", json={
            "model": "m-gem3", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16, "reasoning_effort": "minimal"}, timeout=30)
    assert response.status_code == 400 and "known values" in response.text


def test_llama_levels_pi_offers_are_all_ones_the_router_implements(llama_server, upstream):
    models, records = _run(upstream, llama=llama_server)
    # Derived from the published none..xhigh values; no llama-specific compat or map.
    assert models["llama-test"]["thinkingLevelMap"] == {"off": "none", "xhigh": "xhigh"}
    assert "compat" not in models["llama-test"]
    sent = [r for r in records if r["model"] == "llama-test"]
    assert {r["level"] for r in sent} == {"off", "minimal", "low", "medium", "high", "xhigh"}
    assert all(r["status"] == 200 and r["loss"] is None for r in sent), sent
    # pi sends the effort itself (off as "none"); the router turns it into llama.cpp fields.
    assert {r["level"]: r["request"].get("reasoning_effort") for r in sent} == {
        "off": "none", "minimal": "minimal", "low": "low", "medium": "medium",
        "high": "high", "xhigh": "xhigh"}
    wire = [req["body"] for req in llama_server.requests if req["path"] == "/v1/chat/completions"]
    assert {"enable_thinking": False} in [b.get("chat_template_kwargs") for b in wire]
    assert 32768 in [b.get("thinking_budget_tokens") for b in wire]


def test_pi_discovers_strata_alongside_llama_and_sends_native_efforts(llama_server, upstream):
    strata = FakeLlamaServer(models=[{
        "id": "strata-test", "meta": {"n_ctx": 131072},
        "architecture": {"input_modalities": ["text"]}, "status": {"value": "loaded"},
    }], props={"build_info": "PolyStrata 0.1.45", "modalities": {"vision": False}}).start()
    try:
        models, records = _run(upstream, llama=llama_server, strata=strata)
        model = models["strata-test"]
        assert model["name"] == "strata-test (Strata)"
        assert model["contextWindow"] == 131072
        assert model["maxTokens"] == 16384
        assert model["thinkingLevelMap"] == {"minimal": None, "off": "none"}
        sent = [r for r in records if r["model"] == "strata-test"]
        assert {r["level"] for r in sent} == {"off", "low", "medium", "high"}
        assert all(r["status"] == 200 and r["loss"] is None for r in sent), sent
        wire = [req["body"] for req in strata.requests if req["path"] == "/v1/chat/completions"]
        assert {b["reasoning_effort"] for b in wire} == {"none", "low", "medium", "high"}
        assert all("thinking_budget_tokens" not in b for b in wire)
        assert "llama-test" in models
        assert "xhigh" in models["llama-test"]["thinkingLevelMap"]
        llama_wire = [req["body"] for req in llama_server.requests
                      if req["path"] == "/v1/chat/completions"]
        assert 32768 in [b.get("thinking_budget_tokens") for b in llama_wire]
    finally:
        strata.stop()


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem3"])
def test_conversation_identity_is_the_pi_session_on_every_protocol(upstream, model):
    """Only pi knows its durable session; the router never derives one."""
    _, records = _run(upstream)
    sent = [r for r in records if r["model"] == model]
    assert sent and all(r["status"] == 200 for r in sent), sent
    assert all(r["headers"]["x-session-id"] == "pi-session-0001" for r in sent)
    assert {req["headers"].get("x-opencode-session") for req in upstream.requests} == {
        "pi-session-0001"}


def test_the_router_derives_no_identity_when_the_client_sends_none(upstream):
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["session_header"] = "x-opencode-session"
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/chat/completions", json={
            "model": "m-chat", "messages": [{"role": "user", "content": "same prompt"}],
            "max_tokens": 16}, timeout=30)
    assert response.status_code == 200
    assert all("x-opencode-session" not in req["headers"] for req in upstream.requests)
