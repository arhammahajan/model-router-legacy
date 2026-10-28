"""Strata uses compatible discovery endpoints and native OpenAI effort controls."""
from __future__ import annotations

import json
from contextlib import closing

import httpx
import pytest

import model_router as mr
from test_capabilities import Discovery, entries, local_raw
from test_model_router import llama_server, router_config  # noqa: F401


@pytest.mark.parametrize("build", ["Strata 0.1.45", "PolyStrata 0.1.45"])
def test_strata_inventory_and_offline_restart(monkeypatch, tmp_path, build):
    discovery = Discovery(monkeypatch, rows=[{
        "id": "qwen-test", "aliases": ["qwen"], "meta": {"n_ctx": 131072},
        "architecture": {"input_modalities": ["text"]},
    }], props={"build_info": build, "modalities": {"vision": False},
              "chat_template_caps": {"supports_tools": True}})
    raw = local_raw()
    path = tmp_path / "discovery.json"
    raw["discovery_cache"] = {"path": str(path)}
    with closing(mr.App(mr.build_config(raw))) as app:
        entry = entries(app)["qwen-test"]
        assert entry["metadata"]["engine"] == "strata"
        assert entry["context_window"] == 131072
        facts = entry["metadata"]["router"]
        assert facts["tools"]["supported"]["value"] is True
        assert facts["reasoning"]["effort"]["values"]["value"] == ["none", "low", "medium", "high"]
        assert "router.llama_thinking_map" not in facts["sources"]
        assert facts["sources"]["llama.props"]["server_build"] == build
        assert app.resolve("qwen")._provider.llama_cpp is False
    discovery.status = 503
    with closing(mr.App(mr.build_config(raw))) as app:
        entry = entries(app)["qwen-test"]
        assert entry["metadata"]["engine"] == "strata"
        assert entry["metadata"]["router"]["freshness"]["discovery"]["cached"] is True
        assert "strata.compat" in entry["metadata"]["router"]["sources"]
        assert app.resolve("qwen")._provider.llama_cpp is False


def test_engine_follows_server_report_on_refresh(monkeypatch):
    discovery = Discovery(monkeypatch, rows=[{"id": "strata-named-model"}],
                          props={"build_info": "b9000"})
    with closing(mr.App(mr.build_config(local_raw()))) as app:
        assert app.resolve("strata-named-model")._provider.llama_cpp is True
        discovery.props = {"build_info": "PolyStrata 0.1.45"}
        app.refresh_llama_cpp(force=True)
        assert app.resolve("strata-named-model")._provider.llama_cpp is False
        discovery.status = 503
        app.refresh_llama_cpp(force=True)
        assert app.resolve("strata-named-model")._provider.llama_cpp is False
        discovery.status = 200
        discovery.props = {"build_info": "b9000"}
        app.refresh_llama_cpp(force=True)
        assert app.resolve("strata-named-model")._provider.llama_cpp is True


@pytest.mark.parametrize("ingress", ["chat/completions", "responses", "messages"])
def test_strata_keeps_native_effort_on_every_ingress(llama_server, ingress):
    llama_server.props = {"build_info": "PolyStrata 0.1.45"}
    raw = {"server": {"auth_mode": "none"},
           "llama_cpp": {"servers": [llama_server.url], "refresh_seconds": 0,
                         "compatibility": "lossy" if ingress == "messages" else "strict"}}
    body = {"model": "llama-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]}
    if ingress == "responses":
        body = {"model": "llama-test", "max_output_tokens": 32,
                "input": "hello", "reasoning": {"effort": "low"}}
    elif ingress == "messages":
        body["output_config"] = {"effort": "low"}
    else:
        body["reasoning_effort"] = "low"
    with router_config(raw) as base:
        response = httpx.post(f"{base}/v1/{ingress}", json=body, timeout=30)
    assert response.status_code == 200, response.text
    assert (response.headers.get("x-model-router-loss") is not None) == (ingress == "messages")
    sent = llama_server.last_request("/v1/chat/completions")["body"]
    assert sent["reasoning_effort"] == "low", json.dumps(sent)
    assert "thinking_budget_tokens" not in sent
