"""Versioned, advisory capabilities from offline evidence and local discovery."""
from __future__ import annotations

import time

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest


import model_router as mr
import router_metadata as caps


ROOT = Path(__file__).resolve().parents[1]


class Discovery:
    """Block all external traffic and assert metadata probes cannot load models."""
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, rows: list[Any] | None = None,
                 props: Any = None, by_model: dict[str, Any] | None = None) -> None:
        self.rows = rows if rows is not None else [{"id": "model"}]
        self.props = props if props is not None else {}
        self.by_model = by_model or {}
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.props_status = 200

        def send(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            self.requests.append(request)
            assert request.method == "GET", "metadata discovery must never run inference"
            assert request.url.host == "127.0.0.1", "ordinary tests must stay offline"
            path = request.url.path
            status = self.status
            payload: Any = {}
            if path in ("/catalog", "/v1/models"):
                payload = {"data": self.rows}
            elif path == "/props":
                query = parse_qs(request.url.query.decode())
                assert query.get("autoload") == ["false"]
                model = query.get("model", [None])[0]
                if model is None:
                    payload = self.props
                else:
                    # An unloaded, loading, failed or unknown model must not be
                    # queried even if an older server ignores autoload=false.
                    assert model in self.by_model
                    payload = self.by_model[model]
                status = self.props_status
            else:
                assert path == "/health"
            return httpx.Response(status, json=payload, request=request)

        monkeypatch.setattr(httpx.Client, "send", send)


def local_raw(**kwargs: Any) -> dict[str, Any]:
    return {"llama_cpp": {"servers": ["http://127.0.0.1:9931"],
                          "refresh_seconds": 10, **kwargs}}


def catalog_raw(**kwargs: Any) -> dict[str, Any]:
    return {"providers": {"go": {"dialect": "openai-chat", "base_url": "http://127.0.0.1:9000/v1"}},
            "catalog": {"url": "http://127.0.0.1/catalog", "refresh_seconds": 10,
                        "metadata_source": "opencode-go-v1",
                        "rules": [{"match": ".", "provider": "go"}], **kwargs}}


def entries(app: mr.App) -> dict[str, Any]:
    return {row["id"]: row for row in app.models_payload()["data"]}


def contract(app: mr.App, name: str = "model") -> dict[str, Any]:
    return entries(app)[name]["metadata"]["router"]


def test_props_capabilities_and_corrected_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    discovery = Discovery(monkeypatch, rows=[{
        "id": "model", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "meta": {"n_ctx_train": 131072},
        "metadata": {"other": "preserved", "router": {"schema_version": 99}},
    }], props={
        "build_info": "b9000-abc123", "modalities": {"vision": True, "audio": False},
        "chat_template_caps": {"supports_tools": True, "supports_tool_calls": True,
                               "supports_parallel_tool_calls": False, "supports_reasoning_effort": True},
        "default_generation_settings": {"n_ctx": 8192, "params": {
            "n_predict": 256, "temperature": 0.6, "top_k": 40, "stream": False,
            "samplers": ["top_k", "temperature"], "api_key": "dummy-never-advertise",
        }},
    })
    app = mr.App(mr.build_config(local_raw()))
    try:
        payload = app.models_payload()
        assert set(payload) == {"object", "data", "has_more", "first_id", "last_id"}
        row = payload["data"][0]
        found = row["metadata"]["router"]
        assert found["schema_version"] == 1
        assert row["metadata"]["other"] == "preserved"
        assert found["limits"] == row["metadata"]["limits"]
        assert found["limits"]["context_window"]["value"] == row["context_window"] == 8192
        assert found["limits"]["max_output_tokens"]["value"] is row["max_tokens"] is None
        assert found["limits"]["default_output_tokens"]["value"] == 256
        assert found["tools"]["supported"]["value"] is True
        assert found["tools"]["parallel"]["value"] is False
        assert found["reasoning"]["supported"]["value"] is True
        assert found["reasoning"]["effort"]["supported"]["value"] is True
        assert found["reasoning"]["effort"]["values"]["value"] == list(caps.LLAMA_EFFORT_VALUES)
        assert found["reasoning"]["toggle"]["supported"]["value"] is None
        assert found["reasoning"]["budget"]["supported"]["value"] is None
        assert found["reasoning"]["effort"]["native_parameter"]["value"] == {
            "protocol": "openai-chat", "path": "reasoning_effort"}
        assert found["modalities"]["input"]["image"]["value"] is True
        assert found["modalities"]["input"]["audio"]["value"] is False
        assert found["modalities"]["output"]["image"]["value"] is False
        assert found["sampling"]["temperature"]["default"]["value"] == 0.6
        assert found["sampling"]["top_p"]["supported"]["value"] is None
        assert found["streaming"]["value"] is True
        assert found["structured_output"]["json_schema"]["value"] is None
        assert found["sources"]["llama.props"]["server_build"] == "b9000-abc123"
        assert found["freshness"]["discovery"]["stale"] is False
        assert found["protocols"]["native"]["openai-chat"]["value"] is True
        assert found["protocols"]["native"]["anthropic"]["value"] is None
        assert found["protocols"]["translated"]["anthropic"]["value"] is True
        assert found["protocols"]["translated"]["gemini"]["value"] is False
        assert found["protocols"]["known_lossy_features"]["value"] is None
        assert "dummy-never-advertise" not in json.dumps(payload)
        assert len(discovery.requests) == 3
        # The result is independent from the immutable registry and next read.
        found["sampling"]["temperature"]["default"]["value"] = 99
        assert contract(app)["sampling"]["temperature"]["default"]["value"] == 0.6
    finally:
        app.close()


def test_router_queries_only_loaded_or_sleeping_models(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"id": name, "status": {"value": status}} for name, status in (
        ("loaded", "loaded"), ("sleeping", "sleeping"), ("unloaded", "unloaded"),
        ("loading", "loading"), ("failed", "failed"), ("unknown", "unknown"))]
    discovery = Discovery(monkeypatch, rows=rows,
        props={"role": "router", "modalities": {"vision": True},
               "chat_template_caps": {"supports_reasoning_effort": True}},
        by_model={"loaded": {"modalities": {"vision": False},
                             "chat_template_caps": {"supports_reasoning_effort": False}},
                  "sleeping": {"modalities": {"audio": True},
                               "chat_template_caps": {"supports_reasoning_effort": True}}})
    app = mr.App(mr.build_config(local_raw()))
    try:
        assert contract(app, "loaded")["modalities"]["input"]["image"]["value"] is False
        # The upstream's chat-template report is kept as conflicting evidence;
        # the router's own thinking mapping is what makes effort usable.
        loaded_effort = contract(app, "loaded")["reasoning"]["effort"]["supported"]
        assert loaded_effort["value"] is True and loaded_effort["kind"] == "configured"
        assert loaded_effort["source"] == "router.llama_thinking_map.supported"
        assert [c["value"] for c in loaded_effort["conflicts"]] == [False]
        assert contract(app, "loaded")["reasoning"]["supported"]["value"] is None
        assert contract(app, "sleeping")["modalities"]["input"]["audio"]["value"] is True
        assert contract(app, "sleeping")["sources"]["llama.props"]["endpoint"].startswith("/props?model=")
        for name in ("unloaded", "loading", "failed", "unknown"):
            found = contract(app, name)
            assert found["modalities"]["input"]["image"]["value"] is None
            assert found["reasoning"]["supported"]["value"] is None
            assert found["sources"]["llama.props"]["retrieved_at"] is None
        assert len(discovery.requests) == 5
    finally:
        app.close()




def test_unknown_and_legacy_metadata_preserve_unknowns(monkeypatch: pytest.MonkeyPatch) -> None:
    Discovery(monkeypatch, rows=[{"id": "thinking-128k-high-max", "max_model_len": 4096,
                                 "capabilities": ["multimodal"]}])
    app = mr.App(mr.build_config(local_raw()))
    try:
        found = contract(app, "thinking-128k-high-max")
        assert found["reasoning"]["supported"]["value"] is None
        # Effort is the router's own mapping; whether the model thinks stays unknown.
        assert found["reasoning"]["effort"]["supported"]["value"] is True
        assert found["reasoning"]["effort"]["values"]["value"] == list(caps.LLAMA_EFFORT_VALUES)
        assert found["modalities"]["input"]["image"]["value"] is True
        assert found["modalities"]["output"]["image"]["value"] is None
        assert found["limits"]["context_window"]["value"] == 4096
        assert found["limits"]["max_output_tokens"]["value"] is None
    finally:
        app.close()






def test_go_review_is_opt_in_and_cannot_spread_to_another_native_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Discovery(monkeypatch, rows=[{"id": "qwen3.8-max", "metadata": {"router": {"reasoning": True}}}])
    for source in (None, "opencode-go-v1"):
        app = mr.App(mr.build_config(catalog_raw(metadata_source=source)))
        try:
            found = contract(app, "qwen3.8-max")
            assert found["reasoning"]["supported"]["value"] is None
            assert found["protocols"]["native"]["openai-chat"]["value"] is True
            if source:
                assert found["sources"]["go.review"]["applicable"] is False
        finally:
            app.close()


def test_override_enriches_future_catalog_entry_without_creating_a_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = Discovery(monkeypatch, rows=[])
    raw = catalog_raw()
    raw["capabilities"] = {"overrides": {"future": {"reasoning": {"supported": False}}}}
    app = mr.App(mr.build_config(raw))
    try:
        assert app.resolve("future") is None
        assert app.models_payload()["data"] == []
        discovery.rows = [{"id": "future"}]
        app.refresh_catalog(force=True)
        found = contract(app, "future")
        assert app.resolve("future").provider_name == "go"
        assert found["reasoning"]["supported"]["value"] is False
        assert found["reasoning"]["effort"]["supported"]["value"] is False
    finally:
        app.close()




def test_local_stale_contract_retains_evidence_and_ages_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    discovery = Discovery(monkeypatch, props={"chat_template_caps": {"supports_tools": True}})
    app = mr.App(mr.build_config(local_raw()), clock=lambda: clock[0])
    try:
        initial = contract(app)
        discovery.status = 503
        clock[0] = 11
        app.refresh_llama_cpp(force=True)
        count = len(discovery.requests)
        for current in (11, 18, 25):
            clock[0] = current
            found = contract(app)
            assert found["tools"]["supported"] == initial["tools"]["supported"]
            assert found["sources"]["llama.props"]["retrieved_at"] == initial["sources"]["llama.props"]["retrieved_at"]
            assert found["sources"]["llama.props"]["stale"] is True
            assert found["freshness"]["discovery"]["age_seconds"] == current
        assert len(discovery.requests) == count
    finally:
        app.close()


def test_negative_parent_suppresses_inherited_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    Discovery(monkeypatch, rows=[{"id": "glm-5.3-flash"}])
    raw = catalog_raw()
    raw["capabilities"] = {"overrides": {"glm-5.3-flash": {"reasoning": {"supported": False}}}}
    app = mr.App(mr.build_config(raw))
    try:
        found = contract(app, "glm-5.3-flash")
        assert found["reasoning"]["supported"]["value"] is False
        effort = found["reasoning"]["effort"]
        assert effort["supported"]["value"] is False
        assert effort["values"]["value"] is None
        assert effort["values"]["conflicts"][0]["value"] == ["low", "high", "max"]
        assert effort["native_parameter"]["value"] is None
    finally:
        app.close()






def test_review_data_is_available_for_all_bundled_routes_and_valid() -> None:
    review = caps._go_review()
    endpoints = json.loads((ROOT / "tests/fixtures/opencode_go_endpoints.json").read_text())["models"]
    assert set(review["models"]) == set(endpoints)
    for model, endpoint in endpoints.items():
        found = caps.render(caps.go(model, endpoint["dialect"]), {}, {"stale": False}, 0)
        assert found["sources"]["go.review"]["applicable"] is True
        json.dumps(found, allow_nan=False)


