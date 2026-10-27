"""Limit semantics: runtime reports, configured estimates and unknown maxima."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import parse_qs, urlsplit

import pytest

import model_router as mr
import model_router.discovery as router_discovery


def limits(row: Any = None, props: Any = None, mode: str = "single",
           **kwargs: Any) -> dict[str, Any]:
    return router_discovery._llama_limits(row or {}, props, mode, 12345, kwargs.pop("overrides", {}), **kwargs)


def test_conflicting_reports_keep_distinct_meanings() -> None:
    found = limits({
        "meta": {"n_ctx": 4096, "n_ctx_train": 131072,
                 "max_input_tokens": 4000, "max_output_tokens": 2048},
        "metadata": {"max_tokens": 99999, "max_output_tokens": 1000},
        "status": {"args": ["--ctx-size=32768", "--fit-ctx", "16384", "-n=64"]},
    }, {"n_ctx": 16000, "default_generation_settings": {
        "n_ctx": 8192, "n_predict": 256, "params": {"n_predict": 128, "max_tokens": 512},
    }})
    assert {key: found[key]["value"] for key in mr.LLAMA_LIMIT_FIELDS} == {
        "context_window": 4096, "trained_context_window": 131072,
        "max_input_tokens": 4000, "max_output_tokens": 2048, "default_output_tokens": 128,
    }
    assert found["context_window"]["source"] == "/v1/models.meta.n_ctx"
    assert found["default_output_tokens"]["source"].endswith("params.n_predict")
    assert found["constraints"]["fit_context_min"]["value"] == 16384


@pytest.mark.parametrize("mode", ["single", "legacy"])
def test_props_runtime_context_and_legacy_default_layout(mode: str) -> None:
    found = limits({"max_model_len": 0, "meta": {"n_ctx": 0}}, {
        "n_ctx": 8192, "default_generation_settings": {"n_ctx": 4096, "n_predict": 80},
    }, mode)
    assert found["context_window"] == {
        "value": 4096, "source": "/props.default_generation_settings.n_ctx", "kind": "reported",
    }
    assert found["default_output_tokens"]["value"] == 80
    assert found["max_input_tokens"]["value"] is None
    assert found["max_output_tokens"]["value"] is None






@pytest.mark.parametrize("props", [None, [], "bad", 12, {"default_generation_settings": []}])
def test_absent_or_malformed_props(props: Any) -> None:
    assert limits({"meta": []}, props)["context_window"]["kind"] == "fallback"










@pytest.mark.parametrize("mode", ["router", "llama-swap"])
def test_global_multi_model_props_are_never_applied_to_a_model(mode: str) -> None:
    props = {"n_ctx": 65536, "max_output_tokens": 8192,
             "default_generation_settings": {"n_ctx": 32768, "n_predict": 256}}
    found = limits({"meta": {"n_ctx": 4096, "n_ctx_train": 131072}}, props, mode)
    assert found["context_window"]["value"] == 4096
    assert found["trained_context_window"]["value"] == 131072
    assert found["default_output_tokens"]["value"] is None
    assert found["max_output_tokens"]["value"] is None
    assert limits({}, props, mode)["context_window"]["kind"] == "fallback"


@contextmanager
def local_discovery(rows: list[dict[str, Any]], props: Any,
                    by_model: dict[str, Any] | None = None) -> Iterator[tuple[str, list[Any]]]:
    """Faithful router: generic /props is global; autoload must be disabled."""
    requests: list[Any] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            requests.append((parsed.path, query))
            payload: Any = {}
            status = 200
            if parsed.path == "/v1/models":
                payload = {"data": rows}
            elif parsed.path == "/props":
                assert query.get("autoload") == ["false"]
                model = query.get("model", [None])[0]
                if model is None:
                    payload = props
                elif model in (by_model or {}):
                    payload = by_model[model]
                else:
                    status = 404
            elif parsed.path != "/health":
                status = 404
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def test_router_per_model_props_without_loading_and_no_cross_model_leakage() -> None:
    rows = [
        {"id": "one/a b", "status": {"value": "loaded", "args": ["-c=16000"]}},
        {"id": "two", "meta": {"n_ctx": 2048}, "status": {"value": "sleeping"}},
        {"id": "three", "context_length": 131072,
         "status": {"value": "unloaded", "args": ["--fit-ctx=4096", "-c=0"]}},
        {"id": "old-build", "status": {"value": "loaded", "args": ["-c=8192"]}},
        {"id": "ignored-query", "status": {"value": "loaded"}},
    ]
    with local_discovery(rows, {"role": "router", "n_ctx": 99999}, {
        "one/a b": {"default_generation_settings": {"n_ctx": 4096, "params": {"n_predict": 100}}},
        "two": {"default_generation_settings": {"n_ctx": 8192, "n_predict": 200}},
        "ignored-query": {"role": "router", "n_ctx": 99999, "max_output_tokens": 32768},
    }) as (url, requests):
        app = mr.App(mr.build_config({"llama_cpp": {"servers": [url], "refresh_seconds": 0}}))
        try:
            entries = {item["id"]: item for item in app.models_payload()["data"]}
            assert entries["one/a b"]["context_window"] == 4096
            assert entries["one/a b"]["metadata"]["default_output_tokens"] == 100
            assert entries["two"]["context_window"] == 2048
            assert entries["two"]["metadata"]["default_output_tokens"] == 200
            assert entries["three"]["context_window"] == mr.DEFAULT_LLAMA_CONTEXT
            assert entries["three"]["metadata"]["trained_context_window"] == 131072
            assert entries["three"]["metadata"]["default_output_tokens"] is None
            assert entries["old-build"]["context_window"] == 8192
            assert entries["ignored-query"]["context_window"] == mr.DEFAULT_LLAMA_CONTEXT
            for entry in entries.values():
                assert entry["max_tokens"] is None
                assert entry["metadata"]["max_input_tokens"] is None
            queries = [query for path, query in requests if path == "/props"]
            assert queries == [
                {"autoload": ["false"]},
                {"autoload": ["false"], "model": ["one/a b"]},
                {"autoload": ["false"], "model": ["two"]},
                {"autoload": ["false"], "model": ["old-build"]},
                {"autoload": ["false"], "model": ["ignored-query"]},
            ]
            assert entries["one/a b"]["metadata"]["limits"]["context_window"]["source"].startswith("/props?model=")
        finally:
            app.close()


def test_overrides_win_without_replacing_discovered_route_or_alias() -> None:
    row = {"id": "model", "aliases": ["alias"], "meta": {"n_ctx": 4096, "n_ctx_train": 32768},
           "metadata": {"max_tokens": 99999, "other": "preserved"}}
    override = {"context_window": 2048, "trained_context_window": 65536,
                "max_input_tokens": 2000, "max_output_tokens": 512, "default_output_tokens": 64}
    with local_discovery([row, {"id": "other"}], {"default_generation_settings": {"n_ctx": 8192}}) as (url, _):
        raw = {"llama_cpp": {"servers": [url], "refresh_seconds": 0, "prefix": "local/",
                             "context_fallback": 16384, "limit_overrides": {"model": override}}}
        app = mr.App(mr.build_config(raw))
        try:
            entry, other = app.models_payload()["data"]
            assert entry["id"] == "local/model"
            assert entry["context_window"] == entry["metadata"]["llama_cpp"]["n_ctx"] == 2048
            assert entry["max_tokens"] == entry["metadata"]["max_tokens"] == 512
            for key, value in override.items():
                assert entry["metadata"]["limits"][key] == {
                    "value": value, "source": "llama_cpp.limit_overrides", "kind": "override",
                }
            assert app.resolve("local/alias") is app.resolve("local/model")
            assert entry["metadata"]["other"] == "preserved"
            assert other["context_window"] == 8192
            assert other["max_tokens"] is None
        finally:
            app.close()






def test_zero_and_special_defaults_take_precedence_over_launch_estimates() -> None:
    for value in (0, -1, -2):
        found = limits({"status": {"args": ["-n=256"]}}, {
            "default_generation_settings": {"params": {"n_predict": value, "max_tokens": 512}},
        })
        assert found["default_output_tokens"]["kind"] == "reported"
        assert found["default_output_tokens"]["value"] == (0 if value == 0 else None)




def test_single_server_props_and_swap_compatibility_through_public_listing() -> None:
    scenarios = [
        ({"id": "single"}, {"default_generation_settings": {"n_ctx": 4096, "params": {"n_predict": 64}}},
         "single", 4096, 64),
        ({"id": "swap", "meta": {"n_ctx": 2048, "llamaswap": {"aliases": ["friendly"]}}},
         {"src": "llama-swap", "n_ctx": 99999, "default_generation_settings": {"n_predict": 64}},
         "llama-swap", 2048, None),
    ]
    for row, props, mode, context, default in scenarios:
        with local_discovery([row], props) as (url, requests):
            app = mr.App(mr.build_config({"llama_cpp": {"servers": [url], "refresh_seconds": 0}}))
            try:
                entry, = app.models_payload()["data"]
                assert entry["context_window"] == context
                assert entry["max_tokens"] is None
                assert entry["metadata"]["default_output_tokens"] == default
                assert entry["metadata"]["llama_cpp"]["mode"] == mode
                assert entry["metadata"]["limits"]["context_window"]["kind"] == "reported"
                assert len(requests) == 3
                if mode == "llama-swap":
                    assert app.resolve("friendly") is app.resolve("swap")
            finally:
                app.close()




