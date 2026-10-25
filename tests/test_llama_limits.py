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


def limits(row: Any = None, props: Any = None, mode: str = "single",
           **kwargs: Any) -> dict[str, Any]:
    return mr._llama_limits(row or {}, props, mode, 12345, kwargs.pop("overrides", {}), **kwargs)


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


def test_legacy_row_wins_and_invalid_props_defaults_use_top_level() -> None:
    row = {"max_model_len": 2048, "meta": {"n_ctx": 3000}}
    assert limits(row, {"n_ctx": 4096}, "legacy")["context_window"]["value"] == 2048
    assert limits({}, {"n_ctx": 4096, "default_generation_settings": {"n_ctx": 0}})["context_window"] == {
        "value": 4096, "source": "/props.n_ctx", "kind": "reported",
    }


@pytest.mark.parametrize("bad", [None, 0, -1, True, False, 1.5, float("nan"),
                                 float("inf"), "8192", "auto", [], {}])
def test_invalid_or_auto_values_never_become_limits(bad: Any) -> None:
    found = limits({"id": "model-128k-32k", "meta": {
        "n_ctx": bad, "n_ctx_train": bad, "max_input_tokens": bad, "max_output_tokens": bad,
    }}, {"n_ctx": bad, "default_generation_settings": {"n_ctx": bad}})
    assert found["context_window"] == {
        "value": 12345, "source": "llama_cpp.context_fallback", "kind": "fallback",
    }
    for key in mr.LLAMA_LIMIT_FIELDS[1:]:
        assert found[key] == {"value": None, "source": None, "kind": "unknown"}
    # Listing metadata is JSON-safe even if an upstream supplied NaN/Infinity.
    json.dumps(found, allow_nan=False)


@pytest.mark.parametrize("props", [None, [], "bad", 12, {"default_generation_settings": []}])
def test_absent_or_malformed_props(props: Any) -> None:
    assert limits({"meta": []}, props)["context_window"]["kind"] == "fallback"


@pytest.mark.parametrize("args,expected", [
    (["--ctx-size", "8192"], 8192), (["--ctx-size=8192"], 8192),
    (["-c", "8192"], 8192), (["-c=8192"], 8192),
    (["--ctx-size=16384", "-c", "8192"], 8192),
    (["-c", "8192", "--ctx-size=0"], None),
    (["--ctx-size=8192", "-c", "auto"], None),
    (["--ctx-size=8192", "-c"], None),
    (["--ctx-size", "1.5"], None), (["--ctx-size=nan"], None),
    (["--ctx-size", "inf"], None), (["--ctx-size=-1"], None),
    (["--fit-ctx=8192"], None), (["--fit-ctx", "8192"], None),
    (["-fitc=8192"], None), (["-fitc", "8192"], None),
    ("--ctx-size 8192", None), ([], None),
])
def test_context_argument_forms(args: Any, expected: int | None) -> None:
    found = limits({"status": {"args": args}}, mode="router")
    assert mr._llama_context_from_args(args) == expected
    assert found["context_window"]["value"] == (expected or 12345)
    assert found["context_window"]["kind"] == ("inferred" if expected else "fallback")
    assert found["max_output_tokens"]["value"] is None


@pytest.mark.parametrize("args", [["--fit-ctx", "8192"], ["--fit-ctx=8192"],
                                   ["-fitc", "8192"], ["-fitc=8192"]])
def test_fit_is_only_a_constraint(args: list[str]) -> None:
    found = limits({"meta": {"n_ctx": 4096}, "status": {"args": args}})
    assert found["context_window"]["value"] == 4096
    assert found["constraints"]["fit_context_min"] == {
        "value": 8192, "source": "/v1/models.status.args.fit-ctx", "kind": "constraint",
    }


@pytest.mark.parametrize("raw,special", [(0, None), (256, None), (-1, "unlimited"), (-2, "until_context")])
def test_generation_default_is_never_a_maximum(raw: int, special: str | None) -> None:
    found = limits(props={"default_generation_settings": {"params": {"n_predict": raw}}})
    assert found["default_output_tokens"]["value"] == (raw if raw >= 0 else None)
    assert found["default_output_tokens"].get("special") == special
    assert found["max_output_tokens"]["value"] is None


@pytest.mark.parametrize("flag", ["-n", "--predict", "--n-predict"])
@pytest.mark.parametrize("equals", [False, True])
def test_generation_argument_default(flag: str, equals: bool) -> None:
    args = [flag + "=80"] if equals else [flag, "80"]
    found = limits({"status": {"args": args}})
    assert found["default_output_tokens"] == {
        "value": 80, "source": "/v1/models.status.args.n-predict", "kind": "inferred",
    }
    assert found["max_output_tokens"]["value"] is None


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


def test_explicit_unknown_override_suppresses_reported_value() -> None:
    found = limits({"meta": {"max_output_tokens": 512}}, overrides={"max_output_tokens": None})
    assert found["max_output_tokens"] == {
        "value": None, "source": "llama_cpp.limit_overrides", "kind": "override",
    }


@pytest.mark.parametrize("bad", [True, "256", float("nan"), float("inf"), 1.5, -3])
def test_invalid_defaults_allow_other_usable_sources(bad: Any) -> None:
    found = limits({"status": {"args": ["-n=32"]}}, {
        "default_generation_settings": {"params": {"n_predict": bad}, "n_predict": 64},
    })
    assert found["default_output_tokens"]["value"] == 64
    assert found["max_output_tokens"]["value"] is None


def test_zero_and_special_defaults_take_precedence_over_launch_estimates() -> None:
    for value in (0, -1, -2):
        found = limits({"status": {"args": ["-n=256"]}}, {
            "default_generation_settings": {"params": {"n_predict": value, "max_tokens": 512}},
        })
        assert found["default_output_tokens"]["kind"] == "reported"
        assert found["default_output_tokens"]["value"] == (0 if value == 0 else None)


@pytest.mark.parametrize("field", ["max_input_tokens", "max_output_tokens"])
def test_explicit_maximum_source_precedence(field: str) -> None:
    row = {field: 2048, "metadata": {field: 1024}, "meta": {field: 4096}}
    assert limits(row, {field: 512})[field]["value"] == 4096
    row["meta"][field] = 0
    assert limits(row, {field: 512})[field]["value"] == 2048
    row[field] = None
    assert limits(row, {field: 512})[field]["value"] == 1024
    row["metadata"][field] = -1
    assert limits(row, {field: 512})[field]["value"] == 512


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


@pytest.mark.parametrize("override", [[], {"model": []}, {"": {}}, {" model": {}},
    {"model": {"max_tokens": 32}}, {"model": {"context_window": 0}},
    {"model": {"max_output_tokens": -1}}, {"model": {"default_output_tokens": -1}},
    {"model": {"max_input_tokens": True}}, {"model": {"max_output_tokens": "512"}},
    {"model": {"context_window": 4096.5}}, {"model": {"context_window": float("nan")}},
])
def test_invalid_configured_limits_fail_early(override: Any) -> None:
    with pytest.raises(SystemExit, match="limit_overrides"):
        mr.build_config({"llama_cpp": {"limit_overrides": override}})


@pytest.mark.parametrize("value", [0, -1, True, "8192", 4096.5, float("inf")])
def test_invalid_context_fallback(value: Any) -> None:
    with pytest.raises(SystemExit, match="context_fallback"):
        mr.build_config({"llama_cpp": {"context_fallback": value}})
