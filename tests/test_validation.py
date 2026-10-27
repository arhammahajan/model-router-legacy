"""Prompt 13: configuration boundaries and ingress errors before dispatch."""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import httpx
import pytest
from packaging.requirements import Requirement

import model_router as mr
import model_router.config as router_configuration
from test_model_router import PATHS, _body, _config, router_config, upstream


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_auth_env(monkeypatch):
    monkeypatch.delenv("MODEL_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_ROUTER_API_KEYS", raising=False)


def config():
    return {
        "server": {"auth_mode": "none"},
        "providers": {"local": {"dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1"}},
        "models": [{"name": "public", "provider": "local", "model": "upstream", "aliases": ["alias"]}],
    }


@pytest.mark.parametrize("section,field,value", [
    ("server", "host", False), ("server", "port", True), ("server", "port", "4000"),
    ("server", "port", 0), ("server", "port", 65536),
    ("server", "max_body_bytes", "100"), ("server", "max_body_bytes", False),
    ("server", "max_body_bytes", -1), ("server", "allow_dynamic_routes", "false"),
    ("server", "api_keys", "dummy"), ("server", "api_keys", [123]),
    ("server", "api_keys", [""]), ("server", "api_keys", ["  "]),
    ("server", "api_keys", None), ("server", "header_read_timeout", True),
    ("provider", "timeout", "1"), ("provider", "timeout", True),
    ("provider", "timeout", float("nan")), ("provider", "timeout", float("inf")),
    ("provider", "timeout", 0), ("provider", "timeout", -1),
    ("provider", "timeout", 10**400), ("provider", "headers", []),
    ("provider", "headers", {"x-test": 42}), ("provider", "headers", {"x-test": "a\nb"}),
    ("provider", "api_key_env", False), ("provider", "api_key_env", ""),
    ("provider", "api_key", None),
    ("provider", "api_key_file", []), ("provider", "drop_params", "false"),
    ("model", "enabled", "false"), ("model", "enabled", 0),
    ("model", "aliases", [1]), ("model", "aliases", "alias"),
    ("model", "metadata", []), ("model", "dialect", []),
    ("llama_cpp", "aliases", "false"), ("llama_cpp", "timeout", "2"),
    ("llama_cpp", "refresh_seconds", True), ("llama_cpp", "ports", [True]),
    ("llama_cpp", "ports", ["8080"]), ("llama_cpp", "servers", False),
    ("llama_cpp", "prefix", False), ("llama_cpp", "headers", {"x-test": False}),
    ("catalog", "timeout", False), ("catalog", "refresh_seconds", "30"),
    ("catalog", "refresh_seconds", -1), ("catalog", "refresh_seconds", float("inf")),
])
def test_config_types_and_ranges(section, field, value):
    raw = config()
    target = {"provider": raw["providers"]["local"], "model": raw["models"][0]}.get(section)
    if target is None:
        target = raw.setdefault(section, {})
    target[field] = value
    with pytest.raises(SystemExit):
        mr.build_config(raw)


@pytest.mark.parametrize("section", ["config", "server", "provider", "model", "llama_cpp", "llama_server", "catalog", "rule"])
def test_config_typos_are_rejected(section):
    raw = config()
    targets = {"config": raw, "server": raw["server"],
               "provider": raw["providers"]["local"], "model": raw["models"][0]}
    if section in ("catalog", "rule"):
        raw["catalog"] = {"url": "http://127.0.0.1:1/models", "rules": [{"match": ".*", "provider": "local"}]}
        target = raw["catalog"] if section == "catalog" else raw["catalog"]["rules"][0]
    elif section in ("llama_cpp", "llama_server"):
        raw["llama_cpp"] = {"servers": [{"url": "http://127.0.0.1:1"}]}
        target = raw["llama_cpp"] if section == "llama_cpp" else raw["llama_cpp"]["servers"][0]
    else:
        target = targets[section]
    target["typo_setting"] = True
    with pytest.raises(SystemExit, match="unknown field 'typo_setting'"):
        mr.build_config(raw)


@pytest.mark.parametrize("section,value", [("server", []), ("providers", False), ("models", {}), ("llama_cpp", None)])
def test_empty_invalid_sections_do_not_become_defaults(section, value):
    raw = config()
    raw[section] = value
    with pytest.raises(SystemExit):
        mr.build_config(raw)


@pytest.mark.parametrize("suffix", [".toml", ".json"])
@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_missing_explicit_auth_substitution_never_disables_auth(tmp_path, monkeypatch, suffix, env_value):
    name = "ROUTER_TEST_ONLY_SECRET"
    if env_value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, env_value)
    path = tmp_path / ("router" + suffix)
    if suffix == ".toml":
        path.write_text('[server]\nauth_mode="required"\napi_keys=["${ROUTER_TEST_ONLY_SECRET}"]\n')
    else:
        path.write_text(json.dumps({"server": {"auth_mode": "required", "api_keys": ["${ROUTER_TEST_ONLY_SECRET}"]}}))
    with pytest.raises(SystemExit, match="ROUTER_TEST_ONLY_SECRET.*missing or empty"):
        mr.load_config(path)


def test_environment_expansion_is_single_pass_and_reports_location(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_TEST_SECRET", "dummy-${not-recursively-expanded}")
    raw = config()
    raw["server"] = {"auth_mode": "required", "api_keys": ["${ROUTER_TEST_SECRET}"]}
    path = tmp_path / "router.json"
    path.write_text(json.dumps(raw))
    assert mr.load_config(path).api_keys == ["dummy-${not-recursively-expanded}"]
    raw["providers"]["local"]["headers"] = {"x-test": "${INVALID-SYNTAX}"}
    path.write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="providers.local.headers.x-test: invalid environment substitution"):
        mr.load_config(path)


@pytest.mark.parametrize("source", ["api_key_env", "api_key_file"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_upstream_secret_references_must_resolve(tmp_path, monkeypatch, source, value):
    raw = config()
    if source == "api_key_env":
        ref = "ROUTER_TEST_ONLY_SECRET"
        monkeypatch.delenv(ref, raising=False)
        if value is not None:
            monkeypatch.setenv(ref, value)
    else:
        ref = str(tmp_path / "dummy-secret")
        if value is not None:
            Path(ref).write_text(value)
    raw["providers"]["local"][source] = ref
    with pytest.raises(SystemExit):
        mr.build_config(raw)


def test_auth_modes_and_legacy_key_inference(monkeypatch):
    raw = config()
    monkeypatch.setenv("MODEL_ROUTER_API_KEY", "dummy-env-auth")
    assert mr.build_config(raw).api_keys == []  # explicit none wins over env
    raw["server"] = {"auth_mode": "required"}
    assert mr.build_config(raw).api_keys == ["dummy-env-auth"]
    raw["server"] = {}
    assert mr.build_config(raw).api_keys == ["dummy-env-auth"]
    monkeypatch.delenv("MODEL_ROUTER_API_KEY")
    assert mr.build_config(raw).api_keys == []  # old local configs remain valid
    raw["server"] = {"auth_mode": "required"}
    with pytest.raises(SystemExit, match="requires at least one"):
        mr.build_config(raw)
    raw["server"] = {"auth_mode": "none", "api_keys": ["dummy-key"]}
    with pytest.raises(SystemExit, match="conflicts"):
        mr.build_config(raw)
    raw["server"] = {"auth_mode": "requred"}
    with pytest.raises(SystemExit, match="auth_mode"):
        mr.build_config(raw)


def test_explicit_auth_modes_on_http_requests(upstream, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_API_KEY", "dummy-env-secret")
    raw = _config(upstream)
    raw["server"].update(auth_mode="required", api_keys=["dummy-config-secret"])
    with router_config(raw) as base, httpx.Client() as client:
        body = _body("openai-chat", "m-chat", False)
        assert client.post(base + PATHS["openai-chat"], json=body).status_code == 401
        assert upstream.requests == []
        for key in ("dummy-config-secret", "dummy-env-secret"):
            assert client.post(base + PATHS["openai-chat"], json=body,
                               headers={"Authorization": "Bearer " + key}).status_code == 200
    raw["server"].update(auth_mode="none", api_keys=[])
    with router_config(raw) as base:
        assert httpx.post(base + PATHS["openai-chat"], json=body).status_code == 200


@pytest.mark.parametrize("enabled", [True, False])
def test_dynamic_route_policy_and_disabled_alias_scope(enabled):
    raw = config()
    raw["server"]["allow_dynamic_routes"] = enabled
    raw["models"].append({"name": "disabled", "provider": "local", "model": "other", "enabled": False})
    app = mr.App(mr.build_config(raw))
    try:
        assert app.resolve("public").upstream_model == "upstream"
        assert app.resolve("alias").upstream_model == "upstream"
        assert app.resolve("disabled") is None
        assert (app.resolve("local/other") is not None) == enabled
        assert (app.resolve("local/arbitrary") is not None) == enabled
        assert app.resolve("unknown/model") is None
        assert "local/arbitrary" not in {row["id"] for row in app.models_payload()["data"]}
    finally:
        app.close()


@pytest.mark.parametrize("ingress", tuple(PATHS))
def test_bad_stream_types_are_json_400_before_resolution_or_dispatch(upstream, ingress, monkeypatch):
    raw = _config(upstream)
    raw["server"]["auth_mode"] = "none"
    def forbidden(*args, **kwargs):
        pytest.fail("invalid stream reached route resolution/dispatch")
    monkeypatch.setattr(mr.App, "resolve", forbidden)
    monkeypatch.setattr(mr.App, "dispatch", forbidden)
    with router_config(raw) as base, httpx.Client() as client:
        for value in ("false", "true", [], {}, 0, 1, None):
            body = _body(ingress, "m-chat", False)
            body["stream"] = value
            response = client.post(base + PATHS[ingress], json=body)
            assert response.status_code == 400, response.text
            assert response.headers["content-type"].startswith("application/json")
            assert "stream" in response.json()["error"]["message"]
    assert upstream.requests == []


@pytest.mark.parametrize("ingress", tuple(PATHS))
def test_known_ingress_types_before_native_or_translated_upstream(upstream, ingress):
    raw = _config(upstream)
    bad = [("temperature", "0.5"), ("top_p", True), ("tools", {}),
           ("max_tokens" if ingress != "openai-responses" else "max_output_tokens", False),
           ("input", 42) if ingress == "openai-responses" else ("messages", [{"role": 123}])]
    if ingress != "anthropic":
        bad += [("store", "false"), ("parallel_tool_calls", []),
                ("stream_options", {"include_usage": "false"})]
    if ingress == "openai-responses":
        bad += [("input", [{"role": False, "content": "hello"}]),
                ("input", [{"type": "message", "role": "user", "content": 1}])]
    with router_config(raw) as base, httpx.Client() as client:
        for model in ("m-chat", "m-resp", "m-anth"):
            for field, value in bad:
                body = {**_body(ingress, model, False), field: value}
                response = client.post(base + PATHS[ingress], json=body)
                assert response.status_code == 400, response.text
    assert upstream.requests == []


def test_responses_missing_input_and_unsupported_state_are_client_errors(upstream):
    raw = _config(upstream)
    with router_config(raw) as base, httpx.Client() as client:
        for model in ("m-chat", "m-anth"):
            for extra in ({}, {"instructions": "hello"}, {"previous_response_id": "resp_dummy"},
                          {"conversation": "conv_dummy"}, {"prompt": {"id": "pmpt_dummy"}},
                          {"background": True}, {"input": "hello", "previous_response_id": "resp_dummy"}):
                response = client.post(base + PATHS["openai-responses"], json={"model": model, **extra})
                assert response.status_code == 400, response.text
                assert "missing 1 required" not in response.text
                expected = "unsupported_responses_state" if any(key in extra for key in (
                    "previous_response_id", "conversation", "prompt", "background")) else "invalid_request"
                assert response.json()["error"]["code"] == expected
        response = client.post(base + PATHS["openai-responses"], json={
            "model": "m-resp", "previous_response_id": "resp_dummy", "conversation": "conv_dummy"})
        assert response.status_code == 400
    assert upstream.requests == []


@pytest.mark.parametrize("ingress,model,path", [
    ("openai-chat", "m-chat", "/v1/chat/completions"),
    ("openai-responses", "m-resp", "/v1/responses"),
    ("anthropic", "m-anth", "/v1/messages"),
])
def test_native_vendor_extensions_survive_validation(upstream, ingress, model, path):
    extension = {"vendor_feature": {"enabled": True, "values": [1, "two"]},
                 "extra_body": {"vendor_sampling": {"mode": "custom"}}}
    body = {**_body(ingress, model, False), **extension}
    with router_config(_config(upstream)) as base:
        response = httpx.post(base + PATHS[ingress], json=body)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(path)["body"]
    for field, value in extension.items():
        assert sent[field] == value


def test_native_responses_state_and_prompt_without_input_are_preserved(upstream):
    with router_config(_config(upstream)) as base, httpx.Client() as client:
        for extra in ({"previous_response_id": "resp_dummy"}, {"conversation": {"id": "conv_dummy"}},
                      {"prompt": {"id": "pmpt_dummy"}}, {"instructions": "hello"}):
            response = client.post(base + PATHS["openai-responses"], json={"model": "m-resp", **extra})
            assert response.status_code == 200, response.text
            sent = upstream.last_request("/v1/responses")["body"]
            assert "input" not in sent
            assert all(sent[key] == value for key, value in extra.items())


def test_invalid_json_encoding_and_nonfinite_numbers_are_client_errors(upstream):
    bodies = [b'{"model":"m-chat","messages":[],"vendor_field":"\xff"}',
              b'{"model":"m-chat","messages":[],"temperature":NaN}',
              b'{"model":"m-chat","messages":[],"vendor_field":1e999}']
    with router_config(_config(upstream)) as base, httpx.Client() as client:
        for body in bodies:
            response = client.post(base + PATHS["openai-chat"], content=body,
                                   headers={"Content-Type": "application/json"})
            assert response.status_code == 400, response.text
    assert upstream.requests == []


@pytest.mark.parametrize("block_stdlib", [False, True])
@pytest.mark.parametrize("filename", ["router.toml", "router.unknown"])
def test_default_toml_loading_with_stdlib_and_python310_fallback(tmp_path, monkeypatch, block_stdlib, filename):
    imported = []
    original_import = builtins.__import__
    def importing(name, *args, **kwargs):
        if name in ("tomllib", "tomli"):
            imported.append(name)
        if name == "tomllib" and block_stdlib:
            raise ModuleNotFoundError("simulated Python 3.10: no tomllib", name="tomllib")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setenv("OPENCODE_API_KEY", "dummy-only-test-key")
    path = tmp_path / filename
    path.write_text((ROOT / "router.toml").read_text())
    cfg = mr.load_config(path)
    assert cfg.providers["opencode-chat"].api_key == "dummy-only-test-key"
    assert cfg.api_keys == []
    assert cfg.allow_dynamic_routes is False
    assert ("tomli" in imported) == (block_stdlib or sys.version_info < (3, 11))


def test_python310_tomli_dependency_marker_matches_both_manifests():
    metadata = router_configuration._toml_loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.10,<3.15"
    packaged = [Requirement(dep) for dep in metadata["project"]["dependencies"]]
    requirements = [Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()]
    for deps in (packaged, requirements):
        dep, = [dep for dep in deps if dep.name == "tomli"]
        assert dep.marker.evaluate({"python_version": "3.10"})
        assert not dep.marker.evaluate({"python_version": "3.11"})
        assert not dep.marker.evaluate({"python_version": "3.14"})


@pytest.mark.parametrize("option,value", [("--port", "0"), ("--port", "65536"),
                                          ("--host", ""), ("--api-key", "  ")])
def test_cli_overrides_do_not_bypass_config_validation(tmp_path, option, value):
    path = tmp_path / "router.json"
    path.write_text(json.dumps(config()))
    with pytest.raises(SystemExit):
        mr.main(["--no-default-env", "--config", str(path), "--check", option, value])
