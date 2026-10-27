"""Review regressions at authentication, proxy-header, and config boundaries."""
from __future__ import annotations

from dataclasses import replace
from email.message import Message

import httpx
import pytest

import model_router as mr
import model_router.cli as router_cli
from model_router.headers import (
    _filtered_response_headers,
    _serialized_json_headers,
    _session_id,
    _translation_header_provider,
    _upstream_request_headers,
)
from test_model_router import _body, _config, router_config, upstream


@pytest.mark.parametrize("header", [b"Authorization", b"x-api-key", b"api-key"])
def test_non_ascii_auth_returns_401_and_the_router_recovers(upstream, header):
    raw = _config(upstream, api_keys=["dummy-router-key"])
    value = (b"Bearer " if header == b"Authorization" else b"") + b"caf\xe9"
    with router_config(raw) as base, httpx.Client() as client:
        invalid = [(header, value)]
        assert client.get(base + "/v1/models", headers=invalid).status_code == 401
        body = _body("openai-chat", "m-chat", False)
        response = client.post(base + "/v1/chat/completions", json=body, headers=invalid)
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Invalid API key"
        assert not upstream.requests
        valid = {"Authorization": "Bearer dummy-router-key"}
        assert client.get(base + "/v1/models", headers=valid).status_code == 200
        assert client.post(base + "/v1/chat/completions", json=body,
                           headers=valid).status_code == 200


def test_repeated_connection_fields_remove_only_client_hop_headers(upstream):
    provider = mr.build_config(_config(upstream)).providers["chat"]
    headers = Message()
    headers["Connection"] = " X-Local, AuThOrIzAtIoN "
    headers["connection"] = "x-another-local"
    headers["X-Local"] = "first-hop-only"
    headers["X-Another-Local"] = "also-hop-only"
    headers["Authorization"] = "Bearer client-secret"
    headers["X-End-To-End"] = "keep"
    forwarded = {k.lower(): v for k, v in _upstream_request_headers(
        provider, "openai-chat", headers).items()}
    assert not {"connection", "x-local", "x-another-local"} & forwarded.keys()
    assert forwarded["x-end-to-end"] == "keep"
    assert forwarded["authorization"] == "Bearer upstream-key"


@pytest.mark.parametrize("filter_headers", [_filtered_response_headers, _serialized_json_headers])
def test_connection_options_apply_to_response_and_static_headers(filter_headers):
    headers = Message()
    headers["Connection"] = " X-Local "
    headers["connection"] = "X-Another-Local"
    headers["X-Local"] = "remove"
    headers["X-Another-Local"] = "remove-too"
    headers["X-End-To-End"] = "keep"
    result = {k.lower(): v for k, v in filter_headers(headers).items()}
    assert not {"connection", "x-local", "x-another-local"} & result.keys()
    assert result["x-end-to-end"] == "keep"


def test_translation_does_not_promote_hop_only_beta_or_session_headers(upstream):
    provider = mr.build_config(_config(upstream)).providers["anth"]
    inbound = {
        "Connection": "Anthropic-Beta, X-OpenCode-Session",
        "anthropic-beta": "hop-beta",
        "x-opencode-session": "hop-session",
    }
    assert _session_id(provider, {}, inbound) is None
    translated = _translation_header_provider(provider, inbound)
    assert "anthropic-beta" not in {k.lower() for k in translated.headers}
    # Client hop-by-hop policy cannot suppress trusted operator opt-ins.
    trusted = replace(provider, headers={"Anthropic-Beta": "trusted-beta"})
    translated = _translation_header_provider(trusted, inbound)
    assert translated.headers["anthropic-beta"] == "trusted-beta"


def test_invalid_yaml_is_a_sanitized_cli_configuration_error(tmp_path, capsys):
    pytest.importorskip("yaml")
    path = tmp_path / "router.yaml"
    path.write_text('server: [dummy-private-config-value\n')
    with pytest.raises(SystemExit, match="malformed YAML"):
        mr.load_config(path)
    with pytest.raises(SystemExit) as error:
        router_cli.main(["--no-default-env", "-c", str(path),
                         "--validate-config", "--offline"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "malformed YAML" in message
    assert "dummy-private-config-value" not in message
    assert "Traceback" not in message
