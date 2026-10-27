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








