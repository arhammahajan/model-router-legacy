"""Offline test boundary: installed adapters may connect only to local fixtures."""
from __future__ import annotations

import os
import socket

import pytest

# Select LiteLLM's installed metadata before tests import it. No cost-map fetch
# or telemetry is needed to exercise provider adapters against local HTTP.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["DO_NOT_TRACK"] = "True"


@pytest.fixture(autouse=True)
def local_network_only(monkeypatch):
    original = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple):
            assert address[0] in {"127.0.0.1", "::1", "localhost"}, (
                "Tests must use local upstreams", address)
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    original_getaddrinfo = socket.getaddrinfo
    def getaddrinfo(host, *args, **kwargs):
        assert host in {None, "127.0.0.1", "::1", "localhost", b"127.0.0.1", b"::1", b"localhost"}, (
            "Tests must not resolve external hosts", host)
        return original_getaddrinfo(host, *args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
