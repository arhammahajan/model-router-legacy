"""POST /v1/messages/count_tokens against local fake upstreams only.

The fake records every request, so each test can also assert what the router did
*not* send: no generation call, no client credential, no estimate.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import httpx
import pytest

import model_router as mr
from model_router.headers import _count_tokens_url

COUNT = "/v1/messages/count_tokens"
UPSTREAM_KEY = "upstream-secret-key"
ROUTER_KEY = "router-secret-key"


class CountUpstream:
    """Anthropic-shaped fake. `script` entries are consumed per request, then repeat."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.script: list[dict[str, Any]] = [{"json": {"input_tokens": 42}}]
        self.gate: threading.Event | None = None  # holds /v1/messages until set
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = self.path.split("?", 1)[0]
                with fake._lock:
                    index = sum(1 for r in fake.requests if r["path"] == path)
                    fake.requests.append({
                        "path": path, "raw": raw, "body": json.loads(raw or b"{}"),
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    })
                if path == "/v1/messages":
                    if fake.gate is not None:
                        fake.gate.wait(10)
                    return self._reply({"json": {"type": "message", "content": []}})
                if path != COUNT:
                    return self._reply({"status": 404, "json": {"type": "error", "error": {
                        "type": "not_found_error", "message": "no such route"}}})
                self._reply(fake.script[min(index, len(fake.script) - 1)])

            def _reply(self, step: dict[str, Any]) -> None:
                if step.get("delay"):
                    time.sleep(step["delay"])
                data = step["body"] if "body" in step else json.dumps(step["json"]).encode()
                self.send_response(step.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in step.get("headers", {}).items():
                    self.send_header(name, value)
                self.send_header("X-Upstream-Diagnostic", "Bearer do-not-forward")
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        # The router rightly abandons slow or oversized replies mid-write.
        self._server.handle_error = lambda *_args: None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "CountUpstream":
        self._thread.start()
        return self

    def stop(self) -> None:
        if self.gate is not None:
            self.gate.set()
        self._server.shutdown()
        self._server.server_close()

    def counts(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"] == COUNT]

    def generations(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"] != COUNT]


@pytest.fixture
def upstream() -> Iterator[CountUpstream]:
    fake = CountUpstream().start()
    try:
        yield fake
    finally:
        fake.stop()


def config_for(upstream: CountUpstream, *, server: dict[str, Any] | None = None,
               retry: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
    def anthropic(base: str, **extra: Any) -> dict[str, Any]:
        provider = {"dialect": "anthropic", "base_url": base, "api_key": UPSTREAM_KEY,
                    "session_header": "x-opencode-session", **extra}
        if timeout is not None:
            provider["timeout"] = timeout
        return provider

    raw: dict[str, Any] = {
        "server": {"host": "127.0.0.1", "port": 4000, "api_keys": [ROUTER_KEY],
                   **(server or {})},
        "providers": {
            "counting": anthropic(upstream.root, count_tokens=True),
            "counting-v1": anthropic(upstream.root + "/v1", count_tokens=True),
            "counting-full": anthropic(upstream.root + "/v1/messages", count_tokens=True),
            # Anthropic protocol, but nobody said it implements counting.
            "undeclared": anthropic(upstream.root),
            "disabled": anthropic(upstream.root, count_tokens=False),
            "chat": {"dialect": "openai-chat", "base_url": upstream.root + "/v1",
                     "api_key": UPSTREAM_KEY},
            "resp": {"dialect": "openai-responses", "base_url": upstream.root + "/v1",
                     "api_key": UPSTREAM_KEY},
        },
        "models": [
            {"name": "m-count", "provider": "counting", "model": "upstream-claude"},
            {"name": "m-v1", "provider": "counting-v1", "model": "upstream-v1"},
            {"name": "m-full", "provider": "counting-full", "model": "upstream-full"},
            {"name": "m-undeclared", "provider": "undeclared", "model": "upstream-undeclared"},
            {"name": "m-disabled", "provider": "disabled", "model": "upstream-disabled"},
            {"name": "m-chat", "provider": "chat", "model": "upstream-chat"},
            {"name": "m-resp", "provider": "resp", "model": "upstream-resp"},
        ],
    }
    if retry is not None:
        raw["retry"] = retry
    return raw


@contextmanager
def running(raw: dict[str, Any]) -> Iterator[str]:
    app = mr.App(mr.build_config(raw))
    server = mr.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        app.close()


@pytest.fixture
def base(upstream: CountUpstream) -> Iterator[str]:
    with running(config_for(upstream)) as url:
        yield url


AUTH = {"Authorization": f"Bearer {ROUTER_KEY}"}


def count(base: str, body: dict[str, Any], headers: dict[str, str] | None = None,
          path: str = COUNT) -> httpx.Response:
    return httpx.post(base + path, json=body, headers={**AUTH, **(headers or {})}, timeout=30)


def simple(model: str = "m-count") -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "hello"}]}


# -- success and request preservation ----------------------------------------


def test_exact_upstream_count_is_returned_unchanged(base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"json": {"input_tokens": 1234}}]
    response = count(base, simple())  # note: no max_tokens, as in the Anthropic contract
    assert response.status_code == 200
    assert response.json() == {"input_tokens": 1234}
    assert [r["path"] for r in upstream.requests] == [COUNT]


def test_request_is_preserved_except_for_the_resolved_model(base: str, upstream: CountUpstream) -> None:
    pdf = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                          "data": "JVBERi0xLjQK"}}
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "iVBORw0KGgo="}}
    body = {
        "model": "m-count",
        "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [image, pdf, {"type": "text", "text": "describe"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"location": "SF"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]},
        ],
        "tools": [{"name": "get_weather", "description": "weather",
                   "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}}],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "output_config": {"effort": "high"},
        "cache_control": {"type": "ephemeral"},
        "speed": "standard",
        "vendor_extension": {"nested": [1, 2, 3]},
    }
    assert count(base, body).status_code == 200
    sent = upstream.counts()[0]["body"]
    assert sent.pop("model") == "upstream-claude"
    expected = {k: v for k, v in body.items() if k != "model"}
    assert sent == expected


def test_both_paths_and_dynamic_routes_resolve(base: str, upstream: CountUpstream) -> None:
    assert count(base, simple(), path="/messages/count_tokens").status_code == 200
    assert count(base, simple("counting/claude-dynamic")).status_code == 200
    models = [r["body"]["model"] for r in upstream.counts()]
    assert models == ["upstream-claude", "claude-dynamic"]


@pytest.mark.parametrize("model,expected", [
    ("m-count", COUNT), ("m-v1", COUNT), ("m-full", COUNT),
])
def test_base_url_forms_reach_the_count_endpoint(base: str, upstream: CountUpstream,
                                                 model: str, expected: str) -> None:
    assert count(base, simple(model)).status_code == 200
    assert [r["path"] for r in upstream.requests] == [expected]


def test_count_url_construction() -> None:
    def url(base_url: str) -> str:
        return _count_tokens_url(mr.Provider(
            name="p", dialect="anthropic", base_url=base_url, api_key="k", headers={},
            timeout=1.0, litellm_provider="anthropic", litellm_kwargs={}, drop_params=False))

    assert url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages/count_tokens"
    assert url("https://h.test/v1") == "https://h.test/v1/messages/count_tokens"
    assert url("https://h.test/v1/messages") == "https://h.test/v1/messages/count_tokens"
    assert url("https://h.test/proxy/") == "https://h.test/proxy/v1/messages/count_tokens"


def test_generation_is_never_run_and_stream_is_not_a_stream_request(
        base: str, upstream: CountUpstream) -> None:
    body = {**simple(), "max_tokens": 5, "stream": True}
    response = count(base, body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert upstream.generations() == []


# -- authentication and credential isolation ---------------------------------


def test_router_authentication_is_enforced_before_any_upstream_call(
        base: str, upstream: CountUpstream) -> None:
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"}):
        response = httpx.post(base + COUNT, json=simple(), headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
        assert response.headers["www-authenticate"].startswith("Bearer")
    assert upstream.requests == []
    for headers in ({"Authorization": f"Bearer {ROUTER_KEY}"}, {"x-api-key": ROUTER_KEY}):
        assert httpx.post(base + COUNT, json=simple(), headers=headers).status_code == 200


def test_client_credentials_never_reach_the_upstream(base: str, upstream: CountUpstream) -> None:
    count(base, simple(), headers={
        "x-api-key": ROUTER_KEY, "api-key": ROUTER_KEY, "Cookie": "session=abc",
        "x-opencode-session": "conversation-1", "session-id": "conversation-1",
        "anthropic-beta": "token-counting-2024-11-01", "anthropic-version": "2023-06-01",
    })
    headers = upstream.counts()[0]["headers"]
    assert headers["x-api-key"] == UPSTREAM_KEY
    assert "authorization" not in headers and "cookie" not in headers
    assert "api-key" not in headers
    assert ROUTER_KEY not in json.dumps(headers)
    # Conversation identity is a generation concern; counting must not carry it.
    assert "x-opencode-session" not in headers and "session-id" not in headers
    # Protocol opt-ins are the client's to choose.
    assert headers["anthropic-beta"] == "token-counting-2024-11-01"
    assert headers["anthropic-version"] == "2023-06-01"


def test_default_anthropic_version_is_supplied(base: str, upstream: CountUpstream) -> None:
    count(base, simple())
    assert upstream.counts()[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_unauthenticated_router_still_uses_the_trusted_provider_credential(
        upstream: CountUpstream) -> None:
    raw = config_for(upstream)
    raw["server"] = {"host": "127.0.0.1", "port": 4000, "auth_mode": "none"}
    with running(raw) as url:
        response = httpx.post(url + COUNT, json=simple(),
                              headers={"x-api-key": "client-supplied"})
    assert response.status_code == 200
    assert upstream.counts()[0]["headers"]["x-api-key"] == UPSTREAM_KEY


# -- routing and request validation ------------------------------------------


def test_unknown_and_missing_models(base: str, upstream: CountUpstream) -> None:
    unknown = count(base, simple("nope"))
    assert unknown.status_code == 404
    assert unknown.json()["error"]["type"] == "not_found_error"
    missing = count(base, {"messages": [{"role": "user", "content": "x"}]})
    assert missing.status_code == 400
    assert missing.json()["type"] == "error"
    assert upstream.requests == []


@pytest.mark.parametrize("body", [
    {"model": "m-count"},
    {"model": "m-count", "messages": "hello"},
    {"model": "m-count", "messages": [{"role": "user", "content": 7}]},
    {"model": "m-count", "messages": [], "tools": "x"},
])
def test_invalid_shapes_are_rejected_locally(base: str, upstream: CountUpstream,
                                             body: dict[str, Any]) -> None:
    response = count(base, body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert upstream.requests == []


# -- unsupported capability ---------------------------------------------------


@pytest.mark.parametrize("model,fragment", [
    ("m-chat", "openai-chat"),
    ("m-resp", "openai-responses"),
    ("m-undeclared", "count_tokens = true"),
    ("m-disabled", "disabled"),
])
def test_unsupported_routes_are_refused_without_estimating_or_calling_upstream(
        base: str, upstream: CountUpstream, model: str, fragment: str) -> None:
    response = count(base, simple(model))
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "Exact token counting is unavailable" in payload["error"]["message"]
    assert fragment in payload["error"]["message"]
    assert "input_tokens" not in response.text
    # Nothing at all: not the count endpoint, and certainly no generation call.
    assert upstream.requests == []


def test_dynamic_route_on_unsupported_provider_is_refused(base: str, upstream: CountUpstream) -> None:
    assert count(base, simple("chat/anything")).status_code == 400
    assert count(base, simple("undeclared/anything")).status_code == 400
    assert upstream.requests == []


def _provider(base_url: str, dialect: str = "anthropic", **extra: Any) -> mr.Provider:
    return mr.Provider(name="p", dialect=dialect, base_url=base_url, api_key="k", headers={},
                       timeout=1.0, litellm_provider="anthropic", litellm_kwargs={},
                       drop_params=False, **extra)


@pytest.mark.parametrize("base_url,expected", [
    ("https://api.anthropic.com", True),
    ("https://API.anthropic.com/v1", True),
    ("http://api.anthropic.com", False),
    ("https://api.anthropic.com.evil.test", False),
    ("https://api.anthropic.com:8443", False),
    ("https://user@api.anthropic.com", False),
    ("https://example.test", False),
    ("http://127.0.0.1:9", False),
])
def test_only_the_first_party_host_is_assumed_to_support_counting(
        base_url: str, expected: bool) -> None:
    assert _provider(base_url).count_tokens_enabled is expected


def test_explicit_setting_overrides_the_host_rule_but_not_the_dialect() -> None:
    assert _provider("https://example.test", count_tokens=True).count_tokens_enabled
    assert not _provider("https://api.anthropic.com", count_tokens=False).count_tokens_enabled
    assert not _provider("https://api.anthropic.com", dialect="openai-chat").count_tokens_enabled


@pytest.mark.parametrize("provider", [
    {"dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1", "api_key": "k",
     "count_tokens": True},
    {"dialect": "anthropic", "base_url": "http://127.0.0.1:1", "api_key": "k",
     "count_tokens": "yes"},
])
def test_count_tokens_config_is_validated(provider: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        mr.build_config({"server": {"auth_mode": "none"}, "providers": {"p": provider},
                         "models": [{"name": "m", "provider": "p", "model": "m"}]})


def test_a_chat_provider_may_state_that_it_cannot_count() -> None:
    mr.build_config({"server": {"auth_mode": "none"}, "models": [
        {"name": "m", "provider": "p", "model": "m"}], "providers": {"p": {
            "dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1", "api_key": "k",
            "count_tokens": False}}})


# -- upstream errors ----------------------------------------------------------


def test_upstream_errors_keep_status_and_are_sanitized(base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"status": 400, "json": {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": f"server tools are not supported; key {UPSTREAM_KEY} leaked"}}}]
    body = {**simple(), "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
    response = count(base, body)
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "server tools are not supported" in response.json()["error"]["message"]
    assert UPSTREAM_KEY not in response.text
    # The router does not pre-judge tool types; the upstream's contract decides.
    assert upstream.counts()[0]["body"]["tools"][0]["type"] == "web_search_20250305"


def test_rate_limit_hints_are_forwarded_and_diagnostics_are_not(
        base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"status": 429, "headers": {
        "Retry-After": "7", "anthropic-ratelimit-requests-remaining": "0"},
        "json": {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}}]
    response = count(base, simple())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
    assert response.headers["anthropic-ratelimit-requests-remaining"] == "0"
    assert "x-upstream-diagnostic" not in response.headers
    assert response.json()["error"]["type"] == "rate_limit_error"


@pytest.mark.parametrize("status,kind", [(401, "authentication_error"), (404, "not_found_error"),
                                         (500, "api_error"), (529, "overloaded_error")])
def test_other_upstream_statuses_map_to_anthropic_errors(
        base: str, upstream: CountUpstream, status: int, kind: str) -> None:
    upstream.script = [{"status": status, "body": b"not even json"}]
    response = count(base, simple())
    assert response.status_code == status
    assert response.json()["error"]["type"] == kind


@pytest.mark.parametrize("body", [
    b'{"unexpected": 1}', b"not json", b"[]", b'{"input_tokens": "7"}',
    b'{"input_tokens": true}', b'{"input_tokens": -1}', b'{"input_tokens": 1.5}', b"",
])
def test_a_success_that_is_not_a_count_is_never_passed_off_as_one(
        base: str, upstream: CountUpstream, body: bytes) -> None:
    upstream.script = [{"body": body}]
    response = count(base, simple())
    assert response.status_code == 502
    assert response.json()["type"] == "error"
    assert "input_tokens" not in response.text


def test_oversized_upstream_responses_are_cut_off(base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"body": b'{"input_tokens": 1, "pad": "' + b"x" * 200_000 + b'"}'}]
    assert count(base, simple()).status_code == 502
    upstream.script = [{"status": 500, "body": b"x" * 200_000}]
    assert count(base, simple()).status_code == 502


def test_extra_fields_from_a_genuine_count_are_kept(base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"json": {"input_tokens": 9, "context_management": {
        "original_input_tokens": 12}}}]
    assert count(base, simple()).json() == {
        "input_tokens": 9, "context_management": {"original_input_tokens": 12}}


# -- resource limits ----------------------------------------------------------


def test_body_size_limit_applies_to_counting(upstream: CountUpstream) -> None:
    with running(config_for(upstream, server={"max_body_bytes": 512})) as url:
        big = {**simple(), "messages": [{"role": "user", "content": "x" * 2000}]}
        assert count(url, big).status_code == 413
        assert count(url, simple()).status_code == 200
        # Chunked framing and gzip cannot bypass it either.
        def chunks() -> Iterator[bytes]:
            yield json.dumps(big).encode()
        assert httpx.post(url + COUNT, content=chunks(), headers=AUTH).status_code == 413
        bomb = gzip.compress(json.dumps(big).encode())
        assert httpx.post(url + COUNT, content=bomb, headers={
            **AUTH, "Content-Encoding": "gzip", "Content-Type": "application/json"}).status_code == 413
    assert len(upstream.counts()) == 1


def test_gzip_request_bodies_are_decoded_before_forwarding(base: str, upstream: CountUpstream) -> None:
    response = httpx.post(base + COUNT, content=gzip.compress(json.dumps(simple()).encode()),
                          headers={**AUTH, "Content-Encoding": "gzip",
                                   "Content-Type": "application/json"})
    assert response.status_code == 200
    request = upstream.counts()[0]
    assert request["body"]["model"] == "upstream-claude"
    assert "content-encoding" not in request["headers"]


@pytest.mark.parametrize("content", [b"", b"{not json", b"[1, 2]", b'"text"'])
def test_malformed_bodies_never_reach_the_upstream(base: str, upstream: CountUpstream,
                                                   content: bytes) -> None:
    response = httpx.post(base + COUNT, content=content, headers={
        **AUTH, "Content-Type": "application/json"})
    assert response.status_code == 400
    assert upstream.requests == []


def test_a_slow_upstream_hits_the_provider_timeout_without_billing_language(
        upstream: CountUpstream) -> None:
    upstream.script = [{"delay": 3, "json": {"input_tokens": 1}}]
    with running(config_for(upstream, timeout=0.3)) as url:
        started = time.monotonic()
        response = count(url, simple())
    assert response.status_code == 504
    assert time.monotonic() - started < 2.5
    message = response.json()["error"]["message"]
    assert "token-count request" in message
    assert "billed" not in message and "generation" not in message
    assert len(upstream.counts()) == 1  # not replayed


def test_the_overall_retry_deadline_applies(upstream: CountUpstream) -> None:
    upstream.script = [{"delay": 3, "json": {"input_tokens": 1}}]
    with running(config_for(upstream, retry={"deadline": 0.4})) as url:
        started = time.monotonic()
        response = count(url, simple())
    assert response.status_code == 504
    assert time.monotonic() - started < 2.5
    assert "billed" not in response.json()["error"]["message"]


def test_the_shared_retry_policy_covers_counting(upstream: CountUpstream) -> None:
    upstream.script = [
        {"status": 429, "headers": {"Retry-After": "0"}, "json": {"type": "error", "error": {
            "type": "rate_limit_error", "message": "later"}}},
        {"json": {"input_tokens": 5}},
    ]
    retry = {"max_attempts": 2, "backoff_base": 0.01, "backoff_max": 0.01}
    with running(config_for(upstream, retry=retry)) as url:
        response = count(url, simple())
    assert response.json() == {"input_tokens": 5}
    assert len(upstream.counts()) == 2 and upstream.generations() == []


def test_without_retry_opt_in_a_429_is_not_replayed(base: str, upstream: CountUpstream) -> None:
    upstream.script = [{"status": 429, "json": {"type": "error", "error": {
        "type": "rate_limit_error", "message": "later"}}}]
    assert count(base, simple()).status_code == 429
    assert len(upstream.counts()) == 1


def test_counting_does_not_consume_or_wait_for_generation_slots(upstream: CountUpstream) -> None:
    upstream.gate = threading.Event()
    raw = config_for(upstream, server={"max_generations": 1, "max_connections": 8})
    with running(raw) as url:
        generation = threading.Thread(target=lambda: httpx.post(
            url + "/v1/messages", headers=AUTH, timeout=15,
            json={**simple(), "max_tokens": 8}), daemon=True)
        generation.start()
        deadline = time.monotonic() + 5
        while not upstream.generations() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert upstream.generations(), "generation never reached the upstream"
        # The only generation slot is occupied; a second generation is refused...
        blocked = httpx.post(url + "/v1/messages", headers=AUTH,
                             json={**simple(), "max_tokens": 8}, timeout=5)
        assert blocked.status_code == 503
        # ...but counting is unaffected.
        assert count(url, simple()).json() == {"input_tokens": 42}
        upstream.gate.set()
        generation.join(10)


# -- logging ------------------------------------------------------------------


def test_count_requests_are_logged_without_content_or_credentials(
        base: str, upstream: CountUpstream, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="model-router")
    secret = "PRIVATE-PROMPT-TEXT"
    body = {**simple("counting/claude-dynamic"),
            "messages": [{"role": "user", "content": secret}]}
    assert count(base, body).status_code == 200
    assert count(base, simple("m-chat")).status_code == 400
    outcomes = [json.loads(r.getMessage()) for r in caplog.records
                if hasattr(r, "router_event") and r.router_event["event"] == "request_outcome"]
    assert [o["http_status"] for o in outcomes] == [200, 400]
    assert all(o["path"] == COUNT and o["operation"] == "count_tokens" for o in outcomes)
    assert outcomes[0]["model"] is None and outcomes[0]["upstream_model"] is None  # dynamic
    assert outcomes[1]["provider"] == "chat" and outcomes[1]["translation_mode"] is None
    for forbidden in (secret, UPSTREAM_KEY, ROUTER_KEY):
        assert forbidden not in caplog.text
