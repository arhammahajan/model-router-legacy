"""Prompt 17: a single HTTP retry owner, real adapters, local peers, fake time."""
from __future__ import annotations

import asyncio
import json
import socket
import threading
from contextlib import contextmanager
from email.utils import formatdate

import httpx
import pytest

import model_router as mr
import model_router.adapters as router_adapters
import model_router.common as router_common
import model_router.transport as router_transport
from test_model_router import (
    FakeUpstream, INGRESSES, PATHS, TARGETS, _body, _text_from_response,
    _text_from_sse,
)
from test_operations import running
from test_protocol_contracts import config
from test_stream_lifecycle import ScriptedUpstream as StreamUpstream


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay
        if self.on_sleep:
            self.on_sleep()

    def wall(self):
        return 1800000000.0 + self.now


class SequenceUpstream(FakeUpstream):
    def __init__(self):
        super().__init__()
        self.script = [200]
        self.attempts = 0
        self.before_headers = None
        self.retry_after = "0"
        self.success_headers = {}
        upstream = self
        original = self._server.RequestHandlerClass

        class Handler(original):
            def do_POST(self):
                step = upstream.script[min(upstream.attempts, len(upstream.script) - 1)]
                upstream.attempts += 1
                upstream.error_body = (b'{"error":{"message":"controlled refusal"}}'
                                       if isinstance(step, int) and step >= 400 else None)
                upstream.error_status = step if isinstance(step, int) else 500
                if step in ("drop", "body-abort"):
                    raw = self.rfile.read(int(self.headers["Content-Length"]))
                    upstream.requests.append({"path": self.path, "body": json.loads(raw)})
                    if step == "body-abort":
                        self.send_response(200)
                        self.send_header("Content-Length", "1000")
                        self.end_headers()
                        self.wfile.write(b"{")
                        self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                try:
                    super().do_POST()
                except (BrokenPipeError, ConnectionResetError):
                    # Retry/deadline cancellation deliberately closes this peer.
                    pass

            def send_header(self, name, value):
                if name.lower() == "retry-after":
                    value = upstream.retry_after
                super().send_header(name, value)

            def end_headers(self):
                for name, value in upstream.success_headers.items():
                    self.send_header(name, value)
                if upstream.before_headers:
                    upstream.before_headers()
                super().end_headers()

        self._server.RequestHandlerClass = Handler
        # Keep fixture teardown fast without timing-sensitive correctness checks.
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.01}, daemon=True)


@pytest.fixture
def peer():
    upstream = SequenceUpstream().start()
    try:
        yield upstream
    finally:
        upstream.stop()
        upstream._thread.join(3)
        assert not upstream._thread.is_alive()


@contextmanager
def application(peer, **policy):
    clock = Clock()
    raw = config(peer)
    raw["retry"] = {"max_attempts": 3, "deadline": 30, **policy}
    # Try to override policy through the trusted extension point too. Router
    # policy wins; SDKs must not multiply these values by max_attempts.
    for provider in raw["providers"].values():
        provider["litellm_kwargs"] = {"num_retries": 7, "max_retries": 7}
    app = mr.App(mr.build_config(raw), clock=clock, sleep=clock.sleep,
                 jitter=lambda lo, hi: hi, wall_clock=clock.wall)
    try:
        yield app, clock
    finally:
        app.close()


def dispatch(app, ingress, model, stream=False):
    return app.dispatch(ingress, app.resolve(model), _body(ingress, model, stream), {})


def materialize(plan):
    try:
        if plan.body is not None:
            return plan.body
        if plan.chunks is not None:
            return b"".join(plan.chunks)
        async def collect():
            return b"".join([chunk async for chunk in plan.async_chunks])
        return asyncio.run(collect())
    finally:
        plan.close()


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,text", TARGETS)
def test_recovery_has_one_shared_attempt_budget(peer, status, stream, ingress, model, text):
    peer.script = [status, status, 200]
    with application(peer) as (app, clock):
        plan = dispatch(app, ingress, model, stream)
        result = materialize(plan)
        assert plan.status == 200
        if stream:
            assert _text_from_sse(ingress, result.decode()) == text
        else:
            assert _text_from_response(ingress, json.loads(result)) == text
        assert clock.sleeps == [0.5, 1.0]
    assert peer.attempts == len(peer.requests) == 3
    bodies = [request["body"] for request in peer.requests]
    assert bodies[1:] == bodies[:-1]  # same model and generation, no fallback
    assert all("num_retries" not in body and "max_retries" not in body for body in bodies)


@pytest.mark.parametrize("hint,delay", [("3", 3), ("date", 4), ("bad", 0.5),
                                       ("-1", 0.5), ("0", 0.5), ("past", 0.5)])
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_retry_after_is_a_floor_and_supports_http_dates(peer, hint, delay, model):
    peer.script = [429, 200]
    with application(peer, backoff_max=1) as (app, clock):
        peer.retry_after = (formatdate(clock.wall() + (4 if hint == "date" else -4), usegmt=True)
                            if hint in ("date", "past") else hint)
        materialize(dispatch(app, "openai-chat", model))
        assert clock.sleeps == [delay]
    assert peer.attempts == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 500])
@pytest.mark.parametrize("ingress,model", [("openai-chat", "m-chat"),
    ("openai-chat", "m-anth"), ("openai-responses", "m-chat"), ("anthropic", "m-resp")])
def test_nonretryable_responses_are_not_replayed(peer, status, ingress, model):
    peer.script = [status, 200]
    with application(peer) as (app, clock), running(app) as (client, server):
        reply = client.post(PATHS[ingress], json=_body(ingress, model, False))
        assert reply.status_code == status, reply.text
        assert clock.sleeps == []
    assert peer.attempts == 1


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_exhaustion_keeps_ingress_error_and_rate_limit_hints(peer, ingress, model):
    peer.script = [429]
    peer.error_headers = {"X-RateLimit-Remaining-Requests": "0",
                          "anthropic-ratelimit-tokens-reset": "2030-01-01T00:00:00Z"}
    with application(peer) as (app, clock), running(app) as (client, server):
        reply = client.post(PATHS[ingress], json=_body(ingress, model, True))
        assert reply.status_code == 429, reply.text
        assert reply.headers["retry-after"] == "0"
        assert reply.headers["x-ratelimit-remaining-requests"] == "0"
        assert reply.headers["anthropic-ratelimit-tokens-reset"] == "2030-01-01T00:00:00Z"
        assert "x-upstream-diagnostic" not in reply.headers
        assert reply.json()["error"]["message"]
        if ingress == "anthropic":
            assert reply.json()["type"] == "error"
            assert reply.json()["error"]["type"] == "rate_limit_error"
        assert clock.sleeps == [0.5, 1]
    assert peer.attempts == 3


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
@pytest.mark.parametrize("hint", ["100", "9" * 500, "date"])
def test_unaffordable_retry_after_returns_deadline_with_hints(peer, model, hint):
    peer.script = [503, 200]
    with application(peer, deadline=2) as (app, clock), running(app) as (client, server):
        peer.retry_after = formatdate(clock.wall() + 100, usegmt=True) if hint == "date" else hint
        reply = client.post(PATHS["openai-chat"], json=_body("openai-chat", model, False))
        assert reply.status_code == 504, reply.text
        assert reply.json()["error"]["code"] == "request_deadline_exceeded"
        assert reply.headers["retry-after"] == peer.retry_after
        assert clock.sleeps == []
    assert peer.attempts == 1


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_backoff_consumes_the_same_deadline(peer, model):
    peer.script = [429]
    with application(peer, deadline=1.4) as (app, clock), running(app) as (client, server):
        reply = client.post(PATHS["openai-chat"], json=_body("openai-chat", model, False))
        assert reply.status_code == 504, reply.text
        assert clock.sleeps == [0.5]
    assert peer.attempts == 2


@pytest.mark.parametrize("failure", ["drop", "body-abort"])
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_ambiguous_post_send_failure_is_never_replayed(peer, failure, model):
    peer.script = [failure, 200]
    with application(peer) as (app, clock), running(app) as (client, server):
        reply = client.post(PATHS["openai-chat"], json=_body("openai-chat", model, False))
        assert reply.status_code == 502, reply.text
        assert clock.sleeps == []
    assert peer.attempts == 1


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_response_after_deadline_is_discarded_without_replay(peer, model):
    with application(peer, deadline=2) as (app, clock), running(app) as (client, server):
        peer.before_headers = lambda: setattr(clock, "now", 2)
        reply = client.post(PATHS["openai-chat"], json=_body("openai-chat", model, False))
        assert reply.status_code == 504, reply.text
        assert reply.json()["error"]["code"] == "request_deadline_exceeded"
    assert peer.attempts == 1


@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
@pytest.mark.parametrize("connect_errors", [True, False])
def test_real_connection_refusal_recovers_only_if_enabled(peer, model, connect_errors):
    # Bind a port without listen(): connect gets ECONNREFUSED until fake backoff
    # activates the real local upstream. No synthetic adapter/HTTP exceptions.
    peer._server.shutdown()
    peer._server.server_close()
    peer._thread.join(3)
    peer._server = type(peer._server)(("127.0.0.1", 0), peer._server.RequestHandlerClass,
                                     bind_and_activate=False)
    peer._server.server_bind()
    peer._thread = threading.Thread(target=peer._server.serve_forever,
                                    kwargs={"poll_interval": 0.01}, daemon=True)
    with application(peer, connect_errors=connect_errors) as (app, clock):
        activated = False
        def activate():
            nonlocal activated
            if not activated:
                peer._server.server_activate()
                peer._thread.start()
                activated = True
        clock.on_sleep = activate
        try:
            if connect_errors:
                assert materialize(dispatch(app, "openai-chat", model))
                assert clock.sleeps == [0.5]
                assert peer.attempts == 1
            else:
                with pytest.raises(Exception):
                    dispatch(app, "openai-chat", model)
                assert clock.sleeps == [] and peer.attempts == 0
        finally:
            activate()  # fixture shutdown must always have a serving thread


@pytest.mark.parametrize("phase", ["first", "midstream"])
@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_truncated_stream_is_never_replayed(phase, ingress, model):
    peer = StreamUpstream()
    peer.mode = phase
    count = []
    original = peer.server.RequestHandlerClass
    class Handler(original):
        def do_POST(self):
            count.append(1)
            super().do_POST()
    peer.server.RequestHandlerClass = Handler
    peer.thread.start()
    try:
        with application(peer) as (app, clock), running(app) as (client, server):
            reply = client.post(PATHS[ingress], json=_body(ingress, model, True))
            assert reply.status_code in (200, 502), reply.text
            assert clock.sleeps == []
            assert count == [1]
            if reply.status_code == 200:
                assert '"error"' in reply.text, reply.text
    finally:
        peer.server.shutdown()
        peer.server.server_close()
        peer.thread.join(3)


@pytest.mark.parametrize("field,value", [("max_attempts", 0), ("max_attempts", True),
    ("max_attempts", 21), ("statuses", [400]), ("statuses", [True]), ("statuses", "429"),
    ("connect_errors", 1), ("deadline", 0), ("deadline", float("inf")),
    ("backoff_base", -1), ("backoff_max", 0.1), ("typo", 1)])
def test_invalid_retry_config_is_rejected(field, value):
    with pytest.raises(SystemExit, match="retry"):
        mr.build_config({"retry": {field: value}})


def test_deadline_clamps_each_network_operation_and_tls():
    # Unit-test the pinned backend seam without depending on wall-clock sleeps.
    clock = Clock()
    app = type("App", (), {"config": type("Config", (), {"retry": mr.RetryPolicy(deadline=3)}),
                           "_clock": clock, "_sleep": clock.sleep, "_jitter": lambda lo, hi: hi,
                           "_wall_clock": clock.wall})()
    budget = router_transport._RequestBudget(app)
    seen = []
    class Network:
        def read(self, size, timeout):
            seen.append(("read", timeout))
            return b"x"
        def write(self, buffer, timeout):
            seen.append(("write", timeout))
        def start_tls(self, context, server_hostname, timeout):
            seen.append(("tls", timeout))
            return self
    stream = router_transport._DeadlineNetworkStream(Network(), budget)
    assert stream.read(1, timeout=10) == b"x"
    clock.now = 1
    stream.write(b"x", timeout=10)
    clock.now = 2
    stream.start_tls(None).read(1, timeout=0.5)
    assert seen == [("read", 3), ("write", 2), ("tls", 1), ("read", 0.5)]
    clock.now = 3
    with pytest.raises(mr.RouterError, match="deadline"):
        stream.read(1)
    assert len(seen) == 4


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_stream_deadline_between_pulls_closes_without_replay(peer, ingress, model):
    with application(peer, deadline=2) as (app, clock):
        plan = dispatch(app, ingress, model, True)
        clients = [resource for resource in plan.resources if isinstance(resource, httpx.Client)]
        try:
            if plan.chunks is not None:
                assert next(plan.chunks)
                clock.now = 2
                with pytest.raises(mr.RouterError) as failure:
                    next(plan.chunks)
            else:
                async def pull():
                    assert await anext(plan.async_chunks)
                    clock.now = 2
                    with pytest.raises(mr.RouterError) as failure:
                        await anext(plan.async_chunks)
                    return failure
                failure = asyncio.run(pull())
            assert failure.value.code == "request_deadline_exceeded"
            assert failure.value.status == 504
            assert clock.sleeps == []
        finally:
            plan.close()
        assert clients and all(client.is_closed for client in clients)
    assert peer.attempts == 1


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.WriteTimeout,
    httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError])
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_ambiguous_transport_failures_have_consistent_errors(peer, monkeypatch, failure, model):
    # Local HTTP truncations above exercise real sockets. This seam additionally
    # covers all post-send exception classes without wall-clock timeout tests.
    original = router_transport._GenerationTransport.handle_request
    def handle(transport, request):
        def failed(*args, **kwargs):
            raise failure("controlled post-send failure")
        monkeypatch.setattr(transport.inner, "send", failed)
        return original(transport, request)
    monkeypatch.setattr(router_transport._GenerationTransport, "handle_request", handle)
    with application(peer) as (app, clock), running(app) as (client, server):
        reply = client.post(PATHS["openai-chat"], json=_body("openai-chat", model, False))
        assert reply.status_code == (504 if issubclass(failure, httpx.TimeoutException) else 502)
        assert reply.json()["error"]["code"] == "upstream_outcome_unknown"
        assert "accepted and billed" in reply.json()["error"]["message"]
        assert clock.sleeps == []
    assert peer.attempts == 0


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ConnectTimeout])
def test_connection_failures_exhaust_one_budget(peer, monkeypatch, failure):
    attempts = []
    original = router_transport._GenerationTransport.handle_request
    def handle(transport, request):
        def failed(*args, **kwargs):
            attempts.append(1)
            raise failure("controlled pre-send failure")
        monkeypatch.setattr(transport.inner, "send", failed)
        return original(transport, request)
    monkeypatch.setattr(router_transport._GenerationTransport, "handle_request", handle)
    with application(peer) as (app, clock):
        with pytest.raises(failure):
            dispatch(app, "openai-chat", "m-chat")
        assert attempts == [1, 1, 1]
        assert clock.sleeps == [0.5, 1]


@pytest.mark.parametrize("status", [200, 429])
def test_nested_layer_cannot_send_again_after_final_response(peer, status):
    peer.script = [status]
    with application(peer, max_attempts=1) as (app, clock):
        transport = router_transport._GenerationTransport(router_transport._RequestBudget(app), app.provider("chat"))
        try:
            request = httpx.Request("POST", f"http://127.0.0.1:{peer.port}/v1/chat/completions",
                                    json={"model": "test", "messages": []})
            reply = transport.handle_request(request)
            reply.close()
            with pytest.raises(mr.RouterError, match="replay blocked"):
                transport.handle_request(request)
            assert peer.attempts == 1 and clock.sleeps == []
        finally:
            transport.close()


@pytest.mark.parametrize("ingress", INGRESSES)
def test_client_cannot_change_policy_even_on_passthrough(peer, ingress):
    model = {"openai-chat": "m-chat", "openai-responses": "m-resp", "anthropic": "m-anth"}[ingress]
    peer.script = [429]
    with application(peer, max_attempts=2) as (app, clock):
        payload = {**_body(ingress, model, False), "num_retries": 100, "max_retries": 100,
                   "retry": {"max_attempts": 100}, "timeout": 1000}
        plan = app.dispatch(ingress, app.resolve(model), payload, {})
        assert plan.status == 429
        plan.close()
        assert peer.attempts == 2 and clock.sleeps == [0.5]


@pytest.mark.parametrize("ingress,model", [("openai-chat", "m-anth"),
    ("openai-responses", "m-chat"), ("anthropic", "m-resp")])
@pytest.mark.parametrize("field", ["num_retries", "max_retries", "retry_policy", "timeout"])
def test_client_retry_controls_are_rejected_on_translations(peer, ingress, model, field):
    with application(peer) as (app, clock):
        payload = {**_body(ingress, model, False), field: 100}
        with pytest.raises(mr.RouterError) as error:
            app.dispatch(ingress, app.resolve(model), payload, {})
        assert error.value.status == 400
        assert peer.attempts == 0


@pytest.mark.parametrize("setting,value", [("num_retries", 3), ("retry_policy", {"x": 3}),
    ("fallbacks", ["another-model"]), ("context_window_fallback_dict", {"a": "b"})])
def test_global_litellm_retry_overrides_fail_closed(peer, monkeypatch, setting, value):
    monkeypatch.setattr(router_adapters.litellm, setting, value, raising=False)
    with application(peer) as (app, clock):
        with pytest.raises(mr.RouterError, match="bypasses router retry policy"):
            dispatch(app, "openai-chat", "m-anth")
        assert peer.attempts == 0


def test_default_is_one_attempt_on_both_paths(peer):
    peer.script = [429]
    raw = config(peer)
    app = mr.App(mr.build_config(raw))
    try:
        assert app.config.retry.max_attempts == 1
        for model in ("m-chat", "m-anth"):
            try:
                plan = dispatch(app, "openai-chat", model)
            except Exception as error:
                assert router_common._status_for(error) == 429
            else:
                assert plan.status == 429
                plan.close()
        assert peer.attempts == 2
    finally:
        app.close()


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-chat", "m-anth", "m-resp", "m-gem"])
def test_stream_deadline_after_output_uses_terminal_ingress_error(peer, ingress, model, monkeypatch):
    peer.success_headers = {"X-RateLimit-Remaining-Requests": "1"}
    with application(peer, deadline=2) as (app, clock):
        original = app._limit_plan
        def limit(plan, budget):
            original(plan, budget)
            if plan.chunks is not None:
                source = plan.chunks
                def expired():
                    yield next(source)
                    clock.now = 2
                    yield from source
                plan.resources += (source,)
                plan.chunks = expired()
            else:
                source = plan.async_chunks
                async def expired_async():
                    yield await anext(source)
                    clock.now = 2
                    async for chunk in source:
                        yield chunk
                plan.resources += (source,)
                plan.async_chunks = expired_async()
        monkeypatch.setattr(app, "_limit_plan", limit)
        with running(app) as (client, server):
            reply = client.post(PATHS[ingress], json=_body(ingress, model, True))
            assert reply.status_code == 200, reply.text
            assert reply.headers["x-ratelimit-remaining-requests"] == "1"
            assert '"error"' in reply.text
            assert "deadline exceeded" in reply.text
            if ingress != "anthropic":
                assert "request_deadline_exceeded" in reply.text
            if ingress == "openai-chat":
                assert reply.text.endswith("data: [DONE]\n\n")
            else:
                assert "event: error" in reply.text
            assert clock.sleeps == []
    assert peer.attempts == 1


@pytest.mark.parametrize("prefix", ["azure", "custom_openai", "vertex_ai"])
def test_unreviewed_adapters_cannot_bypass_retry_transport(peer, prefix):
    raw = config(peer)
    raw["providers"]["chat"]["litellm_provider"] = prefix
    app = mr.App(mr.build_config(raw))
    try:
        with pytest.raises(mr.RouterError) as failure:
            dispatch(app, "anthropic", "m-chat")
        assert failure.value.code == "retry_adapter_unsupported"
        assert not peer.requests
        # Direct protocol forwarding still works with arbitrary prefix metadata.
        assert materialize(dispatch(app, "openai-chat", "m-chat"))
    finally:
        app.close()


@pytest.mark.parametrize("statuses,attempts", [([], 1), ([500], 2)])
def test_configurable_status_list_is_used(peer, statuses, attempts):
    peer.script = [500, 200]
    with application(peer, statuses=statuses) as (app, clock):
        plan = dispatch(app, "openai-chat", "m-chat")
        assert plan.status == (500 if attempts == 1 else 200)
        materialize(plan)
        assert peer.attempts == attempts


@pytest.mark.parametrize("status", [429, 503])
def test_synthetic_output_before_lazy_http_also_forbids_replay(peer, status):
    peer.script = [status, 200]
    with application(peer) as (app, clock):
        budget = router_transport._RequestBudget(app)
        plan = mr.Plan(chunks=iter([b"event: response.created\n\n"]))
        app._limit_plan(plan, budget)
        assert next(plan.chunks)  # caller has received an adapter-synthesized event
        assert budget.output_started
        transport = router_transport._GenerationTransport(budget, app.provider("chat"))
        try:
            request = httpx.Request("POST", f"http://127.0.0.1:{peer.port}/v1/chat/completions",
                                    json={"model": "test", "messages": []})
            response = transport.handle_request(request)
            assert response.status_code == status
            response.close()
            assert peer.attempts == 1 and clock.sleeps == []
        finally:
            transport.close()
            plan.close()


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ConnectTimeout])
def test_connection_failure_after_synthetic_output_is_not_replayed(peer, monkeypatch, failure):
    with application(peer) as (app, clock):
        budget = router_transport._RequestBudget(app)
        budget.output_started = True
        transport = router_transport._GenerationTransport(budget, app.provider("chat"))
        attempts = []
        def failed(*args, **kwargs):
            attempts.append(1)
            raise failure("controlled pre-send failure after synthetic output")
        monkeypatch.setattr(transport.inner, "send", failed)
        try:
            with pytest.raises(failure):
                transport.handle_request(httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions",
                                                       json={"model": "test"}))
            assert attempts == [1] and clock.sleeps == []
        finally:
            transport.close()
