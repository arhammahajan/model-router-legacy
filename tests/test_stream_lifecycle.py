"""Streaming contracts: event state, request ownership and local HTTP aborts."""

from __future__ import annotations

import asyncio
import io
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import httpx
import pytest

import model_router as mr
import model_router.adapters as router_adapters
import model_router.common as router_common
import model_router.streams as router_streams
from test_http_capacity import ObservedServer, connect, post, running_router
from test_model_router import (
    INGRESSES, PATHS, _anthropic_sse, _body, _chat_sse, _config,
    _parse_sse, _responses_sse, _with_gemini, _gemini_response,
)


class Tracked:
    def __init__(self, values: list[Any], fail: bool = False) -> None:
        self.values = iter(values)
        self.fail = fail
        self.reads = 0
        self.closes = 0

    def __iter__(self) -> Tracked:
        return self

    def __next__(self) -> Any:
        self.reads += 1
        try:
            return next(self.values)
        except StopIteration:
            if self.fail:
                raise RuntimeError("controlled stream failure")
            raise

    def close(self) -> None:
        self.closes += 1


class AsyncTracked(Tracked):
    def __aiter__(self) -> AsyncTracked:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closes += 1


def event(ingress: str) -> bytes:
    payload = {
        "openai-chat": {"choices": [{"index": 0, "delta": {"content": "hello"}}]},
        "openai-responses": {"type": "response.output_text.delta", "delta": "hello"},
        "anthropic": {"type": "message_start", "message": {"id": "msg_test"}},
    }[ingress]
    name = b"" if ingress == "openai-chat" else f"event: {payload['type']}\n".encode()
    return name + b"data: " + json.dumps(payload).encode() + b"\n\n"






@pytest.mark.parametrize("asynchronous", [False, True])
def test_unstarted_plan_closes_hidden_source(asynchronous: bool) -> None:
    source = AsyncTracked([]) if asynchronous else Tracked([])
    async def async_chunks() -> Any:
        async for part in source:
            yield part
    plan = mr.Plan(resources=(source,),
                   chunks=None if asynchronous else router_streams._chat_sse(source),
                   async_chunks=async_chunks() if asynchronous else None)
    plan.close()
    plan.close()
    assert source.closes == 1
    assert source.reads == 0




@pytest.mark.parametrize("asynchronous", [False, True])
def test_disconnect_stops_consumption_and_closes_every_owner(asynchronous: bool) -> None:
    block = b"data: " + b"x" * (1024 * 1024) + b"\n\n"
    source = (AsyncTracked if asynchronous else Tracked)([block] * 100)
    transport = Tracked([])
    with running_router() as (app, server):
        plan = mr.Plan(resources=(transport,), chunks=None if asynchronous else source,
                       async_chunks=source if asynchronous else None)
        app.dispatch = lambda *args: plan
        with connect(server) as sock:
            post(sock)
            # Headers prove the plan exists; then a disconnect fails the write.
            assert sock.recv(1) == b"H"
            sock.shutdown(socket.SHUT_RDWR)
            server.finished.get(timeout=3)
        assert source.closes == transport.closes == 1
        assert source.reads < 100
        app.dispatch = lambda *args: mr.Plan(body=b"{}")
        reply = httpx.post(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                           json={"model": "test", "messages": []})
        server.finished.get(timeout=3)
        assert reply.status_code == 200
        assert not server._generations


def assert_success(ingress: str, reply: httpx.Response, *, translated: bool = True) -> None:
    frames = _parse_sse(reply.text)
    payloads = [json.loads(frame["data"]) for frame in frames if frame["data"] != "[DONE]"]
    assert "hello" in reply.text
    if ingress == "openai-chat":
        assert frames[-1] == {"data": "[DONE]"}
        assert any(c.get("finish_reason") == "stop" for p in payloads for c in p.get("choices", []))
    elif ingress == "openai-responses":
        assert frames[-1]["event"] == "response.completed"
        assert payloads[-1]["response"]["status"] == "completed"
        if translated:
            assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
    else:
        assert frames[-1]["event"] == "message_stop"
        assert payloads[-1] == {"type": "message_stop"}
        assert payloads[-2]["type"] == "message_delta"
        assert payloads[-2]["delta"]["stop_reason"] == "end_turn"


class ScriptedUpstream:
    """Real SSE with deterministic truncation and enough data for adapters."""
    def __init__(self) -> None:
        upstream = self
        self.mode = "complete"
        self.aborted = threading.Event()
        self.produced = 0
        self.last_wire = b""
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args: Any) -> None:
                pass
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                if self.path.startswith("/v1beta/"):
                    payload = _gemini_response()
                    payload["candidates"][0]["content"]["parts"] = [{"text": "hello"}]
                    frames = [f"data: {json.dumps(payload)}\n\n".encode()]
                elif self.path == "/v1/chat/completions":
                    frames = [f"data: {p}\n\n".encode() for p in _chat_sse("hello")]
                    frames.append(b"data: [DONE]\n\n")
                else:
                    events = (_responses_sse("hello") if self.path == "/v1/responses"
                              else _anthropic_sse("hello"))
                    frames = [f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
                              for name, payload in events]
                mode = upstream.mode
                if mode == "disconnect":
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    if self.path.startswith("/v1beta/"):
                        payload["candidates"][0].pop("finishReason")
                        payload["candidates"][0]["content"]["parts"] = [{"text": "x" * 32768}]
                        delta = f"data: {json.dumps(payload)}\n\n".encode()
                    elif self.path == "/v1/chat/completions":
                        delta = f"data: {_chat_sse('x' * 32768)[0]}\n\n".encode()
                    elif self.path == "/v1/responses":
                        self.wfile.write(b"".join(frames[:4]))
                        name, payload = _responses_sse("x" * 32768)[4]
                        delta = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
                    else:
                        self.wfile.write(b"".join(frames[:2]))
                        name, payload = _anthropic_sse("x" * 32768)[2]
                        delta = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
                    try:
                        for _ in range(4096):
                            self.wfile.write(delta)
                            self.wfile.flush()
                            upstream.produced += 1
                    except OSError:
                        upstream.aborted.set()
                    return
                if mode in ("empty", "first"):
                    wire = b""
                elif mode == "midstream" and self.path.startswith("/v1beta/"):
                    payload["candidates"][0].pop("finishReason")
                    wire = f"data: {json.dumps(payload)}\n\n".encode()
                elif mode == "midstream":
                    wire = b"".join(frames[:5 if self.path == "/v1/responses" else 1 if self.path == "/v1/chat/completions" else 3])
                else:
                    wire = b"".join(frames)
                if mode == "fidelity":
                    wire = (b": heartbeat\nretry: 1000\n\n" +
                            b"".join(f'event: vendor.custom\ndata: {{"type":"vendor.custom","sequence_number":{n}}}\n\n'.encode()
                                     for n in (3, 3, 1)) +
                            b'event: vendor.final\ndata: {"type":"vendor.final"}\n\n')
                upstream.last_wire = wire
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire) + (100 if mode in ("first", "midstream") else 0)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    self.wfile.write(wire)
                    self.wfile.flush()
                except OSError:
                    pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.port = self.server.server_port


@contextmanager
def local_streams() -> Iterator[tuple[ScriptedUpstream, ObservedServer, str]]:
    upstream = ScriptedUpstream()
    upstream.thread.start()
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["litellm_kwargs"] = {"num_retries": 0}
    raw["models"][2]["model"] = "gpt-4.1"  # real Responses SSE, not simulated output
    app = mr.App(mr.build_config(raw))
    server = ObservedServer(("127.0.0.1", 0), type("StreamHandler", (mr.Handler,), {"app": app}))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield upstream, server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        app.close()
        upstream.server.shutdown()
        upstream.server.server_close()
        upstream.thread.join(3)
        assert server.errors.empty(), list(server.errors.queue)






@pytest.mark.parametrize("ingress,model", [
    ("openai-chat", "m-chat"), ("openai-responses", "m-resp"), ("anthropic", "m-anth"),
])
def test_native_sse_bytes_and_vendor_sequences_are_opaque(ingress: str, model: str) -> None:
    with local_streams() as (upstream, server, base):
        for _ in range(2):
            upstream.mode = "fidelity"
            reply = httpx.post(base + PATHS[ingress], json=_body(ingress, model, True))
            server.finished.get(timeout=3)
            assert reply.status_code == 200
            assert reply.content == upstream.last_wire
            assert [json.loads(f["data"])["sequence_number"] for f in _parse_sse(reply.text)[:-1]] == [3, 3, 1]




def test_cleanup_failure_does_not_skip_other_resources() -> None:
    class Broken(Tracked):
        def close(self) -> None:
            super().close()
            raise RuntimeError("cleanup failed")
    broken, good = Broken([]), Tracked([])
    plan = mr.Plan(chunks=broken, resources=(good, broken))
    plan.close()
    plan.close()
    assert broken.closes == good.closes == 1


@pytest.mark.parametrize("ingress", INGRESSES)
def test_translated_error_object_before_first_event_returns_json(ingress: str) -> None:
    state = mr.StreamState()
    if ingress == "openai-responses":
        source = Tracked([{"type": "error", "code": "server_error", "message": "first failure"}])
        chunks = router_streams._responses_sse(source, state=state)
    elif ingress == "openai-chat":
        source = Tracked([{"error": {"type": "api_error", "message": "first failure"}}])
        chunks = router_streams._chat_sse(source, state=state)
    else:
        source = Tracked([b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"first failure"}}\n\n'])
        chunks = (router_streams._translated_sse(chunk, router_common._DEFAULT_SANITIZER, state) for chunk in source)
    plan = mr.Plan(chunks=chunks, resources=(source,), stream_state=state)
    with running_router() as (app, server):
        app.dispatch = lambda *args: plan
        reply = httpx.post(f"http://127.0.0.1:{server.server_port}" + PATHS[ingress],
                           json=_body(ingress, "test", True))
        server.finished.get(timeout=3)
    assert reply.status_code == 502
    assert reply.headers["content-type"] == "application/json"
    assert reply.json()["error"]["message"] == "first failure"
    assert source.closes == source.reads == 1


def test_repeated_cancellation_during_cleanup_releases_remaining_owners() -> None:
    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        class SlowClose(AsyncTracked):
            async def aclose(self) -> None:
                self.closes += 1
                entered.set()
                await release.wait()
        source, transport = SlowClose([]), AsyncTracked([])
        plan = mr.Plan(async_chunks=source, resources=(transport, source))
        handler = object.__new__(mr.Handler)
        handler._send_stream_headers = lambda plan: None
        task = asyncio.create_task(handler._relay_async(plan, "anthropic"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)  # schedule the first cancellation, without a timed wait
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await plan.aclose()
        assert source.closes == transport.closes == 1
        assert plan._closed
        assert handler._stream_outcome == "cancelled"
    asyncio.run(exercise())


@pytest.mark.parametrize("ingress,model", [
    ("openai-chat", "m-anth"), ("openai-chat", "m-resp"), ("openai-chat", "m-gem"),
    ("openai-responses", "m-chat"), ("openai-responses", "m-anth"), ("openai-responses", "m-gem"),
    ("anthropic", "m-chat"), ("anthropic", "m-resp"), ("anthropic", "m-gem"),
])
def test_cancelled_consumer_closes_real_adapter_plan_and_upstream(
    ingress: str, model: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel an embedder between pulls, with real adapter/HTTP resources open."""
    with local_streams() as (upstream, server, _):
        responses = []
        original_send = httpx.Client.send
        def send(client, request, **kwargs):
            reply = original_send(client, request, **kwargs)
            if request.url.port == upstream.port:
                responses.append(reply)
            return reply
        monkeypatch.setattr(httpx.Client, "send", send)
        upstream.mode = "disconnect"
        app = server.app
        plan = app.dispatch(ingress, app.resolve(model), _body(ingress, model, True), {})

        async def run():
            acquired = asyncio.Event()
            hold = asyncio.Event()
            async def consume():
                try:
                    if plan.async_chunks is not None:
                        async for _ in plan.async_chunks:
                            if responses:
                                acquired.set()
                                await hold.wait()
                    else:
                        for _ in plan.chunks:
                            if responses:
                                acquired.set()
                                await hold.wait()
                finally:
                    await plan.aclose()
            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(acquired.wait(), 3)
                assert responses and not all(r.is_closed for r in responses)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
                await plan.aclose()  # idempotent after cancellation
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await plan.aclose()

        asyncio.run(run())
        assert responses and all(r.is_closed for r in responses)
        assert upstream.aborted.wait(5), f"upstream consumed {upstream.produced} events"
        assert upstream.produced < 4096
        upstream.mode = "complete"
        reply = httpx.post(f"http://127.0.0.1:{server.server_port}" + PATHS[ingress],
                          json=_body(ingress, model, True), timeout=15)
        server.finished.get(timeout=3)
        assert_success(ingress, reply)
        assert all(r.is_closed for r in responses)
