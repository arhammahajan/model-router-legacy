"""Local socket tests for admission, phase deadlines and capacity recovery."""

from __future__ import annotations

import http.client
import json
import queue
import socket
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import httpx
import pytest

import model_router as mr


def config(**server: Any) -> mr.Config:
    return mr.build_config({
        "server": server,
        "providers": {"local": {"dialect": "openai-chat",
                                  "base_url": "http://127.0.0.1:1/v1",
                                  "api_key": "dummy"}},
        "models": [{"name": "test", "provider": "local", "model": "test"}],
    })


class ObservedServer(mr.RouterHTTPServer):
    def __init__(self, *args: Any) -> None:
        self.started: queue.Queue[socket.socket] = queue.Queue()
        self.finished: queue.Queue[socket.socket] = queue.Queue()
        self.errors: queue.Queue[Any] = queue.Queue()
        self.thread_count = 0
        super().__init__(*args)

    def process_request_thread(self, request: socket.socket, address: Any) -> None:
        # Force response backpressure quickly, independent of host defaults.
        request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        self.thread_count += 1
        self.started.put(request)
        super().process_request_thread(request, address)
        self.finished.put(request)

    def handle_error(self, request: Any, address: Any) -> None:
        import sys
        self.errors.put(sys.exc_info()[1])


@contextmanager
def running_router(**limits: Any) -> Iterator[tuple[mr.App, ObservedServer]]:
    settings = {"max_connections": 4, "max_generations": 1,
                "header_read_timeout": 0.3, "body_read_timeout": 0.3,
                "write_timeout": 0.15}
    settings.update(limits)
    app = mr.App(config(**settings))
    app.dispatch = lambda *args: mr.Plan(body=b'{"ok":true}')
    handler = type("TestHandler", (mr.Handler,), {"app": app})
    server = ObservedServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield app, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        app.close()
        assert not thread.is_alive()
        assert server.errors.empty(), list(server.errors.queue)


def connect(server: ObservedServer) -> socket.socket:
    return socket.create_connection(server.server_address, timeout=2)


def response(sock: socket.socket) -> tuple[int, dict[str, str], bytes]:
    reply = http.client.HTTPResponse(sock)
    reply.begin()
    data = reply.read()
    return reply.status, dict(reply.getheaders()), data


def post(sock: socket.socket, payload: Any = None, path: str = "/v1/chat/completions") -> None:
    if payload is None:
        payload = {"model": "test", "messages": []}
    body = json.dumps(payload).encode()
    sock.sendall(f"POST {path} HTTP/1.1\r\nHost: local\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)


def ordinary(server: ObservedServer) -> None:
    with connect(server) as sock:
        post(sock)
        assert response(sock)[0] == 200
    server.finished.get(timeout=2)
    assert not server._connections
    assert not server._generations


@pytest.mark.parametrize("name", ["max_connections", "max_generations"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_invalid_capacity(name: str, value: Any) -> None:
    with pytest.raises(SystemExit, match=name):
        config(**{name: value})


@pytest.mark.parametrize("name", ["header_read_timeout", "body_read_timeout", "write_timeout"])
@pytest.mark.parametrize("value", [0, -1, True, "2", None, float("nan"), float("inf")])
def test_invalid_deadline(name: str, value: Any) -> None:
    with pytest.raises(SystemExit, match=name):
        config(**{name: value})


def test_defaults_and_cross_limit_validation() -> None:
    cfg = config()
    assert (cfg.max_connections, cfg.max_generations) == (32, 8)
    assert (cfg.header_read_timeout, cfg.body_read_timeout, cfg.write_timeout) == (10, 30, 30)
    with pytest.raises(SystemExit, match="must not exceed"):
        config(max_connections=1, max_generations=2)


def test_connection_overload_is_rejected_before_threads_start() -> None:
    with running_router(max_connections=2, header_read_timeout=2) as (_, server):
        with connect(server) as first, connect(server) as second:
            first.sendall(b"GET / HTTP/1.1\r\nHost:")
            second.sendall(b"GET / HTTP/1.1\r\nHost:")
            server.started.get(timeout=2)
            server.started.get(timeout=2)
            for _ in range(12):
                with connect(server) as excess:
                    status, headers, _ = response(excess)
                    assert status == 503
                    assert headers["Retry-After"] == "1"
            assert server.thread_count == 2
            assert len(server._connections) == 2
        server.finished.get(timeout=2)
        server.finished.get(timeout=2)
        ordinary(server)


@pytest.mark.parametrize("wire,code", [
    (b"", b"header_read_timeout"),
    (b"POST /v1/chat/completions HTTP/1.1\r\nHost:", b"header_read_timeout"),
    (b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 40\r\n\r\n{", b"body_read_timeout"),
    (b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n20\r\nx", b"body_read_timeout"),
    (b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX:", b"body_read_timeout"),
])
def test_stalled_request_deadlines_release_capacity(wire: bytes, code: bytes) -> None:
    with running_router() as (_, server):
        with connect(server) as sock:
            sock.sendall(wire)
            status, headers, body = response(sock)
            assert status == 408
            assert headers["Connection"] == "close"
            assert code in body
        server.finished.get(timeout=2)
        ordinary(server)


def test_deadline_is_absolute_even_with_continuous_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    class TrickleSocket:
        def settimeout(self, timeout: float) -> None:
            assert 0 < timeout <= 1
        def recv_into(self, buffer: Any) -> int:
            now[0] += 0.4  # Every receive progresses before its inactivity timeout.
            buffer[0] = ord("x")
            return 1
    monkeypatch.setattr(mr.time, "monotonic", lambda: now[0])
    reader = mr._DeadlineReader(TrickleSocket())
    reader.deadline = 101.0
    with pytest.raises(mr._ReadDeadlineExceeded):
        # One buffered call invokes multiple socket reads, each deadline checked.
        import io
        io.BufferedReader(reader).read(20)
    assert now[0] < 101.7


def test_held_generation_rejects_without_discovery_and_recovers() -> None:
    entered, release = threading.Event(), threading.Event()
    with running_router(header_read_timeout=0.05, body_read_timeout=0.05,
                        write_timeout=0.05) as (app, server):
        def dispatch(*args: Any) -> mr.Plan:
            entered.set()
            assert release.wait(2)
            return mr.Plan(body=b'{"ok":true}')
        app.dispatch = dispatch
        try:
            with connect(server) as held:
                post(held)
                assert entered.wait(2)
                original_resolve = app.resolve
                def forbidden_resolve(model: str) -> None:
                    pytest.fail("overloaded work reached discovery/route resolution")
                app.resolve = forbidden_resolve
                with connect(server) as excess:
                    post(excess)
                    status, headers, body = response(excess)
                    assert status == 503
                    assert headers["Retry-After"] == "1"
                    assert b"generation_overloaded" in body
                server.finished.get(timeout=2)
                # Independent locks remain available while generation is held.
                assert app._catalog_lock.acquire(blocking=False)
                app._catalog_lock.release()
                assert app._llama_lock.acquire(blocking=False)
                app._llama_lock.release()
                assert len(server._generations) == 1
                # Inbound and downstream write timeouts must not cap generation.
                assert not release.wait(0.2)
                release.set()
                assert response(held)[0] == 200
                server.finished.get(timeout=2)
                app.resolve = original_resolve
                ordinary(server)
        finally:
            release.set()


@pytest.mark.parametrize("payload,status", [
    ([], 400), ({"model": "test", "messages": "bad"}, 400),
    ({"model": "missing", "messages": []}, 404),
])
def test_validation_failures_do_not_leak_capacity(payload: Any, status: int) -> None:
    with running_router() as (_, server):
        with connect(server) as sock:
            post(sock, payload)
            assert response(sock)[0] == status
        server.finished.get(timeout=2)
        ordinary(server)


@pytest.mark.parametrize("failure,status", [
    (RuntimeError("local failure"), 502), (httpx.ReadTimeout("upstream stalled"), 504),
    (TimeoutError("adapter stalled"), 504),
])
def test_dispatch_failures_release_capacity(failure: Exception, status: int) -> None:
    with running_router() as (app, server):
        original = app.dispatch
        def dispatch(*args: Any) -> None:
            raise failure
        app.dispatch = dispatch
        with connect(server) as sock:
            post(sock)
            assert response(sock)[0] == status
        server.finished.get(timeout=2)
        app.dispatch = original
        ordinary(server)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("disconnect", [False, True])
def test_stalled_response_reader_or_disconnect_releases_generation(
    asynchronous: bool, disconnect: bool,
) -> None:
    closed = threading.Event()
    with running_router() as (app, server):
        original = app.dispatch
        block = b"data: " + b"x" * (1024 * 1024) + b"\n\n"
        def chunks() -> Iterator[bytes]:
            try:
                for _ in range(16):
                    yield block
            finally:
                closed.set()
        async def async_chunks() -> Any:
            try:
                for _ in range(16):
                    yield block
            finally:
                closed.set()
        app.dispatch = lambda *args: mr.Plan(
            chunks=None if asynchronous else chunks(),
            async_chunks=async_chunks() if asynchronous else None,
            content_type="text/event-stream")
        with connect(server) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            post(sock)
            server.started.get(timeout=2)
            if disconnect:
                sock.shutdown(socket.SHUT_RDWR)
            # With no reads, a 16 MiB stream cannot fit these socket buffers.
            assert closed.wait(2)
            server.finished.get(timeout=2)
        assert not server._generations
        app.dispatch = original
        ordinary(server)


def test_stream_may_pause_longer_than_inbound_and_write_timeouts() -> None:
    release = threading.Event()
    with running_router(header_read_timeout=0.03, body_read_timeout=0.03,
                        write_timeout=0.03) as (app, server):
        def chunks() -> Iterator[bytes]:
            yield b"data: first\n\n"
            assert release.wait(2)
            yield b"data: last\n\n"
        app.dispatch = lambda *args: mr.Plan(chunks=chunks(), content_type="text/event-stream")
        try:
            with connect(server) as sock:
                post(sock)
                reply = http.client.HTTPResponse(sock)
                reply.begin()
                assert reply.readline() == b"data: first\n"
                assert not release.wait(0.15)
                release.set()
                assert b"data: last" in reply.read()
            server.finished.get(timeout=2)
            assert not server._generations
        finally:
            release.set()


def test_shutdown_unblocks_stalled_headers_and_writes() -> None:
    closed = threading.Event()
    with running_router(header_read_timeout=60, write_timeout=60) as (app, server):
        def chunks() -> Iterator[bytes]:
            try:
                for _ in range(32):
                    yield b"x" * (1024 * 1024)
            finally:
                closed.set()
        app.dispatch = lambda *args: mr.Plan(chunks=chunks())
        with connect(server) as headers, connect(server) as writer:
            headers.sendall(b"GET / HTTP/1.1\r\nX:")
            writer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            post(writer)
            server.started.get(timeout=2)
            server.started.get(timeout=2)
            # Wait until streaming headers arrive, then leave the body unread.
            assert writer.recv(1) == b"H"
            server.shutdown()
            assert closed.wait(2)
            server.finished.get(timeout=2)
            server.finished.get(timeout=2)
            assert not server._connections
            assert not server._generations


@pytest.mark.parametrize("cancel", [False, True])
def test_real_upstream_timeout_or_client_cancellation_recovers(cancel: bool) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    entered, release = threading.Event(), threading.Event()
    done: queue.Queue[None] = queue.Queue()
    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            try:
                if not entered.is_set():
                    entered.set()
                    assert release.wait(2)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
            except OSError:
                pass  # Expected when the router times out its upstream read.
            finally:
                done.put(None)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever,
                                      kwargs={"poll_interval": 0.01}, daemon=True)
    upstream_thread.start()
    try:
        with running_router(header_read_timeout=0.03, body_read_timeout=0.03) as (app, server):
            app.dispatch = mr.App.dispatch.__get__(app)
            old_provider = app.config.providers["local"]
            from dataclasses import replace
            app.config.providers["local"] = replace(
                old_provider, base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
                timeout=1 if cancel else 0.2)
            # Test setup changes become visible through registry publication;
            # an existing snapshot deliberately retains its provider settings.
            app._publish_registry()
            with connect(server) as held:
                post(held)
                assert entered.wait(2)
                if cancel:
                    held.shutdown(socket.SHUT_RDWR)
                    # Do not recycle a slot while the actual upstream call is
                    # still running: repeated disconnects must not bypass limits.
                    with connect(server) as excess:
                        post(excess)
                        assert response(excess)[0] == 503
                    server.finished.get(timeout=2)
                    release.set()
                else:
                    assert response(held)[0] == 504
                    release.set()
                done.get(timeout=2)
                server.finished.get(timeout=2)
            ordinary(server)
            done.get(timeout=2)
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(2)
        assert not upstream_thread.is_alive()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_at", [0, 1])
def test_stream_exceptions_release_generation(asynchronous: bool, failure_at: int) -> None:
    closed = threading.Event()
    with running_router() as (app, server):
        original = app.dispatch
        def chunks() -> Iterator[bytes]:
            try:
                if failure_at:
                    yield b"data: first\n\n"
                raise RuntimeError("local stream failure")
            finally:
                closed.set()
        async def async_chunks() -> Any:
            try:
                if failure_at:
                    yield b"data: first\n\n"
                raise RuntimeError("local stream failure")
            finally:
                closed.set()
        app.dispatch = lambda *args: mr.Plan(
            chunks=None if asynchronous else chunks(),
            async_chunks=async_chunks() if asynchronous else None,
            content_type="text/event-stream")
        with connect(server) as sock:
            post(sock)
            status, _, body = response(sock)
            assert status == (200 if failure_at else 502)
            assert b"local stream failure" in body
        assert closed.wait(2)
        server.finished.get(timeout=2)
        app.dispatch = original
        ordinary(server)


def test_thread_start_failure_returns_connection_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    with running_router() as (_, server):
        accepted, peer = socket.socketpair()
        with accepted, peer:
            def fail_start(thread: threading.Thread) -> None:
                raise RuntimeError("cannot start handler")
            with monkeypatch.context() as patch:
                patch.setattr(threading.Thread, "start", fail_start)
                with pytest.raises(RuntimeError, match="cannot start handler"):
                    server.process_request(accepted, ("local", 0))
            assert not server._connections
            assert not server._generations
        ordinary(server)
