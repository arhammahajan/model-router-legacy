"""Bounded HTTP server, framing, request handling and response/stream relay."""
from __future__ import annotations

import asyncio
import io
import json
import re
import socket
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlsplit

from .common import (
    RouterError,
    _error_payload,
    _json_bytes,
    _status_for,
    _stream_error_bytes,
    _upstream_error_headers,
)
from .constants import (
    _CHUNK_LINE,
    _DROP_REQUEST_HEADERS,
    _TRAILER_LINE,
    DIALECT_OPENAI_CHAT,
    ENDPOINTS,
    MAX_CHUNK_LINE,
    MAX_CHUNK_METADATA,
    MAX_CHUNK_TRAILERS,
    __version__,
)
from .protocol import _validate_ingress
from .streams import Plan

if TYPE_CHECKING:
    from .app import App


class _ReadDeadlineExceeded(Exception):
    pass


class _ClientWriteTimeout(OSError):
    pass


class _ResponseWriter(io.BufferedIOBase):
    def __init__(self, connection: socket.socket, timeout: float) -> None:
        super().__init__()
        self.connection = connection
        self.timeout = timeout

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        self.connection.settimeout(self.timeout)
        try:
            self.connection.sendall(data)
        except TimeoutError as exc:
            raise _ClientWriteTimeout from exc
        return len(data)


class _DeadlineReader(io.RawIOBase):
    """Check the absolute phase deadline before *each* socket receive.

    A socket inactivity timeout alone lets a slow trickle hold a handler forever.
    BufferedReader may perform several receives inside one read()/readline().
    """

    def __init__(self, connection: socket.socket) -> None:
        super().__init__()
        self.connection = connection
        self.deadline: float | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise _ReadDeadlineExceeded
            self.connection.settimeout(remaining)
        try:
            return self.connection.recv_into(buffer)
        except TimeoutError as exc:
            raise _ReadDeadlineExceeded from exc


class RouterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], handler: type[Handler]) -> None:
        self.app = handler.app
        self.config = handler.app.config
        self._capacity_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._generations: set[socket.socket] = set()
        self._closing = False
        super().__init__(address, handler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        # Admission happens in the accept loop, before ThreadingMixIn can spawn.
        with self._capacity_lock:
            admitted = not self._closing and len(self._connections) < self.config.max_connections
            if admitted:
                self._connections.add(request)
        if not admitted:
            try:
                # Never stall the accept loop on an overloaded, non-reading peer.
                request.setblocking(False)
                request.send(b"HTTP/1.1 503 Service Unavailable\r\n"
                             b"Connection: close\r\nContent-Length: 0\r\n"
                             b"Retry-After: 1\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_connection(request)
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_connection(request)

    def _release_connection(self, request: socket.socket) -> None:
        with self._capacity_lock:
            self._generations.discard(request)
            self._connections.discard(request)

    def acquire_generation(self, request: socket.socket) -> bool:
        with self._capacity_lock:
            if self._closing or len(self._generations) >= self.config.max_generations:
                return False
            self._generations.add(request)
            return True

    def release_generation(self, request: socket.socket) -> None:
        with self._capacity_lock:
            self._generations.discard(request)

    def _stop_connections(self) -> None:
        with self._capacity_lock:
            self._closing = True
            connections = list(self._connections)
            self._connections.clear()
            self._generations.clear()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def shutdown(self) -> None:
        self._stop_connections()
        super().shutdown()
        self.app.stop()

    def server_close(self) -> None:
        self._stop_connections()
        try:
            super().server_close()
        finally:
            self.app.stop()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"litellm-model-router/{__version__}"
    app: App  # provided by serve()/main()

    def setup(self) -> None:
        super().setup()
        self.rfile.close()
        self.wfile.close()
        self._reader = _DeadlineReader(self.connection)
        self.rfile = io.BufferedReader(self._reader)
        self.wfile = _ResponseWriter(self.connection, self.app.config.write_timeout)
        self.connection.settimeout(self.app.config.write_timeout)

    def handle_one_request(self) -> None:
        started = time.monotonic()
        self._request_id = uuid.uuid4().hex
        self._headers_sent = False
        self._http_status = None
        self._stream_outcome = "none"
        self._stream_state = None
        self._streamed_request = False
        self._selection: dict[str, Any] = {
            "provider": None, "model": None, "upstream_model": None, "translation_mode": None,
        }
        self.requestline = ""
        self.request_version = "HTTP/1.1"
        self.command = None
        self._reader.deadline = time.monotonic() + self.app.config.header_read_timeout
        try:
            # The stdlib catches TimeoutError without sending a response. Our
            # distinct read exception lets us return an observable HTTP 408.
            super().handle_one_request()
        except _ReadDeadlineExceeded:
            try:
                self._send_error(DIALECT_OPENAI_CHAT, RouterError(
                    408, "Request headers read deadline exceeded", code="header_read_timeout"
                ))
            except OSError:
                pass
        except OSError:
            pass
        finally:
            self.close_connection = True
            self._reader.deadline = None
            if self.command is not None:
                # Log only recognized paths and resolved, server-owned route
                # identities. Never echo URL queries or an arbitrary model ID.
                path = urlsplit(getattr(self, "path", "")).path.rstrip("/") or "/"
                known = path in ENDPOINTS or path in {
                    "/", "/models", "/v1/models", "/livez", "/readyz",
                    "/health", "/healthz", "/diagnostics",
                }
                self.app.log_event(
                    event="request_outcome", request_id=self._request_id,
                    method=self.command if self.command in {"GET", "POST"} else "other",
                    path=path if known else "unknown", http_status=self._http_status,
                    stream=self._streamed_request, stream_outcome=self._stream_outcome,
                    elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                    **self._selection,
                    discovery=self.app.discovery_summary(),
                )

    def send_response(self, code: int, message: str | None = None) -> None:
        self._http_status = code
        super().send_response(code, message)

    def end_headers(self) -> None:
        self.send_header("X-Request-ID", self._request_id)
        super().end_headers()

    def parse_request(self) -> bool:
        parsed = super().parse_request()
        self._reader.deadline = None
        self.connection.settimeout(self.app.config.write_timeout)
        return parsed

    # -- logging ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        # stdlib access messages contain arbitrary request lines and queries.
        # request_outcome is the sole access log, including in verbose mode.
        pass

    # -- entry points ----------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path in ("/livez", "/health", "/healthz"):
                return self._send_bytes(
                    200,
                    "application/json",
                    {},
                    _json_bytes(self.app.liveness_payload()),
                )
            if path == "/readyz":
                status = self.app.readiness_payload()["status"]
                return self._send_bytes(200 if status == "ready" else 503,
                                        "application/json", {}, _json_bytes({"status": status}))
            if path == "/diagnostics":
                if not self.app.config.api_keys:
                    return self._send_error(DIALECT_OPENAI_CHAT, RouterError(
                        403, "Diagnostics require configured router authentication",
                        code="diagnostics_disabled"))
                if not self.app.authorized(self.headers):
                    return self._send_error(DIALECT_OPENAI_CHAT, RouterError(401, "Invalid API key"))
                return self._send_bytes(200, "application/json", {},
                                        _json_bytes(self.app.diagnostics_payload()))
            if path in ("/v1/models", "/models"):
                if not self.app.authorized(self.headers):
                    return self._send_error(
                        DIALECT_OPENAI_CHAT,
                        RouterError(401, "Invalid API key"),
                    )
                return self._send_bytes(
                    200, "application/json", {}, _json_bytes(self.app.models_payload())
                )
            if path == "/":
                return self._send_bytes(
                    200,
                    "application/json",
                    {},
                    _json_bytes(
                        {
                            "name": "litellm-model-router",
                            "version": __version__,
                            "endpoints": sorted(ENDPOINTS),
                        }
                    ),
                )
            self._send_error(
                DIALECT_OPENAI_CHAT,
                RouterError(404, f"Unknown route: {path}"),
            )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path
        ingress = ENDPOINTS.get(path)
        if ingress is None:
            return self._send_error(
                DIALECT_OPENAI_CHAT, RouterError(404, f"Unknown route: {path}")
            )

        if not self.app.authorized(self.headers):
            return self._send_error(
                ingress,
                RouterError(401, "Invalid API key"),
            )

        self._streamed_request = False
        generation_acquired = False
        try:
            self._reader.deadline = time.monotonic() + self.app.config.body_read_timeout
            try:
                raw = self._read_body()
            except _ReadDeadlineExceeded as exc:
                raise RouterError(408, "Request body read deadline exceeded",
                                  code="body_read_timeout") from exc
            finally:
                self._reader.deadline = None
                self.connection.settimeout(self.app.config.write_timeout)
            if not raw:
                raise RouterError(400, "A JSON request body is required")
            try:
                payload = json.loads(raw)
            except (ValueError, RecursionError) as exc:
                raise RouterError(400, "Invalid JSON body", code="invalid_request") from exc
            if not isinstance(payload, dict):
                raise RouterError(400, "Request body must be a JSON object")

            model = payload.get("model")
            if not isinstance(model, str) or not model.strip():
                raise RouterError(
                    400, "A model ID is required", code="model_required"
                )
            self._validate(ingress, payload)
            self._streamed_request = payload.get("stream", False)
            # No waiting queue and no discovery lock is held during admission.
            # Include route resolution so discovery cannot accumulate an
            # unbounded set of generation callers either.
            generation_acquired = self.server.acquire_generation(self.connection)
            if not generation_acquired:
                raise RouterError(503, "Generation capacity exhausted",
                                  code="generation_overloaded")
            route = self.app.resolve(model)
            if route is None:
                raise RouterError(
                    404,
                    f"Model {model!r} is not configured",
                    code="model_not_found",
                )
            self._selection = {
                "provider": route.provider_name, "model": route.name,
                "upstream_model": route.upstream_model,
                "translation_mode": "passthrough" if ingress == route.dialect else "translated",
                "ingress": ingress, "egress": route.dialect,
            }
            # Dynamic upstream model names are client-supplied; omit them from
            # logs. Keep this provenance on the resolved route across refreshes.
            if route._dynamic:
                self._selection.update(model=None, upstream_model=None)

            plan = self.app.dispatch(ingress, route, payload, self.headers)
            self._send_plan(plan, ingress)
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            self._stream_outcome = "disconnected" if self._streamed_request else "none"
        except RouterError as exc:
            self._send_error(ingress, exc)
        except Exception as exc:  # noqa: BLE001 - a router must not crash
            # Upstream exceptions can echo request/response bodies. The outcome
            # record carries status and selection without logging their text.
            self._send_error(ingress, exc)
        finally:
            if generation_acquired:
                self.server.release_generation(self.connection)

    # -- request parsing --------------------------------------------------

    def _validate(self, ingress: str, payload: Mapping[str, Any]) -> None:
        _validate_ingress(ingress, payload)

    def _read_body(self) -> bytes:
        max_bytes = self.app.config.max_body_bytes
        transfers = self.headers.get_all("Transfer-Encoding", [])
        lengths = self.headers.get_all("Content-Length", [])
        encodings = self.headers.get_all("Content-Encoding", [])
        if transfers and lengths:
            raise RouterError(400, "Transfer-Encoding conflicts with Content-Length")
        if len(transfers) > 1 or (transfers and transfers[0].strip(" \t").lower() != "chunked"):
            raise RouterError(400, "Only a single chunked Transfer-Encoding is supported")
        if len(lengths) > 1:
            raise RouterError(400, "Multiple Content-Length headers are not supported")
        encoding = encodings[0].strip(" \t").lower() if encodings else ""
        if len(encodings) > 1 or encoding not in ("", "identity", "gzip"):
            raise RouterError(415, "Unsupported Content-Encoding")

        if transfers:
            raw = self._read_chunked(max_bytes)
        else:
            length_text = lengths[0].strip(" \t") if lengths else "0"
            if not re.fullmatch(r"[0-9]+", length_text):
                raise RouterError(400, "Invalid Content-Length")
            # Compare before int() so arbitrarily long decimal values stay bounded.
            significant = length_text.lstrip("0") or "0"
            maximum = str(max_bytes)
            if (len(significant) > len(maximum)
                    or (len(significant) == len(maximum) and significant > maximum)):
                raise RouterError(413, "Request body is too large")
            raw = self._read_exact(int(significant))

        if encoding == "gzip":
            raw = _gunzip_limited(raw, max_bytes)

        if len(raw) > max_bytes:
            raise RouterError(413, "Request body is too large")
        return raw

    def _read_exact(self, length: int) -> bytes:
        if length < 0:
            raise RouterError(400, "Invalid body read length")
        data = bytearray()
        while len(data) < length:
            part = self.rfile.read(length - len(data))
            if not part:
                raise RouterError(400, "Truncated request body")
            data.extend(part)
        return bytes(data)

    def _read_chunked(self, max_bytes: int) -> bytes:
        data = bytearray()
        metadata = 0

        def read_line() -> bytes:
            nonlocal metadata
            if metadata >= MAX_CHUNK_METADATA:
                raise RouterError(413, "Chunk metadata is too large")
            line = self.rfile.readline(min(MAX_CHUNK_LINE, MAX_CHUNK_METADATA - metadata) + 1)
            metadata += len(line)
            if len(line) > MAX_CHUNK_LINE or metadata > MAX_CHUNK_METADATA:
                raise RouterError(413, "Chunk metadata is too large")
            if not line.endswith(b"\r\n"):
                raise RouterError(400, "Invalid or truncated chunk framing")
            return line

        while True:
            match = _CHUNK_LINE.fullmatch(read_line())
            if match is None:
                raise RouterError(400, "Invalid chunk size or extension")
            size = int(match[1], 16)
            if size == 0:
                # Validate and discard trailers; they never affect request headers.
                for count in range(MAX_CHUNK_TRAILERS + 1):
                    trailer = read_line()
                    if trailer == b"\r\n":
                        break
                    if count == MAX_CHUNK_TRAILERS:
                        raise RouterError(413, "Too many chunk trailers")
                    if _TRAILER_LINE.fullmatch(trailer) is None:
                        raise RouterError(400, "Invalid chunk trailer")
                    name = trailer.split(b":", 1)[0].decode("ascii").lower()
                    if name in _DROP_REQUEST_HEADERS:
                        raise RouterError(400, "Forbidden chunk trailer")
                break
            if size > max_bytes - len(data):
                raise RouterError(413, "Request body is too large")
            data.extend(self._read_exact(size))
            if self._read_exact(2) != b"\r\n":
                raise RouterError(400, "Invalid chunk delimiter")
            metadata += 2
        return bytes(data)

    # -- response sending -------------------------------------------------

    def _send_plan(self, plan: Plan, ingress: str) -> None:
        self._stream_state = plan.stream_state
        if plan.body is not None:
            if self._streamed_request:
                self._stream_outcome = "failed" if plan.status >= 400 else "completed"
            try:
                self._send_bytes(plan.status, plan.content_type, plan.headers, plan.body)
            finally:
                plan.close()
            return
        self._stream_outcome = "active"
        if plan.async_chunks is not None:
            asyncio.run(self._relay_async(plan, ingress))
        else:
            self._relay_sync(plan, ingress)

    def _write_relay_chunk(self, chunk: bytes) -> bool:
        try:
            self.wfile.write(chunk)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            self._stream_outcome = "disconnected"
            return False

    def _stream_finished(self, plan: Plan) -> None:
        self._stream_outcome = "failed" if plan.stream_state and plan.stream_state.error else "completed"

    def _relay_sync(self, plan: Plan, ingress: str) -> None:
        iterator = plan.chunks
        try:
            if iterator is None:
                self._send_bytes(plan.status, plan.content_type, plan.headers, b"")
                self._stream_finished(plan)
                return
            first = True
            while True:
                try:
                    chunk = next(iterator)
                    if first and plan.stream_state and plan.stream_state.error:
                        raise plan.stream_state.error
                except StopIteration:
                    if first:
                        self._send_stream_headers(plan)
                    self._stream_finished(plan)
                    return
                except Exception as exc:  # includes errors before the first byte
                    self._stream_outcome = "failed"
                    self._send_error(ingress, exc)
                    return
                if first:
                    self._send_stream_headers(plan)
                    first = False
                if not self._write_relay_chunk(chunk):
                    return
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            self._stream_outcome = "disconnected"
        finally:
            plan.close()

    async def _relay_async(self, plan: Plan, ingress: str) -> None:
        iterator = plan.async_chunks
        try:
            if iterator is None:
                self._send_bytes(plan.status, plan.content_type, plan.headers, b"")
                self._stream_finished(plan)
                return
            first = True
            while True:
                try:
                    chunk = await anext(iterator)
                    if first and plan.stream_state and plan.stream_state.error:
                        raise plan.stream_state.error
                except StopAsyncIteration:
                    if first:
                        self._send_stream_headers(plan)
                    self._stream_finished(plan)
                    return
                except Exception as exc:
                    self._stream_outcome = "failed"
                    self._send_error(ingress, exc)
                    return
                if first:
                    self._send_stream_headers(plan)
                    first = False
                if not self._write_relay_chunk(chunk):
                    return
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            self._stream_outcome = "disconnected"
        except asyncio.CancelledError:
            self._stream_outcome = "cancelled"
            raise
        finally:
            try:
                await plan.aclose()
            except asyncio.CancelledError:
                self._stream_outcome = "cancelled"
                raise

    def _send_stream_headers(self, plan: Plan) -> None:
        self._http_status = plan.status
        self.send_response(plan.status)
        self.send_header(
            "Content-Type", plan.content_type or "text/event-stream"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._send_extra_headers(plan.headers)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._headers_sent = True

    def _write_stream_error(self, ingress: str, exc: BaseException) -> None:
        self._stream_outcome = "failed"
        try:
            self.wfile.write(_stream_error_bytes(
                ingress, exc, self.app.sanitize, getattr(self, "_stream_state", None)))
            self.wfile.flush()
            if ingress == DIALECT_OPENAI_CHAT:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            pass

    def _send_extra_headers(self, headers: Mapping[str, str]) -> None:
        """Send forwarded upstream headers without duplicating managed ones."""
        managed = {"content-type", "content-length", "cache-control", "connection", "x-request-id"}
        for name, value in headers.items():
            if name.lower() not in managed:
                self.send_header(name, value)

    def _send_error(self, ingress: str, exc: BaseException) -> None:
        status = _status_for(exc)
        if getattr(self, "_streamed_request", False):
            self._stream_outcome = "failed"
        if getattr(self, "_headers_sent", False):
            self._write_stream_error(ingress, exc)
            return
        body = _json_bytes(_error_payload(ingress, exc, self.app.sanitize))
        extra = _upstream_error_headers(exc, self.app.sanitize)
        if status == 401:
            extra["WWW-Authenticate"] = 'Bearer realm="model-router"'
        if status == 503 and "retry-after" not in extra:
            extra["Retry-After"] = "1"
        self._send_bytes(status, "application/json", extra, body)

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> None:
        self._http_status = status
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_extra_headers(headers)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._headers_sent = True
        if body:
            self.wfile.write(body)
            self.wfile.flush()


def _gunzip_limited(raw: bytes, limit: int) -> bytes:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(raw, limit + 1)
    except zlib.error as exc:
        raise RouterError(400, "Invalid gzip request body") from exc
    if len(data) > limit or decompressor.unconsumed_tail:
        raise RouterError(413, "Decompressed request body is too large")
    if not decompressor.eof:
        raise RouterError(400, "Incomplete gzip request body")
    if decompressor.unused_data:
        raise RouterError(400, "Gzip request body must contain exactly one member and no trailing bytes")
    return data


def serve(app: App, host: str, port: int) -> RouterHTTPServer:
    handler_class = type("BoundHandler", (Handler,), {"app": app})
    server = RouterHTTPServer((host, port), handler_class)
    try:
        app.start()
    except BaseException:
        server.server_close()
        raise
    return server
