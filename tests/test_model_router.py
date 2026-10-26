"""Integration tests: a fake upstream, the real router, real LiteLLM adapters."""

from __future__ import annotations

from dataclasses import replace

import gzip
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx
import pytest


import model_router as mr
import model_router.adapters as router_adapters
import model_router.headers as router_headers
import model_router.protocol as router_protocol


# ---------------------------------------------------------------------------
# Fake upstream
# ---------------------------------------------------------------------------

ANTHROPIC_TEXT = "anthropic-upstream"
CHAT_TEXT = "chat-upstream"
RESPONSES_TEXT = "responses-upstream"
GEMINI_TEXT = "gemini-upstream"


def _gemini_response() -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [
            {"text": "gemini-upstream"}
        ]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3,
                          "totalTokenCount": 8},
        "modelVersion": "gemini-2.5-flash",
    }


def _anthropic_message(text: str, tool: bool = False) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if tool:
        content.append(
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {"q": "hello"},
            }
        )
    content.append({"type": "text", "text": text})
    return {
        "id": "msg_upstream",
        "type": "message",
        "role": "assistant",
        "model": "upstream-anthropic",
        "content": content,
        "stop_reason": "tool_use" if tool else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _chat_completion(text: str, tool: bool = False) -> dict[str, Any]:
    if tool:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"hello"}'},
                }
            ],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"
    return {
        "id": "chatcmpl_upstream",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-chat",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _responses_response(text: str, tool: bool = False) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if tool:
        output.append(
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"hello"}',
                "status": "completed",
            }
        )
    output.append(
        {
            "id": "msg_upstream",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": text, "annotations": []}
            ],
        }
    )
    return {
        "id": "resp_upstream",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "upstream-responses",
        "output": output,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _responses_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    base = {
        "id": "resp_upstream",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "model": "upstream-responses",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    completed = _responses_response(text)
    events = [
        ("response.created", {"type": "response.created", "response": base}),
        (
            "response.in_progress",
            {"type": "response.in_progress", "response": base},
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg_upstream",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": "msg_upstream",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_upstream",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        ),
        (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": "msg_upstream",
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        ),
        ("response.content_part.done", {"type": "response.content_part.done",
             "item_id": "msg_upstream", "output_index": 0, "content_index": 0,
             "part": completed["output"][-1]["content"][0]}),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": completed["output"][-1],
            },
        ),
        ("response.completed", {"type": "response.completed", "response": completed}),
    ]

    return [(name, {**payload, "sequence_number": i})
            for i, (name, payload) in enumerate(events)]

def _chat_sse(text: str) -> list[str]:
    return [
        json.dumps(
            {
                "id": "chatcmpl_upstream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "upstream-chat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_upstream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "upstream-chat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        ),
    ]


def _anthropic_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_upstream",
                    "type": "message",
                    "role": "assistant",
                    "model": "upstream-anthropic",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


class FakeUpstream:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.error_body: bytes | None = None
        self.error_status = 429
        self.stream_error: str | None = None
        self.response_payload: dict[str, Any] | None = None
        self.stream_frames: list[bytes] | None = None
        self.error_headers: dict[str, str] = {}
        self._lock = threading.Lock()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                wire_body = self.rfile.read(length)
                encoding = self.headers.get("Content-Encoding", "identity").lower()
                try:
                    if encoding == "gzip":
                        raw = gzip.decompress(wire_body)
                    elif encoding == "identity":
                        raw = wire_body
                    else:
                        raise ValueError("unsupported content encoding")
                    payload = json.loads(raw or b"{}")
                except (ValueError, OSError, EOFError):
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                path = self.path.split("?", 1)[0]
                with upstream._lock:
                    upstream.requests.append(
                        {
                            "path": path,
                            "body": payload,
                            "raw_body": list(wire_body),
                            "headers": {
                                key.lower(): value for key, value in self.headers.items()
                            },
                        }
                    )
                if upstream.error_body is not None:
                    self.send_response(upstream.error_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(upstream.error_body)))
                    self.send_header("Retry-After", "0")
                    for name, value in upstream.error_headers.items():
                        self.send_header(name, value)
                    self.send_header("X-Upstream-Diagnostic", "Bearer do-not-forward")
                    self.end_headers()
                    self.wfile.write(upstream.error_body)
                elif payload.get("stream") or path.endswith(":streamGenerateContent"):
                    self._stream(path)
                else:
                    self._json(path)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                with upstream._lock:
                    upstream.requests.append(
                        {
                            "path": path,
                            "body": None,
                            "headers": {
                                key.lower(): value for key, value in self.headers.items()
                            },
                        }
                    )
                if path == "/v1/models":
                    payload = {
                        "object": "list",
                        "data": [
                            {"id": "catalog-claude"},
                            {"id": "catalog-other"},
                        ],
                    }
                    data = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _json(self, path: str) -> None:
                if path.startswith("/v1beta/models/"):
                    payload = _gemini_response()
                elif path == "/v1/messages":
                    payload = _anthropic_message(ANTHROPIC_TEXT)
                elif path == "/v1/chat/completions":
                    payload = _chat_completion(CHAT_TEXT)
                elif path == "/v1/responses":
                    payload = _responses_response(RESPONSES_TEXT)
                else:
                    payload = {"error": {"message": f"unknown path {path}"}}
                if upstream.response_payload is not None:
                    payload = upstream.response_payload
                data = json.dumps(payload).encode()
                self.send_response(200 if path in _PATHS or path.startswith("/v1beta/models/") else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, path: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                if upstream.stream_frames is not None:
                    for frame in upstream.stream_frames:
                        self.wfile.write(frame)
                        self.wfile.flush()
                    return
                if upstream.stream_error is not None:
                    error = {"type": "error", "error": {
                        "type": "api_error", "code": "upstream_error",
                        "message": upstream.stream_error,
                    }}
                    if path == "/v1/messages":
                        events = _anthropic_sse(ANTHROPIC_TEXT)[:3]
                    elif path == "/v1/responses":
                        events = _responses_sse(RESPONSES_TEXT)[:5]
                        error = {"type": "error", "code": "server_error",
                                 "message": upstream.stream_error, "param": None,
                                 "sequence_number": 5}
                    else:
                        events = [("message", json.loads(_chat_sse(CHAT_TEXT)[0]))]
                    for event, payload in events + [("error", error)]:
                        self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
                    return
                if path == "/v1/messages":
                    for event, payload in _anthropic_sse(ANTHROPIC_TEXT):
                        self.wfile.write(
                            f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
                        )
                elif path == "/v1/chat/completions":
                    for payload in _chat_sse(CHAT_TEXT):
                        self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                elif path == "/v1/responses":
                    for event, payload in _responses_sse(RESPONSES_TEXT):
                        self.wfile.write(
                            f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
                        )
                elif path.startswith("/v1beta/models/"):
                    self.wfile.write(f"data: {json.dumps(_gemini_response())}\n\n".encode())
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> "FakeUpstream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def last_request(self, path: str) -> dict[str, Any]:
        with self._lock:
            for request in reversed(self.requests):
                if request["path"] == path:
                    return request
        raise AssertionError(f"No request to {path}; got {[r['path'] for r in self.requests]}")


_PATHS = {"/v1/messages", "/v1/chat/completions", "/v1/responses"}


# ---------------------------------------------------------------------------
# Fake llama.cpp / llama-swap server
# ---------------------------------------------------------------------------


class FakeLlamaServer:
    """Minimal llama.cpp server: /health, /v1/models, /props and chat."""

    def __init__(
        self,
        *,
        models: list[dict[str, Any]] | None = None,
        props: dict[str, Any] | None = None,
        api_key: str = "",
    ) -> None:
        self.models = models if models is not None else [
            {
                "id": "llama-test",
                "aliases": ["tiny"],
                "meta": {"n_ctx": 4096},
                "architecture": {"input_modalities": ["text"]},
                "status": {"value": "loaded"},
            }
        ]
        self.props = props if props is not None else {}
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def _record(self, body: Any = None) -> None:
                with fake._lock:
                    fake.requests.append(
                        {
                            "path": self.path.split("?", 1)[0],
                            "body": body,
                            "headers": {
                                key.lower(): value for key, value in self.headers.items()
                            },
                        }
                    )

            def _authorized(self) -> bool:
                if not fake.api_key:
                    return True
                return (
                    self.headers.get("Authorization", "")
                    == "Bearer " + fake.api_key
                )

            def _send(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                self._record()
                if not self._authorized():
                    self._send(401, {"error": {"message": "unauthorized"}})
                    return
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    self._send(200, {"status": "ok"})
                elif path == "/v1/models":
                    self._send(200, {"object": "list", "data": fake.models})
                elif path == "/props":
                    self._send(200, fake.props)
                else:
                    self._send(404, {"error": {"message": f"unknown path {path}"}})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._record(payload)
                if not self._authorized():
                    self._send(401, {"error": {"message": "unauthorized"}})
                    return
                if self.path.split("?", 1)[0] == "/v1/chat/completions":
                    self._send(200, _chat_completion("llama pong"))
                else:
                    self._send(404, {"error": {"message": "unknown path"}})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeLlamaServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def last_request(self, path: str) -> dict[str, Any]:
        with self._lock:
            for request in reversed(self.requests):
                if request["path"] == path:
                    return request
        raise AssertionError(f"No request to {path}; got {[r['path'] for r in self.requests]}")


@pytest.fixture
def llama_server() -> Iterator[FakeLlamaServer]:
    fake = FakeLlamaServer().start()
    try:
        yield fake
    finally:
        fake.stop()


# ---------------------------------------------------------------------------
# Router fixture
# ---------------------------------------------------------------------------


def _config(
    upstream: FakeUpstream,
    *,
    api_keys: list[str] | None = None,
) -> dict[str, Any]:
    root = f"http://127.0.0.1:{upstream.port}"
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 4000,
            "api_keys": api_keys or [],
        },
        "providers": {
            "anth": {
                "dialect": "anthropic",
                "base_url": root,
                "api_key": "upstream-key",
                "session_header": "x-opencode-session",
            },
            "chat": {
                "dialect": "openai-chat",
                "base_url": root + "/v1",
                "api_key": "upstream-key",
                "session_header": "x-opencode-session",
            },
            "resp": {
                "dialect": "openai-responses",
                "base_url": root + "/v1",
                "api_key": "upstream-key",
                "session_header": "x-opencode-session",
            },
        },
        "models": [
            {"name": "m-anth", "provider": "anth", "model": "upstream-anthropic"},
            {"name": "m-chat", "provider": "chat", "model": "upstream-chat"},
            {"name": "m-resp", "provider": "resp", "model": "upstream-responses"},
        ],
    }


@contextmanager
def router_config(config: dict[str, Any]) -> Iterator[str]:
    app = mr.App(mr.build_config(config))
    server = mr.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        app.close()


@contextmanager
def router(upstream: FakeUpstream, **config_kwargs: Any) -> Iterator[str]:
    with router_config(_with_gemini(_config(upstream, **config_kwargs), upstream)) as base_url:
        yield base_url


@pytest.fixture
def upstream() -> Iterator[FakeUpstream]:
    fake = FakeUpstream().start()
    try:
        yield fake
    finally:
        fake.stop()


TARGETS = [("m-anth", ANTHROPIC_TEXT), ("m-chat", CHAT_TEXT), ("m-resp", RESPONSES_TEXT), ("m-gem", GEMINI_TEXT)]
INGRESSES = ["openai-chat", "openai-responses", "anthropic"]

PATHS = {
    "openai-chat": "/v1/chat/completions",
    "openai-responses": "/v1/responses",
    "anthropic": "/v1/messages",
}
UPSTREAM_PATHS = {
    ANTHROPIC_TEXT: "/v1/messages",
    CHAT_TEXT: "/v1/chat/completions",
    RESPONSES_TEXT: "/v1/responses",
    GEMINI_TEXT: "/v1beta/models/gemini-2.5-flash:generateContent",
}


def _body(ingress: str, model: str, stream: bool) -> dict[str, Any]:
    if ingress == "openai-chat":
        return {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "stream": stream,
        }
    if ingress == "openai-responses":
        return {
            "model": model,
            "input": "hello",
            "max_output_tokens": 64,
            "stream": stream,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "stream": stream,
    }


def _text_from_response(ingress: str, payload: dict[str, Any]) -> str:
    if ingress == "openai-chat":
        return payload["choices"][0]["message"]["content"] or ""
    if ingress == "openai-responses":
        return _find_text(payload["output"])
    content = payload.get("content") or []
    return "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )


def _find_text(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("type") == "output_text":
            return value.get("text", "")
        return "".join(_find_text(item) for item in value.values())
    if isinstance(value, list):
        return "".join(_find_text(item) for item in value)
    return ""


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            current["data"] = line[6:]
        elif not line and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def _text_from_sse(ingress: str, text: str) -> str:
    parts: list[str] = []
    for event in _parse_sse(text):
        data = event.get("data")
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if ingress == "openai-chat":
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
        elif ingress == "openai-responses":
            if payload.get("type") == "response.output_text.delta":
                parts.append(payload.get("delta", ""))
        else:
            if payload.get("type") == "content_block_delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,expected_text", TARGETS)
def test_every_ingress_to_every_dialect_non_stream(
    upstream: FakeUpstream, ingress: str, model: str, expected_text: str
) -> None:
    raw_config = _with_gemini(_config(upstream), upstream)
    raw_config["models"][2]["model"] = "gpt-4.1"  # exercise native Responses SSE
    with router_config(raw_config) as base:
        response = httpx.post(
            base + PATHS[ingress],
            json=_body(ingress, model, stream=False),
            timeout=30,
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert _text_from_response(ingress, payload) == expected_text
    upstream_request = upstream.last_request(UPSTREAM_PATHS[expected_text])
    if model == "m-gem":
        assert upstream_request["body"]["contents"][0]["parts"] == [{"text": "hello"}]
        assert upstream_request["body"]["generationConfig"]["max_output_tokens"] == 64
    else:
        assert upstream_request["body"]["model"] == {
            "m-anth": "upstream-anthropic", "m-chat": "upstream-chat",
            "m-resp": "gpt-4.1",
        }[model]
    sent_headers = upstream_request["headers"]
    assert (
        sent_headers.get("authorization") == "Bearer upstream-key"
        or sent_headers.get("x-api-key") == "upstream-key"
        or sent_headers.get("x-goog-api-key") == "upstream-key"
    ), sent_headers


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,expected_text", TARGETS)
def test_every_ingress_to_every_dialect_stream(
    upstream: FakeUpstream, ingress: str, model: str, expected_text: str
) -> None:
    raw_config = _with_gemini(_config(upstream), upstream)
    raw_config["models"][2]["model"] = "gpt-4.1"  # exercise native Responses SSE
    with router_config(raw_config) as base:
        with httpx.stream(
            "POST",
            base + PATHS[ingress],
            json=_body(ingress, model, stream=True),
            timeout=30,
        ) as response:
            assert response.status_code == 200, response.read()
            raw = b"".join(response.iter_bytes()).decode()
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert _text_from_sse(ingress, raw) == expected_text, raw[:1000]
    events = _parse_sse(raw)
    payloads = [json.loads(e["data"]) for e in events if e["data"] != "[DONE]"]
    if ingress == "openai-chat":
        assert events[-1] == {"data": "[DONE]"}
        assert any(c.get("finish_reason") == "stop" for p in payloads for c in p.get("choices", []))
    elif ingress == "openai-responses":
        assert events[0]["event"] == "response.created"
        assert events[-1]["event"] == "response.completed"
        assert payloads[-1]["response"]["status"] == "completed"
        assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
    else:
        assert events[0]["event"] == "message_start"
        assert events[-2]["event"] == "message_delta"
        assert events[-1]["event"] == "message_stop"
    path = UPSTREAM_PATHS[expected_text].replace(":generateContent", ":streamGenerateContent")
    sent = upstream.last_request(path)["body"]
    if model == "m-gem":
        assert sent["contents"][0]["parts"] == [{"text": "hello"}]
        assert sent["generationConfig"]["max_output_tokens"] == 64
    else:
        assert sent["stream"] is True


def test_passthrough_preserves_unknown_fields_and_rewrites_model(
    upstream: FakeUpstream,
) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/chat/completions",
            json={
                "model": "m-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "custom_provider_field": {"nested": [1, 2, 3]},
            },
            timeout=30,
        )
    assert response.status_code == 200
    sent = upstream.last_request("/v1/chat/completions")["body"]
    assert sent["model"] == "upstream-chat"
    assert sent["custom_provider_field"] == {"nested": [1, 2, 3]}


def test_passthrough_preserves_anthropic_fields(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/messages",
            json={
                "model": "m-anth",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "mcp_servers": [{"type": "url", "url": "https://example.test/mcp"}],
                "anthropic_beta_field": True,
            },
            timeout=30,
        )
    assert response.status_code == 200
    sent = upstream.last_request("/v1/messages")["body"]
    assert sent["model"] == "upstream-anthropic"
    assert sent["mcp_servers"] == [
        {"type": "url", "url": "https://example.test/mcp"}
    ]
    assert sent["anthropic_beta_field"] is True


TRANSLATED_ROUTES = [
    ("openai-chat", "m-anth", "/v1/messages"),
    ("openai-chat", "m-resp", "/v1/responses"),
    ("openai-responses", "m-chat", "/v1/chat/completions"),
    ("openai-responses", "m-anth", "/v1/messages"),
    ("anthropic", "m-chat", "/v1/chat/completions"),
    ("anthropic", "m-resp", "/v1/responses"),
]


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES)
def test_translated_requests_cannot_redirect_provider_credentials(
    upstream: FakeUpstream, ingress: str, model: str, path: str,
) -> None:
    """Real adapters, two local HTTP destinations, and only dummy credentials."""
    alternate = FakeUpstream().start()
    try:
        alternate_base = f"http://127.0.0.1:{alternate.port}/v1"
        fallback = {"model": "openai/upstream-chat", "api_base": alternate_base}
        controls = {
            "api_base": alternate_base, "base_url": alternate_base,
            "api_key": "client-dummy-key", "api_version": "client-version",
            "custom_llm_provider": "openai", "fallbacks": [fallback],
            "model_list": [{"model_name": model, "litellm_params": fallback}],
            "models": ["openai/other-model"],
            "num_retries": 0, "max_retries": 0, "timeout": 0.01,
            "context_window_fallback_dict": {model: "openai/other-model"},
            "callbacks": ["client-callback"], "success_callback": ["client"],
            "failure_callback": ["client"], "client": {}, "http_client": {},
            "transport": {}, "shared_session": {}, "headers": {"x-test": "client"},
            "extra_headers": {"Authorization": "Bearer client-dummy-key"},
            "extra_query": {"api_key": "client-dummy-key"},
            "mock_response": "litellm.InternalServerError", "mock_tool_calls": [],
            "mock_delay": 1, "_skip_mcp_handler": False,
            "use_chat_completions_api": False, "litellm_metadata": fallback,
            "mcp_servers": [{"type": "url", "url": alternate_base}],
            "proxy_server_request": {"body": {"fallbacks": [fallback]}},
        }
        attacks = [{key: value} for key, value in controls.items()]
        attacks += [
            # The audit exploit: mock the primary failure, then inherit its key.
            {"mock_response": "litellm.InternalServerError", "fallbacks": [fallback]},
            *({"extra_body": {key: value}} for key, value in controls.items()),
            {"extra_body": {"model": "openai/other-model"}},
            {"extra_body": {"metadata": {"model_group": model}}},
            {"extra_body": {"extra_body": {"api_base": alternate_base}}},
            {"extra_body": {"thinking": {"type": "enabled", "api_base": alternate_base}}},
        ]
        if ingress == "openai-chat":
            attacks += [
                {"reasoning_effort": {"effort": "high", "fallbacks": [fallback]}},
                {"metadata": {"model_group": model}},
                {"metadata": {"requester_metadata": {"guardrails": ["client"]}}},
            ]
        elif ingress == "openai-responses":
            attacks += [
                {"reasoning": {"effort": "high", "api_base": alternate_base}},
                {"metadata": {"model_group": model}},
                {"metadata": {"fallbacks": [fallback]}},
            ]
        else:
            attacks += [
                {"thinking": {"type": "adaptive", "fallbacks": [fallback]}},
                {"output_config": {"effort": "high", "api_base": alternate_base}},
                {"metadata": {"user_id": "client", "model_group": model}},
            ]
        with router(upstream) as base, httpx.Client(timeout=10) as client:
            for attack in attacks:
                response = client.post(
                    base + PATHS[ingress],
                    json={**_body(ingress, model, False), **attack},
                )
                assert response.status_code == 400, (attack, response.text)
                assert upstream.requests == [], attack
                assert alternate.requests == [], attack

            # Unknown kwargs fail closed even if a future LiteLLM adds them.
            response = client.post(base + PATHS[ingress], json={
                **_body(ingress, model, False), "future_litellm_control": True,
            })
            assert response.status_code == 400
            response = client.post(base + PATHS[ingress],
                                   json=_body(ingress, model, False))
            assert response.status_code == 200, response.text
        assert len(upstream.requests) == 1
        sent = upstream.last_request(path)
        assert (sent["headers"].get("authorization") == "Bearer upstream-key"
                or sent["headers"].get("x-api-key") == "upstream-key")
        assert alternate.requests == []  # No request, and thus no inherited key.
    finally:
        alternate.stop()


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES)
def test_translated_tools_and_reasoning_remain_generation_data(
    upstream: FakeUpstream, ingress: str, model: str, path: str,
) -> None:
    schema = {"type": "object", "properties": {"api_base": {"type": "string"}}}
    body = _body(ingress, model, False)
    if ingress == "openai-chat":
        body.update({
            "reasoning_effort": "high",
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": schema,
            }}],
        })
    elif ingress == "openai-responses":
        body.update({
            "reasoning": {"effort": "high"},
            "tools": [{"type": "function", "name": "lookup", "parameters": schema}],
        })
    else:
        body.update({
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "metadata": {"user_id": "dummy-user"},
            "tools": [{"name": "lookup", "input_schema": schema}],
        })
    raw = _config(upstream)
    if ingress == "anthropic":
        # A native token budget becomes an effort bucket on these bridges.
        raw["providers"]["chat" if model == "m-chat" else "resp"]["compatibility"] = "lossy"
    # Use a model with known reasoning support in LiteLLM's local metadata;
    # synthetic "upstream-anthropic" has no adapter reasoning mapping.
    raw["models"][0]["model"] = "claude-3-7-sonnet-20250219"
    if model == "m-anth":
        body["max_tokens" if ingress == "openai-chat" else "max_output_tokens"] = 40000
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=body, timeout=30)
    assert response.status_code == 200, response.text
    sent = upstream.last_request(path)["body"]
    tool = sent["tools"][0]
    if path == "/v1/messages":
        assert tool["name"] == "lookup"
        assert tool["input_schema"] == schema
        assert sent["thinking"]["type"] == "enabled"
        assert sent["thinking"]["budget_tokens"] > 0
    elif path == "/v1/chat/completions":
        assert tool["function"]["parameters"] == schema
        assert sent["reasoning_effort"] == "high"
    else:
        assert tool["parameters"] == schema
        assert sent["reasoning"]["effort"] == "high"


def test_validated_generation_extensions_and_trusted_merge(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["providers"]["chat"]["litellm_kwargs"] = {
        "temperature": 0.25, "extra_body": {"top_k": 7},
    }
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/responses", json={
            **_body("openai-responses", "m-chat", False),
            "temperature": 0.75,
            "extra_body": {"top_k": 42, "min_p": 0.1, "repetition_penalty": 1.1},
        }, timeout=30)
    assert response.status_code == 200, response.text
    sent = upstream.last_request("/v1/chat/completions")["body"]
    assert sent["temperature"] == 0.25
    assert sent["top_k"] == 7
    assert sent["min_p"] == 0.1
    assert sent["repetition_penalty"] == 1.1


@pytest.mark.parametrize("extension", [
    [], {"top_k": True}, {"top_k": {}}, {"min_p": -1},
    {"min_p": float("inf")}, {"repetition_penalty": 0},
    {"thinking": {"type": "enabled", "budget_tokens": {"api_key": "dummy"}}},
])
def test_generation_extensions_have_closed_typed_schemas(extension: Any) -> None:
    with pytest.raises(mr.RouterError) as exc:
        router_protocol._clean_request_params({"extra_body": extension}, "openai-chat")
    assert exc.value.status == 400


def test_thinking_extension_uses_generation_mapping(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["models"][0]["model"] = "claude-3-7-sonnet-20250219"
    with router_config(raw) as base:
        response = httpx.post(base + "/v1/chat/completions", json={
            **_body("openai-chat", "m-anth", False),
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        }, timeout=30)
    assert response.status_code == 200, response.text
    sent = upstream.last_request("/v1/messages")["body"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}


@pytest.mark.parametrize("ingress", ["anthropic", "openai-responses"])
def test_sampling_extensions_reach_chat_upstream(
    upstream: FakeUpstream, ingress: str,
) -> None:
    with router(upstream) as base:
        response = httpx.post(base + PATHS[ingress], json={
            **_body(ingress, "m-chat", False),
            "extra_body": {"top_k": 42, "min_p": 0.05, "repetition_penalty": 1.2},
        }, timeout=30)
    assert response.status_code == 200, response.text
    sent = upstream.last_request("/v1/chat/completions")["body"]
    assert sent["top_k"] == 42
    assert sent["min_p"] == 0.05
    assert sent["repetition_penalty"] == 1.2


def test_translated_mcp_tool_does_not_execute_in_router(upstream: FakeUpstream) -> None:
    alternate = FakeUpstream().start()
    try:
        tool = {"type": "mcp", "server_label": "untrusted",
                "server_url": f"http://127.0.0.1:{alternate.port}/mcp"}
        with router(upstream) as base:
            response = httpx.post(base + "/v1/chat/completions", json={
                **_body("openai-chat", "m-resp", False), "tools": [tool],
            }, timeout=30)
        assert response.status_code == 200, response.text
        assert upstream.last_request("/v1/responses")["body"]["tools"] == [tool]
        assert alternate.requests == []
    finally:
        alternate.stop()


def test_chat_tools_are_translated_for_anthropic_upstream(
    upstream: FakeUpstream,
) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/chat/completions",
            json={
                "model": "m-anth",
                "messages": [{"role": "user", "content": "look it up"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look something up",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                                "required": ["q"],
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            },
            timeout=30,
        )
    assert response.status_code == 200
    sent = upstream.last_request("/v1/messages")["body"]
    assert sent["tools"][0]["name"] == "lookup"
    assert sent["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"


def test_anthropic_tools_are_translated_for_chat_upstream(
    upstream: FakeUpstream,
) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/messages",
            json={
                "model": "m-chat",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "look it up"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Look something up",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "lookup"},
            },
            timeout=30,
        )
    assert response.status_code == 200
    sent = upstream.last_request("/v1/chat/completions")["body"]
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "lookup"


def test_auth_accepts_bearer_and_x_api_key(upstream: FakeUpstream) -> None:
    with router(upstream, api_keys=["router-secret"]) as base:
        body = _body("openai-chat", "m-chat", stream=False)
        assert httpx.post(base + "/v1/chat/completions", json=body).status_code == 401
        assert (
            httpx.post(
                base + "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            httpx.post(
                base + "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer router-secret"},
            ).status_code
            == 200
        )
        assert (
            httpx.post(
                base + "/v1/chat/completions",
                json=body,
                headers={"x-api-key": "router-secret"},
            ).status_code
            == 200
        )
        assert httpx.get(base + "/v1/models").status_code == 401
        assert (
            httpx.get(
                base + "/v1/models",
                headers={"Authorization": "Bearer router-secret"},
            ).status_code
            == 200
        )


def test_anthropic_error_shape_and_model_not_found(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/messages",
            json=_body("anthropic", "does-not-exist", stream=False),
        )
    assert response.status_code == 404
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"
    assert "does-not-exist" in payload["error"]["message"]


def test_openai_error_shape_and_model_not_found(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "does-not-exist", stream=False),
        )
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "model_not_found"
    assert "does-not-exist" in payload["error"]["message"]


def test_dynamic_provider_model_routing(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/chat/completions",
            json={
                "model": "chat/anything-at-all",
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=30,
        )
    assert response.status_code == 200
    assert (
        upstream.last_request("/v1/chat/completions")["body"]["model"]
        == "anything-at-all"
    )


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model,expected", TARGETS)
def test_gzip_request_body(upstream: FakeUpstream, ingress: str, model: str, expected: str) -> None:
    body = json.dumps(_body(ingress, model, stream=False)).encode()
    with router(upstream) as base:
        response = httpx.post(
            base + PATHS[ingress],
            content=gzip.compress(body),
            headers={
                "Content-Type": "application/json; charset=iso-8859-1",
                "Content-Encoding": "gzip",
                "Content-MD5": "stale-md5",
                "Digest": "sha-256=stale-digest",
                "Content-Digest": "sha-256=:stale:",
                "Repr-Digest": "sha-256=:stale:",
                "Content-Range": "bytes 0-20/100",
                "Content-Location": "/original.json.gz",
                "ETag": '"original"',
                "Signature": "stale-signature",
                "Signature-Input": "stale-input",
            },
        )
    assert response.status_code == 200
    assert _text_from_response(ingress, response.json()) == expected
    request = upstream.last_request(UPSTREAM_PATHS[expected])
    headers = request["headers"]
    assert headers.get("content-encoding", "identity") == "identity"
    assert headers["content-type"] == "application/json"
    assert int(headers["content-length"]) == len(request["raw_body"])
    assert json.loads(bytes(request["raw_body"])) == request["body"]
    for name in ("content-encoding", "content-md5", "digest", "content-digest",
                 "repr-digest", "content-range", "content-location", "etag",
                 "signature", "signature-input"):
        assert name not in headers


def test_fake_upstream_rejects_json_advertised_as_gzip(upstream: FakeUpstream) -> None:
    response = httpx.post(
        f"http://127.0.0.1:{upstream.port}/v1/chat/completions",
        content=b'{"model":"test"}',
        headers={"Content-Encoding": "gzip"},
    )
    assert response.status_code == 400


def _raw_post(base: str, headers: bytes, body: bytes) -> tuple[int, bytes]:
    address = urlsplit(base)
    with socket.create_connection((address.hostname, address.port), timeout=5) as sock:
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                     + headers + b"\r\n" + body)
        # EOF makes truncation deterministic without a read-timeout redesign.
        sock.shutdown(socket.SHUT_WR)
        with sock.makefile("rb") as response:
            status = int(response.readline().split()[1])
            return status, response.read()


@pytest.mark.parametrize("headers,body", [
    (b"Transfer-Encoding: chunked\r\n", b"-1\r\n" + b"x" * 10000),
    (b"Transfer-Encoding: chunked\r\n", b"2\r\n{}xx0\r\n\r\n"),
    (b"Transfer-Encoding: chunked\r\n", b"2\r\n{"),
    (b"Transfer-Encoding: chunked\r\n", b"0\r\n"),
    (b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n", b"{}"),
    (b"Transfer-Encoding: gzip, chunked\r\n", b"0\r\n\r\n"),
    (b"Content-Length: 2\r\nContent-Length: 3\r\n", b"{}"),
    (b"Content-Length: 100\r\n", b'{"model":"m-chat","messages":[]}'),
])
def test_raw_malformed_body_never_reaches_upstream(
    upstream: FakeUpstream, headers: bytes, body: bytes,
) -> None:
    config = _config(upstream)
    config["server"]["max_body_bytes"] = 128
    with router_config(config) as base:
        status, response = _raw_post(base, headers, body)
    assert status == 400
    assert b"Connection: close\r\n" in response
    assert not upstream.requests


def test_chunked_gzip_with_extensions_and_trailers(upstream: FakeUpstream) -> None:
    raw = json.dumps(_body("openai-chat", "m-chat", stream=False)).encode()
    zipped = gzip.compress(raw)
    chunks = (f'{len(zipped):x}; flag; label="hello; world"\r\n'.encode()
              + zipped + b"\r\n0\r\nX-Checksum: ignored\r\n\r\n")
    with router(upstream) as base:
        status, response = _raw_post(
            base, b"Transfer-Encoding: Chunked\r\nContent-Encoding: GZIP\r\n", chunks,
        )
    assert status == 200
    assert CHAT_TEXT.encode() in response
    request = upstream.last_request("/v1/chat/completions")
    assert "content-encoding" not in request["headers"]
    assert "transfer-encoding" not in request["headers"]
    assert "x-checksum" not in request["headers"]


def test_models_listing(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.get(base + "/v1/models")
    assert response.status_code == 200
    payload = response.json()
    ids = {row["id"] for row in payload["data"]}
    assert ids == {"m-anth", "m-chat", "m-resp", "m-gem"}
    assert payload["has_more"] is False


def test_health(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        response = httpx.get(base + "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_streaming_unknown_model_returns_local_lookup_error(upstream: FakeUpstream) -> None:
    """A missing route fails before any SSE bytes, with the right error shape."""
    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/messages",
            json=_body("anthropic", "nope", stream=True),
        )
    assert response.status_code == 404
    assert response.json()["type"] == "error"


def test_chunked_request_body(upstream: FakeUpstream) -> None:
    body = json.dumps(_body("openai-chat", "m-chat", stream=False)).encode()

    def chunks() -> Iterator[bytes]:
        yield body[:17]
        yield body[17:]

    with router(upstream) as base:
        response = httpx.post(
            base + "/v1/chat/completions",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == CHAT_TEXT


def test_passthrough_stream_is_byte_identical(upstream: FakeUpstream) -> None:
    expected = "".join(f"data: {chunk}\n\n" for chunk in _chat_sse(CHAT_TEXT))
    expected += "data: [DONE]\n\n"
    with router(upstream) as base:
        with httpx.stream(
            "POST",
            base + "/v1/chat/completions",
            json=_body("openai-chat", "m-chat", stream=True),
        ) as response:
            raw = b"".join(response.iter_bytes()).decode()
    assert raw == expected


def test_disabled_route_is_not_found(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["models"][0]["enabled"] = False
    config = mr.build_config(raw)
    app = mr.App(config)
    server = mr.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        response = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "m-anth", stream=False),
        )
        listing = httpx.get(base + "/v1/models").json()
    finally:
        server.shutdown()
        server.server_close()
        app.close()
    assert response.status_code == 404
    assert "m-anth" not in {row["id"] for row in listing["data"]}


def test_config_validation_errors(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["models"].append({"name": "bad", "provider": "missing"})
    with pytest.raises(SystemExit, match="unknown provider"):
        mr.build_config(raw)

    raw = _config(upstream)
    raw["models"].append({"name": "m-chat", "provider": "chat"})
    with pytest.raises(SystemExit, match="duplicate model name"):
        mr.build_config(raw)

    raw = _config(upstream)
    raw["providers"]["chat"]["dialect"] = "grpc"
    with pytest.raises(SystemExit, match="'dialect' must be one of"):
        mr.build_config(raw)

    raw = _config(upstream)
    raw["providers"]["chat"]["api_key"] = None
    raw["providers"]["chat"]["api_key_env"] = "DEFINITELY_NOT_SET_12345"
    with pytest.raises(SystemExit, match="is not set"):
        mr.build_config(raw)


def test_anthropic_litellm_base_is_normalized(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["providers"]["anth"]["base_url"] = (
        f"http://127.0.0.1:{upstream.port}/v1/messages"
    )
    config = mr.build_config(raw)
    provider = config.providers["anth"]
    assert provider.litellm_call_kwargs()["api_base"] == (
        f"http://127.0.0.1:{upstream.port}"
    )

    raw["providers"]["anth"]["base_url"] = f"http://127.0.0.1:{upstream.port}/v1"
    config = mr.build_config(raw)
    provider = config.providers["anth"]
    assert provider.litellm_call_kwargs()["api_base"] == (
        f"http://127.0.0.1:{upstream.port}"
    )


def test_url_construction() -> None:
    assert router_headers._upstream_url(_provider("anthropic", "https://a.test"), "anthropic") == (
        "https://a.test/v1/messages"
    )
    assert router_headers._upstream_url(_provider("anthropic", "https://a.test/v1"), "anthropic") == (
        "https://a.test/v1/messages"
    )
    assert router_headers._upstream_url(
        _provider("anthropic", "https://a.test/v1/messages"), "anthropic"
    ) == "https://a.test/v1/messages"
    assert router_headers._upstream_url(_provider("openai-chat", "https://b.test/v1"), "openai-chat") == (
        "https://b.test/v1/chat/completions"
    )
    assert router_headers._upstream_url(
        _provider("openai-responses", "https://c.test/v1"), "openai-responses"
    ) == "https://c.test/v1/responses"


def _provider(dialect: str, base_url: str) -> mr.Provider:
    return mr.Provider(
        name="p",
        dialect=dialect,
        base_url=base_url,
        api_key="k",
        headers={},
        timeout=60.0,
        litellm_provider="anthropic" if dialect == "anthropic" else "openai",
        litellm_kwargs={},
        drop_params=True,
    )


def test_env_file_loading(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MR_TEST_NEW", raising=False)
    monkeypatch.delenv("MR_TEST_EXISTING", raising=False)
    monkeypatch.setenv("MR_TEST_EXISTING", "keep-me")
    env_file = tmp_path / "credentials.env"
    env_file.write_text(
        "# comment\n"
        "\n"
        'export MR_TEST_NEW="quoted value"\n'
        "MR_TEST_PLAIN=plain\n"
        "MR_TEST_COMMENT=value # trailing comment\n"
        "MR_TEST_EXISTING=replaced\n"
        "NOT A VALID LINE\n"
    )
    assert mr.load_env_file(env_file) == 4
    assert os.environ["MR_TEST_NEW"] == "quoted value"
    assert os.environ["MR_TEST_PLAIN"] == "plain"
    assert os.environ["MR_TEST_COMMENT"] == "value"
    assert os.environ["MR_TEST_EXISTING"] == "keep-me"


def test_default_env_files_include_xdg(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    paths = mr.default_env_files()
    assert paths[0] == tmp_path / "ai" / "credentials.env"


def test_session_header_is_omitted_or_forwarded(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        first = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "m-anth", stream=False),
            timeout=30,
        )
        second = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "m-anth", stream=False),
            headers={"x-opencode-session": "client-session-1"},
            timeout=30,
        )
    assert first.status_code == 200 and second.status_code == 200
    headers = upstream.last_request("/v1/messages")["headers"]
    assert headers["x-opencode-session"] == "client-session-1"
    assert "x-opencode-session" not in upstream.requests[0]["headers"]


def test_auth_header_injection() -> None:
    provider = mr.Provider(
        name="p",
        dialect="gemini",
        base_url="https://g.test/v1beta",
        api_key="key-1",
        headers={"X-Extra": "yes"},
        timeout=60.0,
        litellm_provider="gemini",
        litellm_kwargs={},
        drop_params=True,
        session_header="x-session",
        auth_header="bearer",
    )
    kwargs = provider.litellm_call_kwargs(session_id="sid-1")
    assert kwargs["api_base"] == "https://g.test/v1beta"
    assert kwargs["extra_headers"] == {
        "X-Extra": "yes",
        "Authorization": "Bearer key-1",
        "x-session": "sid-1",
        "content-type": "application/json",
    }


# Credential isolation and diagnostic redaction (prompt 02).
def _with_gemini(raw: dict[str, Any], upstream: FakeUpstream) -> dict[str, Any]:
    raw["providers"]["gem"] = {
        "dialect": "gemini", "base_url": f"http://127.0.0.1:{upstream.port}/v1beta",
        "api_key": "upstream-key",
    }
    raw["models"].append({"name": "m-gem", "provider": "gem", "model": "gemini-2.5-flash"})
    return raw


@pytest.fixture
def ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    values = {name: "ambient-dummy-" + name for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY",
        "GEMINI_API_KEY", "GOOGLE_API_KEY",
    )}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    for name in ("api_key", "openai_key", "anthropic_key"):
        monkeypatch.setattr(router_adapters.litellm, name, "global-dummy-" + name)
    monkeypatch.setattr(router_adapters.litellm, "headers", {
        "Authorization": "Bearer global-dummy-header", "x-api-key": "global-dummy-header",
    })
    return values


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-anth", "m-chat", "m-resp", "m-gem"])
@pytest.mark.parametrize("credential", [None, "", " \t ", "arbitrary-provider:key+/=?", "env"])
@pytest.mark.parametrize("stream", [False, True])
def test_credentials_are_explicit_across_dialects(
    upstream: FakeUpstream, ambient_credentials: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, ingress: str, model: str, credential: str | None,
    stream: bool,
) -> None:
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider.pop("api_key")
        provider["litellm_kwargs"] = {"api_key": "ignored-duplicate-dummy-key"}
        if credential == "env":
            provider["api_key_env"] = "ROUTER_DUMMY_CONFIGURED_KEY"
        elif credential is not None:
            provider["api_key"] = credential
    monkeypatch.setenv("ROUTER_DUMMY_CONFIGURED_KEY", "configured-env-dummy-key")
    expected = ("configured-env-dummy-key" if credential == "env"
                else (credential or "").strip() or mr.UNAUTHENTICATED_API_KEY)
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, model, stream), timeout=30)
    assert response.status_code == 200, response.text
    if stream:
        assert _text_from_sse(ingress, response.text)
    else:
        assert _text_from_response(ingress, response.json())
    assert len(upstream.requests) == 1
    request = upstream.requests[0]
    headers = request["headers"]
    if model == "m-anth":
        assert headers["x-api-key"] == expected
    elif model == "m-gem":
        assert headers["x-goog-api-key"] == expected
    else:
        assert headers["authorization"] == "Bearer " + expected
    sent = json.dumps(request)
    assert "ambient-dummy-" not in sent
    assert "global-dummy-" not in sent
    assert "ignored-duplicate-dummy-key" not in sent
    assert all(os.environ[name] == value for name, value in ambient_credentials.items())
    assert router_adapters.litellm.api_key == "global-dummy-api_key"


def test_parallel_configured_and_uncredentialed_calls(
    upstream: FakeUpstream, ambient_credentials: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ_type = type(os.environ)
    setitem = environ_type.__setitem__
    delitem = environ_type.__delitem__

    def guarded_setitem(environ: Any, name: str, value: str) -> None:
        assert name not in ambient_credentials, "A request changed an ambient credential"
        setitem(environ, name, value)

    def guarded_delitem(environ: Any, name: str) -> None:
        assert name not in ambient_credentials, "A request removed an ambient credential"
        delitem(environ, name)

    # Catch temporary clear/restore schemes as well as persistent mutations.
    monkeypatch.setattr(environ_type, "__setitem__", guarded_setitem)
    monkeypatch.setattr(environ_type, "__delitem__", guarded_delitem)
    raw = _config(upstream)
    raw["providers"]["anth"]["api_key"] = ""
    raw["providers"]["chat"]["api_key"] = "parallel-configured-key"
    with router_config(raw) as base:
        barrier = threading.Barrier(4)

        def call(index: int) -> httpx.Response:
            barrier.wait(timeout=5)
            model = "m-anth" if index % 2 else "m-chat"
            return httpx.post(base + PATHS["openai-responses"],
                              json=_body("openai-responses", model, False), timeout=30)

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(call, range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert len(upstream.requests) == 4
    for request in upstream.requests:
        headers = request["headers"]
        if request["path"] == "/v1/messages":
            assert headers["x-api-key"] == mr.UNAUTHENTICATED_API_KEY
        else:
            assert headers["authorization"] == "Bearer parallel-configured-key"
    assert all(os.environ[name] == value for name, value in ambient_credentials.items())


@pytest.mark.parametrize("message_attribute", [True, False])
@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("phase", ["json", "first", "sync", "async"])
def test_generated_errors_use_one_sanitizer(
    upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, message_attribute: bool, ingress: str, phase: str,
) -> None:
    secret = "arbitrary-provider:key+/=?"
    raw = _config(upstream, api_keys=["router-auth-dummy"])
    for provider in raw["providers"].values():
        provider["api_key"] = secret
        provider["headers"] = {"X-Custom-Credential": "custom-header-dummy"}
    message = (f"failure {secret} router-auth-dummy custom-header-dummy "
               "Bearer opaque+/=token x-api-key: random-key-format "
               "api_key=provider.not-sk.secret " + "x" * 6000)

    class Failure(Exception):
        pass

    exc = Failure(message)
    if message_attribute:
        exc.message = message
    exc.code = secret
    exc.error_type = secret

    def dispatch(self: mr.App, *_args: Any) -> mr.Plan:
        if phase == "json":
            raise exc

        def chunks() -> Iterator[bytes]:
            if phase != "first":
                yield b"data: {}\n\n"
            raise exc

        async def async_chunks() -> Any:
            yield b"data: {}\n\n"
            raise exc

        return (mr.Plan(async_chunks=async_chunks()) if phase == "async"
                else mr.Plan(chunks=chunks()))

    monkeypatch.setattr(mr.App, "dispatch", dispatch)
    caplog.set_level("DEBUG")
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress],
                              headers={"Authorization": "Bearer router-auth-dummy"},
                              json=_body(ingress, "m-chat", phase != "json"), timeout=10)
    assert response.status_code == (502 if phase in ("json", "first") else 200)
    if phase in ("json", "first"):
        error = response.json()["error"]
    else:
        event = json.loads(_parse_sse(response.text)[-2 if ingress == "openai-chat" else -1]["data"])
        error = event if ingress == "openai-responses" else event["error"]
    assert len(error["message"]) == mr.MAX_ERROR_MESSAGE
    assert "<redacted>" in error["message"]
    for value in (secret, "router-auth-dummy", "custom-header-dummy", "opaque+/=token",
                  "random-key-format", "provider.not-sk.secret"):
        assert value not in response.text
        assert value not in caplog.text
    assert "x" * (mr.MAX_ERROR_MESSAGE + 1) not in response.text
    assert "x" * (mr.MAX_ERROR_MESSAGE + 1) not in caplog.text


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("model", ["m-anth", "m-chat", "m-resp", "m-gem"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_adapter_and_passthrough_errors_are_sanitized(
    upstream: FakeUpstream, ingress: str, model: str, stream: bool,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _with_gemini(_config(upstream), upstream)
    secret = "configured.provider+key/with=punctuation"
    for provider in raw["providers"].values():
        provider["api_key"] = secret
        provider["litellm_kwargs"] = {"num_retries": 0}
    upstream.error_body = json.dumps({"error": {"message": (
        f"echo {secret} Bearer abc+/def== x-api-key: unknown-format-value " + "z" * 6000
    )}}).encode()
    caplog.set_level("DEBUG")
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, model, stream), timeout=30)
    assert upstream.requests  # The error must come from the real local upstream.
    assert response.status_code == 429, response.text
    assert response.headers["content-type"] == "application/json"
    assert "x-upstream-diagnostic" not in response.headers
    error = response.json()["error"]
    assert len(error["message"]) <= mr.MAX_ERROR_MESSAGE
    assert "<redacted>" in error["message"]
    assert "z" * (mr.MAX_ERROR_MESSAGE + 1) not in response.text
    for value in (secret, "abc+/def==", "unknown-format-value"):
        assert value not in response.text
        assert value not in caplog.text
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_redaction_precedes_length_limit_and_handles_escaped_values() -> None:
    from urllib.parse import quote

    secret = 'arbitrary-key/with+quotes"and\\slash'
    sanitize = mr.ErrorSanitizer([secret])
    for value in (secret, json.dumps(secret)[1:-1], quote(secret, safe="")):
        text = sanitize("start " + value + "x" * 6000)
        assert text.startswith("start <redacted>")
        assert len(text) == mr.MAX_ERROR_MESSAGE
    assert sanitize("x" * (mr.MAX_ERROR_MESSAGE - 3) + secret).endswith("<re")


@pytest.mark.parametrize("header", ["bearer", "x-api-key", "api-key"])
def test_custom_auth_headers_use_selected_key(
    upstream: FakeUpstream, ambient_credentials: dict[str, str], header: str,
) -> None:
    raw = _with_gemini(_config(upstream), upstream)
    for provider in raw["providers"].values():
        provider["api_key"] = "selected-custom-dummy-key"
        provider["auth_header"] = header
    with router_config(raw) as base:
        for ingress in INGRESSES:
            for model, _text in TARGETS:
                response = httpx.post(base + PATHS[ingress],
                                      json=_body(ingress, model, False), timeout=30)
                assert response.status_code == 200, response.text
    assert len(upstream.requests) == 12
    for request in upstream.requests:
        name = "authorization" if header == "bearer" else header
        expected = ("Bearer " if header == "bearer" else "") + "selected-custom-dummy-key"
        assert request["headers"][name] == expected
        assert "ambient-dummy-" not in json.dumps(request)
        assert "global-dummy-" not in json.dumps(request)


@pytest.mark.parametrize("setting", ["proxy_auth", "client_session", "aclient_session"])
def test_global_credentials_and_clients_fail_closed_without_mutation(
    upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, setting: str,
) -> None:
    global_value = object()
    monkeypatch.setattr(router_adapters.litellm, setting, global_value)
    with router(upstream) as base:
        response = httpx.post(base + PATHS["openai-chat"],
                              json=_body("openai-chat", "m-anth", False), timeout=10)
    assert response.status_code == 500
    assert "credential isolation" in response.json()["error"]["message"]
    assert not upstream.requests
    assert getattr(router_adapters.litellm, setting) is global_value


def test_credentials_cannot_contain_litellm_environment_references(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["providers"]["anth"]["api_key"] = "os.environ/UNRELATED_DUMMY_KEY"
    with pytest.raises(SystemExit, match="use api_key_env"):
        mr.build_config(raw)


def test_discovery_errors_and_debug_logs_use_sanitizer(
    upstream: FakeUpstream, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    raw = _config(upstream, api_keys=["inbound-dummy-secret"])
    raw["catalog"] = {
        "url": f"http://127.0.0.1:{upstream.port}/v1/models",
        "headers": {"Authorization": "Bearer catalog-dummy-secret"},
        "rules": [{"match": ".*", "provider": "chat"}],
    }
    raw["llama_cpp"] = {
        "servers": [{"url": f"http://127.0.0.1:{upstream.port}",
                     "api_key": "llama-dummy-secret"}],
    }
    message = ("upstream-key catalog-dummy-secret llama-dummy-secret "
               "inbound-dummy-secret Bearer strange+/token== api-key=unrecognized-token " + "z" * 6000)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise mr.RouterError(502, message)

    monkeypatch.setattr(httpx.Client, "get", fail)
    caplog.set_level("DEBUG")
    old_filters = list(mr.LOG.filters)
    app = mr.App(mr.build_config(raw))
    try:
        catalog = app.catalog_status()
        llama = app.llama_cpp_status()
        assert len(catalog["error"]) <= mr.MAX_ERROR_MESSAGE
        assert len(llama["error"]) <= mr.MAX_ERROR_MESSAGE
        assert len(llama["servers"][0]["error"]) <= mr.MAX_ERROR_MESSAGE
        try:
            raise ValueError(message)
        except ValueError:
            mr.LOG.debug("diagnostic failed: %s", message, exc_info=True)
            logging.getLogger("LiteLLM").debug("adapter failed: %s", message, exc_info=True)
    finally:
        app.close()
    assert mr.LOG.filters == old_filters
    for text in (json.dumps(catalog), json.dumps(llama), caplog.text):
        for secret in ("upstream-key", "catalog-dummy-secret", "llama-dummy-secret",
                       "inbound-dummy-secret", "strange+/token==", "unrecognized-token"):
            assert secret not in text
        assert "z" * (mr.MAX_ERROR_MESSAGE + 1) not in text


@pytest.mark.parametrize("ingress,model,path", TRANSLATED_ROUTES)
def test_real_adapter_midstream_diagnostics_are_sanitized(
    upstream: FakeUpstream, ingress: str, model: str, path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream.stream_error = "echo upstream-key Bearer abc+/token== " + "q" * 6000
    caplog.set_level("DEBUG")
    raw = _config(upstream)
    # LiteLLM simulates Responses streaming for unknown model names. A model
    # known to its installed capability table exercises real upstream SSE;
    # the destination remains our local fake server with dummy credentials.
    for route in raw["models"]:
        if route["provider"] == "resp":
            route["model"] = "gpt-4.1"
    with router_config(raw) as base:
        response = httpx.post(base + PATHS[ingress],
                              json=_body(ingress, model, True), timeout=30)
    assert response.status_code == 200
    assert upstream.last_request(path)
    events = [json.loads(event["data"]) for event in _parse_sse(response.text)
              if event.get("data") != "[DONE]"]
    errors = [event.get("error", event) for event in events
              if event.get("type") == "error" or event.get("error")]
    assert errors, response.text
    assert len(errors[-1]["message"]) <= mr.MAX_ERROR_MESSAGE
    for value in ("upstream-key", "abc+/token==", "q" * (mr.MAX_ERROR_MESSAGE + 1)):
        assert value not in response.text
        assert value not in caplog.text


def test_passthrough_http_error_body_is_deliberately_replaced(upstream: FakeUpstream) -> None:
    upstream.error_status = 503
    upstream.error_body = b"non-JSON upstream-key Bearer opaque-token " + b"e" * 6000
    with router(upstream) as base:
        response = httpx.post(base + PATHS["openai-chat"],
                              json=_body("openai-chat", "m-chat", False), timeout=10)
    assert response.status_code == 503
    assert response.content != upstream.error_body
    assert response.headers["retry-after"] == "0"
    assert "x-upstream-diagnostic" not in response.headers
    assert len(response.json()["error"]["message"]) <= mr.MAX_ERROR_MESSAGE
    assert "upstream-key" not in response.text
    assert "opaque-token" not in response.text


@pytest.mark.parametrize("model,payload", [
    ("m-chat", _chat_completion(CHAT_TEXT)),
    ("m-resp", _responses_response(RESPONSES_TEXT)),
    ("m-anth", _anthropic_message(ANTHROPIC_TEXT)),
])
def test_successful_passthrough_json_bytes_are_unchanged(
    upstream: FakeUpstream, model: str, payload: dict[str, Any],
) -> None:
    ingress = {"m-chat": "openai-chat", "m-resp": "openai-responses", "m-anth": "anthropic"}[model]
    with router(upstream) as base:
        response = httpx.post(base + PATHS[ingress], json=_body(ingress, model, False), timeout=10)
    assert response.status_code == 200
    assert response.content == json.dumps(payload).encode()


def test_catalog_discovers_models_and_routes(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["models"] = []
    raw["catalog"] = {
        "url": f"http://127.0.0.1:{upstream.port}/v1/models",
        "refresh_seconds": 0,
        "rules": [
            {"match": "^catalog-claude", "provider": "anth"},
            {"match": ".", "provider": "chat"},
        ],
    }
    config = mr.build_config(raw)
    app = mr.App(config)
    server = mr.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert app.catalog_status()["models"] == 2
        listing = httpx.get(base + "/v1/models").json()
        ids = {row["id"] for row in listing["data"]}
        assert ids == {"catalog-claude", "catalog-other"}
        response = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "catalog-claude", stream=False),
            timeout=30,
        )
        other = httpx.post(
            base + "/v1/chat/completions",
            json=_body("openai-chat", "catalog-other", stream=False),
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        app.close()
    assert response.status_code == 200 and other.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ANTHROPIC_TEXT
    assert other.json()["choices"][0]["message"]["content"] == CHAT_TEXT


def test_catalog_config_validation(upstream: FakeUpstream) -> None:
    raw = _config(upstream)
    raw["catalog"] = {
        "url": f"http://127.0.0.1:{upstream.port}/v1/models",
        "rules": [{"match": ".", "provider": "nope"}],
    }
    with pytest.raises(SystemExit, match="unknown provider"):
        mr.build_config(raw)

    raw["catalog"]["rules"] = [{"match": "(", "provider": "chat"}]
    with pytest.raises(SystemExit, match="invalid 'match' regex"):
        mr.build_config(raw)

    raw["catalog"]["rules"] = []
    with pytest.raises(SystemExit, match="at least one"):
        mr.build_config(raw)


def test_gemini_model_prefix_and_base(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-4-argon",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(router_adapters.litellm, "completion", fake_completion)
    config = mr.build_config(
        {
            "server": {"port": 4000},
            "providers": {
                "gem": {
                    "dialect": "gemini",
                    "base_url": "https://opencode.test/google/v1beta",
                    "api_key": "key",
                    "auth_header": "bearer",
                    "session_header": "x-opencode-session",
                }
            },
            "models": [{"name": "gemini-4-argon", "provider": "gem"}],
        }
    )
    app = mr.App(config)
    try:
        plan = app.dispatch(
            "openai-chat",
            config.routes["gemini-4-argon"],
            {
                "model": "gemini-4-argon",
                "messages": [{"role": "user", "content": "hi"}],
            },
            {"x-opencode-session": "gemini-conversation-1"},
        )
    finally:
        app.close()
    assert plan.body is not None
    assert captured["model"] == "gemini/gemini-4-argon"
    assert captured["api_base"] == "https://opencode.test/google/v1beta"
    assert captured["api_key"] == "key"
    headers = captured["extra_headers"]
    assert headers["Authorization"] == "Bearer key"
    assert headers["x-opencode-session"] == "gemini-conversation-1"


# ---------------------------------------------------------------------------
# llama.cpp auto-discovery
# ---------------------------------------------------------------------------


def _llama_only_config(
    server: FakeLlamaServer, **llama_kwargs: Any
) -> dict[str, Any]:
    return {
        "server": {"host": "127.0.0.1", "port": 4000},
        "llama_cpp": {
            "servers": [server.url],
            "timeout": 1.0,
            "refresh_seconds": 0,
            **llama_kwargs,
        },
    }


def test_llama_cpp_discovers_models_aliases_and_context(
    llama_server: FakeLlamaServer,
) -> None:
    with router_config(_llama_only_config(llama_server)) as base_url:
        models = httpx.get(base_url + "/v1/models").json()
        entry = next(
            item for item in models["data"] if item["id"] == "llama-test"
        )
        assert entry["display_name"] == "tiny"
        assert entry["context_window"] == 4096
        assert entry["max_tokens"] is None
        assert entry["metadata"]["limits"]["context_window"] == {
            "value": 4096, "source": "/v1/models.meta.n_ctx", "kind": "reported",
        }
        assert entry["metadata"]["max_output_tokens"] is None
        assert entry["metadata"]["llama_cpp"] == {
            "server": llama_server.url,
            "mode": "single",
            "status": "loaded",
            "n_ctx": 4096,
        }
        assert entry["owned_by"] == f"llama-127.0.0.1-{llama_server.port}"

        response = httpx.post(
            base_url + "/v1/chat/completions",
            json=_body("openai-chat", "llama-test", False),
        )
        assert response.status_code == 200
        assert _text_from_response("openai-chat", response.json()) == "llama pong"
        sent = llama_server.last_request("/v1/chat/completions")
        assert sent["body"]["model"] == "llama-test"

        # The model's alias is a first-class route to the same upstream.
        response = httpx.post(
            base_url + "/v1/chat/completions",
            json=_body("openai-chat", "tiny", False),
        )
        assert response.status_code == 200
        assert llama_server.last_request("/v1/chat/completions")["body"]["model"] == "llama-test"

        # A non-chat ingress is translated to llama.cpp's chat dialect.
        response = httpx.post(
            base_url + "/v1/messages",
            json=_body("anthropic", "llama-test", False),
        )
        assert response.status_code == 200
        assert _text_from_response("anthropic", response.json()) == "llama pong"


@pytest.mark.parametrize("effort, expected", [
    ("none", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("minimal", {"thinking_budget_tokens": 1024}),
    ("low", {"thinking_budget_tokens": 2048}),
    ("medium", {"thinking_budget_tokens": 8192}),
    ("high", {"thinking_budget_tokens": 16384}),
    ("xhigh", {"thinking_budget_tokens": 32768}),
    ("max", {}),  # unbounded: leave the server default
])
def test_llama_cpp_maps_reasoning_effort_to_native_thinking_fields(
    llama_server: FakeLlamaServer, effort: str, expected: dict[str, Any],
) -> None:
    with router_config(_llama_only_config(llama_server)) as base_url:
        response = httpx.post(
            base_url + "/v1/chat/completions",
            json={**_body("openai-chat", "llama-test", False), "reasoning_effort": effort},
        )
        assert response.status_code == 200
        sent = llama_server.last_request("/v1/chat/completions")["body"]
        assert "reasoning_effort" not in sent
        for key in ("thinking_budget_tokens", "chat_template_kwargs"):
            if key in expected:
                assert sent[key] == expected[key]
            else:
                assert key not in sent


def test_llama_cpp_explicit_native_thinking_fields_win(llama_server: FakeLlamaServer) -> None:
    with router_config(_llama_only_config(llama_server)) as base_url:
        httpx.post(base_url + "/v1/chat/completions", json={
            **_body("openai-chat", "llama-test", False), "reasoning_effort": "high",
            "thinking_budget_tokens": 7,
        })
        assert llama_server.last_request("/v1/chat/completions")["body"]["thinking_budget_tokens"] == 7
        httpx.post(base_url + "/v1/chat/completions", json={
            **_body("openai-chat", "llama-test", False), "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": True, "x": 1},
        })
        sent = llama_server.last_request("/v1/chat/completions")["body"]
        assert sent["chat_template_kwargs"] == {"enable_thinking": True, "x": 1}


def test_llama_cpp_responses_ingress_maps_reasoning_effort(llama_server: FakeLlamaServer) -> None:
    with router_config(_llama_only_config(llama_server)) as base_url:
        response = httpx.post(base_url + "/v1/responses", json={
            **_body("openai-responses", "llama-test", False), "reasoning": {"effort": "low"},
        })
        assert response.status_code == 200, response.text
        sent = llama_server.last_request("/v1/chat/completions")["body"]
        assert sent["thinking_budget_tokens"] == 2048
        assert "reasoning_effort" not in sent and "reasoning" not in sent


def test_non_llama_passthrough_keeps_reasoning_effort(upstream: FakeUpstream) -> None:
    with router(upstream) as base:
        httpx.post(base + "/v1/chat/completions", json={
            "model": "m-chat", "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }, timeout=30)
    sent = upstream.last_request("/v1/chat/completions")["body"]
    assert sent["reasoning_effort"] == "high"
    assert "thinking_budget_tokens" not in sent


def test_llama_cpp_router_mode_context_from_launch_args() -> None:
    server = FakeLlamaServer(
        models=[
            {
                "id": "router-model",
                "status": {
                    "value": "unloaded",
                    "args": ["--model", "x.gguf", "--ctx-size", "8192"],
                },
            }
        ],
        props={"role": "router"},
    ).start()
    app = None
    try:
        app = mr.App(mr.build_config(_llama_only_config(server)))
        entry = next(
            item
            for item in app.models_payload()["data"]
            if item["id"] == "router-model"
        )
        assert entry["context_window"] == 8192
        assert entry["metadata"]["llama_cpp"]["mode"] == "router"
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_legacy_mode_context_and_capabilities() -> None:
    server = FakeLlamaServer(
        models=[
            {
                "id": "legacy-model",
                "max_model_len": 2048,
                "capabilities": ["completion", "multimodal"],
            }
        ]
    ).start()
    app = None
    try:
        app = mr.App(mr.build_config(_llama_only_config(server)))
        entry = next(
            item
            for item in app.models_payload()["data"]
            if item["id"] == "legacy-model"
        )
        assert entry["context_window"] == 2048
        assert entry["metadata"]["capabilities"] == ["text", "image"]
        assert entry["metadata"]["llama_cpp"]["mode"] == "legacy"
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_legacy_zero_max_model_len_falls_back_to_props() -> None:
    server = FakeLlamaServer(
        models=[{"id": "legacy-model", "max_model_len": 0}],
        props={"default_generation_settings": {"n_ctx": 3072}},
    ).start()
    app = None
    try:
        app = mr.App(mr.build_config(_llama_only_config(server)))
        entry = next(
            item
            for item in app.models_payload()["data"]
            if item["id"] == "legacy-model"
        )
        assert entry["context_window"] == 3072
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_prefix_and_alias_exposure() -> None:
    server = FakeLlamaServer().start()
    app = None
    try:
        config = mr.build_config(
            _llama_only_config(server, prefix="local/", aliases=True)
        )
        app = mr.App(config)
        ids = {item["id"] for item in app.models_payload()["data"]}
        assert ids == {"local/llama-test"}
        assert app.resolve("local/llama-test") is not None
        assert app.resolve("local/tiny") is not None
        assert app.resolve("llama-test") is None

        disabled = mr.App(
            mr.build_config(_llama_only_config(server, aliases=False))
        )
        try:
            assert disabled.resolve("tiny") is None
        finally:
            disabled.close()
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_per_server_name_and_api_key() -> None:
    server = FakeLlamaServer(api_key="llama-key").start()
    app = None
    try:
        config = mr.build_config(
            {
                "server": {"host": "127.0.0.1", "port": 4000},
                "llama_cpp": {
                    "servers": [
                        {
                            "url": server.url,
                            "name": "local-llama",
                            "api_key": "llama-key",
                        }
                    ],
                    "refresh_seconds": 0,
                },
            }
        )
        app = mr.App(config)
        entry = next(
            item
            for item in app.models_payload()["data"]
            if item["id"] == "llama-test"
        )
        assert entry["owned_by"] == "local-llama"
        route = app.resolve("llama-test")
        assert route is not None
        plan = app.dispatch(
            "openai-chat",
            route,
            _body("openai-chat", "llama-test", False),
            {},
        )
        assert plan.status == 200
        assert plan.body is not None
        assert json.loads(plan.body)["choices"][0]["message"]["content"] == "llama pong"
        sent = server.last_request("/v1/chat/completions")
        assert sent["body"]["model"] == "llama-test"
        assert sent["headers"]["authorization"] == "Bearer llama-key"
        assert app.provider(route.provider_name).api_key == "llama-key"
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_unreachable_server_does_not_break_startup(
    upstream: FakeUpstream,
) -> None:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]

    config = _config(upstream, api_keys=["diagnostics-test-key"])
    config["llama_cpp"] = {
        "servers": [f"http://127.0.0.1:{closed_port}"],
        "timeout": 0.2,
        "refresh_seconds": 0,
    }
    with router_config(config) as base_url:
        # Diagnostic inventory moved off the public liveness aliases.
        health = httpx.get(base_url + "/diagnostics",
                           headers={"Authorization": "Bearer diagnostics-test-key"}).json()
        assert health["status"] == "ok"
        assert health["llama_cpp"]["enabled"] is True
        assert health["llama_cpp"]["servers"][0]["reachable"] is False
        ids = {item["id"] for item in httpx.get(base_url + "/v1/models", headers={
            "Authorization": "Bearer diagnostics-test-key"}).json()["data"]}
        assert "m-chat" in ids
        assert len(ids) == 3


def test_llama_cpp_keeps_last_routes_when_server_goes_away() -> None:
    server = FakeLlamaServer().start()
    app = None
    try:
        app = mr.App(mr.build_config(_llama_only_config(server)))
        assert app.resolve("llama-test") is not None
        server.stop()
        # Drop keep-alive connections so the next probe really fails.
        app.http.close()
        app.refresh_llama_cpp(force=True)
        assert app.resolve("llama-test") is not None
        status = app.llama_cpp_status()
        assert status["servers"][0]["reachable"] is False
        assert status["servers"][0]["stale"] is True
        assert status["models"] == 1
    finally:
        if app is not None:
            app.close()
        server.stop()


def test_llama_cpp_env_and_default_server_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLAMA_SERVER_URL", "http://127.0.0.1:8080; http://127.0.0.1:8081/"
    )
    config = mr.build_config({"llama_cpp": {}})
    assert [server.url for server in config.llama_cpp.servers] == [
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
    ]

    monkeypatch.delenv("LLAMA_SERVER_URL")
    config = mr.build_config({"llama_cpp": {}})
    assert [server.url for server in config.llama_cpp.servers] == [
        mr.DEFAULT_LLAMA_URL
    ]

    config = mr.build_config({"llama_cpp": {"ports": [8080, 8081]}})
    assert [server.url for server in config.llama_cpp.servers] == [
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
    ]


def test_llama_cpp_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"servers": ["ftp://127.0.0.1:9931"]}})
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"servers": [{"name": "no-url"}]}})
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"ports": [70000]}})
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"timeout": 0}})
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"refresh_seconds": -1}})
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"context_fallback": 0}})

    monkeypatch.delenv("LLAMA_TEST_KEY", raising=False)
    with pytest.raises(SystemExit):
        mr.build_config({"llama_cpp": {"api_key_env": "LLAMA_TEST_KEY"}})


@pytest.mark.parametrize("second_url", ["http://127.0.0.1:9002", "http://127.0.0.1:9001"])
def test_llama_cpp_rejects_audit_duplicate_names_before_probing(
    monkeypatch: pytest.MonkeyPatch, second_url: str,
) -> None:
    def unexpected_probe(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Ambiguous identities must fail before network discovery")

    monkeypatch.setattr(mr.App, "_probe_llama_server", unexpected_probe)
    with pytest.raises(SystemExit, match="duplicate explicit provider name 'local'") as exc:
        mr.App(mr.build_config({"llama_cpp": {"servers": [
            {"url": "http://127.0.0.1:9001", "name": "local"},
            {"url": second_url, "name": "local"},
        ]}}))
    assert "servers[1]" in str(exc.value)
    assert "servers[0]" in str(exc.value)
    assert "unique name or omit names" in str(exc.value)


@pytest.mark.parametrize("reserved", [False, True])
def test_llama_cpp_generated_collisions_keep_identity_through_health_changes(
    monkeypatch: pytest.MonkeyPatch, reserved: bool,
) -> None:
    # Different URLs normalize to the same generated base name.
    urls = ["http://127.0.0.1:9001/a/b", "http://127.0.0.1:9001/a-b"]
    base = "llama-127.0.0.1-9001-a-b"
    servers: list[Any] = [{"url": url, "api_key": f"dummy-{i}"}
                          for i, url in enumerate(urls)]
    raw: dict[str, Any] = {"llama_cpp": {"servers": servers, "refresh_seconds": 0}}
    expected = [base, base + "-2"]
    if reserved:
        raw["providers"] = {base: {"dialect": "openai-chat", "base_url": urls[0],
                                   "api_key": "dummy-static"}}
        # Even a later, unreachable explicit server reserves its name before
        # either generated identity is allocated.
        servers.append({"url": "http://127.0.0.1:9003", "name": base + "-2"})
        expected = [base + "-3", base + "-4"]

    reachable = {urls[1]}

    def probe(app: mr.App, llama: mr.LlamaCppConfig, server: mr.LlamaCppServer) -> dict[str, Any]:
        # Identities must already exist when the very first probe begins.
        assert [app._llama_names[url] for url in urls] == expected
        if server.url not in reachable:
            return {"server": server, "error": "controlled outage"}
        i = urls.index(server.url)
        return {"server": server, "rows": [{"id": f"model-{i}", "aliases": [f"alias-{i}"]}],
                "props": {}}

    def send(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        assert request.method == "POST"
        request.read()
        model = json.loads(request.content)["model"]
        i = int(model.removeprefix("model-"))
        assert str(request.url) == urls[i] + "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer dummy-{i}"
        return httpx.Response(200, json={"selected": i}, request=request)

    monkeypatch.setattr(mr.App, "_probe_llama_server", probe)
    monkeypatch.setattr(httpx.Client, "send", send)
    app = mr.App(mr.build_config(raw))
    try:
        assert app._llama_names[urls[0]] == expected[0]
        assert app._llama_names[urls[1]] == expected[1]
        seen: set[str] = set()
        for live in ({urls[1]}, {urls[0]}, set(urls), {urls[1]}, set(), set(urls)):
            reachable.clear()
            reachable.update(live)
            seen.update(live)
            app.refresh_llama_cpp(force=True)
            status = app.llama_cpp_status()
            for i, url in enumerate(urls):
                state = status["servers"][i]
                assert state["provider"] == expected[i]
                assert state["reachable"] == (url in live)
                assert state["stale"] == (url in seen and url not in live)
                route = app.resolve(f"model-{i}")
                if url not in seen:
                    assert route is None
                    continue
                assert route is not None
                assert route.provider_name == expected[i]
                assert app.resolve(f"alias-{i}") is route
                plan = app.dispatch("openai-chat", route, _body("openai-chat", route.name, False), {})
                assert json.loads(plan.body) == {"selected": i}
            entries = app.models_payload()["data"]
            assert {entry["id"] for entry in entries} == {
                f"model-{i}" for i, url in enumerate(urls) if url in seen
            }
            for entry in entries:
                i = int(entry["id"].removeprefix("model-"))
                assert entry["owned_by"] == expected[i]
                assert entry["metadata"]["llama_cpp"]["server"] == urls[i]
    finally:
        app.close()


def test_llama_cpp_explicit_name_cannot_shadow_static_provider() -> None:
    with pytest.raises(SystemExit, match="already configured in providers.*unique server name"):
        mr.build_config({
            "providers": {"local": {"dialect": "openai-chat", "base_url": "http://127.0.0.1:9000"}},
            "llama_cpp": {"servers": [{"url": "http://127.0.0.1:9001", "name": "local"}]},
        })


@pytest.mark.parametrize("ingress", INGRESSES)
@pytest.mark.parametrize("selection", ["llama-test", "tiny", "local/manual-model"])
@pytest.mark.parametrize("change", ["remove", "replace"])
def test_refresh_between_resolution_and_dispatch_pins_provider(
    monkeypatch: pytest.MonkeyPatch, ingress: str, selection: str, change: str,
) -> None:
    server = FakeLlamaServer().start()
    app = mr.App(mr.build_config(_llama_only_config(server, servers=[{
        "url": server.url, "name": "local", "api_key": "dummy-old-key",
    }])))
    resolved = threading.Event()
    resume = threading.Event()
    original_resolve = app.resolve

    def paused_resolve(model: str) -> mr.Route | None:
        route = original_resolve(model)
        resolved.set()
        assert resume.wait(10), "test did not release resolution"
        return route

    monkeypatch.setattr(app, "resolve", paused_resolve)
    router_server = mr.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=router_server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{router_server.server_address[1]}"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(httpx.post, base_url + PATHS[ingress],
                                  json=_body(ingress, selection, False), timeout=15)
            try:
                assert resolved.wait(10), "request never reached resolution"
                assert app.config.llama_cpp is not None
                if change == "remove":
                    app.config.llama_cpp.servers.clear()
                else:
                    old = app.config.llama_cpp.servers[0]
                    app.config.llama_cpp.servers[0] = replace(old, api_key="dummy-new-key")
                    server.models = [{"id": "new-model"}]
                app.refresh_llama_cpp(force=True)
                assert original_resolve("llama-test") is None
                assert original_resolve("tiny") is None
                if change == "remove":
                    assert original_resolve("local/manual-model") is None
                    assert app.provider_count() == 0
                else:
                    assert app.provider("local").api_key == "dummy-new-key"
                    assert original_resolve("new-model") is not None
            finally:
                resume.set()
            response = pending.result(timeout=15)
        assert response.status_code == 200, response.text
        assert _text_from_response(ingress, response.json()) == "llama pong"
        sent = server.last_request("/v1/chat/completions")
        assert sent["headers"]["authorization"] == "Bearer dummy-old-key"
        assert sent["body"]["model"] == (
            "manual-model" if selection == "local/manual-model" else "llama-test"
        )
        if change == "replace":
            response = httpx.post(base_url + PATHS[ingress], json=_body(ingress, "new-model", False))
            assert response.status_code == 200, response.text
            assert server.last_request("/v1/chat/completions")["headers"]["authorization"] == "Bearer dummy-new-key"
    finally:
        resume.set()
        router_server.shutdown()
        router_server.server_close()
        app.close()
        server.stop()


def test_model_listing_retains_snapshot_during_refresh(
    monkeypatch: pytest.MonkeyPatch, llama_server: FakeLlamaServer,
) -> None:
    app = mr.App(mr.build_config(_llama_only_config(llama_server)))
    entered = threading.Event()
    resume = threading.Event()
    original_entry = app._route_entry

    def paused_entry(route: mr.Route, now: int, snapshot: mr.RegistrySnapshot) -> dict[str, Any]:
        entered.set()
        assert resume.wait(10)
        return original_entry(route, now, snapshot)

    monkeypatch.setattr(app, "_route_entry", paused_entry)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(app.models_payload)
            try:
                assert entered.wait(10)
                app.config.llama_cpp.servers.clear()
                app.refresh_llama_cpp(force=True)
                assert app.models_payload()["data"] == []
            finally:
                resume.set()
            old_listing = pending.result(timeout=10)
        assert [entry["id"] for entry in old_listing["data"]] == ["llama-test"]
        assert old_listing["data"][0]["metadata"]["provider_dialect"] == "openai-chat"
        assert old_listing["data"][0]["metadata"]["router"]["freshness"]["discovery"]["stale"] is False
    finally:
        resume.set()
        app.close()


def test_health_retains_inventory_and_status_generation(
    monkeypatch: pytest.MonkeyPatch, llama_server: FakeLlamaServer,
) -> None:
    app = mr.App(mr.build_config(_llama_only_config(llama_server)))
    entered = threading.Event()
    resume = threading.Event()
    original_status = app.catalog_status

    def paused_status(snapshot: mr.RegistrySnapshot | None = None) -> dict[str, Any]:
        entered.set()
        assert resume.wait(10)
        return original_status(snapshot)

    monkeypatch.setattr(app, "catalog_status", paused_status)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(app.health_payload)
            try:
                assert entered.wait(10)
                app.config.llama_cpp.servers.clear()
                app.refresh_llama_cpp(force=True)
            finally:
                resume.set()
            old_health = pending.result(timeout=10)
        assert old_health["models"] == old_health["llama_cpp"]["models"] == 1
        assert old_health["providers"] == 1
        assert len(old_health["llama_cpp"]["servers"]) == 1
        new_health = app.health_payload()
        assert new_health["models"] == new_health["llama_cpp"]["models"] == 0
        assert new_health["providers"] == 0
        assert new_health["llama_cpp"]["servers"] == []
    finally:
        resume.set()
        app.close()


@contextmanager
def controlled_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[mr.App, dict[str, Any]]]:
    state: dict[str, Any] = {
        "local": [{"id": "shared"}, {"id": "local-wins", "aliases": ["local-alias"]},
                  {"id": "explicit-alias"}, {"id": "disabled"}],
        "catalog": ["shared", "local-wins", "catalog-only", "explicit-alias", "disabled"],
        "local_error": None, "catalog_error": False,
    }

    def probe(app: mr.App, llama: mr.LlamaCppConfig, server: mr.LlamaCppServer) -> dict[str, Any]:
        return {"server": server, "rows": state["local"], "props": {},
                "error": state["local_error"]}

    def send(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        assert request.method == "GET" and str(request.url) == "http://127.0.0.1:9000/catalog"
        return httpx.Response(503 if state["catalog_error"] else 200,
                              json={"data": state["catalog"]}, request=request)

    monkeypatch.setattr(mr.App, "_probe_llama_server", probe)
    monkeypatch.setattr(httpx.Client, "send", send)
    app = mr.App(mr.build_config({
        "providers": {"static": {"dialect": "openai-responses", "base_url": "http://127.0.0.1:9000/v1"}},
        "models": [{"name": "shared", "provider": "static", "aliases": ["explicit-alias"]},
                   {"name": "disabled", "provider": "static", "enabled": False}],
        "catalog": {"url": "http://127.0.0.1:9000/catalog", "refresh_seconds": 0,
                    "rules": [{"match": ".", "provider": "static"}]},
        "llama_cpp": {"servers": [{"url": "http://127.0.0.1:9001", "name": "local"}],
                      "refresh_seconds": 0},
    }))
    try:
        yield app, state
    finally:
        app.close()


def test_registry_preserves_route_precedence_and_stale_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    with controlled_registry(monkeypatch) as (app, state):
        assert app.resolve("shared").provider_name == "static"
        assert app.resolve("explicit-alias") is app.resolve("shared")
        assert app.resolve("local-wins").provider_name == "local"
        assert app.resolve("local-alias") is app.resolve("local-wins")
        assert app.resolve("catalog-only").provider_name == "static"
        assert app.resolve("disabled") is None
        public = {entry["id"]: entry for entry in app.models_payload()["data"]}
        assert set(public) == {"shared", "local-wins", "catalog-only"}
        assert public["shared"]["owned_by"] == "static"
        assert public["local-wins"]["owned_by"] == "local"

        state["local_error"] = "controlled local failure"
        state["catalog_error"] = True
        app.refresh_llama_cpp(force=True)
        app.refresh_catalog(force=True)
        assert app.resolve("local-wins").provider_name == "local"
        assert app.resolve("local-alias") is app.resolve("local-wins")
        assert app.resolve("catalog-only").provider_name == "static"
        assert app.llama_cpp_status()["servers"][0]["stale"] is True
        assert app.catalog_status()["error"] is not None

        # A successful empty model list replaces the old source inventory;
        # lower-priority catalog routes become available again.
        state["local_error"] = None
        state["local"] = []
        app.refresh_llama_cpp(force=True)
        assert app.resolve("local-wins").provider_name == "static"
        assert app.resolve("local-alias") is None
        assert app.llama_cpp_status()["servers"][0]["stale"] is False
        assert app.llama_cpp_status()["models"] == 0


def test_concurrent_catalog_and_local_publications_keep_both_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    with controlled_registry(monkeypatch) as (app, state):
        before = app._registry
        state["local"] = [{"id": "new-local"}]
        state["catalog"] = ["new-catalog"]
        ready = threading.Barrier(3, timeout=10)
        publish = threading.Event()
        original_publish = app._publish_registry

        def paused_publish(**kwargs: Any) -> None:
            ready.wait()
            assert publish.wait(10)
            original_publish(**kwargs)

        monkeypatch.setattr(app, "_publish_registry", paused_publish)
        with ThreadPoolExecutor(max_workers=2) as pool:
            local = pool.submit(app.refresh_llama_cpp, force=True)
            catalog = pool.submit(app.refresh_catalog, force=True)
            try:
                ready.wait()
                # Neither half-built inventory nor status has been published.
                assert app._registry is before
                health = app.health_payload()
                assert health["llama_cpp"]["models"] == 1
                assert health["catalog"]["models"] == 5
            finally:
                publish.set()
            local.result(timeout=10)
            catalog.result(timeout=10)
        assert app.resolve("new-local").provider_name == "local"
        assert app.resolve("new-catalog").provider_name == "static"
        assert app.resolve("local-wins") is None
        assert app.resolve("catalog-only") is None
        health = app.health_payload()
        assert health["models"] == 3  # explicit shared plus the two new models
        assert health["llama_cpp"]["models"] == health["catalog"]["models"] == 1


def test_registry_metadata_and_transport_settings_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with controlled_registry(monkeypatch) as (app, state):
        snapshot = app._registry
        route = app.resolve("local-wins")
        with pytest.raises(TypeError):
            snapshot.providers["other"] = app.provider("local")
        with pytest.raises(TypeError):
            route.metadata["llama_cpp"]["server"] = "http://127.0.0.1:9999"
        with pytest.raises(TypeError):
            route._provider.headers["Authorization"] = "Bearer changed"
        entries = app.models_payload()["data"]
        local = next(entry for entry in entries if entry["id"] == "local-wins")
        local["metadata"]["llama_cpp"]["server"] = "changed"
        local["metadata"]["capabilities"].append("changed")
        state["local"][1]["aliases"].append("changed")
        fresh = next(entry for entry in app.models_payload()["data"] if entry["id"] == "local-wins")
        assert fresh["metadata"]["llama_cpp"]["server"] == "http://127.0.0.1:9001"
        assert fresh["metadata"]["capabilities"] == ["text"]
        assert app.resolve("changed") is None
