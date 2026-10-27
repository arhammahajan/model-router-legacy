"""Response plans, explicit resource ownership and translated SSE serialization."""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator

import httpx

from .adapters import adapter
from .common import (
    _DEFAULT_SANITIZER,
    ErrorSanitizer,
    RouterError,
    _close_quietly,
    _json_bytes,
    _jsonable,
    _sanitize_adapter_sse,
    _sanitize_generated_errors,
)


@dataclass
class StreamState:
    """Request-local translated event state, also used by relay-generated errors."""

    sequence: int = 0
    error: RouterError | None = None

    def number(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {**payload, "sequence_number": self.sequence}
        self.sequence += 1
        return payload

    def observe(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        error = payload.get("error")
        if payload.get("type") == "response.failed":
            error = (payload.get("response") or {}).get("error") or {}
        elif payload.get("type") == "error":
            error = error or payload
        if error is not None:
            message = error.get("message", "Upstream stream failed") if isinstance(error, dict) else str(error)
            self.error = RouterError(502, message, code=error.get("code") if isinstance(error, dict) else None)


async def _close_stream_resource(resource: Any, seen: set[int]) -> None:
    if resource is None or id(resource) in seen:
        return
    seen.add(id(resource))
    for child in adapter.stream_resources(resource):
        await _close_stream_resource(child, seen)
    if isinstance(resource, httpx.Response) and resource.is_closed:
        return
    close = getattr(resource, "close", None)
    aclose = getattr(resource, "aclose", None)
    # Sync httpx/SDK objects must use close, async objects must use aclose.
    if isinstance(resource, httpx.Response):
        close = resource.close if isinstance(resource.stream, httpx.SyncByteStream) else resource.aclose
    elif aclose is not None:
        close = aclose
    if close is not None:
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except (Exception, asyncio.CancelledError):
            # A failing/cancelled closer must not skip the remaining owners.
            pass


@dataclass
class Plan:
    """Own response iterators and request transports until close/aclose.

    App/shared SDK connection pools are borrowed, never owned. Resources are
    registered explicitly, including sources hidden behind serializer generators
    (closing an unstarted generator does not execute its finally block).
    """

    status: int = 200
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    chunks: Iterator[bytes] | None = None
    async_chunks: AsyncIterator[bytes] | None = None
    closer: Callable[[], None] | None = None
    resources: tuple[Any, ...] = ()
    stream_state: StreamState | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _cleanup_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            asyncio.run(self.aclose())

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        cancelled = False
        # Cancellation may arrive during normal completion's cleanup, or be
        # repeated while a cancelled relay releases its transports. Finish
        # releasing all owners before propagating it to the caller.
        while True:
            try:
                await asyncio.shield(self._cleanup_task)
                break
            except asyncio.CancelledError:
                if self._cleanup_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _cleanup(self) -> None:
        try:
            seen: set[int] = set()
            for resource in (self.chunks, self.async_chunks, *self.resources):
                await _close_stream_resource(resource, seen)
            # Compatibility for embedders using the original callback API.
            owner = getattr(self.closer, "__self__", None)
            if owner is None or id(owner) not in seen:
                _close_quietly(self.closer)
        finally:
            self._closed = True


def _chat_sse(
    iterator: Iterator[Any], sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
    state: StreamState | None = None,
) -> Iterator[bytes]:
    for chunk in iterator:
        payload = _sanitize_generated_errors(_jsonable(chunk), sanitize)
        if state is not None:
            state.observe(payload)
        yield b"data: " + _json_bytes(payload) + b"\n\n"
        if state is not None and state.error is not None:
            break
    yield b"data: [DONE]\n\n"


def _responses_sse(
    iterator: Iterator[Any], sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
    state: StreamState | None = None,
) -> Iterator[bytes]:
    state = state if state is not None else StreamState()
    for event in iterator:
        payload = _sanitize_generated_errors(_jsonable(event), sanitize)
        if not isinstance(payload, dict):
            payload = {"type": "response.event", "data": payload}
        payload = state.number(payload)
        state.observe(payload)
        event_type = str(payload.get("type") or "message")
        yield b"event: " + event_type.encode("utf-8") + b"\ndata: " + _json_bytes(payload) + b"\n\n"
        if state.error is not None:
            return


def _translated_sse(chunk: bytes, sanitize: ErrorSanitizer, state: StreamState) -> bytes:
    chunk = _sanitize_adapter_sse(chunk, sanitize)
    for line in chunk.splitlines():
        if line.startswith(b"data: "):
            state.observe(json.loads(line[6:]))
    return chunk
