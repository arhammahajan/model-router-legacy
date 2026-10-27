"""Native HTTP transport and generation retry/deadline budgets."""
from __future__ import annotations

import math
import re
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import httpx

from .common import _DEFAULT_SANITIZER, RouterError, _upstream_error_headers

if TYPE_CHECKING:
    from .app import App
    from .config import Provider


class _RequestBudget:
    """One monotonic budget, including backoff and all upstream stream reads."""

    def __init__(self, app: App) -> None:
        self.policy = app.config.retry
        self.clock, self.sleep, self.jitter = app._clock, app._sleep, app._jitter
        self.wall_clock = app._wall_clock
        self.end = self.clock() + self.policy.deadline
        self.headers: dict[str, str] = {}
        self.failure: RouterError | None = None
        self.output_started = False

    def error(self) -> RouterError:
        exc = RouterError(504, "Overall upstream request deadline exceeded; generation may "
                          "have been accepted and billed. The router will not replay it.",
                          code="request_deadline_exceeded")
        exc.headers = self.headers
        self.failure = exc
        return exc

    def timeout(self, timeout: float | None = None) -> float:
        if self.failure is not None:
            raise self.failure
        remaining = self.end - self.clock()
        if remaining <= 0:
            raise self.error()
        return min(remaining, timeout) if timeout is not None else remaining

    def ambiguous(self, exc: BaseException) -> RouterError:
        self.timeout()
        error = RouterError(504 if isinstance(exc, httpx.TimeoutException) else 502,
                            "The upstream failed after the request may have been sent; "
                            "generation may have been accepted and billed. The router "
                            "will not replay it.", code="upstream_outcome_unknown")
        error.headers = self.headers
        self.failure = error
        return error

    def delay(self, attempt: int, headers: Mapping[str, str]) -> None:
        cap = min(self.policy.backoff_max, self.policy.backoff_base * 2 ** (attempt - 1))
        delay = self.jitter(cap / 2, cap)
        # Retry-After is a minimum, never capped by the exponential backoff cap.
        value = headers.get("retry-after", "").strip()
        hint = 0.0
        if re.fullmatch(r"[0-9]+", value):
            try:
                hint = float(value)
            except ValueError:
                hint = math.inf
        elif value:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    hint = max(0.0, date.timestamp() - self.wall_clock())
            except (ValueError, TypeError, OverflowError):
                pass
        delay = max(delay, hint)
        if delay >= self.timeout():
            raise self.error()
        self.sleep(delay)
        self.timeout()


class _DeadlineNetworkStream:
    """Clamp each socket operation, so a trickling stream cannot reset the budget."""

    def __init__(self, stream: Any, budget: _RequestBudget) -> None:
        self.stream, self.budget = stream, budget

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self.stream.read(max_bytes, timeout=self.budget.timeout(timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        # Bound individual sendall calls as well as the complete upload.
        for offset in range(0, len(buffer), 65536):
            self.stream.write(buffer[offset:offset + 65536], timeout=self.budget.timeout(timeout))

    def start_tls(self, ssl_context: Any, server_hostname: str | None = None,
                  timeout: float | None = None) -> _DeadlineNetworkStream:
        return _DeadlineNetworkStream(self.stream.start_tls(
            ssl_context, server_hostname=server_hostname, timeout=self.budget.timeout(timeout)),
            self.budget)

    def close(self) -> None:
        self.stream.close()

    def get_extra_info(self, info: str) -> Any:
        return self.stream.get_extra_info(info)


class _DeadlineNetworkBackend:
    def __init__(self, backend: Any, budget: _RequestBudget) -> None:
        self.backend, self.budget = backend, budget

    def connect_tcp(self, *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        return _DeadlineNetworkStream(self.backend.connect_tcp(
            *args, timeout=self.budget.timeout(timeout), **kwargs), self.budget)

    def connect_unix_socket(self, *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        return _DeadlineNetworkStream(self.backend.connect_unix_socket(
            *args, timeout=self.budget.timeout(timeout), **kwargs), self.budget)


class _BudgetStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, budget: _RequestBudget) -> None:
        self.stream, self.budget = stream, budget

    def __iter__(self) -> Iterator[bytes]:
        iterator = iter(self.stream)
        while True:
            self.budget.timeout()
            try:
                chunk = next(iterator)
            except StopIteration:
                return
            except (httpx.ReadError, httpx.WriteError, httpx.ReadTimeout,
                    httpx.WriteTimeout, httpx.RemoteProtocolError) as exc:
                raise self.budget.ambiguous(exc) from exc
            except Exception:
                self.budget.timeout()
                raise
            self.budget.timeout()
            yield chunk

    def close(self) -> None:
        self.stream.close()


class _GenerationTransport(httpx.BaseTransport):
    """The sole retry owner. Never retries body reads or a returned response."""

    def __init__(self, budget: _RequestBudget, provider: Provider) -> None:
        self.budget, self.provider = budget, provider
        # Use an inner client to retain httpx's environment proxy/CA selection.
        # The outer client has no proxy mounts, so every send enters this owner.
        self.inner = httpx.Client(follow_redirects=False, timeout=provider.timeout)
        # httpx 0.28.1 / httpcore 1.0.9: the public timeout API is per-operation.
        # Wrap the pinned backend to also enforce a deadline on every socket IO.
        transports = {self.inner._transport, *self.inner._mounts.values()} - {None}
        for transport in transports:
            pool = transport._pool
            pool._retries = 0
            pool._network_backend = _DeadlineNetworkBackend(pool._network_backend, budget)
        self.finished = False
        self.attempts = 0
        self.last_status = 502

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.finished:
            exc = RouterError(self.last_status if self.last_status >= 400 else 502,
                              "An adapter attempted to replay a completed upstream attempt; "
                              "replay blocked by router retry policy", code="retry_blocked")
            exc.headers = self.budget.headers
            raise exc
        # All supported ingress bodies are materialized JSON, never upload iterators.
        request.read()
        try:
            while True:
                self.budget.timeout()
                self.attempts += 1
                request.extensions["timeout"] = {
                    name: self.budget.timeout(min(20.0, self.provider.timeout)
                                              if name == "connect" else self.provider.timeout)
                    for name in ("connect", "read", "write", "pool")}
                try:
                    response = self.inner.send(request, stream=True)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    self.budget.timeout()
                    if (not self.budget.policy.connect_errors
                            or self.budget.output_started
                            or self.attempts >= self.budget.policy.max_attempts):
                        raise
                    self.budget.delay(self.attempts, {})
                    continue
                except (httpx.ReadError, httpx.WriteError, httpx.ReadTimeout,
                        httpx.WriteTimeout, httpx.RemoteProtocolError) as exc:
                    raise self.budget.ambiguous(exc) from exc
                self.last_status = response.status_code
                response.request = request
                error = httpx.HTTPStatusError("upstream", request=request, response=response)
                self.budget.headers = _upstream_error_headers(error, _DEFAULT_SANITIZER)
                if (response.status_code not in self.budget.policy.statuses
                        or self.budget.output_started
                        or self.attempts >= self.budget.policy.max_attempts):
                    response.stream = _BudgetStream(response.stream, self.budget)
                    return response
                # No error body is consumed and no downstream output exists yet.
                # Close before waiting; never hold an unused upstream socket.
                response.close()
                self.budget.delay(self.attempts, response.headers)
        finally:
            # This also blocks hidden SDK/adapter retries on any escaped failure.
            self.finished = True

    def close(self) -> None:
        self.inner.close()
