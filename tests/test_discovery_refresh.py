"""Discovery scheduling contracts with a fake clock and local HTTP responses."""

from __future__ import annotations

import random

import http.client
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx
import pytest


import model_router as mr


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.now

    def advance_to(self, deadline: float) -> None:
        with self.lock:
            assert deadline >= self.now
            self.now = deadline


class ObservedWakeup:
    """Expose worker idle boundaries without sleeping or polling its state."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.condition = threading.Condition()
        self.waits = 0

    def clear(self) -> None:
        self.event.clear()

    def set(self) -> None:
        self.event.set()

    def wait(self, timeout: float | None = None) -> bool:
        with self.condition:
            self.waits += 1
            self.condition.notify_all()
        return self.event.wait(timeout)

    def until_idle(self, count: int) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: self.waits >= count, timeout=10)


class LocalDiscovery:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.statuses = {"catalog": 200, "one": 200, "two": 200}
        self.models = {"catalog": ["catalog-old"], "one": ["local-one"], "two": ["local-two"]}
        self.blocks: dict[str, tuple[threading.Event, threading.Event]] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                source = path.strip("/").split("/")[0]
                primary = path == "/catalog" or path.endswith("/health")
                with owner.lock:
                    owner.counts[path] += 1
                    status = owner.statuses[source] if primary else 200
                    models = list(owner.models[source])
                    block = owner.blocks.get(source) if primary else None
                if block:
                    block[0].set()
                    if not block[1].wait(10):
                        status = 504
                payload: Any = {"data": [{"id": model} for model in models]}
                if path.endswith("/health") or path.endswith("/props"):
                    payload = {}
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def attempts(self) -> tuple[int, int, int]:
        with self.lock:
            return tuple(self.counts[path] for path in ("/catalog", "/one/health", "/two/health"))

    def fail(self, *sources: str) -> None:
        with self.lock:
            for source in sources:
                self.statuses[source] = 503

    def recover(self) -> None:
        with self.lock:
            self.statuses = dict.fromkeys(self.statuses, 200)
            self.models["catalog"] = ["catalog-new"]
            self.models["one"] = ["local-new"]

    def block(self, source: str) -> tuple[threading.Event, threading.Event]:
        entered, release = threading.Event(), threading.Event()
        with self.lock:
            self.blocks[source] = (entered, release)
        return entered, release

    def close(self) -> None:
        for _, release in self.blocks.values():
            release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(10)
        assert not self.thread.is_alive()


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalDiscovery]:
    monkeypatch.delenv("MODEL_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_ROUTER_API_KEYS", raising=False)
    local = LocalDiscovery()
    try:
        yield local
    finally:
        local.close()


def config(upstream: LocalDiscovery, *, catalog_interval: float = 10,
           llama_interval: float = 10) -> mr.Config:
    return mr.build_config({
        "providers": {"static": {"dialect": "openai-chat", "base_url": upstream.url + "/v1",
                                 "api_key": "dummy-static-key"}},
        "models": [{"name": "explicit", "provider": "static"}],
        "catalog": {"url": upstream.url + "/catalog?api_key=dummy-catalog-key",
                    "headers": {"x-api-key": "dummy-catalog-key"}, "timeout": 5,
                    "refresh_seconds": catalog_interval,
                    "rules": [{"match": ".", "provider": "static"}]},
        "llama_cpp": {"servers": [{"url": upstream.url + "/one", "name": "one"},
                                  {"url": upstream.url + "/two", "name": "two"}],
                      "timeout": 5, "refresh_seconds": llama_interval},
    })


@contextmanager
def application(upstream: LocalDiscovery, **kwargs: Any) -> Iterator[tuple[mr.App, FakeClock]]:
    clock = FakeClock()
    app = mr.App(config(upstream, **kwargs), clock=clock, jitter=lambda low, high: high)
    try:
        yield app, clock
    finally:
        app.close()


def read_burst(app: mr.App) -> None:
    def read(_: int) -> None:
        assert app.health_payload()["status"] == "ok"
        app.models_payload()
        assert app.resolve("explicit") is not None
        app.resolve("catalog-old")
        app.resolve("local-one")
        assert app.resolve("never-discovered") is None

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(read, range(128)))


@pytest.mark.parametrize("failure", ["http", "malformed"])
def test_success_failure_backoff_recovery(upstream: LocalDiscovery, failure: str) -> None:
    with application(upstream) as (app, clock):
        assert app.catalog_status()["refresh"]["last_success"] == 0
        assert app.llama_cpp_status()["refresh"]["last_success"] == 0
        assert upstream.attempts() == (1, 1, 1)
        upstream.fail("one", "two")
        if failure == "http":
            upstream.fail("catalog")
        else:
            # A successful HTTP response with no usable model list also backs off.
            original_get = app.http.get

            def malformed(url: str, **kwargs: Any) -> httpx.Response:
                if "/catalog" in url:
                    return httpx.Response(200, json={"wrong": []}, request=httpx.Request("GET", url))
                return original_get(url, **kwargs)

            app.http.get = malformed
        clock.advance_to(10)
        for count in range(1, 11):
            before_attempts = upstream.attempts()
            app.refresh_catalog()
            app.refresh_llama_cpp()
            for status in (app.catalog_status(), app.llama_cpp_status()):
                refresh = status["refresh"]
                assert status["loaded"] is True
                assert status["error"]
                assert "dummy-catalog-key" not in json.dumps(status)
                assert refresh["last_success"] == 0
                assert refresh["last_attempt"] == clock()
                assert refresh["consecutive_failures"] == count
                assert refresh["stale"] is True
                assert refresh["age_seconds"] == clock()
                assert refresh["stale_age_seconds"] == clock() - 10
                delay = min(5 * 2 ** (count - 1), 300)
                assert refresh["next_attempt"] == clock() + delay
            assert app.resolve("catalog-old") is not None
            assert app.resolve("local-one") is not None
            after_attempts = upstream.attempts()
            assert after_attempts[1:] == tuple(value + 1 for value in before_attempts[1:])
            if failure == "http":
                assert after_attempts[0] == before_attempts[0] + 1
            read_burst(app)
            app.refresh_catalog()
            app.refresh_llama_cpp()
            assert upstream.attempts() == after_attempts
            deadline = app.catalog_status()["refresh"]["next_attempt"]
            clock.advance_to(deadline)
        if failure == "malformed":
            app.http.get = original_get
        upstream.recover()
        app.refresh_catalog()
        app.refresh_llama_cpp()
        for status in (app.catalog_status(), app.llama_cpp_status()):
            assert status["error"] is None
            assert status["refresh"]["consecutive_failures"] == 0
            assert status["refresh"]["last_success"] == clock()
            assert status["refresh"]["next_attempt"] == clock() + 10
            assert status["refresh"]["stale"] is False
        assert app.resolve("catalog-new") is not None
        assert app.resolve("catalog-old") is None
        assert app.resolve("local-new") is not None
        assert app.resolve("local-one") is None
        upstream.fail("catalog", "one", "two")
        clock.advance_to(clock() + 10)
        app.refresh_catalog()
        app.refresh_llama_cpp()
        assert app.catalog_status()["refresh"]["next_attempt"] == clock() + 5
        assert app.llama_cpp_status()["refresh"]["next_attempt"] == clock() + 5


def test_partial_failures_keep_per_server_success_and_independent_schedules(upstream: LocalDiscovery) -> None:
    with application(upstream, catalog_interval=20) as (app, clock):
        upstream.fail("one")
        with upstream.lock:
            upstream.models["two"] = ["local-two-new"]
        clock.advance_to(10)
        app.refresh_catalog()
        app.refresh_llama_cpp()
        assert upstream.attempts() == (1, 2, 2)
        assert app.catalog_status()["refresh"]["next_attempt"] == 20
        state = app.llama_cpp_status()
        assert state["refresh"]["last_success"] == 0  # last complete success
        assert state["refresh"]["next_attempt"] == 15
        one, two = state["servers"]
        assert one["last_success"] == 0
        assert one["stale"] is True
        assert two["last_success"] == 10
        assert two["stale"] is False
        assert app.resolve("local-one") is not None
        assert app.resolve("local-two-new") is not None
        assert app.resolve("local-two") is None
        clock.advance_to(15)
        app.refresh_catalog()
        app.refresh_llama_cpp()
        assert upstream.attempts() == (1, 3, 3)
        assert app.llama_cpp_status()["refresh"]["consecutive_failures"] == 2
        assert app.llama_cpp_status()["refresh"]["next_attempt"] == 25
        assert app.llama_cpp_status()["servers"][0]["stale_age_seconds"] == 5
        clock.advance_to(20)
        app.refresh_catalog()
        app.refresh_llama_cpp()
        assert upstream.attempts() == (2, 3, 3)


@pytest.mark.parametrize("interval", [0, 10])
def test_startup_failure_has_no_discovered_routes_and_respects_mode(upstream: LocalDiscovery, interval: int) -> None:
    upstream.fail("catalog", "one", "two")
    with application(upstream, catalog_interval=interval, llama_interval=interval) as (app, clock):
        for status in (app.catalog_status(), app.llama_cpp_status()):
            assert status["loaded"] is False
            assert status["refresh"]["last_success"] is None
            assert status["refresh"]["last_attempt"] == 0
            assert status["refresh"]["age_seconds"] is None
            assert status["refresh"]["next_attempt"] == (5 if interval else None)
        assert app.resolve("local-one") is None
        assert app.resolve("catalog-old") is None
        assert app.resolve("one/manual") is None
        assert app.resolve("explicit") is not None
        assert [row["id"] for row in app.models_payload()["data"]] == ["explicit"]
        app.start()
        if interval == 0:
            assert app._refresh_worker is None
            clock.advance_to(10000)
            app.refresh_catalog()
            app.refresh_llama_cpp()
            read_burst(app)
            assert upstream.attempts() == (1, 1, 1)
            # Startup-only still allows an explicit administrative refresh.
            upstream.recover()
            app.refresh_catalog(force=True)
            assert app.resolve("catalog-new") is not None
            assert app.catalog_status()["refresh"]["next_attempt"] is None
        else:
            app.stop()
            upstream.recover()
            clock.advance_to(5)
            wakeup = ObservedWakeup()
            app._refresh_wakeup = wakeup
            app.start()
            wakeup.until_idle(1)
            assert upstream.attempts() == (2, 2, 2)
            assert app.resolve("local-new") is not None
            assert app.resolve("catalog-new") is not None


def test_first_partial_startup_success_is_usable_and_still_backs_off(upstream: LocalDiscovery) -> None:
    upstream.fail("one")
    with application(upstream) as (app, clock):
        status = app.llama_cpp_status()
        assert status["loaded"] is True
        assert status["refresh"]["last_success"] is None
        assert status["refresh"]["next_attempt"] == 5
        assert app.resolve("local-one") is None
        assert app.resolve("local-two") is not None
        assert app.resolve("two/manual") is not None
        assert app.resolve("one/manual") is None


def test_attempt_time_and_retry_deadline_are_independent_of_completion(upstream: LocalDiscovery) -> None:
    with application(upstream) as (app, clock):
        upstream.fail("catalog")
        entered, release = upstream.block("catalog")
        clock.advance_to(10)
        before = app._registry
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(app.refresh_catalog)
            try:
                assert entered.wait(10)
                clock.advance_to(40)
                assert app._registry is before
                status = app.catalog_status()
                assert status["refresh"]["last_attempt"] == 0
                assert status["refresh"]["stale_age_seconds"] == 30
                read_burst(app)
            finally:
                release.set()
            pending.result(timeout=10)
        status = app.catalog_status()
        assert status["refresh"]["last_success"] == 0
        assert status["refresh"]["last_attempt"] == 10
        assert status["refresh"]["next_attempt"] == 45
        app.refresh_catalog()
        assert upstream.attempts()[0] == 2


def test_http_liveness_listing_and_unknown_resolution_never_refresh(upstream: LocalDiscovery) -> None:
    upstream.fail("catalog", "one", "two")
    with application(upstream, catalog_interval=0, llama_interval=0) as (app, clock):
        server = mr.serve(app, "127.0.0.1", 0)
        serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        serving.start()
        clock.advance_to(10000)

        def request(index: int) -> None:
            path = ("/health", "/healthz", "/v1/models", "/v1/chat/completions")[index % 4]
            connection = http.client.HTTPConnection(*server.server_address, timeout=10)
            try:
                if path == "/v1/chat/completions":
                    connection.request("POST", path, json.dumps({
                        "model": "local-one", "messages": [{"role": "user", "content": "test"}],
                    }), {"Content-Type": "application/json"})
                else:
                    connection.request("GET", path)
                response = connection.getresponse()
                body = json.loads(response.read())
                if path == "/v1/chat/completions":
                    assert response.status == 404
                    assert body["error"]["code"] == "model_not_found"
                elif path == "/v1/models":
                    assert response.status == 200
                    assert [row["id"] for row in body["data"]] == ["explicit"]
                else:
                    assert response.status == 200
                    assert body == {"status": "ok"}
            finally:
                connection.close()

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(request, range(64)))
            assert upstream.attempts() == (1, 1, 1)
        finally:
            server.shutdown()
            server.server_close()
            serving.join(10)
            assert not serving.is_alive()


@pytest.mark.parametrize("source", ["catalog", "one"])
def test_worker_deduplicates_refresh_and_serves_cached_routes_while_blocked(
    upstream: LocalDiscovery, source: str,
) -> None:
    with application(upstream) as (app, clock):
        wakeup = ObservedWakeup()
        app._refresh_wakeup = wakeup
        app.start()
        worker = app._refresh_worker
        app.start()
        assert app._refresh_worker is worker
        wakeup.until_idle(1)
        entered, release = upstream.block(source)
        upstream.fail("catalog", "one", "two")
        clock.advance_to(10)
        before = app._registry
        wakeup.set()
        assert entered.wait(10)
        try:
            # Network work is held indefinitely by the test; callers still finish.
            read_burst(app)
            if source == "one":
                assert app._registry is before  # no partially published llama round
            else:
                assert app._registry.catalog is before.catalog
            refresh = app.refresh_catalog if source == "catalog" else app.refresh_llama_cpp
            with ThreadPoolExecutor(max_workers=16) as pool:
                pending = [pool.submit(refresh, force=True) for _ in range(128)]
                for result in pending:
                    result.result(timeout=5)
            if source == "catalog":
                assert upstream.attempts()[0] == 2
            else:
                assert upstream.attempts()[1] == 2
        finally:
            release.set()
        wakeup.until_idle(2)
        assert upstream.attempts() == (2, 2, 2)
        assert app.catalog_status()["refresh"]["next_attempt"] == 15
        assert app.llama_cpp_status()["refresh"]["next_attempt"] == 15
        read_burst(app)
        assert upstream.attempts() == (2, 2, 2)
        upstream.recover()
        clock.advance_to(15)
        wakeup.set()
        wakeup.until_idle(3)
        assert upstream.attempts() == (3, 3, 3)
        assert app.catalog_status()["error"] is None
        assert app.llama_cpp_status()["error"] is None
        assert app.resolve("local-new") is not None


@pytest.mark.parametrize("source", ["catalog", "one"])
def test_close_joins_active_worker_before_transport_and_filter_cleanup(
    upstream: LocalDiscovery, source: str,
) -> None:
    old_filters = list(mr.LOG.filters)
    with application(upstream) as (app, clock):
        wakeup = ObservedWakeup()
        app._refresh_wakeup = wakeup
        app.start()
        wakeup.until_idle(1)
        worker = app._refresh_worker
        entered, release = upstream.block(source)
        clock.advance_to(10)
        wakeup.set()
        assert entered.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            closing = pool.submit(app.close)
            try:
                assert app._closed.wait(10)
                assert not closing.done()
                assert not app.http.is_closed
                assert app._log_filter in mr.LOG.filters
                read_burst(app)
            finally:
                release.set()
            closing.result(timeout=10)
        assert worker is not None and not worker.is_alive()
        assert app._refresh_worker is None
        assert app.http.is_closed
        assert mr.LOG.filters == old_filters
        before = upstream.attempts()
        clock.advance_to(1000)
        app.refresh_catalog(force=True)
        app.refresh_llama_cpp(force=True)
        assert upstream.attempts() == before
        if source == "one":
            assert before[0] == 1  # shutdown skips the next due source
        with pytest.raises(RuntimeError, match="closed router"):
            app.start()
        app.close()  # idempotent


def test_serve_owns_worker_start_and_stop(upstream: LocalDiscovery) -> None:
    with application(upstream) as (app, clock):
        server = mr.serve(app, "127.0.0.1", 0)
        try:
            worker = app._refresh_worker
            assert worker is not None and worker.is_alive()
        finally:
            server.server_close()  # also works without entering serve_forever
        assert not worker.is_alive()
        app.start()
        assert app._refresh_worker is not worker
        app.stop()
        assert not app.http.is_closed  # stop permits resuming; close releases it


def test_jitter_stays_in_bounded_backoff_range_and_varies(upstream: LocalDiscovery) -> None:
    with application(upstream) as (app, clock):
        app._jitter = random.Random(42).uniform
        upstream.fail("catalog")
        clock.advance_to(10)
        delays = []
        for failures in range(1, 14):
            app.refresh_catalog()
            deadline = app.catalog_status()["refresh"]["next_attempt"]
            delay = deadline - clock()
            ceiling = min(5 * 2 ** (failures - 1), 300)
            assert ceiling / 2 <= delay <= ceiling
            delays.append(delay)
            clock.advance_to(deadline)
        assert len(set(delays[-5:])) == 5


@pytest.mark.parametrize("source", ["catalog", "llama_cpp"])
@pytest.mark.parametrize("field", ["refresh_seconds", "timeout"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), threading.TIMEOUT_MAX * 2])
def test_scheduler_rejects_nonfinite_or_unwaitable_config(
    upstream: LocalDiscovery, source: str, field: str, value: float,
) -> None:
    if source == "catalog":
        raw = {"providers": {"static": {"dialect": "openai-chat", "base_url": upstream.url}},
               "catalog": {"url": upstream.url + "/catalog", field: value,
                           "rules": [{"match": ".", "provider": "static"}]}}
    else:
        raw = {"llama_cpp": {"servers": [upstream.url + "/one"], field: value}}
    with pytest.raises(SystemExit, match="finite"):
        mr.build_config(raw)
