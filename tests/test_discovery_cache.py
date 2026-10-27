"""Opt-in persistence contracts, with fake clocks and local discovery only."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

import model_router as mr
import model_router.config as router_configuration
import model_router.discovery as router_discovery
from test_discovery_refresh import FakeClock, LocalDiscovery, config, upstream  # noqa: F401


def cached_config(local: LocalDiscovery, path: Path, **kwargs) -> mr.Config:
    result = config(local, **kwargs)
    result.discovery_cache = mr.DiscoveryCacheConfig(str(path), 60)
    return result


@contextmanager
def application(settings: mr.Config, wall: float = 1000) -> Iterator[mr.App]:
    app = mr.App(settings, clock=FakeClock(), wall_clock=lambda: wall)
    try:
        yield app
    finally:
        app.close()


def cache_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))
    path.chmod(0o600)


def entries(app: mr.App) -> dict:
    return {entry["id"]: entry for entry in app.models_payload()["data"]}


def test_offline_restart_and_live_recovery(upstream: LocalDiscovery, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "discovery.json"
    settings = cached_config(upstream, path)
    with application(settings) as app:
        assert not app.catalog_status()["cached"]
        original = entries(app)
        assert path.exists()
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    upstream.fail("catalog", "one", "two")
    with application(settings, 1010) as app:
        assert app._refresh_worker is None
        assert set(entries(app)) == set(original)
        assert app.diagnostics_payload()["discovery_cache"]["load"] == "loaded"
        assert app.catalog_status()["cached"]
        for name in ("catalog-old", "local-one", "local-two"):
            route = app.resolve(name)
            assert route is not None
            assert route._provider.api_key == ("dummy-static-key" if name == "catalog-old" else "")
            freshness = entries(app)[name]["metadata"]["router"]["freshness"]["discovery"]
            assert freshness["cached"] is True
            assert freshness["stale"] is True
            assert freshness["age_seconds"] == 10
            assert freshness["succeeded_at"] == 1000
        assert entries(app)["local-one"]["context_window"] == original["local-one"]["context_window"]
        assert all(s["cached"] and not s["reachable"] for s in app.llama_cpp_status()["servers"])
        upstream.recover()
        app.refresh_catalog(force=True)
        app.refresh_llama_cpp(force=True)
        assert app.resolve("catalog-old") is None
        assert app.resolve("local-one") is None
        for name in ("catalog-new", "local-new", "local-two"):
            assert entries(app)[name]["metadata"]["router"]["freshness"]["discovery"]["cached"] is False
        assert all(not s["cached"] for s in app.llama_cpp_status()["servers"])
    # A subsequent offline restart uses the replacement live inventory.
    upstream.fail("catalog", "one", "two")
    with application(settings, 1020) as app:
        assert app.resolve("catalog-old") is None
        assert app.resolve("local-new") is not None
        assert app.catalog_status()["succeeded_at"] == 1010


@pytest.mark.parametrize("damage, expected", [
    ("json", "invalid"), ("schema", "incompatible"), ("schema_bool", "incompatible"),
    ("shape", "invalid"), ("fact", "invalid"), ("date", "invalid"),
    ("permissions", "invalid"), ("symlink", "invalid"), ("future", "invalid"),
])
def test_corrupt_incompatible_and_unsafe_files(upstream: LocalDiscovery, tmp_path: Path,
                                               damage: str, expected: str) -> None:
    path = tmp_path / "cache.json"
    settings = cached_config(upstream, path)
    with application(settings):
        pass
    payload = json.loads(path.read_text())
    if damage == "json":
        path.write_text('{"truncated":')
    elif damage in ("schema", "schema_bool"):
        payload["schema_version"] = 999 if damage == "schema" else True
        cache_json(path, payload)
    elif damage == "shape":
        payload["llama"]["0"]["models"] = ["broken"]
        cache_json(path, payload)
    elif damage == "fact":
        del payload["llama"]["0"]["models"][0]["capabilities"]["tools"]
        cache_json(path, payload)
    elif damage == "date":
        payload["catalog"]["models"][0]["capabilities"]["sources"]["catalog"]["retrieved_at"] = "bad date"
        cache_json(path, payload)
    elif damage == "future":
        payload["catalog"]["succeeded_at"] = 1011
        cache_json(path, payload)
    elif damage == "permissions":
        path.chmod(0o644)
    elif damage == "symlink":
        target = tmp_path / "target.json"
        path.rename(target)
        path.symlink_to(target)
    upstream.fail("catalog", "one", "two")
    with application(settings, 1010) as app:
        assert app.resolve("catalog-old") is None
        assert app.resolve("local-one") is None
        assert app.resolve("explicit") is not None
        assert app.diagnostics_payload()["discovery_cache"]["load"] == expected
        assert app.http.is_closed is False
    # A successful discovery can replace even a broken/symlink cache safely.
    upstream.recover()
    with application(settings, 1011) as app:
        assert app.resolve("catalog-new") is not None
    assert not path.is_symlink()
    assert json.loads(path.read_text())["schema_version"] == mr.DISCOVERY_CACHE_VERSION
    assert path.stat().st_mode & 0o777 == 0o600






def test_opt_in_and_startup_only(upstream: LocalDiscovery, tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    with application(config(upstream)) as app:
        assert app.diagnostics_payload()["discovery_cache"]["enabled"] is False
    assert not path.exists()
    settings = cached_config(upstream, path, catalog_interval=0, llama_interval=0)
    with application(settings):
        pass
    upstream.fail("catalog", "one", "two")
    with application(settings, 1010) as app:
        app.start()
        assert app._refresh_worker is None
        assert app.resolve("local-one") is not None
        assert entries(app)["catalog-old"]["metadata"]["router"]["freshness"]["discovery"]["stale"]




def test_write_failure_preserves_previous_cache_and_live_routes(upstream: LocalDiscovery, tmp_path: Path,
                                                               monkeypatch) -> None:
    path = tmp_path / "cache.json"
    settings = cached_config(upstream, path)
    with application(settings):
        pass
    before = path.read_bytes()
    original_replace = os.replace
    def fail_replace(source, destination):
        # A complete private temporary file exists, but the old cache is intact.
        assert Path(source).stat().st_mode & 0o777 == 0o600
        assert path.read_bytes() == before
        raise OSError("fake-key-never-logged")
    monkeypatch.setattr(os, "replace", fail_replace)
    upstream.recover()
    with application(settings, 1010) as app:
        assert app.resolve("catalog-new") is not None
        assert app.diagnostics_payload()["discovery_cache"]["write_error"] == "cache write failed"
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(os, "replace", original_replace)






def test_no_credentials_urls_or_freeform_metadata_persisted(upstream: LocalDiscovery, tmp_path: Path,
                                                           monkeypatch) -> None:
    import httpx
    path = tmp_path / "cache.json"
    settings = cached_config(upstream, path)
    origin = upstream.url.replace("http://", "http://fake-user:fake-password@")
    settings.catalog = replace(settings.catalog,
        url=origin + "/catalog?unusual-auth-field=opaque-query-credential",
        headers={"x-custom-auth": "opaque-header-credential"})
    settings.providers["static"] = replace(settings.providers["static"],
        base_url=origin + "/v1?unusual-auth-field=opaque-endpoint-credential",
        headers={"x-custom-auth": "opaque-provider-header"})
    llama = settings.llama_cpp
    settings.llama_cpp = replace(llama, headers={"x-auth": "opaque-local-header"},
        servers=[replace(server, url=server.url.replace("http://", "http://local-user:local-password@"),
                         api_key="opaque-local-key") for server in llama.servers])
    with application(settings) as app:
        get = app.http.get
        def metadata_get(url, **kwargs):
            if "/catalog" in url:
                return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{
                    "id": "catalog-old", "metadata": {
                        "context_window": 4096, "display_name": "Public model",
                        "headers": {"x-secret": "unknown-upstream-secret"},
                        "url": "https://name:unconfigured-password@example.invalid/?anything=private",
                        "notes": "private-freeform-data", "authorization": "Bearer unknown-token",
                    }}]})
            return get(url, **kwargs)
        monkeypatch.setattr(app.http, "get", metadata_get)
        app.refresh_catalog(force=True)
        data = path.read_text()
        assert app.diagnostics_payload()["discovery_cache"]["write_error"] is None
        assert "Public model" in data and "4096" in data
        for secret in ("fake-user", "fake-password", "opaque-query-credential", "opaque-header-credential",
                       "opaque-endpoint-credential", "opaque-provider-header", "opaque-local-header",
                       "opaque-local-key", "dummy-static-key", "local-user", "local-password",
                       "unknown-upstream-secret", "unconfigured-password", "private-freeform-data", "unknown-token"):
            assert secret not in data
        assert "http://" not in data and "https://" not in data
        assert "headers" not in data and "api_key" not in data
    upstream.fail("catalog", "one", "two")
    with application(settings, 1010) as app:
        route = app.resolve("catalog-old")
        assert route._provider is settings.providers["static"]
        assert route.metadata["context_window"] == 4096
        assert "notes" not in route.metadata
        route = app.resolve("local-one")
        assert route._provider.base_url == router_configuration._llama_base_url(settings.llama_cpp.servers[0].url)
        assert route._provider.api_key == "opaque-local-key"
        assert route._provider.headers["x-auth"] == "opaque-local-header"


def test_cache_cannot_supply_transport_configuration(upstream: LocalDiscovery, tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    settings = cached_config(upstream, path)
    with application(settings):
        pass
    data = json.loads(path.read_text())
    data["catalog"]["models"][0]["base_url"] = "http://127.0.0.1:1/untrusted"
    cache_json(path, data)
    upstream.fail("catalog", "one", "two")
    with application(settings, 1010) as app:
        assert app.resolve("catalog-old") is None
        assert app.diagnostics_payload()["discovery_cache"]["load"] == "invalid"




