"""Discovery, cache and atomic registry snapshots; no generation dispatch."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qsl, unquote, urlsplit

import httpx

import router_metadata

from . import gemini
from .model_names import MODEL_DISPLAY_NAMES
from .common import (
    ErrorSanitizer,
    RouterError,
    _DiagnosticLogFilter,
    _error_message,
    _freeze,
    _sanitize_details,
    _thaw,
)
from .config import (
    CatalogConfig,
    Config,
    LlamaCppConfig,
    LlamaCppServer,
    Provider,
    Route,
    _config_finite,
    _config_secrets,
    _llama_base_url,
    _llama_provider_names,
)
from .constants import (
    DEFAULT_TIMEOUT,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
    DISCOVERY_CACHE_MAX_BYTES,
    DISCOVERY_CACHE_VERSION,
    LLAMA_LIMIT_FIELDS,
    LOG,
    REFRESH_BACKOFF_BASE,
    REFRESH_BACKOFF_MAX,
    __version__,
)


@dataclass(frozen=True)
class RefreshState:
    # Monotonic process-local times; None also works with clocks starting at 0.
    last_success: float | None = None
    last_attempt: float | None = None
    next_attempt: float | None = None
    consecutive_failures: int = 0


@dataclass(frozen=True)
class CatalogSnapshot:
    routes: Mapping[str, Route] = field(default_factory=lambda: _freeze({}))
    refresh: RefreshState = field(default_factory=RefreshState)
    error: str | None = None
    unmapped: frozenset[str] = frozenset()
    succeeded_at: float | None = None  # Wall time, for disk age across processes.
    cached: bool = False


@dataclass(frozen=True)
class LlamaSnapshot:
    providers: Mapping[str, Provider] = field(default_factory=lambda: _freeze({}))
    routes: Mapping[str, Route] = field(default_factory=lambda: _freeze({}))
    public: tuple[Route, ...] = ()
    by_server: Mapping[str, tuple[Provider, Mapping[str, Route], tuple[Route, ...]]] = field(
        default_factory=lambda: _freeze({})
    )
    servers: tuple[Mapping[str, Any], ...] = ()
    refresh: RefreshState = field(default_factory=RefreshState)
    error: str | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    providers: Mapping[str, Provider]
    routes: Mapping[str, Route]
    public: tuple[Route, ...]
    catalog: CatalogSnapshot
    llama: LlamaSnapshot


def _llama_mode(props: Any, rows: list[Mapping[str, Any]]) -> str:
    """Classify a llama.cpp endpoint the same way pi-llama-cpp does."""
    if isinstance(props, Mapping):
        if props.get("src") == "llama-swap":
            return "llama-swap"
        if props.get("role") == "router":
            return "router"
    if rows and "max_model_len" in rows[0]:
        return "legacy"
    return "single"


def _llama_arg(args: Any, flags: tuple[str, ...]) -> int | None:
    """Read the last occurrence, including equals syntax; never execute args.

    A final invalid/auto value must not revive an earlier positive value.
    """
    if not isinstance(args, (list, tuple)):
        return None
    result = None
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        flag, sep, value = arg.partition("=")
        if flag not in flags:
            continue
        if not sep:
            value = args[index + 1] if index + 1 < len(args) else None
        try:
            result = int(value) if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value) else None
        except ValueError:  # also rejects integer strings beyond Python's size guard
            result = None
    return result


def _llama_context_from_args(args: Any) -> int | None:
    value = _llama_arg(args, ("--ctx-size", "-c"))
    return value if value is not None and value > 0 else None


def _limit_number(value: Any, *, minimum: int = 1) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    return int(value) if value >= minimum else None


def _llama_limits(
    row: Mapping[str, Any], props: Any, mode: str, fallback: int,
    overrides: Mapping[str, int | None], *, per_model_props: bool = False,
) -> dict[str, Any]:
    """Keep reported runtime capacity, training metadata and defaults distinct.

    Router/swap global props cannot describe any particular model. The caller
    must explicitly identify props fetched for that model with autoload=false.
    """
    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    meta = mapping(row.get("meta"))
    metadata = mapping(row.get("metadata"))
    props = mapping(props) if mode in ("single", "legacy") or per_model_props else {}
    props_source = "/props?model=<id>" if per_model_props else "/props"
    defaults = mapping(props.get("default_generation_settings"))
    params = mapping(defaults.get("params"))
    args = mapping(row.get("status")).get("args")
    result: dict[str, Any] = {
        key: {"value": None, "source": None, "kind": "unknown"}
        for key in LLAMA_LIMIT_FIELDS
    }

    def choose(key: str, candidates: list[tuple[Any, str]], *, minimum: int = 1) -> None:
        for raw, source in candidates:
            value = _limit_number(raw, minimum=minimum)
            if value is not None:
                result[key] = {"value": value, "source": source, "kind": "reported"}
                return

    context = [(meta.get("n_ctx"), "/v1/models.meta.n_ctx")]
    if mode == "legacy":
        context.insert(0, (row.get("max_model_len"), "/v1/models.max_model_len"))
    choose("context_window", context + [
        (defaults.get("n_ctx"), props_source + ".default_generation_settings.n_ctx"),
        (props.get("n_ctx"), props_source + ".n_ctx"),
    ])
    if result["context_window"]["value"] is None:
        requested = _llama_context_from_args(args)
        result["context_window"] = (
            {"value": requested, "source": "/v1/models.status.args.ctx-size", "kind": "inferred"}
            if requested is not None else
            {"value": fallback, "source": "llama_cpp.context_fallback", "kind": "fallback"}
        )
    choose("trained_context_window", [
        (meta.get("n_ctx_train"), "/v1/models.meta.n_ctx_train"),
        # llama.cpp router listings use context_length for trained capacity.
        (row.get("context_length") if mode == "router" else None, "/v1/models.context_length"),
        (props.get("n_ctx_train"), props_source + ".n_ctx_train"),
    ])
    # These names explicitly denote hard limits. Ambiguous max_tokens and
    # n_predict are never promoted to maximum output capacity.
    for key in ("max_input_tokens", "max_output_tokens"):
        choose(key, [(meta.get(key), "/v1/models.meta." + key),
                     (row.get(key), "/v1/models." + key),
                     (metadata.get(key), "/v1/models.metadata." + key),
                     (props.get(key), props_source + "." + key)])

    for raw, source in [
        (params.get("n_predict"), props_source + ".default_generation_settings.params.n_predict"),
        (defaults.get("n_predict"), props_source + ".default_generation_settings.n_predict"),
        (params.get("max_tokens"), props_source + ".default_generation_settings.params.max_tokens"),
        (defaults.get("max_tokens"), props_source + ".default_generation_settings.max_tokens"),
        (_llama_arg(args, ("--predict", "--n-predict", "-n")), "/v1/models.status.args.n-predict"),
    ]:
        value = _limit_number(raw, minimum=-2)
        if value is None:
            continue
        result["default_output_tokens"] = {
            "value": value if value >= 0 else None, "source": source,
            "kind": "inferred" if source.startswith("/v1/models.status.args") else "reported",
        }
        if value < 0:
            result["default_output_tokens"]["special"] = "unlimited" if value == -1 else "until_context"
        break

    for key, value in overrides.items():
        result[key] = {"value": value, "source": "llama_cpp.limit_overrides", "kind": "override"}

    fit = _llama_arg(args, ("--fit-ctx", "-fitc"))
    result["constraints"] = {}
    if fit is not None and fit > 0:
        result["constraints"]["fit_context_min"] = {
            "value": fit, "source": "/v1/models.status.args.fit-ctx", "kind": "constraint",
        }
    return result


def _llama_capabilities(row: Mapping[str, Any], mode: str) -> list[str]:
    architecture = row.get("architecture")
    if isinstance(architecture, Mapping):
        modalities = architecture.get("input_modalities")
        if isinstance(modalities, (list, tuple)):
            return ["text", "image"] if "image" in modalities else ["text"]
    if mode == "legacy":
        capabilities = row.get("capabilities")
        if isinstance(capabilities, (list, tuple)) and any(
            str(item).lower() == "multimodal" for item in capabilities
        ):
            return ["text", "image"]
    return ["text"]


def _llama_model_aliases(row: Mapping[str, Any]) -> list[str]:
    """Aliases reported by llama.cpp, preferring llama-swap's nested list."""
    meta = row.get("meta")
    candidates: Any = None
    if isinstance(meta, Mapping):
        llamaswap = meta.get("llamaswap")
        if isinstance(llamaswap, Mapping):
            candidates = llamaswap.get("aliases")
    if not isinstance(candidates, (list, tuple)):
        candidates = row.get("aliases")
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        return []
    return [str(alias).strip() for alias in candidates if str(alias).strip()]


def _discovery_cache_identity(config: Config) -> str:
    """Conservatively invalidate the entire inventory on discovery/routing edits.

    Only a digest reaches disk, including for URLs and header values. Rotating
    a credential also invalidates the cache (it may select a different tenant).
    Discovery intervals, ingress auth and retry settings do not affect identity.
    """
    catalog, llama = config.catalog, config.llama_cpp
    identity = {
        "providers": {name: _thaw(vars(provider)) for name, provider in config.providers.items()},
        "models": {name: [route.name, route.provider_name, route.upstream_model,
                          route.dialect, route.enabled] for name, route in config.routes.items()},
        "catalog": {"url": catalog.url, "headers": catalog.headers,
                    "model_providers": catalog.model_providers,
                    "rules": [[r.pattern.pattern, r.pattern.flags, r.provider_name] for r in catalog.rules],
                    "metadata_source": catalog.metadata_source} if catalog else None,
        # Order matters: the first configured local server wins model collisions.
        "llama": {"servers": [vars(server) for server in llama.servers],
                  "headers": llama.headers, "prefix": llama.prefix, "aliases": llama.expose_aliases,
                  "context_fallback": llama.context_fallback,
                  "limit_overrides": _thaw(llama.limit_overrides)} if llama else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _discovery_cache_sanitizer(config: Config) -> ErrorSanitizer:
    secrets = _config_secrets(config)
    urls = [p.base_url for p in config.providers.values()]
    if config.catalog:
        urls.append(config.catalog.url)
    if config.llama_cpp:
        urls.extend(s.url for s in config.llama_cpp.servers)
    for url in urls:
        parsed = urlsplit(url)
        # Unknown query/header names can still carry credentials. Guard
        # against upstream metadata echoing even a bare query value/user name.
        secrets.extend(unquote(v) for v in (parsed.username, parsed.password) if v)
        secrets.extend(v for _, v in parse_qsl(parsed.query, keep_blank_values=True) if v)
    return ErrorSanitizer(secrets)


def _cache_text(value: Any, sanitize: ErrorSanitizer) -> bool:
    # Never persist full URLs, even with unknown credential parameter names, or
    # strings changed by the application's secret sanitizer. Model IDs must not
    # be rewritten: omit an unsafe entry rather than changing its destination.
    return (isinstance(value, str) and bool(value) and len(value) <= 2048
            and not re.search(r"(?i)https?://", value) and sanitize(value) == value)


def _cache_contract(raw: Mapping[str, Any], dialect: str, sanitize: ErrorSanitizer) -> dict[str, Any]:
    """Project generated facts; never persist arbitrary upstream metadata/props.

    Unknown fields, freeform stop strings and conflict histories are omitted.
    The same projection validates disk input before it can reach a renderer.
    """
    result = router_metadata.empty(dialect)
    for path, validate in {**router_metadata.FIELDS, **{
        "limits." + key: lambda v: type(v) is int and v >= 0 for key in LLAMA_LIMIT_FIELDS
    }}.items():
        if path.startswith("sampling.dry_sequence_breakers."):
            continue
        value: Any = raw
        for key in path.split("."):
            value = value[key]
        if not isinstance(value, Mapping) or not {"value", "source", "kind"} <= value.keys():
            raise ValueError("invalid cached capability fact")
        fact_value = _thaw(value["value"])
        if fact_value is not None and not validate(fact_value):
            raise ValueError("invalid cached capability value")
        # Native parameter objects have fixed keys verified by FIELDS. Inspect
        # strings at every depth too, including default enums and native values.
        def safe_strings(item: Any) -> bool:
            if isinstance(item, str):
                return _cache_text(item, sanitize)
            if isinstance(item, Mapping):
                return all(_cache_text(k, sanitize) and safe_strings(v) for k, v in item.items())
            if isinstance(item, (list, tuple)):
                return all(safe_strings(v) for v in item)
            return True
        if not safe_strings(fact_value):
            continue
        source, kind = value["source"], value["kind"]
        if source is not None and not _cache_text(source, sanitize):
            continue
        if not _cache_text(kind, sanitize):
            continue
        router_metadata.put(result, path, fact_value, source, kind)
    sources = raw.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("invalid cached provenance")
    for name in ("catalog", "go.review", "llama.models", "llama.props", "strata.compat"):
        if name not in sources:
            continue
        source = sources[name]
        if not isinstance(source, Mapping):
            raise ValueError("invalid cached source")
        result["sources"][name] = {key: value for key, value in source.items()
            if key in {"retrieved_at", "stale_after_seconds", "verification", "endpoint", "available", "applicable"}
            and (value is None or type(value) in (int, float, bool)
                 or _cache_text(value, sanitize))}
    # Validate dates/TTLs and the complete shape before admission/publication.
    router_metadata.render(result, {}, {"stale": True}, time.time())
    return result


def _cache_metadata(raw: Mapping[str, Any], sanitize: ErrorSanitizer) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if raw.get("engine") in ("llama.cpp", "strata"):
        result["engine"] = raw["engine"]
    for key in (*LLAMA_LIMIT_FIELDS, "max_tokens"):
        value = raw.get(key)
        if key in raw and (value is None or type(value) is int and value >= 0):
            result[key] = value
    if _cache_text(raw.get("display_name"), sanitize):
        result["display_name"] = raw["display_name"]
    if isinstance(raw.get("capabilities"), (list, tuple)):
        result["capabilities"] = [v for v in raw["capabilities"] if v in ("text", "image")]
    # Preserve limit facts through the same typed capability projection.
    if isinstance(raw.get("limits"), Mapping):
        contract = router_metadata.empty(DIALECT_OPENAI_CHAT)
        contract["limits"] = raw["limits"]
        result["limits"] = _cache_contract(contract, DIALECT_OPENAI_CHAT, sanitize)["limits"]
    local = raw.get("llama_cpp")
    if isinstance(local, Mapping):
        result["llama_cpp"] = {k: v for k, v in local.items()
            if k in {"mode", "status", "n_ctx"} and
            (v is None or type(v) is int or _cache_text(v, sanitize))}
    return result


def _atomic_discovery_cache_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DiscoveryRegistry:
    """Own discovery state, snapshot publication and probe-client lifetime."""

    def __init__(
        self, config: Config, *, clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._clock = clock
        self._jitter = jitter
        self._sleep, self._wall_clock = sleep, wall_clock
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._refresh_wakeup = threading.Event()
        self._refresh_worker: threading.Thread | None = None
        self.sanitize = ErrorSanitizer(_config_secrets(config))
        self._log_filter = _DiagnosticLogFilter(self.sanitize)
        # Logger filters do not propagate to children. Include the dependency
        # loggers themselves, which can log HTTP URLs and adapter exceptions.
        names = {"model-router", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy",
                 "httpx", "httpcore", "openai"}
        names.update(name for name in logging.Logger.manager.loggerDict
                     if name.startswith(("LiteLLM", "litellm", "httpx", "httpcore", "openai")))
        self._diagnostic_loggers = [logging.getLogger(name) for name in names]
        for logger in self._diagnostic_loggers:
            logger.addFilter(self._log_filter)
        self.http = httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self.verbose = False
        self._registry_lock = threading.Lock()
        self._llama_names = _llama_provider_names(config)
        self._registry = RegistrySnapshot(
            providers=_freeze({}), routes=_freeze({}), public=(),
            catalog=CatalogSnapshot(), llama=LlamaSnapshot(),
        )
        self._publish_registry()
        self._catalog_lock = threading.Lock()
        self._llama_lock = threading.Lock()
        self._llama_unreachable: set[str] = set()
        self._llama_counts: dict[str, int] = {}
        self._cache_lock = threading.Lock()
        self._cache_sanitize = _discovery_cache_sanitizer(config) if config.discovery_cache else self.sanitize
        self._cache_identity = _discovery_cache_identity(config) if config.discovery_cache else None
        self._cache_status: dict[str, Any] = {"enabled": config.discovery_cache is not None,
                                              "load": "disabled", "write_error": None}
        # One synchronous initial inventory preserves startup/--check behavior.
        # No background thread is owned until start() (or serve()) is called.
        try:
            self._load_discovery_cache()
            self.refresh_llama_cpp(force=True)
            self.refresh_catalog(force=True)
        except BaseException:
            self.close()
            raise

    def _llama_provider(self, server: LlamaCppServer, *, engine: str = "llama.cpp") -> Provider:
        llama = self.config.llama_cpp
        assert llama is not None
        return Provider(
            name=self._llama_names[server.url], dialect=DIALECT_OPENAI_CHAT,
            base_url=_llama_base_url(server.url), api_key=server.api_key,
            headers={**llama.headers, **server.headers}, timeout=DEFAULT_TIMEOUT,
            litellm_provider="openai", litellm_kwargs={},
            drop_params=llama.compatibility == "lossy",
            llama_cpp=engine != "strata",
        )

    def _cached_route(self, row: Any, provider: Provider, prefix: str = "") -> Route:
        if (not isinstance(row, dict) or set(row) != {"id", "metadata", "capabilities", "aliases"}
                or not _cache_text(row["id"], self._cache_sanitize)
                or not isinstance(row["metadata"], dict) or not isinstance(row["capabilities"], dict)
                or not isinstance(row["aliases"], list)
                or any(not _cache_text(v, self._cache_sanitize) for v in row["aliases"])):
            raise ValueError("invalid cached model")
        return Route(name=prefix + row["id"], upstream_model=row["id"],
                     provider_name=provider.name, dialect=provider.dialect,
                     metadata=_cache_metadata(row["metadata"], self._cache_sanitize),
                     _capabilities=_cache_contract(row["capabilities"], provider.dialect, self._cache_sanitize))

    def _load_discovery_cache(self) -> None:
        cache = self.config.discovery_cache
        if cache is None:
            return
        try:
            fd = os.open(Path(cache.path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or info.st_size > DISCOVERY_CACHE_MAX_BYTES):
                    raise ValueError("unsafe cache file")
                data = stream.read(DISCOVERY_CACHE_MAX_BYTES + 1)
                if len(data) > DISCOVERY_CACHE_MAX_BYTES:
                    raise ValueError("cache too large")
                payload = json.loads(data)
            _config_finite(payload)
            if (not isinstance(payload, dict) or set(payload) != {"schema_version", "identity", "catalog", "llama"}
                    or type(payload["schema_version"]) is not int
                    or payload["schema_version"] != DISCOVERY_CACHE_VERSION):
                self._cache_status = {**self._cache_status, "load": "incompatible"}
                return
            if payload["identity"] != self._cache_identity:
                self._cache_status = {**self._cache_status, "load": "configuration_changed"}
                return
            wall, now = self._wall_clock(), self._clock()
            expired = 0
            def timestamp(unit: Any) -> float | None:
                nonlocal expired
                if not isinstance(unit, dict) or set(unit) != {"succeeded_at", "models"} or not isinstance(unit["models"], list):
                    raise ValueError("invalid cache inventory")
                at = unit["succeeded_at"]
                if type(at) not in (int, float) or not math.isfinite(at) or at > wall or at < 0:
                    raise ValueError("invalid cache time")
                if wall - at >= cache.max_age_seconds:
                    expired += 1
                    return None
                return float(at)

            catalog = CatalogSnapshot()
            unit = payload["catalog"]
            if unit is not None:
                at = timestamp(unit)
                if at is not None and self.config.catalog:
                    routes = {}
                    for row in unit["models"]:
                        model_id = row["id"]
                        name = self._catalog_provider_name(model_id)
                        if name is None:
                            continue
                        route = self._cached_route(row, self.config.providers[name])
                        if route.name in routes or row["aliases"]:
                            raise ValueError("duplicate cached catalog ID or alias")
                        routes[route.name] = route
                    catalog = CatalogSnapshot(routes=_freeze(routes),
                        refresh=RefreshState(last_success=now - (wall - at)),
                        error="cached discovery; live availability unverified", succeeded_at=at, cached=True)
            expected_servers = {str(i) for i, _ in enumerate(self.config.llama_cpp.servers)} if self.config.llama_cpp else set()
            if not isinstance(payload["llama"], dict) or payload["llama"].keys() - expected_servers:
                raise ValueError("invalid cached servers")
            providers, routes, by_server = {}, {}, {}
            public, states = [], []
            llama = self.config.llama_cpp
            for index, server in enumerate(llama.servers if llama else []):
                unit = payload["llama"].get(str(index))
                if unit is None:
                    continue
                at = timestamp(unit)
                if at is None:
                    continue
                engines = {_cache_metadata(row["metadata"], self._cache_sanitize).get("engine", "llama.cpp")
                           for row in unit["models"]}
                if len(engines) > 1:
                    raise ValueError("inconsistent cached local engines")
                provider = self._llama_provider(server, engine=next(iter(engines), "llama.cpp"))
                server_routes, server_public = {}, []
                for row in unit["models"]:
                    route = self._cached_route(row, provider, llama.prefix)
                    if row["aliases"] and (not llama.expose_aliases or
                            any(not alias.startswith(llama.prefix) for alias in row["aliases"])):
                        raise ValueError("invalid cached local aliases")
                    if route.name in server_routes:
                        raise ValueError("duplicate cached local ID")
                    server_public.append(route)
                    server_routes[route.name] = route
                    for alias in row["aliases"]:
                        if alias in server_routes:
                            raise ValueError("duplicate cached alias")
                        server_routes[alias] = route
                providers[provider.name] = provider
                by_server[server.url] = (provider, server_routes, tuple(server_public))
                for name, route in server_routes.items():
                    if name not in self.config.routes and name not in routes:
                        routes[name] = route
                public.extend(r for r in server_public if routes.get(r.name) is r)
                states.append({"url": server.url, "provider": provider.name, "models": len(server_public),
                    "reachable": False, "mode": None, "stale": True, "cached": True,
                    "error": "cached discovery; live availability unverified", "succeeded_at": at,
                    "last_success": now - (wall - at), "last_attempt": None, "consecutive_failures": 0})
            local = LlamaSnapshot(providers=_freeze(providers), routes=_freeze(routes), public=tuple(public),
                by_server=_freeze(by_server), servers=_freeze(states),
                error="cached discovery; live availability unverified" if states else None)
            # Publish only after the entire file has been validated/reconstructed.
            self._publish_registry(catalog=catalog, llama=local)
            self._cache_status = {**self._cache_status, "load": "loaded" if catalog.cached or states else "empty",
                                  "expired_sources": expired}
        except FileNotFoundError:
            self._cache_status = {**self._cache_status, "load": "missing"}
        except (Exception, SystemExit):
            # File contents, paths and parser errors may contain secrets. Keep
            # this deliberately generic, including on permission/schema errors.
            self._cache_status = {**self._cache_status, "load": "invalid"}
            LOG.warning("Discovery cache ignored: invalid, unsafe or unreadable file")

    def _write_discovery_cache(self) -> None:
        cache = self.config.discovery_cache
        if cache is None:
            return
        with self._cache_lock:
            try:
                snapshot = self._registry
                def inventory(at: float, public: Any, routes: Mapping[str, Route]) -> dict[str, Any]:
                    models = []
                    for route in public:
                        aliases = [name for name, r in routes.items() if r is route and name != route.name]
                        if not all(_cache_text(v, self._cache_sanitize) for v in [route.upstream_model, *aliases]):
                            continue
                        models.append({"id": route.upstream_model, "aliases": aliases,
                            "metadata": _cache_metadata(route.metadata, self._cache_sanitize),
                            "capabilities": _cache_contract(route._capabilities, route.dialect, self._cache_sanitize)})
                    return {"succeeded_at": at, "models": models}
                catalog = snapshot.catalog
                payload = {"schema_version": DISCOVERY_CACHE_VERSION, "identity": self._cache_identity,
                           "catalog": inventory(catalog.succeeded_at, catalog.routes.values(), catalog.routes)
                                      if catalog.succeeded_at is not None else None, "llama": {}}
                states = {s["url"]: s for s in snapshot.llama.servers}
                for index, server in enumerate(self.config.llama_cpp.servers if self.config.llama_cpp else []):
                    previous = snapshot.llama.by_server.get(server.url)
                    at = states.get(server.url, {}).get("succeeded_at")
                    if previous is not None and at is not None:
                        _, routes, public = previous
                        payload["llama"][str(index)] = inventory(at, public, routes)
                data = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
                if len(data) > DISCOVERY_CACHE_MAX_BYTES:
                    raise ValueError("cache too large")
                _atomic_discovery_cache_write(Path(cache.path).expanduser(), data)
                self._cache_status = {**self._cache_status, "write_error": None}
            except Exception:
                self._cache_status = {**self._cache_status, "write_error": "cache write failed"}
                LOG.warning("Discovery cache write failed; continuing with in-memory discovery")

    def start(self) -> None:
        """Start scheduled discovery. Idempotent; serve() calls this after bind."""
        with self._worker_lock:
            if self._closed.is_set():
                raise RuntimeError("Cannot start a closed router")
            if self._refresh_worker is not None:
                return
            if not any(source and source.refresh_seconds > 0
                       for source in (self.config.catalog, self.config.llama_cpp)):
                return
            self._worker_stop.clear()
            self._refresh_worker = threading.Thread(
                target=self._run_refresh_worker, name="model-router-discovery", daemon=False,
            )
            try:
                self._refresh_worker.start()
            except BaseException:
                self._refresh_worker = None
                raise

    def stop(self) -> None:
        """Stop scheduling and join active discovery before returning.

        An active HTTP probe finishes under its configured timeout. start() can
        resume scheduling after stop(); close() permanently releases resources.
        """
        with self._worker_lock:
            self._worker_stop.set()
            self._refresh_wakeup.set()
            if self._refresh_worker is not None:
                self._refresh_worker.join()
                self._refresh_worker = None

    def _run_refresh_worker(self) -> None:
        while not self._worker_stop.is_set():
            self._refresh_wakeup.clear()
            for refresh in (self.refresh_llama_cpp, self.refresh_catalog):
                if self._worker_stop.is_set():
                    return
                refresh()
            snapshot = self._registry
            # A manual refresh may own a source. Its completion wakes us;
            # exclude its old deadline so an overdue attempt cannot spin here.
            deadlines = [state.refresh.next_attempt
                         for state, lock in ((snapshot.catalog, self._catalog_lock),
                                             (snapshot.llama, self._llama_lock))
                         if state.refresh.next_attempt is not None and not lock.locked()]
            delay = max(0.0, min(deadlines) - self._clock()) if deadlines else None
            self._refresh_wakeup.wait(
                min(delay, threading.TIMEOUT_MAX) if delay is not None else None
            )

    def _next_refresh(
        self, previous: RefreshState, interval: float, attempted: float, *, failed: bool,
    ) -> RefreshState:
        completed = self._clock()
        failures = previous.consecutive_failures + 1 if failed else 0
        if failed:
            ceiling = min(REFRESH_BACKOFF_MAX,
                          REFRESH_BACKOFF_BASE * 2 ** min(failures - 1, 16))
            delay = self._jitter(ceiling / 2, ceiling)
        else:
            delay = interval
        return RefreshState(
            last_success=previous.last_success if failed else completed,
            last_attempt=attempted,
            next_attempt=completed + delay if interval > 0 else None,
            consecutive_failures=failures,
        )

    def _refresh_status(
        self, state: RefreshState, interval: float, error: str | None, now: float,
    ) -> dict[str, Any]:
        age = max(0.0, now - state.last_success) if state.last_success is not None else None
        return {
            "last_success": state.last_success,
            "last_attempt": state.last_attempt,
            "next_attempt": state.next_attempt,
            "consecutive_failures": state.consecutive_failures,
            "age_seconds": age,
            "stale": state.last_success is None or error is not None
                     or (interval > 0 and age is not None and age >= interval),
            "stale_age_seconds": (max(0.0, age - interval) if interval > 0 else 0.0)
                                 if age is not None else None,
        }

    def _publish_registry(
        self, *, catalog: CatalogSnapshot | None = None,
        llama: LlamaSnapshot | None = None,
    ) -> None:
        """Merge the latest source generations and publish one immutable value.

        Discovery source locks serialize their own network work. This short
        lock prevents simultaneous catalog/local publications losing updates;
        request dispatch never holds it, including during upstream I/O.
        """
        with self._registry_lock:
            previous = self._registry
            catalog = catalog if catalog is not None else previous.catalog
            llama = llama if llama is not None else previous.llama
            providers = {**self.config.providers, **llama.providers}
            bound: dict[int, Route] = {}

            def bind(route: Route) -> Route:
                if id(route) not in bound:
                    bound[id(route)] = replace(
                        route, _provider=providers[route.provider_name]
                    )
                return bound[id(route)]

            routes = {**catalog.routes, **llama.routes, **self.config.routes}
            public: list[Route] = []
            # Explicit aliases also reserve names against public discovery.
            names: set[str] = set()
            for source, entries in (
                (self.config.routes, self.config.public_routes),
                (llama.routes, llama.public),
                (catalog.routes, catalog.routes.values()),
            ):
                for route in entries:
                    if route.enabled and route.name not in names:
                        public.append(bind(route))
                names.update(source)
            self._registry = RegistrySnapshot(
                providers=_freeze(providers),
                routes=_freeze({name: bind(route) for name, route in routes.items()}),
                public=tuple(public), catalog=catalog, llama=llama,
            )

    def authorized(self, headers: Mapping[str, str], query_keys: Iterable[str] = ()) -> bool:
        """Router client authentication.

        ``query_keys`` are the values of a ``?key=`` parameter; only the Gemini
        endpoints pass them, since that is how Gemini clients may send their key.
        """
        if not self.config.api_keys:
            return True
        candidates: list[str] = []
        authorization = headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        for name in ("x-api-key", "api-key", "x-goog-api-key"):
            value = headers.get(name)
            if value:
                candidates.append(value.strip())
        candidates.extend(value.strip() for value in query_keys if value and value.strip())
        if not candidates:
            return False
        return any(
            hmac.compare_digest(candidate.encode("utf-8"), key.encode("utf-8"))
            for candidate in candidates
            for key in self.config.api_keys
        )

    def _catalog_is_stale(self) -> bool:
        deadline = self._registry.catalog.refresh.next_attempt
        return deadline is not None and self._clock() >= deadline

    def refresh_catalog(self, *, force: bool = False) -> None:
        """Explicit refresh; overlapping calls return immediately, even if forced."""
        catalog = self.config.catalog
        if catalog is None or not self._catalog_lock.acquire(blocking=False):
            return
        try:
            if self._closed.is_set() or (not force and not self._catalog_is_stale()):
                return
            previous = self._registry.catalog
            attempted = self._clock()
            try:
                self._load_catalog(catalog, previous, attempted)
            except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                error = _error_message(exc, self.sanitize)
                self._publish_registry(catalog=replace(
                    previous, error=error,
                    refresh=self._next_refresh(previous.refresh, catalog.refresh_seconds,
                                               attempted, failed=True),
                ))
                LOG.warning("Model catalog refresh failed: %s", error)
            self._log_refresh("catalog", attempted)
        finally:
            self._catalog_lock.release()
            if threading.current_thread() is not self._refresh_worker:
                self._refresh_wakeup.set()

    def _load_catalog(
        self, catalog: CatalogConfig, previous: CatalogSnapshot, attempted: float,
    ) -> None:
        request_headers = {"User-Agent": f"litellm-model-router/{__version__}"}
        request_headers.update(catalog.headers)
        response = self.http.get(
            catalog.url,
            headers=request_headers,
            timeout=httpx.Timeout(catalog.timeout, connect=catalog.timeout),
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("catalog response has no model list")

        routes: dict[str, Route] = {}
        unmapped: set[str] = set()
        for row in rows:
            if isinstance(row, Mapping):
                model_id = row.get("id")
                metadata = row.get("metadata")
            else:
                model_id = row
                metadata = None
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            provider_name = self._catalog_provider_name(model_id)
            if provider_name is None:
                unmapped.add(model_id)
                continue
            provider = self.config.providers[provider_name]
            capabilities = (router_metadata.go(model_id, provider.dialect)
                            if catalog.metadata_source == "opencode-go-v1"
                            else router_metadata.empty(provider.dialect))
            capabilities["sources"]["catalog"] = {
                "retrieved_at": time.time(), "verification": "availability only",
            }
            routes[model_id] = Route(
                name=model_id,
                provider_name=provider_name,
                upstream_model=model_id,
                dialect=provider.dialect,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                _capabilities=capabilities,
            )

        # A valid catalog containing only unreviewed IDs is a successful
        # empty routing inventory, not a reason to retain removed routes.
        self._publish_registry(catalog=CatalogSnapshot(
            routes=_freeze(routes), unmapped=frozenset(unmapped),
            refresh=self._next_refresh(previous.refresh, catalog.refresh_seconds,
                                       attempted, failed=False),
            succeeded_at=self._wall_clock(),
        ))
        self._write_discovery_cache()
        unmapped = set(self.catalog_status()["unmapped_models"])
        if unmapped:
            LOG.warning(
                "Catalog has %d model(s) with unknown endpoint mapping; "
                "omitted from /v1/models: %s. Configure catalog.model_providers "
                "or an explicit [[models]] route after reviewing the endpoint.",
                len(unmapped), ", ".join(sorted(unmapped)),
            )
        LOG.info(
            "Model catalog loaded %d models from %s", len(routes), catalog.url
        )

    def _catalog_provider_name(self, model_id: str) -> str | None:
        catalog = self.config.catalog
        assert catalog is not None
        name = catalog.model_providers.get(model_id)
        return name if name is not None else next(
            (r.provider_name for r in catalog.rules if r.pattern.search(model_id)), None)

    def catalog_status(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        state = snapshot.catalog
        catalog = self.config.catalog
        return {
            "enabled": catalog is not None,
            "url": self.sanitize(catalog.url) if catalog else None,
            "models": len(state.routes),
            "loaded": state.refresh.last_success is not None,
            "error": state.error,
            "cached": state.cached,
            "succeeded_at": state.succeeded_at,
            "refresh": self._refresh_status(state.refresh, catalog.refresh_seconds if catalog else 0,
                                            state.error, self._clock()),
            "unmapped_models": sorted(
                state.unmapped - snapshot.routes.keys()
            ),
        }

    def provider(self, name: str) -> Provider:
        """Look up a provider in the current snapshot (resolved routes pin theirs)."""
        return self._registry.providers[name]

    def provider_count(self) -> int:
        return len(self._registry.providers)

    def _llama_is_stale(self) -> bool:
        deadline = self._registry.llama.refresh.next_attempt
        return deadline is not None and self._clock() >= deadline

    def _probe_llama_server(
        self, llama: LlamaCppConfig, server: LlamaCppServer
    ) -> dict[str, Any]:
        """Fetch /health, /v1/models and /props from one llama.cpp server."""
        headers = {"User-Agent": f"litellm-model-router/{__version__}"}
        headers.update(llama.headers)
        headers.update(server.headers)
        if server.api_key:
            headers["Authorization"] = "Bearer " + server.api_key
        timeout = httpx.Timeout(
            llama.probe_timeout, connect=llama.probe_timeout
        )
        base = server.url.rstrip("/")
        try:
            health = self.http.get(base + "/health", headers=headers, timeout=timeout)
            if health.status_code >= 400:
                raise RuntimeError(f"/health returned HTTP {health.status_code}")
            response = self.http.get(
                base + "/v1/models", headers=headers, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            return {"server": server, "error": _error_message(exc, self.sanitize)}

        try:
            props_response = self.http.get(
                base + "/props",
                params={"autoload": "false"},
                headers=headers,
                timeout=timeout,
            )
            props = (
                props_response.json()
                if props_response.status_code < 400
                else {}
            )
        except Exception:  # noqa: BLE001 - props is optional
            props = {}

        if isinstance(payload, Mapping):
            rows = payload.get("data")
            if not isinstance(rows, list):
                rows = payload.get("models")
        else:
            rows = payload
        if not isinstance(rows, list):
            return {"server": server, "error": "response has no model list"}

        rows = [row for row in rows if isinstance(row, Mapping)]
        model_props: dict[str, Mapping[str, Any]] = {}
        if _llama_mode(props, rows) == "router":
            for row in rows:
                model_id = row.get("id")
                status = row.get("status")
                if (not isinstance(model_id, str) or not model_id.strip()
                        or not isinstance(status, Mapping)
                        or status.get("value") not in ("loaded", "sleeping")):
                    continue
                try:
                    response = self.http.get(
                        base + "/props", params={"model": model_id, "autoload": "false"},
                        headers=headers, timeout=timeout,
                    )
                    if response.status_code < 400:
                        value = response.json()
                        # Older routers may ignore the model query and return
                        # their global router props. Those are not model data.
                        if (isinstance(value, Mapping)
                                and value.get("role") != "router"
                                and value.get("src") != "llama-swap"):
                            model_props[model_id.strip()] = value
                except Exception:  # noqa: BLE001 - older servers may lack per-model props
                    pass

        return {
            "server": server,
            "rows": rows,
            "props": props,
            "model_props": model_props,
        }

    def _log_llama_probe(
        self, server: LlamaCppServer, error: str | None, models: int
    ) -> None:
        """Log reachability changes without spamming every refresh."""
        if error:
            if server.url not in self._llama_unreachable:
                LOG.warning("llama.cpp server %s unavailable: %s", server.url, error)
            else:
                LOG.debug("llama.cpp server %s still unavailable: %s", server.url, error)
            self._llama_unreachable.add(server.url)
            return
        was_unreachable = server.url in self._llama_unreachable
        previous = self._llama_counts.get(server.url)
        if was_unreachable or previous != models:
            LOG.info(
                "llama.cpp server %s ready with %d model%s",
                server.url,
                models,
                "" if models == 1 else "s",
            )
        self._llama_unreachable.discard(server.url)
        self._llama_counts[server.url] = models

    def refresh_llama_cpp(self, *, force: bool = False) -> None:
        """Probe every configured llama.cpp server and rebuild its routes."""
        llama = self.config.llama_cpp
        if llama is None:
            return
        if not self._llama_lock.acquire(blocking=False):
            return
        try:
            if self._closed.is_set() or (not force and not self._llama_is_stale()):
                return
            attempted = self._clock()
            previous = self._registry.llama
            try:
                self._load_llama_cpp(llama, previous, attempted)
            except Exception as exc:  # noqa: BLE001 - retain last usable inventory
                error = _error_message(exc, self.sanitize)
                self._publish_registry(llama=replace(
                    previous, error=error,
                    refresh=self._next_refresh(previous.refresh, llama.refresh_seconds,
                                               attempted, failed=True),
                ))
                LOG.warning("llama.cpp discovery failed: %s", error)
            self._log_refresh("llama_cpp", attempted)
        finally:
            self._llama_lock.release()
            if threading.current_thread() is not self._refresh_worker:
                self._refresh_wakeup.set()

    def _load_llama_cpp(
        self, llama: LlamaCppConfig, previous_state: LlamaSnapshot, attempted: float,
    ) -> None:
        outcomes: list[dict[str, Any] | None] = [None] * len(llama.servers)
        if llama.servers:
            workers = min(8, len(llama.servers))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._probe_llama_server, llama, server): index
                    for index, server in enumerate(llama.servers)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        outcomes[index] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        outcomes[index] = {
                            "server": llama.servers[index],
                            "error": _error_message(exc, self.sanitize),
                        }

        explicit_names = set(self.config.routes)
        providers: dict[str, Provider] = {}
        routes: dict[str, Route] = {}
        public: list[Route] = []
        by_server: dict[
            str, tuple[Provider, Mapping[str, Route], tuple[Route, ...]]
        ] = {}
        states: list[dict[str, Any]] = []
        errors: list[str] = []
        reachable = 0
        previous_servers = {server["url"]: server for server in previous_state.servers}

        for index, server in enumerate(llama.servers):
            outcome = outcomes[index]
            if outcome is None or outcome.get("error"):
                error = self.sanitize(
                    outcome["error"]
                    if outcome and outcome.get("error")
                    else "probe failed"
                )
                # Keep the last good routes for this server during a
                # transient failure instead of dropping them.
                previous = previous_state.by_server.get(server.url)
                if previous is not None:
                    old_provider, old_routes, old_public = previous
                    providers[old_provider.name] = old_provider
                    for public_name, route in old_routes.items():
                        if (
                            public_name not in explicit_names
                            and public_name not in routes
                        ):
                            routes[public_name] = route
                    known = {route.name for route in public}
                    for route in old_public:
                        if (
                            route.name not in explicit_names
                            and route.name not in known
                        ):
                            public.append(route)
                            known.add(route.name)
                    by_server[server.url] = previous
                    model_count = len(old_public)
                else:
                    model_count = 0
                self._log_llama_probe(server, error, model_count)
                errors.append(self.sanitize(f"{server.url}: {error}"))
                states.append(
                    {
                        "url": server.url,
                        "reachable": False,
                        "mode": None,
                        "provider": self._llama_names[server.url],
                        "models": model_count,
                        "stale": previous is not None,
                        "error": error,
                        "last_success": previous_servers.get(server.url, {}).get("last_success"),
                        "last_attempt": attempted,
                        "consecutive_failures": previous_servers.get(server.url, {}).get(
                            "consecutive_failures", 0) + 1,
                        "cached": previous_servers.get(server.url, {}).get("cached", False),
                        "succeeded_at": previous_servers.get(server.url, {}).get("succeeded_at"),
                    }
                )
                continue

            reachable += 1
            rows = outcome["rows"]
            props = outcome["props"]
            mode = _llama_mode(props, rows)
            engine = router_metadata.local_engine(props)
            provider_name = self._llama_names[server.url]
            provider = self._llama_provider(server, engine=engine)
            providers[provider_name] = provider
            server_routes: dict[str, Route] = {}
            server_public: list[Route] = []
            alias_candidates: list[tuple[list[str], Route]] = []

            for row in rows:
                model_id = row.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                model_id = model_id.strip()
                aliases = _llama_model_aliases(row)
                display_name = f"Local ({engine})" if model_id == "local" else (aliases[0] if aliases else model_id)
                model_props = outcome.get("model_props", {})
                per_model_props = model_id in model_props
                limits = _llama_limits(
                    row, model_props[model_id] if per_model_props else props,
                    mode, llama.context_fallback, llama.limit_overrides.get(model_id, {}),
                    per_model_props=per_model_props,
                )
                context = limits["context_window"]["value"]
                capabilities = _llama_capabilities(row, mode)
                status = row.get("status")
                status_value = (
                    status.get("value") if isinstance(status, Mapping) else None
                )
                if status_value is None and mode in ("single", "legacy"):
                    # These servers only ever run one loaded model.
                    status_value = "loaded"
                metadata = (
                    dict(row["metadata"])
                    if isinstance(row.get("metadata"), Mapping)
                    else {}
                )
                metadata.update(
                    {
                        "engine": engine,
                        "display_name": display_name,
                        "context_window": context,
                        # Compatibility alias: maximum generation length only.
                        # null means unknown, never the context or a default.
                        "max_tokens": limits["max_output_tokens"]["value"],
                        "limits": limits,
                        "capabilities": capabilities,
                        "llama_cpp": {
                            "server": server.url,
                            "mode": mode,
                            "status": status_value,
                            "n_ctx": context,
                        },
                    }
                )
                for key in LLAMA_LIMIT_FIELDS:
                    metadata[key] = limits[key]["value"]
                public_name = llama.prefix + ("local-llama" if model_id == "local" else model_id)
                if (
                    public_name in explicit_names
                    or public_name in routes
                    or public_name in server_routes
                ):
                    continue
                route = Route(
                    name=public_name,
                    provider_name=provider_name,
                    upstream_model=model_id,
                    dialect=DIALECT_OPENAI_CHAT,
                    metadata=metadata,
                    _capabilities=router_metadata.llama(
                        row, model_props[model_id] if per_model_props else props,
                        mode, limits, time.time(), per_model_props=per_model_props, engine=engine,
                    ),
                )
                server_routes[public_name] = route
                routes[public_name] = route
                public.append(route)
                server_public.append(route)
                if llama.expose_aliases and aliases:
                    alias_candidates.append((aliases, route))

            for aliases, route in alias_candidates:
                for alias in aliases:
                    public_alias = llama.prefix + alias
                    if (
                        public_alias in explicit_names
                        or public_alias in routes
                        or public_alias in server_routes
                    ):
                        continue
                    server_routes[public_alias] = route
                    routes[public_alias] = route

            by_server[server.url] = (
                provider,
                server_routes,
                tuple(server_public),
            )
            self._log_llama_probe(server, None, len(server_public))
            states.append(
                {
                    "url": server.url,
                    "reachable": True,
                    "mode": mode,
                    "provider": provider_name,
                    "models": len(server_public),
                    "stale": False,
                    "error": None,
                    "last_success": self._clock(),
                    "last_attempt": attempted,
                    "consecutive_failures": 0,
                    "cached": False,
                    "succeeded_at": self._wall_clock(),
                }
            )

        self._publish_registry(llama=LlamaSnapshot(
            providers=_freeze(providers), routes=_freeze(routes),
            public=tuple(public), by_server=_freeze(by_server),
            servers=_freeze(states),
            refresh=self._next_refresh(previous_state.refresh, llama.refresh_seconds,
                                       attempted, failed=bool(errors)),
            error=self.sanitize("; ".join(errors)) if errors else None,
        ))
        if reachable or not llama.servers:
            self._write_discovery_cache()
        LOG.debug(
            "llama.cpp discovery: %d server(s) reachable, %d model route(s)",
            reachable,
            len(public),
        )

    def llama_cpp_status(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        state = snapshot.llama
        llama = self.config.llama_cpp
        now = self._clock()
        interval = llama.refresh_seconds if llama else 0

        def server_status(server: Mapping[str, Any]) -> dict[str, Any]:
            freshness = self._refresh_status(
                RefreshState(last_success=server["last_success"]), interval, server["error"], now,
            )
            return {**server, "url": self.sanitize(server["url"]),
                    "age_seconds": freshness["age_seconds"],
                    "stale_age_seconds": freshness["stale_age_seconds"]}

        return {
            "enabled": llama is not None,
            "servers": [server_status(server) for server in state.servers],
            "models": len(state.public),
            "loaded": bool(state.by_server) or state.refresh.last_success is not None,
            "error": state.error,
            "refresh": self._refresh_status(state.refresh, interval, state.error, now),
        }

    def _route_entry(self, route: Route, now: int, snapshot: RegistrySnapshot) -> dict[str, Any]:
        provider = route._provider
        assert provider is not None, "public routes must be bound to their provider"
        if route.provider_name in snapshot.llama.providers:
            server = next((s for s in snapshot.llama.servers if s["provider"] == route.provider_name), {})
            interval = self.config.llama_cpp.refresh_seconds if self.config.llama_cpp else 0
            freshness = self._refresh_status(
                RefreshState(last_success=server.get("last_success")), interval,
                server.get("error"), self._clock(),
            )
            freshness["cached"] = server.get("cached", False)
            freshness["succeeded_at"] = server.get("succeeded_at")
        elif route.name in snapshot.catalog.routes and route.name not in self.config.routes:
            interval = self.config.catalog.refresh_seconds if self.config.catalog else 0
            freshness = self._refresh_status(snapshot.catalog.refresh, interval,
                                             snapshot.catalog.error, self._clock())
            freshness["cached"] = snapshot.catalog.cached
            freshness["succeeded_at"] = snapshot.catalog.succeeded_at
        else:
            freshness = {"stale": None, "age_seconds": None}
        # Upstream metadata.router is never the router's own capability
        # contract. Only reviewed reports and trusted sparse overrides feed it.
        contract = router_metadata.render(
            _thaw(route._capabilities) if route._capabilities else router_metadata.empty(route.dialect),
            _thaw(self.config.capability_overrides.get(route.name, {})), freshness, now,
        )
        entry: dict[str, Any] = {
            "id": route.name,
            "object": "model",
            "type": "model",
            "created": now,
            "created_at": now,
            "owned_by": route.provider_name,
            "display_name": MODEL_DISPLAY_NAMES.get(
                route.name, _thaw(route.metadata.get("display_name", route.name))
            ),
            "metadata": {
                **_thaw(route.metadata),
                "display_name": MODEL_DISPLAY_NAMES.get(
                    route.name, _thaw(route.metadata.get("display_name", route.name))
                ),
                "provider": route.provider_name,
                "upstream_model": route.upstream_model,
                "dialect": route.dialect,
                "provider_dialect": provider.dialect,
                "router": contract,
            },
        }
        for key in ("context_window", "max_tokens"):
            if key in route.metadata:
                entry[key] = _thaw(route.metadata[key])
        return entry

    def models_payload(self) -> dict[str, Any]:
        return self._models_payload(self._registry)

    def gemini_models_payload(self, page_size: str | None = None,
                              page_token: str | None = None) -> dict[str, Any]:
        """models.list: every public route as a Gemini Model resource."""
        return gemini.models_page(self.models_payload()["data"], page_size, page_token)

    def gemini_model_payload(self, name: str) -> dict[str, Any] | None:
        """models.get for a listed route; unlisted (dynamic) names are not exposed."""
        for entry in self.models_payload()["data"]:
            if entry["id"] == name:
                return gemini.model_resource(entry)
        return None

    def muse_models_payload(self) -> dict[str, Any]:
        """Project the full cached inventory into Muse's Responses catalog."""
        payload = self.models_payload()
        data = payload["data"]
        for entry in data:
            metadata = entry["metadata"]
            limits = metadata["router"]["limits"]
            metadata["muse-code"] = {
                "name": entry["display_name"],
                "display_label": entry["display_name"],
                "is_hidden": False,
                "limit": {
                    "context": limits["context_window"]["value"] or 128000,
                    "output": limits["max_output_tokens"]["value"] or 16384,
                },
            }
            reasoning = metadata["router"]["reasoning"]
            muse = metadata["muse-code"]
            muse["reasoning"] = reasoning["supported"]["value"] is True
            efforts = reasoning["effort"]["values"]["value"]
            if isinstance(efforts, list):
                muse["variants"] = {
                    effort: {"reasoningEffort": effort, "selector": {"description": effort}}
                    for effort in efforts
                }
                if efforts:
                    default = reasoning["effort"]["default"]["value"]
                    if default not in efforts:
                        default = "high" if "high" in efforts else efforts[0]
                    muse["options"] = {"reasoningEffort": default}
        return {
            **payload,
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def _models_payload(self, snapshot: RegistrySnapshot) -> dict[str, Any]:
        now = int(time.time())
        data = [self._route_entry(route, now, snapshot) for route in snapshot.public]
        return {
            "object": "list",
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def liveness_payload(self) -> dict[str, str]:
        # Deliberately independent of inventory, locks, HTTP clients and clocks.
        return {"status": "ok"}

    def readiness_payload(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        unavailable = {server["provider"] for server in snapshot.llama.servers
                       if not server["reachable"]}
        enabled = [route for route in snapshot.routes.values() if route.enabled]
        usable = sum(route.provider_name not in unavailable for route in enabled)
        return {"status": "ready" if usable else "not_ready", "ready": bool(usable),
                "enabled_routes": len(enabled), "usable_routes": usable,
                "reason": "usable_routes" if usable else "no_usable_named_routes"}

    def discovery_summary(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        now = self._clock()
        result = {}
        for name, config, state in (("catalog", self.config.catalog, snapshot.catalog),
                                    ("llama_cpp", self.config.llama_cpp, snapshot.llama)):
            if config is not None:
                status = self._refresh_status(state.refresh, config.refresh_seconds, state.error, now)
                result[name] = {key: status[key] for key in (
                    "stale", "age_seconds", "stale_age_seconds", "consecutive_failures")}
        return result

    def log_event(self, **fields: Any) -> None:
        # Remain safe/useful if a finishing handler logs after App.close() has
        # detached its filters. Filters for other live Apps may redact further.
        event = _sanitize_details(fields, self.sanitize)
        LOG.info(json.dumps(event, separators=(",", ":"), ensure_ascii=True),
                 extra={"router_event": event})

    def _log_refresh(self, source: str, attempted: float) -> None:
        self.log_event(event="discovery_refresh", source=source,
                       elapsed_ms=round(max(0.0, self._clock() - attempted) * 1000, 3),
                       discovery=self.discovery_summary())

    def diagnostics_payload(self) -> dict[str, Any]:
        # One snapshot for the whole report. No headers, credentials, transport
        # kwargs, request bodies or model metadata are included in this surface.
        snapshot = self._registry
        unavailable = {server["provider"] for server in snapshot.llama.servers
                       if not server["reachable"]}
        return _sanitize_details({
            "status": "ok",
            "providers": len(snapshot.providers),
            "models": len(snapshot.public),
            "readiness": self.readiness_payload(snapshot),
            "provider_details": [
                {"name": provider.name, "dialect": provider.dialect,
                 "reachability": "unavailable" if name in unavailable else
                                 "available" if name in snapshot.llama.providers else "unprobed",
                 "enabled_routes": sum(route.enabled and route.provider_name == name
                                       for route in snapshot.routes.values())}
                for name, provider in sorted(snapshot.providers.items())
            ],
            "catalog": self.catalog_status(snapshot),
            "llama_cpp": self.llama_cpp_status(snapshot),
            "discovery_cache": dict(self._cache_status),
        }, self.sanitize)

    def health_payload(self) -> dict[str, Any]:
        """Legacy embedder API; HTTP details now require /diagnostics auth."""
        return self.diagnostics_payload()

    def resolve(self, model: str) -> Route | None:
        snapshot = self._registry
        route = snapshot.routes.get(model)
        if route is not None:
            return route if route.enabled else None
        if model in snapshot.catalog.unmapped:
            raise RouterError(
                404,
                f"Model {model!r} is listed by the catalog but its endpoint compatibility "
                "is unknown. Configure catalog.model_providers or an explicit "
                "[[models]] route after reviewing the provider endpoint.",
                code="model_endpoint_unknown",
            )
        # Dynamic "provider/model" routing is a convenience for people who do
        # not want to list every model explicitly.
        if self.config.allow_dynamic_routes and "/" in model:
            provider_name, _, upstream = model.partition("/")
            provider = snapshot.providers.get(provider_name)
            if provider is not None and upstream.strip():
                return Route(
                    name=model,
                    provider_name=provider_name,
                    upstream_model=upstream.strip(),
                    dialect=provider.dialect,
                    _provider=provider,
                    _dynamic=True,
                )
        return None

    def close(self) -> None:
        """Join discovery (including manual refreshes), then release transports."""
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self.stop()
            with self._catalog_lock, self._llama_lock:
                try:
                    self.http.close()
                finally:
                    for logger in self._diagnostic_loggers:
                        logger.removeFilter(self._log_filter)
